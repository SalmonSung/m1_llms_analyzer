# Design decisions

Choices made while building this that were **not** discussed, with the reasoning behind
each. Decisions that *were* agreed (in-process services, pooled-by-default output, JSON +
optional sidecar, Colab-secret auth, tiny-then-real default models, layer selection built
now, and the four extras) are recorded in `architecture.md` and the README instead.

If you disagree with anything here, it is meant to be cheap to reverse — each entry names
the file to change.

---

## Architecture

### Protocols (`typing.Protocol`) instead of ABCs
`services/interfaces.py` defines structural types, not base classes. A test double or a
future HTTP client satisfies `ModelProvider` by having the right methods; it does not have
to import and subclass anything. That keeps the services genuinely decoupled — `services/`
has no inheritance graph to reason about — and it is what makes the in-process design
liftable into real HTTP services later.

### A facade (`Analyzer`) on top of the container
Three services and a config object is more surface than a notebook cell wants. `Analyzer`
is the single object the notebook touches, and it owns the loaded model for the session so
a large model is downloaded and placed on the GPU exactly once. The services stay
independently constructible; `container.py` only does wiring, and each dependency can be
injected (`Analyzer(storage_service=...)`).

### Dataclasses, not YAML/Hydra
The notebook *is* the config surface. Dataclasses give tab-completion, `?` docstrings,
type checking, and `__post_init__` validation with no extra dependency and no config file
to keep in sync with a notebook cell. Add YAML only if runs ever need to be launched from
outside Python.

### `AutoModel` by default; `AutoModelForCausalLM` as an explicit opt-in (`head`)
Hidden-state extraction does not need the LM head, which is a `vocab × hidden` matrix —
hundreds of MB on a large-vocab model — so `head="base"` loads `AutoModel`. The experiments
need log-probabilities, so `ModelConfig(head="causal_lm")` loads `AutoModelForCausalLM`
through the same token / revision / dtype / pad-token path. A causal-LM model still returns
hidden states, so extraction works with either head; scoring works only with the causal one
and says so (`Analyzer.for_scoring` is the shorthand). The head is validated against
transformers' causal-LM registry at load time, so an encoder fails with a message rather
than a missing `logits` attribute mid-run.

### Validation at construction, not at use
Every config dataclass validates in `__post_init__`. A typo like `pooling="max"` fails on
the config line, not thirty seconds later inside a forward pass with a `KeyError`. In a
notebook, where the failure may arrive after a multi-minute model download, this is the
difference between a one-second fix and a repeated download.

---

## Model loading

### Layer index 0 is the embedding output
`hidden_states` has length `num_hidden_layers + 1`; index 0 is the embedding output, not
block 1. Rather than hide that with an off-by-one shim, the convention is exposed
directly and written into every output file (`layer_index_convention`), along with both the
requested label and the resolved absolute index. Silent renumbering would make a saved
`"layer 12"` mean different things in different tools.

### Encoder-decoder models are rejected, not silently handled
For T5/BART, `outputs.hidden_states` is the **encoder** stack; the decoder's are under
`decoder_hidden_states`. Saving the former as "the last layer" would be quietly wrong, and
quietly wrong is worse than unsupported. `model_service._reject_unsupported` raises with an
explanation. Supporting them properly means a config choice of which stack to read.

### `pad_token = eos_token` when a tokenizer has none
GPT-2-family tokenizers ship no PAD token, which makes batching impossible. Reusing EOS is
the standard fix and is safe here because the attention mask excludes those positions from
both the forward pass and pooling. The vocabulary is **not** resized, so the model's
embedding matrix is untouched. The substitution is logged and recorded in the manifest as
`pad_token_substituted`.

### bfloat16 preferred over float16 on GPUs that run it natively
Same speed, but fp32's exponent range. Deep residual streams in large models can produce
activations that overflow fp16 to `inf`; bf16 does not. CPU always gets fp32 — many
kernels have no fp16 CPU implementation and those that do are slower than fp32. The check
is `torch.cuda.is_bf16_supported(including_emulation=False)`: since torch 2.3 the bare
call answers *True* on a Turing GPU (a Colab T4) because bf16 can be emulated, and emulated
bf16 is several times slower than fp16. A T4 therefore gets fp16, and the scorer treats a
resulting overflow as a failure with a `dtype="float32"` hint rather than a number.

### Hidden states are always cast to float32 before leaving the device
The compute dtype is a performance choice; it should not change what a saved file
contains. Casting to fp32 on the way out means a bf16 run on an A100 and an fp32 run on a
CPU produce comparable numbers, and JSON never has to represent a bf16 value.

### `trust_remote_code` defaults to `False`
It executes arbitrary Python from a Hub repo at load time. Opt-in per model, with an error
message that says exactly that when a model needs it.

### Hub errors are translated, not re-raised raw
`GatedRepoError` and a bare 401 are useless from a notebook. `_explain_load_failure` turns
them into numbered instructions naming the exact Colab secret to add and the licence page
to accept. This is the most likely first-run failure, so it gets the best error message.

---

## Inference

### `invoke()` is `batch()` of one
One code path, so the two APIs cannot drift in pooling, truncation, dtype, or masking.
`test_invoke_and_batch_agree_exactly` pins this. `invoke` raises on failure while `batch`
collects failures, because a single call has no partial-success story to tell.

### Last-token pooling scans the mask from the right
The common idiom `hidden[torch.arange(n), attention_mask.sum(1) - 1]` is correct **only**
for right padding; on a left-padded batch it lands on the *first* real token. This code
uses `mask.flip(1).argmax(1)` to find the last real token from the right, which is correct
under either padding side. This is not hypothetical — it was caught by
`test_last_token_ignores_padding_side` during development, and
`test_naive_last_index_would_be_wrong` now guards the naive alternative.

### Inputs are length-sorted before batching, then restored to input order
Padding every batch to its own longest member wastes compute proportional to the length
spread. Sorting groups similar lengths together; the original indices are carried through
so the returned records are in input order. Off via `sort_by_length=False`.

### OOM halves the batch instead of aborting
A CUDA OOM three hours into a run is expensive. `_run_chunk` catches it, halves, empties
the cache, and retries down to size 1; only a single input that still OOMs becomes a
failure, with a hint about `max_length` and pooling. Colab GPU memory varies by session,
so a batch size that worked yesterday can fail today — the pipeline absorbs that.

### One bad item does not kill a run
Per-item exceptions become `ExtractionFailure` records that travel with the results and
land in the output JSON. Losing 9,999 good results to item 5,000 is not acceptable
behaviour for a batch API. Set `continue_on_error=False` to get the old behaviour.

### Untruncated length is probed with a second tokenizer call
Detecting truncation requires knowing the length that *would* have been produced. That
costs one extra (cheap, CPU-side) tokenizer pass per input. Worth it: silently analysing a
clipped prompt is a wrong result that looks like a right one. Both counts land in the
record (`token_count`, `original_token_count`, `truncated`).

### Item ids are `sha256(text)[:16]`, not positional indices
Content-derived ids are stable across runs and files, which makes two result files
joinable and duplicates detectable. 16 hex characters (64 bits) is ample here. A positional
index would break the moment the input list is reordered or filtered.

### Empty and whitespace-only inputs are rejected, not silently pooled
They tokenize to zero real tokens, so there is nothing to pool — the result would be a
degenerate or garbage vector. An explicit `ValueError` naming the offending index is
better than a plausible-looking number.

### `max_length_cap` (default 4096) on top of the model's context
A 128k-context model would otherwise let a single long input allocate an enormous
activation tensor and OOM a Colab GPU instantly. The cap is a guard rail, logged when it
bites, and raised in config when you actually want long contexts.

---

## Scoring (log-probabilities)

### A BOS token is prepended uniformly, by hand
The cost of a span is a difference of mean log-probabilities per *predicted* token. Without
a start token the first word is never predicted, and tokenizers disagree about adding one:
Llama's does, GPT-2's and Qwen's do not (they use `<|endoftext|>` as the document separator,
so prepending it conditions on "start of document", which is what the pretraining data
looked like). Tokenising with `add_special_tokens=False` and prepending `bos_token_id`
(else `eos_token_id`) gives every model the same treatment and every real token a
prediction. `bos_policy="none"` scores the text as given; the token used is recorded in the
cost cache header.

### Padding is forced to the right
Some models (Qwen3 among them) build position ids from `arange` when none are passed, so a
left-padded batch shifts every real token's position and changes its score. The scorer
builds its own right-padded batch instead of trusting the tokenizer's `padding_side`.

### `gather − logsumexp`, never a full `log_softmax`
A 64-sequence batch of 40 tokens over Qwen's 152k vocabulary is 1.5 GB of float32 for
`log_softmax` alone, which is what runs a T4 out of memory. The scorer takes the target
logit and the log-partition per position, in float32, over small sub-chunks
(`ScoringConfig.logit_chunk`). HF's `labels=` is not used: its loss is a batch mean, not a
per-sequence sum.

### Non-finite scores are failures, not numbers
A float16 overflow inside the model yields `inf`/`nan` for that text. It is recorded as a
failure with a hint, never cached, and phase A counts it, so a run with overflow is visible
and re-runnable in fp32 rather than silently wrong.

### Raw sums are cached, not costs
`SpanCostTable` stores `(sum_logprob, n_tokens)` for the base text and each proform × span.
Mean-per-token vs total normalisation, and the policy that reduces per-proform costs to
one, are then phase-B choices that cost no GPU time. Mean-per-token is the default because
a variant is shorter than the sentence it came from and totals carry that length bias.

### The cost cache is JSONL, appended and fsynced
Colab pre-empts free sessions in one to two hours; phase A on 1000 sentences takes about
half an hour on a T4 and hours on a CPU. One line per finished sentence, flushed and
fsynced, means a disconnect loses at most one sentence; on resume the header must match on
model, proforms, and BOS policy (the mismatching field is named), a truncated last line is
dropped and rescored, and an optional Drive mirror survives the wipe of `/content`.

---

## Experiments (Task 1b)

### Base models, not instruct models
The cost is a raw-text log-probability. A base model's log-probability *is* its fluency
judgement. An instruct model is further trained to produce chat-formatted answers, expects
a template it is not given, and its preference tuning is known to miscalibrate raw
likelihoods — noise unrelated to syntax, and incomparable with the theory doc and the
unsupervised-parsing literature, which score base LMs. The default is `Qwen/Qwen3-0.6B-Base`;
`gpt2` reproduces the theory doc's one-sentence numbers.

### The free NLTK PTB sample is the gold
The full Penn Treebank needs an LDC licence. NLTK ships 10% of the WSJ section (3,914
sentences) with the annotators' constituency trees, downloadable from a notebook. 3,160
sentences survive the 5–30-word window, so N = 1000 is a seeded sample, returned in corpus
order for reproducibility. Universal Dependencies is a documented stub: its arcs convert to
spans under conventions of their own (no VP node, prepositions under the noun) and the
numbers are not comparable.

### Treebank conventions follow the unsupervised-parsing literature
Traces (`-NONE-`) removed with any node they empty, recursively; punctuation removed by the
ON-LSTM / Compound-PCFG tag list (`, . : `` '' -LRB- -RRB- # $` — `$` and `#` are currency
tags in PTB and go with the rest); unaries collapsed by making gold a *set* of spans;
labels dropped (substitution yields none); single words and the whole sentence excluded.
Spans are defined over PTB tokens, so a boundary can fall inside a contraction
(`do | n't`); the detokenised string is then slightly odd, which is accepted and noted.
Sentences with no gold span after all this (flat trees) are excluded and counted.

### Sentence-level mean F1 with a bootstrap over sentences
The figure's whiskers are "95% bootstrap CI over sentences", which is only meaningful for a
per-sentence statistic, so `methods[*].f1` is the mean of per-sentence F1 (percentile
bootstrap, seeded). The corpus-level micro F1 the literature reports is in `diagnostics`,
with the paired-bootstrap gap against each baseline, which is what the verdict uses.

### Greedy induction by default, CKY as an option
"Cheapest first, never cross" is the theory doc's procedure and the simplest to reason
about; its known failure (an early cheap non-constituent blocking two gold spans) is part
of what the experiment measures. `cky_induce` finds the minimum-total-cost tree over the
same costs for comparison; the record names the inducer.

### The random baseline is a recursive uniform split
Pick a split point uniformly at random, recurse — the standard "random tree" baseline (not
uniform over the Catalan set). K = 10 draws per sentence are averaged before the bootstrap.

### The answer key lives inside the cost cache
The cache identifies sentences by `fileid:index`; rebuilding gold elsewhere would need NLTK,
the corpus, and byte-identical conventions. Rather than a second file that must stay paired
with the cache (an earlier design), every cache row carries the sentence's words, gold spans,
provenance and, when `trees` are passed, its original bracketed parse; the header records the
conventions that produced the spans. `load_span_costs_with_gold` therefore rebuilds tables
and sentences from the one file with no NLTK at all, and the stored trees let gold be
re-derived under *different* conventions (keeping punctuation, say) without the corpus.
`save_gold_jsonl` / `load_gold_jsonl` remain for caches written before schema 2.

### The proform policy is min over the real proforms; controls are cached but never chosen
The notebook scores 15 proforms (`it, there, did, then, do so, does so, did so, done so,
doing so, is, was, be, been, happens, happened`) and two *controls*: `blorp`, a nonsense
word with no category (the floor every real proform should beat on its own category), and
`<del>`, deletion of the span (is a cheap span cheap because any shortening helps the mean,
or because the proform fits?). Every span is scored with all 17 and cached. The cost used
for tree induction is the minimum over the 15 proforms only: if the controls competed,
deletion would win most spans and the parse would measure shortening, not fit. Deletion
needs its own substitution rule because an empty replacement leaves a lowercase
sentence-initial word and a dangling period; `substitute` drops the span and re-capitalises
the word that becomes initial.

### Batch size is learned from the GPU, not typed in
A Colab runtime is a T4 one day and an L4 the next, and the memory a batch needs depends on
the model, dtype and sequence length, so no fixed number is right for both. With
`batch_size="auto"` the scorer probes once, doubling a batch of the longest variant until it
no longer fits, and starts there. During the run `AdaptiveBatchSize` keeps what it learns:
a halving after an out-of-memory error sticks for the following chunks (the old per-chunk
halving forgot, and paid the OOM again on every chunk), and after a streak of clean chunks
the size doubles back, never above half of a size that already failed. An integer pins the
size for reproducibility; it still halves on OOM but never grows.

---

## Experiments (Task 9a)

### Next-token states are a separate service, fed token ids
The scorer never materialises a full log-softmax (that is what runs a T4 out of memory), so
the state vector Acts 6-8 measure distances between needed its own narrow service
(`services/state_service.py`). It takes **token ids**, not text: the splice is made on ids,
so spliced position `i+1+k` and original position `j+1+k` are the same token by
construction; re-tokenising a rejoined string could move the join. Only the requested
positions are reduced to float32 and moved off the device. The decoded splice is kept for
the audit, and each cut records whether it re-tokenises to the same ids.

### The corpus is streamed Wikipedia, not wikitext or NLTK
The unit is a paragraph of 150-300 tokens with five or more sentences, in prose the model
reads as written. wikitext, even the "raw" variant, is Moses-tokenised (`word ,`, `@-@`)
and NLTK's Brown and Gutenberg corpora are pre-tokenised too, so all three would put a
detokeniser between corpus and model. `wikimedia/wikipedia` streamed through `datasets` is
untokenised with real paragraph breaks and downloads only the articles it consumes. The
selected paragraphs are written to a JSONL next to the cache, so a re-run reads them back.
A text or JSONL file is accepted for any other corpus. Contamination is not addressed: a
0.6B model trained on 36T tokens has seen every public corpus.

### Sentence-final only is the primary boundary; clause-final is stored, not compared
Admissibility is meant to make the splice grammatical *by construction*. That holds for a
sentence end joined to a sentence start, and it does not hold for comma-to-comma joins
(list commas, appositives, `however,`), which would inflate the audit's failure rate for a
rule the comparison does not use. Clause cuts (`, ; :` and dashes) are generated and cached
under `boundary: "clause"` and summarised in `diagnostics.secondary_boundaries`; only
sentence cuts enter `record["cuts"]`. The first sentence of a paragraph is never deletable
(no boundary precedes it); no virtual boundary at BOS is invented.

### A boundary is validated on both sides, because a cut uses it at both ends
The first real run's hand audit found three endpoints that ended a sentence on the left but
did not begin one on the right: `"... Vol." + "4 (1972) and its successors"` (punkt split on
an abbreviation it does not know), `"... a monastery." + "at the court of the Frankish
monarchy"` (the extractor dropped an italicised term, so the sentence starts lowercase), and
`"... the Gallic Wars." + ", 39 volumes have been released"` (a dropped `As of <date>`
template left the sentence headless). Only 12 of 1036 endpoints (1.2%), but **the defect is
correlated with the exposure variable**: such a position is mid-sentence, so its state is
atypical for a sentence end and therefore far from genuine ones — 13.5% of the far group
touched one against 0.5% of the close group. That is a confound path straight to the
hypothesis (false boundary → "far" label *and* a broken splice), so it is rejected by
construction rather than left to a 30-cut audit to catch. `starts_sentence` skips whitespace
and opening quotes, then requires a capital; a lowercase letter, a digit or punctuation means
mid-sentence. Running out of text is fine (end of paragraph). Caseless scripts (CJK, Hebrew,
Arabic) are accepted, having no capitalisation to test — the cost is that an all-lowercase
corpus yields no usable boundary at all, which is the right refusal.

On the run that motivated this, excluding the affected cuts moved the stratified ratio from
1.8635 to 1.8557 — so it did *not* change that conclusion. The rule exists for the run where
the effect is smaller, and because a control that is known to be broken should be fixed
rather than argued around.

### Sentence ends come from punkt, mapped through tokenizer offsets, every rejection counted
NLTK's punkt knows `Mr.` and `U.S.` are not sentence ends; a regex splitter exists for the
offline tests only, and the record names which was used. Each sentence's last character is
mapped to the token containing it; the position is kept only when that token ends exactly
there, the sentence ends in `. ! ?` (plus closing quotes), and whitespace follows. Rejected
ends are counted in the cache row rather than silently dropped.

### Close / far are deciles within length-matching bins, not pooled deciles
Divergence tracks deleted length (Spearman ~ +0.9) and so does endpoint distance (~ +0.65),
so pooled deciles put every far cut in the long stratum and every close cut in the short one.
Deciles within each stratum were the first fix, and a synthetic *null* corpus exposed a
spurious 1.2x: inside a stratum of doubling width the far decile still deletes more than the
close decile. Labels are therefore taken within `match_width`-token bins (default 10) inside
each stratum, so close and far delete the same amount of text to within a bin; the
per-stratum median deleted lengths are reported as the balance check. The pooled labels are
kept as `pair_pooled` and reported under `pooled` with the warning the runbook asks for.

### One stratified permutation test is the verdict; everything else is diagnostics
"p < 0.05 within length strata" is made concrete as a single test: the statistic is the
stratum-size-weighted mean of `log(median far / median close)` over strata with at least
`min_per_group` cuts in both groups, and the null shuffles close / far labels within each
matching bin, so the null keeps every bin's label counts. Per-stratum one-sided Mann-Whitney
values are shown in the figure as detail. Cuts from one paragraph share its states, so the
interval on the stratified ratio is a paragraph-level cluster bootstrap. The demo's residual
analysis (Spearman after removing the log-length trend) is in `diagnostics.continuous`.

### The pre-registration is the cache header, and `schema` is part of it
Measure, window, boundary kinds, primary boundary, strata, decile, matching width and the
boundary rule are written as the first line of the cache before any cut is scored, and phase B
reads them from there. `schema` is deliberately *not* in the unchecked set: schema 2 tightened
the boundary rule, so a schema-1 cache holds endpoints selected under a different definition
of admissible and resuming it would silently mix the two. Such a cache can still be *read* —
`analyse_9a` reports its contamination in `diagnostics.boundary_check` and the verdict warns
about it — it simply cannot be extended. Passing different strata or decile to `analyse_9a` is allowed and stamped into the
record's notes as `DEVIATION`.

### The audit stays, even though the rule now rejects the failure mode it found
Tightening the boundary rule does not retire the audit: the audit is what found that gap, and
an automated check can only test failure modes someone has already thought of. A changed rule
also needs fresh evidence that it holds. `boundary_check` re-verifies the known failure mode on
every run and the audit remains the instrument for the unknown ones.

### The audit is blind, stratified, and its failures are reported, not dropped
The sheet holds 10 close, 10 far and 10 other admissible sentence cuts drawn with the run
seed, shows the spliced text with the join marked and nothing numeric, and is filled by a
person. Unaudited cuts carry `grammatical: true, audited: false`, so the two are never
conflated; an audited failure is counted in `audit`, named in the verdict as a violation of
the rule at that rate, excluded from every summary and drawn hollow by `fig_9a`. An existing
sheet is never overwritten.

### The exploratory extras are cheap, so they are kept
Every cut stores its twenty per-position distances and the spliced text's fluency. The first
allows a window-sensitivity check without a re-run; the second makes the "fluency and
divergence disagree" point reproducible (`diagnostics.continuous.spearman_fluency_delta_div`).
Neither enters the verdict.

## Experiments (Task 9b)

### Two text measures, neither a single decode
One greedy continuation cannot say whether a deletion changed what the model writes: after a
sentence boundary the next sentence is nearly free, so greedy decodes from the original and the
spliced context part within a few tokens either way, and that agreement length barely tracks the
state divergence. Measure A scores the author's *actual* next `W` tokens under both contexts (a
deterministic number per cut); measure B samples `K` continuations from each context and compares
the two *sets* by bag-of-token F1. `first_diff_greedy` stays in the record as a descriptive because
the theory document quotes it; nothing is judged on it.

### `lp_orig` is recomputed, and the cache is the invariant
The original side of measure A is available in the 9a cache as `-surprisal`, but it was scored in
another session. Both sides of `true_dlogp` are taken from *this* session -- one original pass per
paragraph, one spliced pass per cut, the same dtype and kernels -- so the difference is not
contaminated by a change of GPU or library version. The cached surprisal is then compared against
`lp_orig` and the maximum difference reported per cut and overall (invariant 1): a tolerance
warning, never an adjustment.

### Sampling uses its own loop and a per-cut `torch.Generator`
`generate()` draws from the global RNG and its sampling path changes between versions. The
protocol needs the original and spliced samples of one cut to be *paired by random state*, so the
loop draws all `K` rows together with one `torch.multinomial` per step from a generator created and
seeded (`seed_base + cut_index`) immediately before each context. Two contexts then consume the
identical stream, and nothing else in the process observes a seed change. Nucleus filtering is the
standard "smallest set whose mass reaches p, crossing token included", applied to float32
probabilities. The KV cache is used for speed; `kv_cache=False` re-feeds the whole sequence and is
the oracle the tests compare against.

### EOS is an ordinary token and every sample has exactly `L` ids
"EOS not forced" is read as *neither suppressed nor terminal*. Masking EOS would alter the
distribution being compared; stopping at it would give bags of unequal size and an F1 that
depends on length. So a sampled EOS is kept and decoding continues to `L`; the record counts how
many samples contain one, per cut and overall.

### `cross / within`, on token ids
Raw cross-set F1 conflates "the spliced context writes something else" with "this context writes
many different things anyway"; dividing by the within-set F1 of the original samples normalises
the second away, so 1.0 means the same distribution of text. F1 is over token *ids* as multisets,
with no detokenisation, so it is exactly recomputable from the stored samples -- which is why all
`2K` continuations are in the record: any other overlap function can be applied to them later. A
`within` of 0 leaves the ratio undefined; that cut is reported as collapsed, never patched.

### The 9b record is additive and the 9a files are read-only
Nothing in the 9a cache or record moves. The generation cache header carries the sha256 of both
inputs, so resuming against a different 9a record or cache refuses and names the field, and the
tests assert the input bytes are unchanged. The record keeps the requested fields first and the
additive ones (`key`, `cut_index`, `seed`, `grammatical`, EOS counts, the greedy pairs) after them.

### The analysis is not in the repository
The pre-registered comparison (far vs close on both measures within 9a's length strata, the
cluster bootstrap, the permutation test, Spearman with `state_div`) is run from the record on the
analyst's side, by decision. The repo delivers the measurements and the invariants; `fig_9b`
remains vendored for a record that carries `text_overlap`.

### `fig_9a` was edited, not vendored verbatim
The runbook's figure printed a pooled Mann-Whitney in its title and cut strata at data
tertiles, both of which the task text rules out. It now prints the stratified ratio, interval
and permutation p when the record carries them, uses the record's pre-registered strata for
panel B, and falls back to the original behaviour for a record without them, so the mock data
and any older record still draw.

---

## Storage

### JSON is the index; the `.npz` sidecar holds the bulk
Text floats are big and slow: per-token states from a 4096-hidden model can be tens of MB
of JSON that no editor opens and that takes seconds to parse. Above
`npy_threshold_floats` (default 1M) the raw float32 arrays move into a compressed `.npz`
and the JSON keeps `values: null` plus an `npy_ref`. `load_run()` rehydrates transparently,
so callers never branch on which mode was used. Force either mode with
`npy_mode="always"` / `"never"`.

### Floats are rounded to 6 decimals in JSON
Full float32 repr is ~17 characters of mostly noise per value. Six decimals keeps the file
readable and roughly 40% smaller, and the sidecar keeps the lossless copy when precision
matters. Tune with `float_precision`.

### NaN/inf are serialised as `null`, with a warning
JSON has no `NaN` or `Infinity` literals; Python's `json` emits them anyway, producing
files that other parsers reject. Nulling them keeps the file valid, and the warning points
at the real cause (usually low-precision overflow).

### Writes are atomic (temp file + `os.replace`)
A Colab runtime can be recycled mid-write. Without this you get a truncated JSON that
still parses as *something*. With it, the output file either does not exist or is complete.

### `run_id` is a UTC timestamp, and filenames are sanitised
Timestamps sort chronologically in a file listing, which is what you want when scanning a
results directory. Names are stripped to `[A-Za-z0-9-_.]` so they survive Drive, Windows,
and shell globbing.

### The effective config travels with the result
`BatchResult.extraction` records the settings that actually ran, including per-call
overrides, and `Analyzer.save()` builds the manifest from it. Otherwise
`analyzer.batch(texts, pooling="none")` would be saved under a manifest claiming
`pooling="last_token"` — a file that lies about its own contents.

### Input text can be withheld (`include_input_text=False`)
Ids still allow joining results back to the inputs, but the corpus itself need not be
duplicated into every output file. Useful for sensitive or large text.

---

## Environment and secrets

### Secrets resolve Colab → env → `None`, and `None` is a success
The same code runs unchanged in Colab, in CI, and locally. Crucially, no token is the
*normal* path: ungated models load fine anonymously, so a missing `HF_TOKEN` must not be
an error. Switching to a gated model later requires adding a secret, not editing code.

### The clone token never enters the URL at all
Auth is passed as a per-command `git -c http.extraHeader=...` value rather than embedded
in the clone URL. A URL-embedded PAT is written into `.git/config` on the runtime's disk
and has to be scrubbed afterwards -- which only works if the scrub actually runs, so any
failure between clone and scrub leaves a live credential behind. A per-command config
value writes nothing, so there is nothing to clean up. Git errors are filtered before
printing, and the header itself is replaced with `<auth>` in any command echoed to output.

### The bootstrap cell preflights the token against the API
`git clone` reports every credential problem as `Invalid username or token`, which covers
an expired token, a typo, a missing repository grant, and un-authorised SAML SSO alike --
four different fixes behind one message. Two API calls (`/user`, then
`/repos/{owner}/{repo}`) separate them before git runs, so a 401 says "the token is bad",
a 404 says "the token is fine but has no grant on this repo", and a 403 points at SSO.
This costs two HTTP requests and turns the single most likely first-run failure from a
dead end into an instruction. A git failure *after* a passing preflight is reported
separately, since the obvious causes have just been ruled out.

### `torch` is not pinned to an exact version
Colab ships a torch build matched to its CUDA driver. Forcing a different version triggers
a multi-minute reinstall and can break GPU support outright. `requirements.txt` asks for
`torch>=2.0` and the notebook installs with `--upgrade-strategy only-if-needed`, so an
existing satisfying torch is kept.

### Logging attaches exactly one handler, and never propagates
Re-running a notebook cell that imports the package would otherwise stack handlers and
print every line 2×, 3×, 4× — a classic Colab annoyance. `configure_logging` is idempotent
and sets `propagate = False` so Colab's root logger does not double them either.

### Strict determinism is opt-in
Seeding is always on. `torch.use_deterministic_algorithms(True)` is not: it is slower, and
it raises on ops with no deterministic implementation. Even with it, results are only
reproducible on the *same* GPU model and library versions — cuBLAS reduction order differs
across hardware. The manifest records the seed and every relevant version so a run can be
described exactly, which is the achievable goal.

---

## Testing

### Tests build a tiny model locally instead of downloading one
`testing.build_tiny_local_model` writes a randomly-initialised 4-layer, 16-hidden GPT-2 and
a word-level tokenizer to a temp directory, then loads it through the *same*
`from_pretrained` path a real model uses. The suite runs in seconds, needs no network, and
cannot be broken by a Hub outage. Its tokenizer deliberately ships **no** pad token, so
every test run exercises the eos-as-pad fallback.

### The notebook is tested by joining its source the way a reader does

`tests/test_notebook.py` joins each cell's `source` with `""`, not `"\n"`. This is not a
detail: the notebook shipped once with every `source` element missing its trailing
newline. `nbformat.validate` passed, the JSON parsed, and a syntax check that joined with
`"\n"` passed too — but Colab, which concatenates verbatim, rendered every cell as a
single unrunnable line. Only the `""` join reproduces what a reader actually sees, so it
is the only join the tests use.

The suite also executes the notebook end-to-end during development (via `nbclient`, with
the model ids redirected at a locally-built tiny model) — a check that a static parse
cannot substitute for.

### Shared test constants live in the package, not in `conftest.py`

`from tests.conftest import TINY_HIDDEN` resolves only when the repo root is on
`sys.path`. `python -m pytest` prepends the cwd, so it works; the `pytest` console script
does not, so it fails at collection. The suite passed under one invocation and could not
collect under the other -- and the README documents the broken one. The tiny-model shape
now lives in `m1_analyzer.testing` beside the builder that produces it, which is
importable however pytest is started, and `tests/test_imports.py` fails if any test module
imports `tests.*` again.

### `test_architecture_doc.py` enforces the documentation rule
The requirement to keep `architecture.md` current is enforced by a test rather than by
memory: adding a file without documenting it fails the suite, naming the file.

---

## Deliberately not built yet

Each of these is a real gap, listed so it is a decision rather than an oversight.

| Not built | Why, and what it would take |
|---|---|
| **Chat templates for instruct models** | Inputs go in as raw text. `Qwen2.5-0.5B-**Instruct**` was trained with a chat template; hidden states for `"hello"` differ from those for the templated version. For layer *analysis* raw text is usually what you want, and applying a template silently would be a hidden transformation. Add `apply_chat_template: bool` to `ExtractionConfig` when you need it — it is a few lines in `_forward`. |
| **Attention weights** | `output_attentions=True` in the same forward pass, plus a records/storage shape. Attention is `layers × heads × seq × seq` — far bigger than hidden states, so it needs its own storage policy. |
| **Middle-layer *aggregation*** (e.g. mean of layers 8–16) | Selection is built; combining several layers into one vector is not. Belongs next to `utils/pooling.py` as a second, layer-axis reduction. |
| **Long-document chunking** | Inputs longer than the context are truncated and flagged, not split-and-stitched. Sliding-window chunking plus a merge policy is a design question of its own. |
| **Quantized loading (`bitsandbytes`, GPTQ)** | Would let bigger models fit a T4. Quantization changes hidden-state values, so it needs to be recorded in the manifest and thought about before results are compared across runs. |
| **Multi-GPU / `device_map="auto"`** | Single-device only. Colab gives one GPU; `accelerate` sharding adds real complexity for no benefit there. |
| **Cross-run comparison utilities** | Cosine similarity, layer-wise drift, projections. Deliberately out of scope for the extraction pipeline: its job is producing trustworthy, self-describing files. The experiments package is the exception, because a runbook task *is* an analysis with a fixed record schema. |
| **Universal Dependencies gold** | `load_ud_conllu` is a stub. Subtree yields → spans, drop non-projective yields, and say so in the record's `treebank` field; not comparable with PTB numbers. |
| **In-memory score cache** | Phase A de-duplicates variants per sentence and the JSONL is the cache; a dict of 700k texts would be memory for nothing. |
| **Truncating the spliced sequence at `i + 1 + window`** | Under causal attention it would give identical states at two thirds of the compute, but the full spliced text is what yields the exploratory fluency for free, and phase A on 200 paragraphs is minutes on a T4 either way. |
| **Streaming / incremental writes** | A run is held in memory and written once. For a very large corpus, an append-mode sink (JSONL + a growing `.npz`) satisfying `ResultSink` would be the change. |
