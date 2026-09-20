# m1_llms_analyzer

Extract hidden states from open-source Hugging Face models and save them as JSON, and run
sentence-structure experiments on the same models' log-probabilities. Built as in-process
microservices, driven from Google Colab notebooks.

- **`invoke(text)`** — one input, one record.
- **`batch(texts)`** — many inputs, order preserved, OOM-safe, failures isolated.
- **Any layer** — `-1` (last), `0` (embeddings), `"middle"`, `"all"`, or a list.
- **Pooled or per-token** — `last_token` (default), `mean`, `cls`, or `none`.
- **JSON out** — self-describing, with a lossless `.npz` sidecar when a run is large.
- **`score(texts)`** — sentence log-probabilities (mean per token = the fluency of the
  teaching doc), when the model is loaded with its LM head.
- **`next_token_states(ids, positions)`** — the full next-token log-probability vector at
  chosen positions, for state-distance experiments.
- **Experiments** — Task 1b (bracket induction from substitution costs), Task 9a
  (omittability by splicing, length-matched, with a hand audit) and Task 9b (does the deletion
  change the generated text?) end to end, each with a resumable cache. See [Experiments](#experiments).

**Documentation site:** [salmonsung.github.io/m1_llms_analyzer](https://salmonsung.github.io/m1_llms_analyzer/) —
one page per task with the files it uses, a pipeline diagram, how the data is collected and
worked examples, plus the architecture and maintainer docs.

---

## Use it from Colab (the intended path)

1. **Add two Colab secrets** (key icon in the left sidebar, *Notebook access* on):

   | Secret | Required | What it is |
   |---|---|---|
   | `GITHUB_TOKEN` | **Yes** (the bootstrap cell clones with it; required while the repo is private) | GitHub fine-grained PAT with *Contents: Read* on this repo — [create one](https://github.com/settings/personal-access-tokens/new) |
   | `HF_TOKEN` | Only for gated models | Hugging Face read token — [create one](https://huggingface.co/settings/tokens) |

2. **Open [`notebooks/colab_entrypoint.ipynb`](https://github.com/SalmonSung/m1_llms_analyzer/blob/main/notebooks/colab_entrypoint.ipynb)**, copy it
   into Colab (or `File → Upload notebook`), and hit **Runtime → Run all**.

That is the whole setup. The notebook clones this repo, installs its dependencies, runs a
5 MB smoke test so you know the pipeline works within seconds, then runs the real model and
writes the results. Every cell is re-run safe.

Neither token is ever printed, logged, or written into an output file — and the clone
token is scrubbed out of `.git/config` immediately after cloning.

---

## Use it locally

```bash
pip install -e ".[dev]"
python scripts/smoke_test.py --offline    # no network needed: builds a tiny model
pytest -q
```

```python
from m1_analyzer import Analyzer

analyzer = Analyzer.quick("Qwen/Qwen2.5-0.5B-Instruct", layers=[-1, "middle"])

record = analyzer.invoke("The capital of France is Paris.")
record.layer(-1)          # numpy array, shape (hidden_size,)
record.layer("middle")    # the middle block
record.truncated          # was the input clipped?

result = analyzer.batch(["one", "two", "three"])
result.records            # in input order
result.failures           # anything that went wrong, per item

paths = analyzer.save(result, name="my_run")   # -> outputs/my_run.json
```

Read it back — sidecar or not, you get numpy arrays either way:

```python
from m1_analyzer import load_run

run = load_run("outputs/my_run.json")
run["run"]["model_id"], run["run"]["layers_resolved"]
run["records"][0]["layers"]["-1"]["values"]     # np.ndarray
```

Or from a terminal (note the `=` — argparse would read a bare `-1` as a flag):

```bash
python scripts/run_extraction.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --layers=-1,middle --pooling last_token \
  --input-file prompts.txt --out outputs/run.json
```

---

## Experiments

`notebooks/experiment_1b.ipynb` runs **Task 1b — does substitution FIND constituents?**
Same two secrets as above, a GPU runtime, *Run all*. It smoke-tests the whole chain on a
tiny model, loads 1000 sentences of the free NLTK Penn Treebank sample with their gold
brackets, scores every span × replacement variant (15 proforms plus the `blorp` and deletion
controls) on `Qwen/Qwen3-0.6B-Base` (phase A, GPU, batch size probed per GPU, resumable;
the cache embeds the gold), induces a bracketing per sentence and scores it against gold and the trivial
baselines (phase B, CPU, seconds), then draws `fig_1b` and `fig_0c` and copies everything to
Drive. Background: `theory_1b_bracket_induction.md` (outside this repo).

The same pieces from Python:

```python
from m1_analyzer import Analyzer
from m1_analyzer.experiments import (
    MinOverSet, analyse_1b, compute_span_costs, load_ptb_nltk, verdict, experiment_figures as EF,
)

analyzer  = Analyzer.for_scoring("Qwen/Qwen3-0.6B-Base")        # loads the LM head
analyzer.score_one("The tall man opened the door.").mean_logprob  # nats per predicted token

sentences = load_ptb_nltk(1000, min_len=5, max_len=30, seed=42)  # gold brackets included
policy    = MinOverSet(["it", "there", "did", "then"],             # blind: cheapest proform wins
                       controls=["blorp", "<del>"])               # scored + cached, never chosen
tables    = compute_span_costs(analyzer, sentences, policy,       # phase A, cached + resumable
                               cache_path="outputs/task_1b/span_costs.jsonl",
                               provenance={"model_id": "Qwen/Qwen3-0.6B-Base"})
record    = analyse_1b(tables, sentences, policy=policy, inducer="greedy",
                       model="Qwen/Qwen3-0.6B-Base")              # phase B: the fig_1b record
print(verdict(record))
EF.fig_1b(record, path="outputs/task_1b/fig_1b.png")
```

Conventions that the numbers depend on (all recorded in the output): traces and
punctuation removed from the treebank, unaries collapsed, unlabelled spans, single words
and the whole sentence excluded; cost = mean log-probability per token of the sentence
minus that of the variant; a BOS token prepended so every real token is predicted; F1 is
the sentence-level mean with a 95% bootstrap over sentences (corpus micro F1 in
`diagnostics`). Use a **base** model: instruct tuning distorts raw-text likelihoods.

`notebooks/experiment_9a.ipynb` runs **Task 9a — omittability by splicing** (Task 9 with its
grammaticality control, at power). A cut `(i, j)` deletes the tokens strictly after `i`
through `j` and rejoins the text; it is *admissible by construction* when both endpoints are
sentence-final and `window` tokens follow `j`. The notebook streams ~200 Wikipedia paragraphs
of 150–300 tokens, scores every admissible cut on `Qwen/Qwen3-0.6B-Base` (phase A, resumable;
the cache header is the pre-registration: divergence measure, window, strata, decile, matching
width), writes a 30-cut **audit sheet** you fill in by hand on text alone, then compares close
and far endpoint pairs **within deleted-length strata** with labels matched on length (phase B),
and draws `fig_9a`. Divergence tracks how much was deleted, so a pooled ratio is never the verdict.

```python
from m1_analyzer import Analyzer
from m1_analyzer.experiments import (
    add_surprisal, analyse_9a, compute_splices, load_audit_csv, load_wikipedia_paragraphs, verdict_9a,
    experiment_figures as EF,
)

analyzer   = Analyzer.for_scoring("Qwen/Qwen3-0.6B-Base")
paragraphs = load_wikipedia_paragraphs(lambda t: len(analyzer.states.encode(t)), n=200, seed=42)
header, rows = compute_splices(analyzer.states, paragraphs, window=20,       # phase A, cached + resumable
                               cache_path="outputs/task_9a/splices.jsonl",
                               provenance={"model_id": "Qwen/Qwen3-0.6B-Base"})
header, rows, report = add_surprisal(analyzer.states, "outputs/task_9a/splices.jsonl")  # additive: per-token
print(report["n_bad_paragraph_length"], report["max_abs_mean_all_minus_fluency"])       # surprisal, del_surp per cut
audit  = load_audit_csv("outputs/task_9a/audit.csv")                          # after you fill it in
record = analyse_9a(rows, header, audit=audit, model="Qwen/Qwen3-0.6B-Base")  # phase B
print(verdict_9a(record))
EF.fig_9a(record, path="outputs/task_9a/fig_9a.png")
```

What the numbers mean (all in the record): state = float32 log-softmax over the vocabulary;
endpoint distance = L2 between the original text's states at `i` and `j`; divergence = median
over `k < window` of the L2 between the spliced state at `i+1+k` and the original at `j+1+k`;
close / far = bottom / top decile of endpoint distance within 10-token bins of deleted length,
pooled across paragraphs; the estimate is the stratum-weighted far/close median ratio with a
paragraph-level cluster-bootstrap interval; the p-value is a permutation test that shuffles labels
within bins. Raw distances scale with the vocabulary size, so compare within one model only.
`surprisal[t] = -log p(ids[t] | ids[:t])` in nats, from the same original-text pass and the same
float32 log-softmax as the states (token `t` is read from the state at position `t-1`; token 0 from
the BOS row); `del_surp` is its sum over the deleted tokens `i+1..j` and `del_surp_mean` that sum
over `seg_len` -- a second matching variable, so close and far can be matched on information as
well as length. A paragraph's `fluency` is the mean of `surprisal` over **all** tokens, token 0
included, so `mean(surprisal)` reproduces it and `mean(surprisal[1:])` does not. `add_surprisal`
fills these into an existing cache without changing any other byte, and reports both invariants.

`notebooks/experiment_9b.ipynb` runs **Task 9b — does the deletion change what the model *writes*,
not only what it predicts?** It reads the 9a cache and record (never writes them) and, for each of the
record's labelled close / far cuts, measures two things that do not depend on a single decode. **A:**
the log-probability of the author's actual next `W = 20` tokens under the original and the spliced
context (`true_dlogp`, nats/token, plus the 20 per-token values). **B:** `K = 16` nucleus samples
(`p = 0.95`, `T = 1.0`, `L = 30` new tokens, EOS an ordinary token) from context O = `ids[:j+1]` and
S = `spliced[:i+1]`, seeded `20250914 + cut_index` with the same seed for both, compared as bags of
token ids: `within` (mean F1 over the O–O pairs), `cross` (over the O–S pairs) and
`sample_overlap = cross / within`. Every sampled continuation is stored so another overlap function can
be applied later; `first_diff_greedy` is kept as a descriptive. The output is a separate record,
`record_9b_<RUN_TAG>.json`, with three invariants reported: the cached `surprisal` reproduces `lp_orig`,
the cut keys match the 9a record exactly, and `within > 0` everywhere. The pre-registered analysis is
run from the record, outside the repo.

```python
from m1_analyzer.experiments import GenerationProtocol, build_record_9b, compute_generation_9b

header, rows = compute_generation_9b(                                   # phase A, cached + resumable
    analyzer.states, analyzer.models, "outputs/task_9a/record_9a.json", "outputs/task_9a/splices.jsonl",
    cache_path="outputs/task_9b/gen_9b.jsonl", protocol=GenerationProtocol(),  # W_true=20, K=16, L=30, p=0.95, T=1.0
    provenance={"model_id": "Qwen/Qwen3-0.6B-Base"})
record = build_record_9b(header, rows, json.load(open("outputs/task_9a/record_9a.json")), model="Qwen/Qwen3-0.6B-Base")
record["invariants"]["max_abs_cached_lp_diff"], record["invariants"]["keys_match"], record["invariants"]["n_collapsed"]
```

`lp_orig` comes from a fresh pass of the original ids in the same session as the spliced pass, so
both sides of `true_dlogp` share one set of numerics; the 9a cache's `surprisal` is what it is checked
against, not the input. Samples are drawn from a per-cut `torch.Generator`, so a re-run on the same GPU
and dtype reproduces them and the global RNGs are never touched.

---

## Configuration

```python
from m1_analyzer import Analyzer, RunConfig, ModelConfig, ExtractionConfig, ScoringConfig, StorageConfig

config = RunConfig(
    model=ModelConfig(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        revision=None,             # pin a commit SHA for reproducibility
        device="auto",             # auto | cpu | cuda | mps
        dtype="auto",              # auto | float32 | float16 | bfloat16 (bf16 only when native)
        trust_remote_code=False,   # executes repo code -- opt in per model
        head="base",               # base (hidden states) | causal_lm (adds score())
    ),
    extraction=ExtractionConfig(
        layers=-1,                 # int, list, "all", "last", "middle"
        pooling="last_token",      # last_token | mean | cls | none
        max_length=None,           # None -> model context, capped below
        max_length_cap=4096,
        batch_size=8,              # auto-halves on GPU OOM
        continue_on_error=True,    # record failures instead of raising
        include_tokens=False,      # save token strings with per-token output
    ),
    scoring=ScoringConfig(         # used by score() when head="causal_lm"
        batch_size=64,
        max_length_cap=512,
        bos_policy="auto",         # prepend BOS (else EOS) so every token is predicted | none
        logit_chunk=8,             # sequences per float32 logit block (memory knob)
    ),
    storage=StorageConfig(
        output_dir="outputs",
        float_precision=6,         # decimals kept in JSON
        npy_mode="auto",           # auto | always | never
        include_input_text=True,
    ),
    seed=42,
)

analyzer = Analyzer(config)
```

### Layer indexing

`hidden_states` has `num_hidden_layers + 1` entries:

| Index | Tensor |
|---|---|
| `0` | embedding output (before block 1) |
| `1 … N` | output of transformer block *i* |
| `-1` | same as `N` — the last block |

`"middle"` resolves to block `N // 2`. Every output file records both the label you asked
for and the resolved absolute index, so a saved result is never ambiguous.

---

## Output format

```jsonc
{
  "schema_version": "1.0",
  "run": {
    "run_id": "run-20260831T120000Z",
    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "revision": "<commit sha>",
    "device": "cuda", "dtype": "bfloat16",
    "layers_requested": [-1, "middle"], "layers_resolved": [24, 12],
    "layer_index_convention": "hidden_states[0] is the embedding output; ...",
    "pooling": "last_token", "max_length": 4096, "seed": 42,
    "num_hidden_layers": 24, "hidden_size": 896,
    "library_versions": { "torch": "...", "transformers": "..." },
    "config_fingerprint": "..."
  },
  "counts":  { "records": 4, "failures": 0, "total_floats": 7168 },
  "storage": { "float_precision": 6, "values_in_sidecar": false, "sidecar_file": null },
  "records": [
    {
      "id": "a1b2c3d4e5f6a7b8",
      "text": "The capital of France is Paris.",
      "token_count": 8, "original_token_count": 8, "truncated": false,
      "layers": {
        "-1":     { "index": 24, "requested": "-1",     "shape": [896], "values": [0.123456, ...], "npy_ref": null },
        "middle": { "index": 12, "requested": "middle", "shape": [896], "values": [0.234567, ...], "npy_ref": null }
      }
    }
  ],
  "failures": []
}
```

When a run exceeds `npy_threshold_floats` (1M by default), `values` becomes `null`,
`npy_ref` points into a companion `.npz`, and `load_run()` stitches them back together.
**Keep the `.json` and `.npz` together** when moving results.

---

## Docs

The docs are a website: **[https://salmonsung.github.io/m1_llms_analyzer/](https://salmonsung.github.io/m1_llms_analyzer/)**.

| Page | What is in it |
|---|---|
| [Tasks](https://salmonsung.github.io/m1_llms_analyzer/tasks/) | Each pipeline (extraction, Task 1b, 9a, 9b): the `.py` files it uses, a diagram, how the data is collected, outputs, worked examples |
| [Architecture](https://salmonsung.github.io/m1_llms_analyzer/architecture/) | Structure tree, every file's responsibility, request flows, extension points (the source is [`architecture.md`](https://github.com/SalmonSung/m1_llms_analyzer/blob/main/architecture.md)) |
| [Services and request flows](https://salmonsung.github.io/m1_llms_analyzer/maintainers/services/) | The Protocols, the wiring, one sequence per request path, module map |
| [Design decisions](https://salmonsung.github.io/m1_llms_analyzer/design_decisions/) | Every undiscussed choice and its reasoning, plus what is deliberately not built |
| [Edge cases](https://salmonsung.github.io/m1_llms_analyzer/edge_cases/) | 30 handled edge cases with the code that handles each, and the known limits |

Build it locally:

```bash
pip install -r requirements-docs.txt
mkdocs serve                              # http://127.0.0.1:8000, live reload
mkdocs build --strict                     # what the docs workflow runs before deploying
python scripts/make_doc_examples.py       # regenerate docs/assets/examples/ (torch optional)
```

## Tests

```bash
pytest -q                              # ~310 tests, fully offline, under a minute
python scripts/smoke_test.py --offline # end-to-end without the Hub
```

The Task 1b analyses are tested against `m1_analyzer.testing.FakeSpanScorer`, a scorer
that knows the gold tree, because the tiny model's word-level tokenizer cannot spell the
proforms. The Task 9a pipeline runs on the tiny model itself (its vocabulary carries sentence
punctuation) with the offline `regex` sentence splitter; the analysis is tested on synthetic
caches with and without a planted effect. The NLTK loader test is skipped until the corpus has
been downloaded once (`python -c "import nltk; nltk.download('treebank')"`).

The suite builds a tiny random GPT-2 on disk and loads it through the same
`from_pretrained` path a real model uses, so it needs no network and cannot be broken by a
Hub outage. `tests/test_architecture_doc.py` fails the build if a source file is missing
from `architecture.md`.
