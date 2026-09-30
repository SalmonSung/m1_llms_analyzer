# Job D: frame-only substitution scores on five more models and 2,161 new sentences

**What and why.** Both substitution measures in the paper come from one pass, computed exactly as in job A: every span of every sentence, replaced by each of the same 17 strings, scored token by token (`t4_rows`). Job A covered one model, Qwen3-0.6B-Base, on the 1,000 sentences. Job D adds three things:
- more models, so the results are not one model's;
- a held-out set of 2,161 new sentences, so the findings can be checked on text they were not found on;
- one extra number per row (`first_sub`, below), for a reviewer check.

**The submission deadline is 12 October.** We need tier 1 by Sunday 4 October and everything by Thursday 8 October. Anything that arrives after 9 October cannot go into the submission. Please send each file as soon as it is done; do not wait for the batch.

## Inputs (all in `runner_package_job_d.zip`)

| file | what | sha256 |
|---|---|---|
| `tree_runner_ref.py` | reference code **v4** | `3b1e5aac659e22f7afdbe0755134e04bd416e27888ab2b7f205039d8e2f5394b` |
| `sentences_3161.jsonl` | 3,161 sentences: `set` = `main` (the 1,000 of job A, identical words, text, tree and gold) or `heldout` (the other 2,161 of the NLTK sample with 5–30 words, same cleaning rule) | `f2a25c1c18a967b1767eae1473d298d52d6bec3865a4e7e30007ef0370716e55` |
| `anchor_job_d.json` | our local scores (float32, CPU) for 3 held-out sentences on Qwen3-0.6B and 5 sentences on GPT-2 | — |

**v4 changes four things.**
1. `bos_id(tok)` picks the start token: the tokenizer's own BOS if it defines one, otherwise `<|endoftext|>`.
   - For Qwen3 this gives the same id v1 used, 151643.
   - Llama-3.1 gets `<|begin_of_text|>`, as in your 1b run. OLMo-3 gets `<|endoftext|>` (100257), and GPT-2 its own `<|endoftext|>`.
   - `load()` calls it.
2. `span_keys(n)` lists every span, as `i,j` with `j > i`, the whole sentence excluded. It equals the key set of the 1b cache.
3. **Each row gains an 8th field, `first_sub`:** the log-prob, in the substituted sentence, of the first token after the span (or of the first end-mark token if nothing follows).
   - The first seven fields are unchanged and in the same order.
   - Why: that token's leading space also encodes where the replacement word ends. Having it lets us check that the frame-only cost is not partly a score of the replacement itself.
4. `tokenizer_checks(tok, sentences)`: tokenizer-only checks, no model needed (see step 4 below).

**Tested here.** v4 plus the sentence file reproduce job A's own rows for the two shortest main sentences: 306 rows, every integer field identical, largest difference 3.5e-04 nats. `first_sub` equals `suf_sub` or `end_sub` exactly wherever it must (206 rows).

## Models

| model | revision | dtype | sets | est. A100 hours (bf16 speed-up … none) | note |
|---|---|---|---|---|---|
| `Qwen/Qwen3-0.6B-Base` | `da87bfb608c1…` | float32 | main + heldout | 1–2 h / 2–5 h | main again for first_sub; fields 1-7 must match job A |
| `Qwen/Qwen3-1.7B-Base` | `ea980cb0a6c2…` | float32 | main + heldout | 3–6 h / 7–14 h | same revision as your 1b run |
| `Qwen/Qwen3-8B-Base` | `49e3418fbbbc…` | bfloat16 | main + heldout | 15–30 h / 34–68 h | same revision as your omittability run |
| `meta-llama/Llama-3.1-8B` | `d04e592bb4f6…` | bfloat16 | main + heldout | 15–29 h / 33–66 h | same revision as your 1b run |
| `allenai/Olmo-3-1025-7B` | `a81bae42db39…` | bfloat16 | main + heldout | 13–27 h / 30–60 h | new; fully open training data |
| `openai-community/gpt2` | `607a30d783df…` | float32 | main + heldout | <2 h / <2 h | new; cheap, older reference point |

- **Revisions:** please load each model at the revision shown and record the full commit sha in the header.
- **dtype:** models of 2B parameters or fewer run in float32, as in job A; larger ones in bfloat16. `forward()` always does the log-softmax in float32. Our 0.6B cache (bfloat16) and job A (float32) give tree F1 0.3455 and 0.3458, so this split is safe.
- **Hours** are scaled from job A (7,853 s on one A100-40GB) by parameter count and span-weighted length. The held-out set is 2.28× job A's work.
- **Dropped:** gpt-oss-20B (not a base model) and Gemma-4-31B (too slow for the deadline). Their earlier 1b files are enough.

## Procedure: your job A notebook, with four changes

1. **Use v4.** Put its sha in `reference_code_sha`; `reference_code_modified` must stay false.
2. **Read sentences from `sentences_3161.jsonl`**, not the 1b cache. For each row call `t4_rows(tok, mdl, bos, row['words'], span_keys(len(row['words'])), row['end'])` with the 17 default fillers. `row['end']` equals `end_string(row['tree'])`.
3. **One output file per model and set:** `tree_t4_<model>_<set>.jsonl.gz`, e.g. `tree_t4_qwen3-8b_main.jsonl.gz`. The format is in the next section.
4. **Checks before each full run,** written to `checks_<model>_<set>.json` and to `meta.checks`:
   - **tokenizer** (every model, once per set): `tokenizer_checks(tok, rows_of_that_set)`. It takes about 20 minutes on one CPU core per set, and can run while the model loads.
     - `is_fast` must be true; `special_in_text`, `roundtrip_fail` and `offsets_bad` must be 0. **If any fails, stop and tell us;** do not work around it.
     - `frame_pre_short`, `frame_suf_short` and `end_merge` are counts to report, not gates. `t4_rows` already handles them: the frame is the tokens the two sentences share, and the end mark is always appended separately, as in job A.
     - We ran it here for the Qwen3, GPT-2 and OLMo-3 tokenizers (Qwen3 0.6B, 1.7B and 8B share one tokenizer file) on the first 300 sentences (904,564 texts): all gates pass. Llama-3.1 is gated, so it is untested; please send its counts first.
   - **anchor** (Qwen3-0.6B, GPT-2): the sentences in `anchor_job_d.json`, tolerance 0.05 nats on `total`, `suf_sub`, `end_sub` and `first_sub`; counts identical.
   - **job A** (Qwen3-0.6B main, after the run): fields 1–7 against your job A file, every row. Integer fields identical; float fields within 0.01 nats.
   - **cache** (Qwen3-1.7B, Llama-3.1-8B, main set): 50 sentences against your own 1b span-cost file for that model. Check the four strings it has (it, there, did, then): `n_tok` identical and Spearman of `total` at least 0.999, as in job A.
   - **causality** (every model): `pre_check`. Report the maximum and how many rows exceed 1e-3. For bfloat16 models, flag any row over 0.05 per prefix token. Job A had 144 of 2,957,422 rows slightly over 1e-3 (largest 1.01e-03), all from batch numerics; that is fine.
   - Qwen3-8B and OLMo-3 have no earlier file. For them causality plus the finite/complete checks are the test; we compare across models here.

## Output format

Line 1 is `{"meta": {...}}`: the same fields as job A (`job`, `date`, `model_id`, `revision`, `dtype`, `logits_dtype`, `versions`, `gpu`, `wall_seconds`, `reference_code_sha`, `reference_code_modified`, `checks`, `n_sentences`, `n_spans`, `fillers`, `bos_id`, `batch`, `notebook`). Set `job` to `"d"` and add:
- `set` (`main` or `heldout`);
- `sentences_sha256` (of `sentences_3161.jsonl`);
- `bos_token` (the token string).

Every later line is one sentence, in the file's order:

| key | content |
|---|---|
| `id` | as in `sentences_3161.jsonl` |
| `end` | the end mark used |
| `orig` | `text`, `ids`, `lp` (per text token), `end_ids`, `end_lp` (per end-mark token) |
| `rows` | `rows[string]["i,j"]` = 8 numbers: `total, n_tok, n_pre, n_suf, suf_sub, end_sub, pre_check, first_sub`, for all 17 strings and every key of `span_keys(n)` |

Numbers are plain floats and integers. Log-probs are natural logs (nats), not rounded.

## Order (a delivery order, not a cut)

| tier | when | what |
|---|---|---|
| 1 | by Sun 4 Oct | Qwen3-0.6B main + held-out, Qwen3-1.7B main, Qwen3-8B main, Llama-3.1-8B main |
| 2 | by Tue 6 Oct | OLMo-3-7B main, GPT-2 main + held-out, Qwen3-1.7B held-out |
| 3 | by Thu 8 Oct | held-out for Qwen3-8B, Llama-3.1-8B, OLMo-3-7B |

Run models in parallel on separate GPUs if you can. **Please do not trim** sentences, spans or strings: every span is ranked against all spans of its length. If something will not fit, tell us first and we will choose what to cut.

## What to send back
- Per model and set: `tree_t4_<model>_<set>.jsonl.gz` and `checks_<model>_<set>.json` (tokenizer, anchor or cache, causality, and for Qwen3-0.6B main the job A comparison).
- Your notebook.

If anything is unclear, or you have suggestions (batching, a prefix cache, a different model), please ask. A prefix KV cache is welcome only if it reproduces the anchors within tolerance.
