---
title: Job D — T4 on six models
---

# Job D: frame-only substitution scores on six models, main and held-out

## The question

Job A scored one pass on Qwen3-0.6B-Base over the 1,000 main sentences: every span of every sentence, replaced by
each of 17 strings, scored token by token with the sentence's own end mark (`t4_rows`). Job D repeats that pass so
the findings are not one model's, and extends it to a held-out set of 2,161 sentences they were not found on. The
request is `data/job_d/runner_note_job_d.md`.

| model | dtype | sets | tier |
|---|---|---|---|
| Qwen3-0.6B-Base | float32 | held-out (main is job A) | 1 |
| Qwen3-1.7B-Base | float32 | main, held-out | 1, 2 |
| Qwen3-8B-Base | bfloat16 | main, held-out | 1, 3 |
| Llama-3.1-8B | bfloat16 | main, held-out | 1, 3 |
| OLMo-3-1025-7B | bfloat16 | main, held-out | 2, 3 |
| GPT-2 | float32 | main, held-out | 2 |

Main has 2,957,422 substitutions and held-out has 6,664,952. Everything runs on **one A100-40GB**.

## The reference code

`experiments/tree_runner_ref_v2.py` is the package's reference v2, **byte-identical** (sha256
`1a4fe95fdbfc…`, pinned by `tests/test_job_d.py`). It changes two things from v1:
- `bos_id(tok)`: the tokenizer's own BOS, otherwise `<|endoftext|>`;
- `span_keys(n)`.

v1 (`tree_runner_ref.py`) stays, so jobs A and C remain reproducible.

## The engine

`ref.t4_rows` scores one sentence at a time, in file order, 64 substitutions per pass. `job_d.score_sentences`
computes the same values faster:

1. **Same jobs.** `prepare` builds `(filler, key, ids, n_pre, n_suf)` for each substitution with the reference's
   own `text_of`, `encode`, `sub_text`, `word_chars` and `_common_prefix`, with the same caps and the same overlap
   clamp.
2. **Better batches.** The originals and substitutions of several sentences (about `GROUP_SEQS` = 20,000) are
   pooled, sorted by length and cut into batches of at most `budget` padded tokens. This removes most padding and
   keeps the GPU full even for 5-word sentences.
3. **Same pass.** `[BOS] + ids + end_ids`, right-padded with an attention mask; `log_softmax(logits.float())`
   gathered at the next token, run on row chunks so the float32 copy stays small. The rows use the reference's
   formulas.

A prefix KV cache was not used: it would make `pre_check` zero by construction.

## Auto-sizing

`calibrate_budget(mdl, bos, device, max_len=...)`:
1. Estimate the bytes one padded token costs (the logits in the model's dtype, their float32 copy, and
   activations). Divide 85 % of the free memory after the weights by that.
2. Run a pass at that budget with the longest sequence that can occur. Halve on out-of-memory until it fits.
3. Time the budget and ½ and ¼ of it at a typical length. Keep the fastest, or the largest within 3 % of the fastest.

`forward_safe` still halves a batch that runs out of memory mid-run and shrinks the budget for later batches. The
budget and the number of backoffs go into `checks` and the meta block.

## The checks — per model and set, before the full run

| check | models | pass | blocking |
|---|---|---|---|
| `engine_equivalence` | all | the engine against the unmodified `ref.t4_rows` on 3 sentences (the shortest, a middle one and the longest of the first 50). Integer fields and original ids identical. float32: every float within 0.01 nats. bfloat16: median within 0.05 nats and Spearman(total) ≥ 0.9999 | yes |
| `anchor` | Qwen3-0.6B, GPT-2 | the note's: `anchor_job_d.json`'s sentences, counts identical, `total` / `suf_sub` / `end_sub` within 0.05 nats | yes |
| `cache` | Qwen3-1.7B, Llama-3.1-8B (main) | the note's: the first 50 main sentences against the 1b span-cost file, on it / there / did / then. `n_tok` identical, Spearman(total) ≥ 0.999 | yes |
| `extra_cache` | Qwen3-8B (main) | the same against Qwen3-8B's 1b file. The note says 8B has no earlier file, but the 1b run left one on Drive | no, reported |
| `causality` | all | the note's: max \|`pre_check`\|, rows over 1e-3, and for bfloat16 rows over 0.05 per prefix token | no, reported |
| `complete` (at the end) | all | every sentence × filler × span present, every number finite | yes |

Every model also checks its load. The full revision sha must start with the note's revision, and must equal the
anchor's where the anchor has one. The BOS id and token must be the note's and the anchor's `bos` block (Qwen3
151643, OLMo-3 100257, GPT-2 50256, Llama `<|begin_of_text|>`). The dtype must be the note's. TF32 is off.

## The queue

`notebooks/experiment_job_d.ipynb` runs `job_d.QUEUE`, the note's delivery order, smallest first within a tier:

1. tier 1: Qwen3-1.7B main, Qwen3-0.6B held-out, Llama-3.1-8B main, Qwen3-8B main;
2. tier 2: GPT-2 main + held-out, Qwen3-1.7B held-out, OLMo-3 main;
3. tier 3: held-out for Qwen3-8B, Llama-3.1-8B, OLMo-3.

A model is loaded once for consecutive items. After its checks, each item prints a rough ETA against its tier's due
date. While it runs, the progress bar shows the budget and a live ETA in hours. A finished item is skipped. After a
disconnect, re-running all cells resumes the current item from its working cache (restored from Drive).

## Files

All under `outputs/job_d/`, pushed to Drive `m1_llms_analyzer/tree_runner_job_d/`.

| file | what |
|---|---|
| `tree_t4_<model>_<set>.jsonl.gz` | the deliverable. First line `{"meta": ...}`: job A's fields (model, full revision, dtype, versions, GPU, wall time, reference sha, `reference_code_modified`, checks) plus `set`, `sentences_sha256`, `bos_token`, `bos_id`, `engine`, `token_budget`, `tf32`. Then `{"id", "end", "orig", "rows"}` per sentence in file order |
| `checks_<model>_<set>.json` | every check's numbers, the calibration, the ETA estimate |
| `work_t4_<model>_<set>.jsonl` | the resumable working cache: a header (model, revision, dtype, set, sha256s), then one line per sentence with `seconds` |
| `span_costs_*_min-over-it-there-did-then.jsonl` | the 1b caches of Qwen3-1.7B, Llama-3.1-8B and Qwen3-8B, copied from `task_1b/` for the checks |

**Drive reuse.** Nothing else on Drive can stand in for a job D score:
- the 1b caches hold `total` and `n_tok` for 4 fillers without the end mark, and `suf_sub`, `end_sub` and
  `pre_check` need the forward pass anyway;
- Qwen3-0.6B main is job A's `tree_t4_qwen3_0.6b.jsonl.gz`, which the note does not ask for again.

## Configuration knobs

| knob | default | effect |
|---|---|---|
| `ITEMS` | `all` | the queue, or e.g. `qwen3-8b:main,gpt2:heldout` |
| `LIMIT` | 0 | score only the first `LIMIT` sentences per item (a dry run; no deliverable) |
| `GROUP_SEQS` | 20000 | substituted sentences per packing pool: larger packs better, a disconnect loses more |
| `MIRROR_MINUTES` | 15 | how often the working cache is copied to Drive |
| `STOP_ON_FAIL` | True | stop the queue on a failed blocking check |
