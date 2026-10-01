---
title: Job E — masked LMs
---

# Job E: job D's substitutions scored by masked language models

## The question

Job D's causal models predict each word from the words before it. For a span that ends the sentence, the only
thing they read after the replacement is the end mark, and their predictions of the words before the span never see
the replacement. Job E scores **exactly the same substitutions** with masked language models, which predict every
word from both sides, to find out whether that blind spot comes from the reading direction. The request is
`data/job_e/runner_note_job_e.md`.

| model | kind | tier | output |
|---|---|---|---|
| RoBERTa-base | masked, pairs with GPT-2 (job D) | 1 main, 3 held-out | `tree_mlm_roberta-base_<set>.jsonl.gz` |
| GPT-2-medium | causal, job D's code unchanged | 2 main, 3 held-out | `tree_t4_gpt2-medium_<set>.jsonl.gz` |
| RoBERTa-large | masked, pairs with GPT-2-medium | 2 main, optional held-out | `tree_mlm_roberta-large_<set>.jsonl.gz` |
| ModernBERT-large | masked, a 2024 model | optional main | `tree_mlm_modernbert-large_main.jsonl.gz` |

Everything runs in float32 with TF32 off (the note: bfloat16 moved one anchor sentence by up to 0.087 nats per token).

## The score

**PLL-word-l2r** (Kauf & Ivanova, ACL 2023): for every token, its log-probability when that token *and every later
token of the same word* are replaced by the mask token. The model reads `<s> text end-mark </s>`. The reference is
`experiments/mlm_runner_ref.py`, byte-identical (sha256 `b3dc1941…`). It imports job D's v4 for the texts, the
span keys and the frame rule. A row has 9 numbers:

`total, n_tok, n_pre, n_suf, pre_sub, suf_sub, end_sub, first_sub, last_pre_sub`

`pre_sub` is the summed score of the tokens before the span in the substituted sentence. Unlike a causal model's,
it reacts to the replacement. `last_pre_sub` is the token just before the replacement.

## The engine

`mlm_rows` scores one sentence at a time, 256 masked copies per pass. `job_e.score_sentences_mlm`:
1. builds the same substitutions and word ids with the reference's helpers (batched tokenisation);
2. makes one masked copy per token for every sequence of a group of sentences (`GROUP_SUBS` substitutions);
3. sorts the copies by length and packs them under job D's auto-sized token budget. Inputs are built with numpy;
4. reads the logits through the reference's `masked_logits` (RoBERTa: the head on the masked position only), then
   `log_softmax` in float32, gathered at the target. Values are float64, as in `pll`.

`calibrate_budget_mlm` is job D's probe-and-sweep with a masked probe.

## The checks (per model and set, before the full run)

| check | pass | blocking |
|---|---|---|
| `tokenizer` | masked: `mlm_checks` counts **equal the note's table exactly** (RoBERTa main 1,000 / 2,957,422 / 53,960,342 / 0 / 0; held-out 2,161 / 6,664,952 / 122,230,977 / 0 / 0; ModernBERT 54,485,905 / 123,737,010 forwards). GPT-2-medium: job D's `tokenizer_checks` gates | yes |
| `engine_equivalence` | the engine against the unmodified `mlm_rows` (or `t4_rows`) on 3 sentences: integers identical, every float within 0.01 nats | yes |
| `anchor` | `anchor_job_e.json`'s five sentences: integers identical, every float field within 0.05 nats | yes |
| `timing` | the first 20 sentences, timed; projected hours for the set (cost = spans × words², or spans × words for GPT-2-medium), printed and pushed to Drive before the full run | reported |
| `causality` | GPT-2-medium only | reported |
| `complete` (at the end) | every sentence × string × span present, every number finite; 9 numbers per masked row, 8 for GPT-2-medium | yes |

## The queue

`notebooks/experiment_job_e.ipynb`:
1. tokenizer checks for every queued model, cached per tokenizer and set (RoBERTa-base and -large share one);
2. the items in delivery order (`INCLUDE_OPTIONAL` adds the optional ones). Each item gets the model, auto-sizing,
   the checks, the full run and the deliverable, and is pushed to Drive `m1_llms_analyzer/tree_runner_job_e/`.

`PAUSE_AFTER_TIMING` stops after an item's timing, for when you want the requesters' go-ahead first. Job D has
priority: run job E on a GPU job D is not using.

## Output

`tree_mlm_<model>_<set>.jsonl.gz`. Line 1 is `{"meta": ...}` with `job: "e"`, `date`, `model_id`, `revision`,
`dtype`, `versions`, `gpu`, `wall_seconds`, `reference_code_sha` (of `mlm_runner_ref.py`), `tree_runner_ref_sha`,
`reference_code_modified`, `checks`, `n_sentences`, `n_spans`, `fillers`, `set`, `sentences_sha256`,
`score: "PLL-word-l2r"`, `batch` (the token budget) and `notebook`, plus `engine`, `token_budget`, `tf32` and
`calibration`. Then `{"id", "end", "orig", "rows"}` per sentence in file order. GPT-2-medium's file is job D's
format.
