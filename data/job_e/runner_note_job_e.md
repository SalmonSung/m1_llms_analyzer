# Runner note: job E (masked language models on job D's substitutions)

**What and why.** Job D scores each substitution with causal models, which predict every word from the words *before* it.
- For a span that ends the sentence, the only thing such a model reads after the replacement is the end mark, and its predictions of the words before the span cannot see the replacement at all.
- Job E scores **exactly the same substitutions** with masked language models, which predict every word from **both sides**. The question is whether the sentence-end blind spot comes from reading direction.
- The models are chosen in matched pairs. RoBERTa uses GPT-2's byte-pair merges, so a RoBERTa model and a GPT-2 model of the same size see the same token strings and the same frames. We checked this on all 9,622,374 substituted sentences and the 3,161 originals: 0 differ.

## Files

| file | what |
|---|---|
| `mlm_runner_ref.py` | **new**: masked scoring (`load_mlm`, `pll`, `mlm_rows`, `mlm_checks`) |
| `tree_runner_ref.py` | job D's **v4, unchanged** (sha `3b1e5aac659e`); imported for the texts, the span keys and the frame rule |
| `sentences_3161.jsonl` | the same file as job D (sha `f2a25c1c18a9`) |
| `anchor_job_e.json` | float32 CPU rows for all four models on job D's five anchor sentences |

## The score

**PLL-word-l2r** (Kauf & Ivanova, ACL 2023): for every token, its log-probability when that token *and every later token of the same word* are replaced by the mask token. The model sees its own start token, the text, the end mark and its own end token (`<s> … </s>` for RoBERTa, `[CLS] … [SEP]` for ModernBERT), as one sequence.
- Words are the tokenizer's own word ids, as in the authors' library (minicons, `PLL_metric="within_word_l2r"`).
- The end mark is appended as its own tokens, exactly as in job D.

**Tested here** (CPU, float32):
- Our `pll` equals minicons' token scores on 37 of the 40 shortest sentences for RoBERTa-base (largest difference 3.4e-05 nats) and on 14 of the 15 shortest for ModernBERT-large (4.1e-05). The rest were skipped because minicons scores the attached string, which tokenizes the end mark differently there.
- For RoBERTa, the output head runs only on the masked position, which is identical to the full logits within 1.2e-05. Other models take the full logits.
- `mlm_rows` gives the same `n_tok`, `n_pre`, `n_suf` as job D's GPT-2 anchor rows: 765 rows, 0 differ.
- Unlike a causal model, the words before the span do react to the replacement: median change in the prefix sum 7.3 nats on the anchors.

## Models (all float32)

| model | revision | kind | role | rough GPU hours, main / held-out |
|---|---|---|---|---|
| `FacebookAI/roberta-base` | `e2da8e2f811d1448a5b465c236feacd80ffbac7b` | masked | pairs with GPT-2 (124M, job D) | 6 h / 14 h |
| `FacebookAI/roberta-large` | `722cf37b1afa9454edce342e7895e588b6ff1d59` | masked | pairs with GPT-2-medium | 21 h / 49 h |
| `openai-community/gpt2-medium` | `6dcaa7a952f72f9298047fd5137cd6e4f05f41da` | causal | pairs with RoBERTa-large; job D's code, unchanged | 1.3 h / 3 h |
| `answerdotai/ModernBERT-large` | `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13` | masked | optional: a 2024 masked model | 28 h / 64 h |

- **float32 is required.** bfloat16 differed from float32 by up to 0.087 nats per token on one anchor sentence, too much for these comparisons. Keep TF32 matmuls off (PyTorch's default). If a run is too slow, tell us before changing precision.
- The hours are a **rough projection**: job A's measured float32 rate on NVIDIA A100-SXM4-40GB, scaled by each model's compute per pass. Small models often use a GPU less efficiently, so please **time the first 20 sentences and send the projected hours** before the full run.

## Procedure

1. **Masked models.** `tok, mdl = load_mlm(dir, revision)`. For each row of the set: `mlm_rows(tok, mdl, row['words'], span_keys(len(row['words'])), row['end'])`, with the 17 default fillers.
2. **GPT-2-medium.** Job D's procedure and output format, unchanged (`t4_rows`, `bos_id`).
3. **Speed.** `batch` is the number of masked copies per forward pass (default 256); raise it on a GPU. `chunk` is the number of substituted sentences per `pll` call. Sorting copies by length, or any other batching, is fine as long as the anchors still pass.
4. **Checks before each full run,** written to `checks_<model>_<set>.json` and to `meta.checks`:
   - **tokenizer**: `mlm_checks(tok, rows_of_that_set, causal_tok=...)`. Pass the GPT-2 tokenizer for the RoBERTa models and `None` for ModernBERT. The counts are deterministic and must equal these exactly; **if any differs, stop and tell us.**

| tokenizer | set | sentences / substitutions / masked passes | splits differing from GPT-2 | frames differing |
|---|---|---|---|---|
| RoBERTa (base and large share the files) | main | 1,000 / 2,957,422 / 53,960,342 | 0 | 0 |
| RoBERTa | held-out | 2,161 / 6,664,952 / 122,230,977 | 0 | 0 |
| ModernBERT-large | main | 1,000 / 2,957,422 / 54,485,905 | not compared | not compared |
| ModernBERT-large | held-out | 2,161 / 6,664,952 / 123,737,010 | not compared | not compared |

   - **anchor**: every model, the sentences in `anchor_job_e.json`. Integer fields identical; float fields within 0.05 nats.
   - **finite and complete**: every row has all 17 strings and every key of `span_keys(n)`; every number is finite. Masked rows have 9 numbers, GPT-2-medium rows 8.
   - **timing**: the first 20 sentences, as above.

## Output format

- **Masked models:** `tree_mlm_<model>_<set>.jsonl.gz`, e.g. `tree_mlm_roberta-base_main.jsonl.gz`.
  - Line 1 is `{"meta": {...}}`: `job` (`"e"`), `date`, `model_id`, `revision`, `dtype`, `versions`, `gpu`, `wall_seconds`, `reference_code_sha` (of `mlm_runner_ref.py`), `tree_runner_ref_sha`, `reference_code_modified` (must be false), `checks`, `n_sentences`, `n_spans`, `fillers`, `set`, `sentences_sha256`, `score` (`"PLL-word-l2r"`), `batch`, `notebook`.
  - Every later line is one sentence, in the file's order:

| key | content |
|---|---|
| `id` | as in `sentences_3161.jsonl` |
| `end` | the end mark used |
| `orig` | `text`, `ids`, `lp` (PLL-word-l2r per text token), `end_ids`, `end_lp` |
| `rows` | `rows[string]["i,j"]` = 9 numbers: `total, n_tok, n_pre, n_suf, pre_sub, suf_sub, end_sub, first_sub, last_pre_sub` |

  - `pre_sub` is the summed score of the tokens before the span in the substituted sentence; `last_pre_sub` is the token just before the replacement. The other fields mean what they mean in job D.
- **GPT-2-medium:** `tree_t4_gpt2-medium_<set>.jsonl.gz`, job D's format.
- Plain floats and integers; natural logs (nats), not rounded.

## Order (a delivery order, not a cut)

| tier | when | what | rough GPU hours |
|---|---|---|---|
| 1 | Sun 4 Oct | RoBERTa-base main | 6 |
| 2 | Tue 6 Oct | GPT-2-medium main, RoBERTa-large main | 23 |
| 3 | Thu 8 Oct | RoBERTa-base held-out, GPT-2-medium held-out | 17 |
| optional | if GPUs are free | RoBERTa-large held-out, ModernBERT-large main | 77 |

- **Job D comes first.** Run job E on GPUs job D is not using; tier 1 of job E is a single small model. If both cannot fit, tell us and we will choose.
- Results that arrive after Fri 9 Oct cannot go into the Mon 12 Oct submission.
- **Please do not trim** sentences, spans or strings. If something will not fit, tell us first.

## What to send back
- Per model and set: the output file and `checks_<model>_<set>.json` (tokenizer, anchor, finite/complete, timing).
- Your notebook.

If anything is unclear, or you have suggestions (batching, a different masked model), please ask.
