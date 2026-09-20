---
title: Tasks at a glance
---

# The pipelines at a glance

This repository runs one plain pipeline (hidden-state extraction) and three numbered
experiments on the same loaded model. Every one of them is a Colab notebook that clones
this repo and calls into the `m1_analyzer` package; the notebooks hold configuration and
glue, the package holds the logic.

| Task | Question | Notebook | Entry functions | Data source | Outputs |
|---|---|---|---|---|---|
| [Hidden-state extraction](extraction.md) | Save selected layers' hidden states for some texts as JSON | `colab_entrypoint.ipynb` | `Analyzer.invoke / batch / save`, `load_run` | Texts you supply | `outputs/<name>.json` (+ `.npz`) |
| [Task 1b](task_1b.md) | Does substitution cost *find* constituents? | `experiment_1b.ipynb` | `load_ptb_nltk` → `compute_span_costs` → `analyse_1b` → `fig_1b` | NLTK Penn Treebank sample (gold brackets) | `span_costs_*.jsonl`, `record_1b_*.json`, `fig_1b_*.png`, `fig_0c_*.png` |
| [Task 9a](task_9a.md) | Do cuts between *close* endpoint states change what follows less than cuts between *far* ones? | `experiment_9a.ipynb` | `load_wikipedia_paragraphs` → `compute_splices` → `add_surprisal` → audit → `analyse_9a` → `fig_9a` | Streamed English Wikipedia paragraphs | `paragraphs_*.jsonl`, `splices_*.jsonl`, `audit_*.csv`, `record_9a_*.json`, `fig_9a_*.png` |
| [Task 9b](task_9b.md) | Does the deletion change what the model *writes*, not only what it predicts? | `experiment_9b.ipynb` | `load_9a_inputs` → `compute_generation_9b` → `build_record_9b` | Task 9a's record and cache (read-only) | `gen_9b_*.jsonl`, `record_9b_*.json` (no figure) |

## How is the data collected?

| Task | Where the text comes from | Which function fetches it | What is stored |
|---|---|---|---|
| Extraction | You: `--text`, a file of lines, a JSON array or JSONL with a `text` key, or the notebook's `INPUTS` list | `cli.read_inputs` / the notebook | The texts travel inside the output JSON (`include_input_text`) |
| Task 1b | The free NLTK Penn Treebank sample (10 % of WSJ, 3 914 parsed sentences), downloaded once with `nltk.download("treebank")` | `treebank.load_ptb_nltk(n, min_len, max_len, seed)`: seeded sample of eligible sentences, gold brackets derived from the trees | Every cache row carries the sentence, its gold spans and its bracketed tree, so phase B needs no treebank |
| Task 9a | English Wikipedia, `wikimedia/wikipedia` `20231101.en`, **streamed** through `datasets` (no bulk download) | `paragraphs.load_wikipedia_paragraphs(count_tokens, n, ...)`: clean → token-count and sentence-count filter → seeded sample from a pool | `paragraphs_<RUN_TAG>.jsonl` pins the draw; the notebook's *Pin the corpus* cell restores it by id on a later run |
| Task 9b | Nothing new: Task 9a's `record_9a_*.json` (the labelled cuts) and `splices_*.jsonl` (texts and per-token surprisal) | `task_9b.load_9a_inputs(record, cache)`, sha256 of both stamped into the 9b cache header | The 9a files are never written to |

## The shared notebook skeleton

All four notebooks have the same shape, so once you have read one you can read the others:

1. **Bootstrap**: clone or update the repo with a `GITHUB_TOKEN` Colab secret, scrub the token from `.git/config`.
2. **Dependencies**: `pip install` with `--upgrade-strategy only-if-needed`, so Colab's CUDA-matched torch stays.
3. **Environment report**: GPU, versions.
4. **Configuration**: `#@param` form fields → a `RunConfig`; the `RUN_TAG` and every output path are derived here.
5. **Smoke test**: the whole experiment on `sshleifer/tiny-gpt2` and the hand fixtures, in seconds.
6. **Load the model** (`Analyzer`), then **load the data**.
7. **One item end to end**: a didactic walk-through of a single sentence, paragraph or cut.
8. **Phase A (GPU, resumable)**: the expensive scoring pass, appended to a JSONL cache with a header that records the pre-registration.
9. **Phase B (CPU, seconds)**: the analysis, re-runnable from the cache without a GPU.
10. **Figure → atomic save → copy to Drive → download → `pytest` inside Colab.**

Three pieces of logic live only in the notebooks and not in the package: the bootstrap
cell, an `atomic_write_json` helper used to save the records, and Task 9a's *Pin the
corpus* cell.

## Naming: `RUN_TAG`

Every output file is keyed by what produced it, so two runs never share a cache:

| Task | `RUN_TAG` | Example |
|---|---|---|
| 1b | `slug(MODEL_ID)_slug(POLICY.name)` | `span_costs_Qwen-Qwen3-0.6B-Base_min-over-it-there-did-then-.jsonl` |
| 9a | `slug(MODEL_ID)_slug(CORPUS)_w<WINDOW>` | `splices_Qwen-Qwen3-0.6B-Base_wikipedia_w20.jsonl` |
| 9b | the same tag as 9a, so the 9a files are found by name | `gen_9b_Qwen-Qwen3-0.6B-Base_wikipedia_w20.jsonl` |

`slug` replaces anything that is not `[A-Za-z0-9._-]` with `-`.

## Resumable caches

Phase A of every experiment appends one JSON line per item to a cache, `flush` + `fsync`
after each line, behind a header line that records everything fixed before scoring
(model, proforms or window, strata, decile, matching width, boundary rule). A re-run with
`RESUME=True` reads the cache back, refuses if any checked header field differs (naming
the field), repairs a truncated last line, and continues from the first unscored item.
The mechanics live in `experiments/jsonl_cache.py` and are shared by all three tasks.
With `MIRROR_TO_DRIVE=True` the cache is copied to Google Drive every few items, and
restored from there when the local file is missing.
