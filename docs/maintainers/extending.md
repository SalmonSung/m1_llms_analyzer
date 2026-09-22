---
title: Extending and maintaining
---

# Extending and maintaining

## Working locally

```bash
pip install -e ".[dev]"
python scripts/smoke_test.py --offline    # builds a tiny model; no network
pytest -q                                 # ~310 tests, offline, under a minute
```

The suite builds a random GPT-2 on disk (`m1_analyzer.testing.build_tiny_local_model`)
and loads it through the same `from_pretrained` path a real model uses, so it needs no
Hub access. The Task 1b analyses are tested against `FakeSpanScorer`, a scorer that knows
the gold tree, because the tiny word-level tokenizer cannot spell the proforms; the Task
9a and 9b pipelines run on the tiny model itself, whose vocabulary carries sentence
punctuation and capitalised openers for that purpose. The NLTK loader test is skipped
until `nltk.download("treebank")` has run once.

## Extension points

| Goal | Touch |
|---|---|
| New pooling mode | `utils/pooling.py` (add to `POOLERS`), `config/settings.py` (`POOLING_MODES`), a test in `tests/test_pooling.py` |
| New output format (Parquet, HDF5) | A class satisfying `ResultSink`; pass it as `Analyzer(storage_service=...)` |
| Attention weights as well as hidden states | `inference_service._forward` (`output_attentions=True`), `domain/records.py`, `storage_service` |
| Encoder-decoder support | `model_service._reject_unsupported` and `_forward` (decoder states are a separate output) |
| Images for a multimodal model (text-only already works) | `model_service` (`AutoProcessor` beside the tokenizer), `inference_service._forward` (`pixel_values`) |
| Remote or HTTP model host | A class satisfying `ModelProvider`; nothing else changes |
| **A new experiment task** | New `experiments/task_<id>.py` producing the record its `fig_<id>` docstring in `experiment_figures.py` specifies; reuse `jsonl_cache.py`, `stats.py`, and the 1b / 9a building blocks; a notebook copied from `experiment_1b.ipynb` or `experiment_9a.ipynb`; a page under `docs/tasks/` |
| A state-level experiment (distances between next-token vectors) | `analyzer.states.states(ids, positions)`; see `experiments/splice.py` for the alignment bookkeeping |
| A generation experiment (text after a context) | `experiments/decoding.py` with `analyzer.models` as the provider; see `experiments/task_9b.py` for the per-cut driver, cache and record |
| A multi-model experiment (the same items on several checkpoints) | `experiments/task_8a.py`: a `MODELS_8A`-style registry, a tokenizer-only phase 0 that pins the items, one resumable cache per model, a CPU merge from any runtime; `ModelConfig(device_map="auto")` for checkpoints larger than the CPU RAM |
| Another treebank (Universal Dependencies) | `experiments/treebank.py` (`load_ud_conllu` is a documented stub: subtree yields → spans, drop non-projective); name the conversion in the record's `treebank` field |
| Another replacement policy | A class satisfying `ReplacementPolicy` in `experiments/proforms.py`; if its proforms are a subset of a cache's header, phase B alone suffices |

### The shape of a new task

1. **Fixtures first**: a hand example that needs no model (`treebank.hand_examples`,
   `paragraphs.hand_paragraphs` are the pattern), so the smoke test and the docs can run
   offline.
2. **Phase A** writes a JSONL cache through `jsonl_cache`: `write_header` with everything
   fixed before scoring, `check_header` on resume, `append_row` per item, `mirror` to
   Drive. Store raw quantities, derive the analysed ones on read.
3. **Phase B** reads the cache back and builds the record the figure expects;
   `validate_record` and `verdict` sit beside it.
4. **Tests** on the tiny model or a fake scorer, plus a synthetic cache with and without a
   planted effect for the analysis.
5. **The notebook** copies the shared skeleton (bootstrap → dependencies → configuration
   → smoke test → model → data → one item → phase A → phase B → figure → save → Drive).
6. **Docs**: a line per new file in `architecture.md` (the test insists), a page under
   `docs/tasks/` from the existing template, an entry in `mkdocs.yml`'s `nav`, and
   examples added to `scripts/make_doc_examples.py`.

## Maintenance rules

### `architecture.md` is part of the contract

`tests/test_architecture_doc.py` fails the build if any `.py`, `.ipynb` or `.md` file
under `src/`, `scripts/`, `tests/`, `notebooks/` or `docs/` is not named in
`architecture.md`. When a file is added, renamed or removed, or changes responsibility,
update the tree and its one-line description in the same commit. The same test asserts
that `docs/design_decisions.md` and `docs/edge_cases.md` exist.

### Notebooks are tested as text

`tests/test_notebook.py` checks every notebook: valid nbformat, every source line keeps
its trailing newline, code cells compile when joined the way a reader joins them, no
hardcoded secrets, header-based clone auth, no committed outputs. Edit the JSON, not a
Colab export with outputs in it.

### Design decisions and edge cases are living documents

A choice made without discussion goes into `docs/design_decisions.md` with its
reasoning (and, when relevant, what was deliberately not built). A pitfall handled in
code goes into `docs/edge_cases.md` with the code that handles it. Both pages are part
of this site unchanged.

### The docs site

- The site is built from `docs/` by `mkdocs.yml`; `README.md` and `architecture.md` stay
  at the repo root and are pulled in by the stubs `docs/index.md` and
  `docs/architecture.md`, so there is one source for each.
- `mkdocs build --strict` is the gate: a broken link, a missing snippet or a page absent
  from `nav` fails it. Run it before pushing docs changes; the `docs` workflow runs it on
  every push to `main` and deploys the result to GitHub Pages.
- The examples under `docs/assets/examples/` are generated, not written:
  `python scripts/make_doc_examples.py` regenerates them (the tiny-model artifacts need
  torch; the rest does not), and `python scripts/make_doc_examples.py --check` fails
  when the committed torch-free fragments have drifted from the code. Keep that folder
  free of `.md` and `.py` files.
- A new page goes in three places: the file under `docs/`, `mkdocs.yml`'s `nav`, and the
  `docs/` block of `architecture.md`.

```bash
pip install -r requirements-docs.txt
mkdocs serve                              # http://127.0.0.1:8000 with live reload
mkdocs build --strict                     # what CI runs
python scripts/make_doc_examples.py       # regenerate docs/assets/examples/
```

### Continuous integration

Two workflows under `.github/workflows/`:

| Workflow | Runs on | Does |
|---|---|---|
| `ci.yml` | pushes to `main`, pull requests | CPU torch, `pip install -e ".[dev]"`, `pytest -q`, `scripts/smoke_test.py --offline`, `scripts/make_doc_examples.py --check` |
| `docs.yml` | pushes to `main` that touch the docs, pull requests (build only), manual dispatch | `mkdocs build --strict`, then deploy to GitHub Pages through the Actions source |

Pages must be set to **Source: GitHub Actions** once, under the repository's
*Settings → Pages*.

## Deliberately not built

See [Deliberately not built yet](../design_decisions.md#deliberately-not-built-yet) for
the list and the reasoning: no HTTP between services, no YAML configuration, no
encoder-decoder support, no attention extraction, no exact torch pin, and the Task 9b and
Task 8a analyses kept outside the repository.
