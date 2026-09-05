# Architecture

> **Maintenance rule:** this file is part of the contract. Whenever a file or folder is
> added, renamed, removed, or changes responsibility, update the tree **and** its
> description here in the same commit. `tests/test_architecture_doc.py` fails the build
> if any `.py`, `.ipynb`, or `.md` file under `src/`, `scripts/`, `tests/`, `notebooks/`,
> or `docs/` is not mentioned below.

## What this repo is

An in-process **microservice** pipeline that loads an open-source Hugging Face model,
extracts hidden states from selected layers, and saves them as JSON — and, on the same
loaded model, scores sentence log-probabilities for a series of sentence-structure
experiments (`experiments/`). It is driven from Google Colab notebooks that clone this
private repo and install everything themselves: one for extraction, one per experiment.

"Microservice" here means **separately-owned services with narrow contracts**, wired in one
place, running in one process. Each service knows nothing about the others' internals — they
depend only on the Protocols in `services/interfaces.py`. There are no ports and no HTTP:
inside a single Colab VM that would mean multiple processes and a model loaded per service,
for no benefit. Because the seams are Protocols, any service can later be moved behind HTTP
by writing a client that satisfies the same Protocol, with no change to pipeline code.

## Structure

```
m1_llms_analyzer/
├── README.md                      Quickstart: Colab flow, local install, API tour.
├── architecture.md                This file: structure tree + per-file responsibility.
├── pyproject.toml                 Package metadata, deps, pytest config, `m1-extract` script.
├── requirements.txt               Runtime deps. torch deliberately unpinned (see design doc).
├── requirements-dev.txt           requirements.txt + pytest.
├── .gitignore                     Ignores outputs/, caches, *.npy/*.npz, .venv, notebook checkpoints.
│
├── docs/
│   ├── design_decisions.md        Every choice made without discussion, and why.
│   └── edge_cases.md              Known pitfalls, how each is handled, and what is not.
│
├── notebooks/
│   ├── colab_entrypoint.ipynb     THE extraction entrypoint. Paste into Colab, Run all.
│   └── experiment_1b.ipynb        Task 1b (does substitution FIND constituents?): same
│                                  bootstrap, then smoke test -> treebank -> model ->
│                                  one-sentence walkthrough -> phase A scoring (resumable)
│                                  -> phase B analysis -> fig_1b / fig_0c -> Drive.
│
├── src/m1_analyzer/
│   ├── __init__.py                Public API surface: re-exports Analyzer, configs, records.
│   ├── container.py               `Analyzer` facade + wiring. Builds the services from one
│   │                              RunConfig; owns the loaded model for the session;
│   │                              exposes invoke / batch / save / run (hidden states),
│   │                              score / score_one (log-probabilities, when the model was
│   │                              loaded with head="causal_lm"; `Analyzer.for_scoring`),
│   │                              describe / unload, and builds the RunManifest.
│   ├── cli.py                     argparse entrypoint (`m1-extract`). Same pipeline from a
│   │                              terminal or CI; reads inputs from flags, lines, JSON, JSONL.
│   ├── testing.py                 Builds a tiny random GPT-2 + tokenizer on disk, offline.
│   │                              Backs the test suite and `smoke_test.py --offline`.
│   │                              Also owns TINY_LAYERS / TINY_HIDDEN /
│   │                              TINY_MAX_POSITIONS, the shape tests assert against, so
│   │                              no test needs to import `tests.conftest`; and
│   │                              FakeSpanScorer, a SequenceScorer that knows the gold
│   │                              tree, for testing the experiment analyses without a
│   │                              model (the tiny tokenizer cannot spell the proforms).
│   │
│   ├── config/
│   │   ├── __init__.py            Re-exports the config dataclasses.
│   │   └── settings.py            ModelConfig (incl. head: base | causal_lm) /
│   │                              ExtractionConfig / ScoringConfig / StorageConfig / RunConfig.
│   │                              Validation in __post_init__ so bad settings fail at
│   │                              construction, not mid-forward-pass. RunConfig.fingerprint()
│   │                              hashes the settings; to_dict() strips any token.
│   │
│   ├── domain/
│   │   ├── __init__.py            Re-exports the record types.
│   │   └── records.py             The data services exchange: LayerState, ExtractionRecord,
│   │                              ExtractionFailure, BatchResult, SentenceScore, ScoreResult,
│   │                              RunManifest, WrittenPaths, make_item_id (sha256 content
│   │                              id). Holds numpy, never torch, so results need no CUDA
│   │                              context to read.
│   │
│   ├── services/
│   │   ├── __init__.py            Re-exports the services.
│   │   ├── interfaces.py          The seams: ModelProvider, InferenceEngine, SequenceScorer,
│   │                              ResultSink Protocols. Services depend on these, never on each other.
│   │   ├── model_service.py       Only module that calls transformers' loading APIs. Owns
│   │                              the head choice (AutoModel vs AutoModelForCausalLM),
│   │                              device/dtype choice, HF token use, pad-token fallback,
│   │                              architecture rejection, layer-spec resolution, context
│   │                              length, and provenance metadata. Raises ModelLoadError /
│   │                              UnsupportedArchitectureError with actionable messages.
│   │   ├── inference_service.py   The forward pass. invoke() is batch() of one. Length-sorted
│   │                              batching, OOM halving, per-item failure isolation,
│   │                              truncation detection, layer picking, pooling, float32 cast.
│   │   ├── scoring_service.py     LogProbService: text -> log-probability (sum, per-token
│   │                              mean, optional per-token vector). Uniform BOS prefix,
│   │                              right padding, gather-minus-logsumexp in float32 (never a
│   │                              full log_softmax over a 150k vocabulary), non-finite
│   │                              scores recorded as failures, OOM halving.
│   │   └── storage_service.py     JSON writer (atomic), float rounding, .npz sidecar policy,
│   │                              and `load_run()` which rehydrates arrays on read.
│   │
│   ├── experiments/               The sentence-structure experiments (runbook tasks).
│   │   ├── __init__.py            Re-exports the public names below.
│   │   ├── experiment_figures.py  The runbook's figure module, vendored verbatim: every
│   │   │                          fig_<task>(record, path) plus mock_<task> data and the
│   │   │                          record schema each figure expects (in its docstring).
│   │   ├── stats.py               bootstrap_ci, paired_bootstrap_diff, trapezoid_area.
│   │   ├── spans.py               Span algebra: enumerate_spans, crosses, right/left/random
│   │   │                          baselines, greedy_induce (cheapest first, never cross),
│   │   │                          cky_induce (minimum total cost), bracket_prf, rank_curve.
│   │   ├── proforms.py            Blind replacement policies (ReplacementPolicy Protocol):
│   │   │                          MinOverSet, ByLengthClass, parse_policy, substitute.
│   │   ├── treebank.py            TreebankSentence + gold spans from trees (traces and
│   │   │                          punctuation removed, unaries collapsed), PTB detokeniser,
│   │   │                          load_ptb_nltk (the free NLTK WSJ sample), hand examples
│   │   │                          incl. the theory doc's sentence, and save/load_gold_jsonl
│   │   │                          (+ ptb_tree_strings) which export the answer key so a cost
│   │   │                          cache can be re-analysed with no corpus installed.
│   │   │                          UD is a documented stub.
│   │   ├── span_costs.py          Phase A: SpanCostTable (raw sum/count per proform x span,
│   │   │                          cost derived on read), compute_span_costs with a resumable
│   │   │                          JSONL cache (header check, fsync, truncated-line repair,
│   │   │                          Drive mirror), load_span_costs, variant_count.
│   │   └── task_1b.py             Phase B: evaluate_sentence, analyse_1b -> the fig_1b
│   │                              record (sentence-level mean F1 + bootstrap CI, by length,
│   │                              pooled rank curve, diagnostics), validate_record, verdict,
│   │                              run_task_1b (A then B).
│   │
│   └── utils/
│       ├── __init__.py            Marks the package; holds no logic.
│       ├── env.py                 Secret resolution (Colab userdata -> env var -> None),
│       │                          in_colab(), redact(). None is a valid, ungated result.
│       ├── batching.py            chunk / maybe_progress / OOM_ERRORS and
│       │                          run_with_oom_halving, shared by inference and scoring.
│       ├── device.py              Device and dtype auto-selection (bf16 only when native:
│       │                          a T4 gets fp16) + describe_device().
│       ├── pooling.py             Mask-aware last_token / mean / cls pooling and
│       │                          unpad_sequence. Padding-position agnostic by construction.
│       ├── seeding.py             seed_everything(): python/numpy/torch RNGs, optional
│       │                          torch deterministic algorithms.
│       └── logging.py             One idempotent handler on the package logger, so
│                                  re-running a Colab cell does not duplicate every line.
│
├── scripts/
│   ├── smoke_test.py              End-to-end check (load, invoke, batch, save, reload,
│   │                              verify). `--offline` needs no network. Exits non-zero
│   │                              on failure, so it works as a CI gate.
│   └── run_extraction.py          Thin wrapper that runs cli.py from a checkout without
│                                  installing the package.
│
└── tests/
    ├── conftest.py                Session-scoped tiny local model, Analyzer factory,
    │                              sample inputs. Entirely offline.
    ├── test_config.py             Config validation and fingerprinting; token never leaks.
    ├── test_pooling.py            Pooling math against hand-computed values; left- vs
    │                              right-padding equivalence; guards the hidden[:, -1] bug.
    ├── test_layer_selection.py    Negative/positive indexing, keywords, dedup, out-of-range
    │                              errors, max-length resolution, metadata.
    ├── test_inference.py          invoke/batch equivalence, order preservation, truncation
    │                              flagging, empty-input rejection, failure isolation, OOM
    │                              fallback, per-token shapes.
    ├── test_storage.py            JSON schema, rounding, sidecar policy, lossless round-trip,
    │                              atomicity, filename sanitising, manifest correctness.
    ├── test_imports.py            Guards import hygiene: no test module may import
    │                              `tests.*`, which resolves only when the repo root is on
    │                              sys.path (true for `python -m pytest`, false for the
    │                              `pytest` console script).
    ├── test_notebook.py           Guards both Colab notebooks: nbformat validity, every
    │                              source line keeps its trailing newline, code cells compile
    │                              when joined the way a reader joins them, no hardcoded
    │                              secrets, header-based clone auth, no committed outputs.
    ├── test_scoring.py            LogProbService against a manual log-softmax on the tiny
    │                              model; BOS policy; batch == single; non-finite -> failure;
    │                              OOM halving; base head refuses; both heads coexist.
    ├── test_spans.py              Crossing rule, baselines, greedy/CKY induction, F1 and
    │                              rank curve on the theory doc's 13-word example.
    ├── test_proforms.py           Substitution text, policies, spec parsing.
    ├── test_treebank.py           Tree -> gold spans (traces, punctuation, unaries),
    │                              detokeniser, the NLTK loader (skipped without the corpus).
    ├── test_span_costs.py         Cost table, normalisations, JSONL resume, header mismatch,
    │                              truncated last line, failed variants, Drive mirror.
    ├── test_task_1b.py            Record schema, statistics, verdict, and the real fig_1b /
    │                              fig_0c rendered from a FakeSpanScorer run.
    └── test_architecture_doc.py   Fails if this file omits any source file.
```

## How a run flows

```
notebooks/colab_entrypoint.ipynb
        │  builds a RunConfig (config/settings.py)
        ▼
container.Analyzer
        │  seeds RNGs (utils/seeding.py)
        ├──▶ services/model_service.ModelService.load()
        │        resolves device+dtype (utils/device.py)
        │        resolves HF token     (utils/env.py)
        │        loads tokenizer+model, fixes pad token, rejects unsupported archs
        │
        ├──▶ services/inference_service.InferenceService.batch(texts)
        │        validates inputs · sorts by length · chunks
        │        tokenize(truncation) + probe untruncated length
        │        forward(output_hidden_states=True) under inference_mode
        │        pick layers (ModelProvider.resolve_layers)
        │        pool (utils/pooling.py) or unpad · cast float32 · to numpy
        │        on OOM: halve and retry · on item error: record and continue
        │     ──▶ BatchResult[ExtractionRecord] + failures + effective config
        │
        └──▶ services/storage_service.StorageService.write(manifest, result)
                 round floats · decide sidecar · atomic write
              ──▶ outputs/<name>.json  (+ outputs/<name>.npz when large)

read back: services/storage_service.load_run(path) -> dict with numpy arrays
```

## How an experiment flows (Task 1b)

```
notebooks/experiment_1b.ipynb
        │  RunConfig(model=ModelConfig(head="causal_lm"), scoring=ScoringConfig(...))
        ▼
container.Analyzer  ──▶ ModelService.load()  (AutoModelForCausalLM)
        │           ──▶ LogProbService        (analyzer.score / score_one)
        │
        ├──▶ experiments.treebank.load_ptb_nltk(n, min_len, max_len, seed)
        │        NLTK PTB sample -> TreebankSentence(words, text, gold_spans)
        │     ──▶ outputs/task_1b/gold_<treebank>_<seed>.jsonl   (save_gold_jsonl)
        │
        ├──▶ PHASE A (GPU)  experiments.span_costs.compute_span_costs(analyzer, sentences, policy)
        │        every span x proform -> substitute -> detokenise -> score
        │        raw (sum_logprob, n_tokens) per variant -> SpanCostTable
        │     ──▶ outputs/task_1b/span_costs_<model>_<policy>.jsonl   (append, fsync, resume)
        │
        └──▶ PHASE B (CPU)  experiments.task_1b.analyse_1b(tables, sentences, policy, inducer)
                 cost(i, j) = policy.choose({p: mean_lp(x) - mean_lp(x'_p)})
                 greedy_induce (or cky) -> bracket_prf vs gold; RB / LB / random baselines
                 bootstrap over sentences; F1 by length; pooled rank curve
              ──▶ record (fig_1b schema) ──▶ experiment_figures.fig_1b / fig_0c ──▶ PNG
```

Phase B is seconds, so a different inducer, normalisation, or proform subset is re-run
from the cache without the GPU. The cost cache names sentences by id but holds no gold, so
the gold export travels with it: `load_gold_jsonl` + `load_span_costs` reconstruct every
phase-B input on a machine with no treebank installed. The same cache is Task 1a's raw data (every span is a
"box" if it is gold, a "straddle" if it crosses gold).

## Layer indexing convention

`output_hidden_states=True` returns `num_hidden_layers + 1` tensors:

| index | tensor |
|---|---|
| `0` | embedding output, before block 1 |
| `1 … N` | output of transformer block *i* |
| `-1` | same as `N`, the last block |

`"middle"` resolves to block `N // 2`. Every output file records both the label you asked
for and the resolved absolute index, so a saved result is never ambiguous.

## Extending it

| Goal | Touch |
|---|---|
| New pooling mode | `utils/pooling.py` (add to `POOLERS`), `config/settings.py` (`POOLING_MODES`), a test in `test_pooling.py` |
| New output format (Parquet, HDF5) | New class satisfying `ResultSink`; pass it to `Analyzer(storage_service=...)` |
| Attention weights as well as hidden states | `inference_service._forward` (`output_attentions=True`), `domain/records.py`, `storage_service` |
| Encoder-decoder support | `model_service._reject_unsupported` and `_forward` (decoder states are a separate output) |
| Remote/HTTP model host | New class satisfying `ModelProvider`; nothing else changes |
| A new experiment task | New `experiments/task_<id>.py` producing the record its `fig_<id>` docstring specifies; reuse `treebank.py`, `spans.py`, `span_costs.py`, `stats.py`; a notebook copied from `experiment_1b.ipynb` |
| Another treebank (UD) | `experiments/treebank.py` (`load_ud_conllu`: subtree yields -> spans, drop non-projective); name the conversion in the record's `treebank` field |
| Another replacement policy | A class satisfying `ReplacementPolicy` in `experiments/proforms.py`; if its proforms are a subset of a cache's header, phase B alone suffices |
