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
│   ├── experiment_1b.ipynb        Task 1b (does substitution FIND constituents?): same
│   │                              bootstrap, then smoke test -> treebank -> model ->
│   │                              one-sentence walkthrough -> phase A scoring (resumable)
│   │                              -> phase B analysis -> fig_1b / fig_0c -> Drive.
│   └── experiment_9a.ipynb        Task 9a (omittability by splicing, length-matched, with
│                                  the hand audit): same bootstrap, smoke test -> model ->
│                                  Wikipedia paragraphs -> one-paragraph walkthrough ->
│                                  phase A splice scoring (resumable, pre-registered
│                                  header) -> audit sheet out / back in -> phase B
│                                  stratified analysis -> fig_9a -> Drive.
│
├── src/m1_analyzer/
│   ├── __init__.py                Public API surface: re-exports Analyzer, configs, records.
│   ├── container.py               `Analyzer` facade + wiring. Builds the services from one
│   │                              RunConfig; owns the loaded model for the session;
│   │                              exposes invoke / batch / save / run (hidden states),
│   │                              score / score_one (log-probabilities) and
│   │                              next_token_states (full next-token vectors), both when
│   │                              the model was loaded with head="causal_lm"
│   │                              (`Analyzer.for_scoring`), describe / unload, and builds
│   │                              the RunManifest.
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
│   │                              The tiny vocabulary carries sentence punctuation so
│   │                              the splice experiment runs on it end to end.
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
│   │                              StateResult (next-token vectors at chosen positions),
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
│   │   ├── jsonl_cache.py         The resumable JSONL cache both phase-A scorers use:
│   │   │                          read (truncated last line dropped), header check that
│   │   │                          names the differing field, atomic rewrite, fsynced
│   │   │                          append, Drive mirror.
│   │   ├── spans.py               Span algebra: enumerate_spans, crosses, right/left/random
│   │   │                          baselines, greedy_induce (cheapest first, never cross),
│   │   │                          cky_induce (minimum total cost), bracket_prf, rank_curve.
│   │   ├── proforms.py            Blind replacement policies (ReplacementPolicy Protocol):
│   │   │                          MinOverSet (+ controls: scored and cached, never chosen),
│   │   │                          ByLengthClass, parse_policy, substitute, and DELETION
│   │   │                          ("<del>": drop the span, re-capitalise at position 0).
│   │   ├── treebank.py            TreebankSentence + gold spans from trees (traces and
│   │   │                          punctuation removed, unaries collapsed), PTB detokeniser,
│   │   │                          load_ptb_nltk (the free NLTK WSJ sample), hand examples
│   │   │                          incl. the theory doc's sentence, sentence_to/from_json +
│   │   │                          gold_header_fields (the answer key as JSON, embedded in
│   │   │                          the cost cache), ptb_tree_strings, and the standalone
│   │   │                          save/load_gold_jsonl export. UD is a documented stub.
│   │   ├── span_costs.py          Phase A: SpanCostTable (raw sum/count per proform x span,
│   │   │                          cost derived on read), compute_span_costs with a resumable
│   │   │                          JSONL cache (header check, fsync, truncated-line repair,
│   │   │                          Drive mirror; every row also carries the sentence's gold
│   │   │                          spans and tree), load_span_costs,
│   │   │                          load_span_costs_with_gold, variant_count.
│   │   ├── task_1b.py             Phase B: evaluate_sentence, analyse_1b -> the fig_1b
│   │   │                          record (sentence-level mean F1 + bootstrap CI, by length,
│   │   │                          pooled rank curve, diagnostics), validate_record, verdict,
│   │   │                          run_task_1b (A then B).
│   │   ├── boundaries.py          Task 9a: sentence-final token positions (punkt or a
│   │   │                          regex splitter, mapped through tokenizer offsets; every
│   │   │                          rejected sentence end counted) and clause-final ones
│   │   │                          (`, ; :` and dashes), the two boundary kinds a cut may join.
│   │   │                          A boundary is validated on BOTH sides: the token ends a
│   │   │                          sentence AND `starts_sentence` confirms the text after it
│   │   │                          begins one, so a mid-sentence position (an abbreviation
│   │   │                          punkt split on, a dropped wiki template) cannot serve as
│   │   │                          an endpoint. Clause boundaries are exempt: they are
│   │   │                          followed by lowercase by design.
│   │   ├── paragraphs.py          Task 9a corpus: Paragraph, select_paragraphs (token-count
│   │   │                          and sentence-count filter, seeded sample from a
│   │   │                          source-order pool), Wikipedia streamed through
│   │   │                          `datasets`, plain-text / JSONL files, hand_paragraphs.
│   │   ├── splice.py              Task 9a phase A: admissible_cuts (same boundary type,
│   │   │                          `window` tokens after j), splice_ids / splice_text,
│   │   │                          score_paragraph (endpoint distance d, per-position
│   │   │                          divergences div_k and their median, spliced fluency,
│   │   │                          re-tokenisation check, join snippet), compute_splices
│   │   │                          with the pre-registration as the cache header (measure,
│   │   │                          window, strata, decile, matching width, boundary rule),
│   │   │                          load_splices. `schema` is a CHECKED header field: a
│   │   │                          schema-1 cache was built under the one-sided boundary
│   │   │                          rule and must not be resumed under the two-sided one.
│   │   └── task_9a.py             Task 9a phase B: close / far deciles within
│   │                              length-matching bins inside each stratum, the audit
│   │                              sample / CSV sheet / read-back, the stratified
│   │                              permutation test, the paragraph-level cluster bootstrap,
│   │                              pooled and continuous diagnostics, `boundary_check`
│   │                              (re-verifies every stored endpoint and splits the
│   │                              failures by close/far, since they concentrate in one
│   │                              arm), analyse_9a -> the fig_9a record, validate_record,
│   │                              verdict.
│   │
│   └── utils/
│       ├── __init__.py            Marks the package; holds no logic.
│       ├── env.py                 Secret resolution (Colab userdata -> env var -> None),
│       │                          in_colab(), redact(). None is a valid, ungated result.
│       ├── batching.py            AdaptiveBatchSize (sticky halving, growth after clean
│       │                          batches, ceiling below any size that ran out of memory),
│       │                          chunk / maybe_progress / OOM_ERRORS and
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
    │                              OOM halving that sticks across calls; the batch-size
    │                              calibration probe; base head refuses; both heads coexist.
    ├── test_batching.py           AdaptiveBatchSize: sticky shrink, growth streaks, ceilings,
    │                              lazy chunking, and the OOM hook in run_with_oom_halving.
    ├── test_spans.py              Crossing rule, baselines, greedy/CKY induction, F1 and
    │                              rank curve on the theory doc's 13-word example.
    ├── test_proforms.py           Substitution text (incl. deletion and multi-word
    │                              proforms), policies with controls, spec parsing.
    ├── test_treebank.py           Tree -> gold spans (traces, punctuation, unaries),
    │                              detokeniser, the NLTK loader (skipped without the corpus).
    ├── test_span_costs.py         Cost table, normalisations, JSONL resume, header mismatch,
    │                              truncated last line, failed variants, Drive mirror,
    │                              controls, and the answer key embedded in each row.
    ├── test_task_1b.py            Record schema, statistics, verdict, and the real fig_1b /
    │                              fig_0c rendered from a FakeSpanScorer run.
    ├── test_state_service.py      Next-token vectors against a manual log-softmax, BOS
    │                              policies, batch == single, prefix invariance, guards.
    ├── test_boundaries.py         Sentence / clause boundary positions, rejections counted,
    │                              punkt named when missing.
    ├── test_paragraphs.py         Cleaning, the token / sentence filter, seeded sampling,
    │                              text and JSONL sources.
    ├── test_splice.py             Admissible cuts, splicing on ids and text, one cut
    │                              recomputed by hand from the state service, resume,
    │                              the pre-registration guard, truncated-line repair.
    ├── test_task_9a.py            Labels within matching bins, a planted effect passes and
    │                              a null fails, the pooled confound, the permutation test,
    │                              the audit round trip with failures excluded, deviations
    │                              recorded, the real and mock fig_9a.
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
        │        NLTK PTB sample -> TreebankSentence(words, text, gold_spans) (+ tree strings)
        │
        ├──▶ PHASE A (GPU)  experiments.span_costs.compute_span_costs(analyzer, sentences, policy, trees)
        │        LogProbService.calibrate_batch_size(longest variants) -> AdaptiveBatchSize
        │        every span x (proform | control) -> substitute -> detokenise -> score
        │        raw (sum_logprob, n_tokens) per variant -> SpanCostTable
        │     ──▶ outputs/task_1b/span_costs_<model>_<policy>.jsonl   (append, fsync, resume)
        │          one line per sentence: raw scores + words, gold_spans, source, info, tree
        │
        └──▶ PHASE B (CPU)  experiments.task_1b.analyse_1b(tables, sentences, policy, inducer)
                 cost(i, j) = policy.choose({p: mean_lp(x) - mean_lp(x'_p)})
                 greedy_induce (or cky) -> bracket_prf vs gold; RB / LB / random baselines
                 bootstrap over sentences; F1 by length; pooled rank curve
              ──▶ record (fig_1b schema) ──▶ experiment_figures.fig_1b / fig_0c ──▶ PNG
```

Phase B is seconds, so a different inducer, normalisation, or proform subset is re-run
from the cache without the GPU. Controls (`blorp`, `<del>`) are in the cache like any
proform but `MinOverSet.choose` never picks them, so the induced trees are unchanged by
adding them. Each cache row carries the sentence's answer key, so
`load_span_costs_with_gold` reconstructs every phase-B input on a machine with no treebank
installed (the older two-file export, `save_gold_jsonl` + `load_gold_jsonl`, still works).
The same cache is Task 1a's raw data (every span is a "box" if it is gold, a "straddle" if
it crosses gold).

## How an experiment flows (Task 9a)

```
notebooks/experiment_9a.ipynb
        │  RunConfig(model=ModelConfig(head="causal_lm"))
        ▼
container.Analyzer  ──▶ ModelService.load()  (AutoModelForCausalLM)
        │           ──▶ NextTokenStateService (analyzer.states / next_token_states)
        │
        ├──▶ experiments.paragraphs.load_wikipedia_paragraphs(count_tokens, n, min/max tokens, seed)
        │        streamed articles -> lines -> clean -> token + sentence filter -> seeded sample
        │
        ├──▶ PHASE A (GPU)  experiments.splice.compute_splices(states, paragraphs, window, cache_path, ...)
        │        header = preregistration(measure, window, strata, decile, match_width)  written FIRST
        │        per paragraph: boundaries (boundaries.py, validated on both sides)
        │          -> admissible cuts (same type, window after j)
        │          one pass of the original -> endpoint states + downstream states
        │          one pass per cut of ids[:i+1] + ids[j+1:] -> states at i+1..i+window
        │          d = |S_i - S_j|, div_k = |S'_{i+1+k} - S_{j+1+k}|, div = median_k, fl, retokenises
        │     ──▶ outputs/task_9a/splices_<model>_<corpus>_w<window>.jsonl   (append, fsync, resume)
        │
        ├──▶ AUDIT  task_9a.assign_pairs -> audit_sample(10 close, 10 far, 10 other) -> write_audit_csv
        │        a person fills `grammatical` (y/n) on text alone -> load_audit_csv
        │
        └──▶ PHASE B (CPU)  experiments.task_9a.analyse_9a(rows, header, audit)
                 labels: bottom / top decile of d within MATCH_WIDTH-token bins of each stratum
                 stratified ratio (weighted log median ratio) + permutation p (labels shuffled within bins)
                 paragraph-level cluster bootstrap CI; pooled and residual diagnostics; audit summary
                 boundary_check: every stored endpoint re-verified, failures split by close/far
              ──▶ record (fig_9a schema + strata/stratified/pooled/audit) ──▶ experiment_figures.fig_9a ──▶ PNG
```

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
| A new experiment task | New `experiments/task_<id>.py` producing the record its `fig_<id>` docstring specifies; reuse `treebank.py`, `spans.py`, `span_costs.py`, `stats.py`, `jsonl_cache.py`; a notebook copied from `experiment_1b.ipynb` or `experiment_9a.ipynb` |
| A state-level experiment (distances between next-token vectors) | `NextTokenStateService.states(ids, positions)` via `analyzer.states`; see `experiments/splice.py` for the alignment bookkeeping |
| Another treebank (UD) | `experiments/treebank.py` (`load_ud_conllu`: subtree yields -> spans, drop non-projective); name the conversion in the record's `treebank` field |
| Another replacement policy | A class satisfying `ReplacementPolicy` in `experiments/proforms.py`; if its proforms are a subset of a cache's header, phase B alone suffices |
