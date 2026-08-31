# Architecture

> **Maintenance rule:** this file is part of the contract. Whenever a file or folder is
> added, renamed, removed, or changes responsibility, update the tree **and** its
> description here in the same commit. `tests/test_architecture_doc.py` fails the build
> if any `.py`, `.ipynb`, or `.md` file under `src/`, `scripts/`, `tests/`, `notebooks/`,
> or `docs/` is not mentioned below.

## What this repo is

An in-process **microservice** pipeline that loads an open-source Hugging Face model,
extracts hidden states from selected layers, and saves them as JSON. It is driven from a
single Google Colab notebook that clones this private repo and installs everything itself.

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
│   └── colab_entrypoint.ipynb     THE entrypoint. Paste into Colab, Run all.
│
├── src/m1_analyzer/
│   ├── __init__.py                Public API surface: re-exports Analyzer, configs, records.
│   ├── container.py               `Analyzer` facade + wiring. Builds the three services from
│   │                              one RunConfig; owns the loaded model for the session;
│   │                              exposes invoke / batch / save / run / describe / unload,
│   │                              and builds the RunManifest.
│   ├── cli.py                     argparse entrypoint (`m1-extract`). Same pipeline from a
│   │                              terminal or CI; reads inputs from flags, lines, JSON, JSONL.
│   ├── testing.py                 Builds a tiny random GPT-2 + tokenizer on disk, offline.
│   │                              Backs the test suite and `smoke_test.py --offline`.
│   │
│   ├── config/
│   │   ├── __init__.py            Re-exports the config dataclasses.
│   │   └── settings.py            ModelConfig / ExtractionConfig / StorageConfig / RunConfig.
│   │                              Validation in __post_init__ so bad settings fail at
│   │                              construction, not mid-forward-pass. RunConfig.fingerprint()
│   │                              hashes the settings; to_dict() strips any token.
│   │
│   ├── domain/
│   │   ├── __init__.py            Re-exports the record types.
│   │   └── records.py             The data services exchange: LayerState, ExtractionRecord,
│   │                              ExtractionFailure, BatchResult, RunManifest, WrittenPaths,
│   │                              make_item_id (sha256 content id). Holds numpy, never torch,
│   │                              so results need no CUDA context to read.
│   │
│   ├── services/
│   │   ├── __init__.py            Re-exports the services.
│   │   ├── interfaces.py          The seams: ModelProvider, InferenceEngine, ResultSink
│   │                              Protocols. Services depend on these, never on each other.
│   │   ├── model_service.py       Only module that calls transformers' loading APIs. Owns
│   │                              device/dtype choice, HF token use, pad-token fallback,
│   │                              architecture rejection, layer-spec resolution, context
│   │                              length, and provenance metadata. Raises ModelLoadError /
│   │                              UnsupportedArchitectureError with actionable messages.
│   │   ├── inference_service.py   The forward pass. invoke() is batch() of one. Length-sorted
│   │                              batching, OOM halving, per-item failure isolation,
│   │                              truncation detection, layer picking, pooling, float32 cast.
│   │   └── storage_service.py     JSON writer (atomic), float rounding, .npz sidecar policy,
│   │                              and `load_run()` which rehydrates arrays on read.
│   │
│   └── utils/
│       ├── __init__.py            Marks the package; holds no logic.
│       ├── env.py                 Secret resolution (Colab userdata -> env var -> None),
│       │                          in_colab(), redact(). None is a valid, ungated result.
│       ├── device.py              Device and dtype auto-selection + describe_device().
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
