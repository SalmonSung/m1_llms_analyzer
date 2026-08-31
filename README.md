# m1_llms_analyzer

Extract hidden states from open-source Hugging Face models and save them as JSON.
Built as in-process microservices, driven from a single Google Colab notebook.

- **`invoke(text)`** — one input, one record.
- **`batch(texts)`** — many inputs, order preserved, OOM-safe, failures isolated.
- **Any layer** — `-1` (last), `0` (embeddings), `"middle"`, `"all"`, or a list.
- **Pooled or per-token** — `last_token` (default), `mean`, `cls`, or `none`.
- **JSON out** — self-describing, with a lossless `.npz` sidecar when a run is large.

---

## Use it from Colab (the intended path)

1. **Add two Colab secrets** (key icon in the left sidebar, *Notebook access* on):

   | Secret | Required | What it is |
   |---|---|---|
   | `GITHUB_TOKEN` | **Yes** (this repo is private) | GitHub fine-grained PAT with *Contents: Read* on this repo — [create one](https://github.com/settings/personal-access-tokens/new) |
   | `HF_TOKEN` | Only for gated models | Hugging Face read token — [create one](https://huggingface.co/settings/tokens) |

2. **Open [`notebooks/colab_entrypoint.ipynb`](notebooks/colab_entrypoint.ipynb)**, copy it
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

## Configuration

```python
from m1_analyzer import Analyzer, RunConfig, ModelConfig, ExtractionConfig, StorageConfig

config = RunConfig(
    model=ModelConfig(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        revision=None,             # pin a commit SHA for reproducibility
        device="auto",             # auto | cpu | cuda | mps
        dtype="auto",              # auto | float32 | float16 | bfloat16
        trust_remote_code=False,   # executes repo code -- opt in per model
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

| File | What is in it |
|---|---|
| [`architecture.md`](architecture.md) | Structure tree, every file's responsibility, request flow, extension points |
| [`docs/design_decisions.md`](docs/design_decisions.md) | Every undiscussed choice and its reasoning, plus what is deliberately not built |
| [`docs/edge_cases.md`](docs/edge_cases.md) | 30 handled edge cases with the code that handles each, and the known limits |

## Tests

```bash
pytest -q                              # ~80 tests, fully offline, a few seconds
python scripts/smoke_test.py --offline # end-to-end without the Hub
```

The suite builds a tiny random GPT-2 on disk and loads it through the same
`from_pretrained` path a real model uses, so it needs no network and cannot be broken by a
Hub outage. `tests/test_architecture_doc.py` fails the build if a source file is missing
from `architecture.md`.
