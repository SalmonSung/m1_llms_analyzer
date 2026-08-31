# Edge cases

What can go wrong, what this code does about it, and where. Rationale for the *choices*
lives in `design_decisions.md`; this file is the checklist.

## Handled

| # | Edge case | What would go wrong | How it is handled | Where |
|---|---|---|---|---|
| 1 | **Padding side vs. last-token pooling** | The usual `sum(mask)-1` index is right-padding-only; on a left-padded batch it silently pools the *first* token. `hidden[:, -1]` pools a PAD token outright. | Last real token found by scanning the mask from the right (`flip().argmax()`), correct under either padding side. | `utils/pooling.py`; `test_pooling.py::test_last_token_ignores_padding_side` |
| 2 | **Tokenizer has no PAD token** | GPT-2-family tokenizers raise as soon as you batch. | `pad_token = eos_token` (then `unk_token`), vocabulary unchanged, logged and recorded as `pad_token_substituted`. | `model_service._ensure_pad_token` |
| 3 | **float16 on CPU** | Many kernels have no fp16 CPU implementation; the rest are slower than fp32. | CPU always resolves to float32; an explicit fp16-on-CPU request is warned about and overridden. | `utils/device.py` |
| 4 | **Hidden-states off-by-one** | `hidden_states` has `N+1` entries; treating index 12 as "block 12" is wrong by one for anyone assuming 0-based blocks. | Convention documented, and both the requested label and resolved absolute index written into every file. | `model_service.resolve_layers`, `RunManifest.layer_index_convention` |
| 5 | **Encoder-decoder models (T5, BART)** | `hidden_states` is the *encoder* stack; the decoder's is a separate output. Saving it as "the last layer" is quietly wrong. | Detected at load and rejected with an explanation. | `model_service._reject_unsupported` |
| 6 | **Vision/multimodal models** | Text-only input paths produce meaningless or failing forward passes. | Detected at load and rejected. | `model_service._reject_unsupported` |
| 7 | **Silent truncation** | A prompt longer than the context is clipped, and the resulting vector describes only the prefix while looking perfectly normal. | Untruncated length probed; `truncated`, `token_count`, and `original_token_count` recorded; a warning is logged. | `inference_service._forward` |
| 8 | **Empty / whitespace-only input** | Zero real tokens means nothing to pool — a degenerate vector, or a division by zero in mean pooling. | Rejected up front with the offending index named; pooling also asserts non-empty masks. | `inference_service._validate_inputs`, `pooling._assert_non_empty` |
| 9 | **Duplicate inputs** | Positional ids make identical texts look like different items. | `sha256(text)[:16]` — identical text always gets the same id. | `domain/records.make_item_id` |
| 10 | **CUDA OOM mid-batch** | A long run dies hours in; Colab GPU memory varies session to session. | Batch halved and retried down to size 1, cache emptied; only a single input that still OOMs fails, with a hint. | `inference_service._run_chunk` |
| 11 | **One bad item kills the run** | 9,999 good results lost to item 5,000. | Per-item exceptions become `ExtractionFailure` records that travel with the results into the JSON. | `inference_service._record_failure` |
| 12 | **JSON size explosion** | Per-token states on a big model are millions of floats — tens of MB of text, slow to parse, unopenable in an editor. | Above a threshold the arrays move to a compressed `.npz`; the JSON keeps a pointer and `load_run()` rehydrates transparently. | `storage_service`, `StorageConfig.npy_mode` |
| 13 | **Float precision in text** | 17-digit reprs bloat the file with noise. | Rounded to `float_precision` (6) in JSON; the sidecar keeps lossless float32. | `storage_service._round_nested` |
| 14 | **NaN / inf in output** | Python's `json` emits `NaN`, which is invalid JSON and rejected by other parsers. | Serialised as `null` with a warning pointing at low-precision overflow. | `storage_service._round_nested` |
| 15 | **Interrupted write** | A recycled runtime leaves a half-written file that still parses as *something*. | Temp file + `os.replace`: the output either does not exist or is complete. | `storage_service._atomic_write` |
| 16 | **Compute dtype leaking into results** | bf16 on GPU vs fp32 on CPU would give visibly different saved numbers. | Hidden states cast to float32 before leaving the device. | `inference_service._forward` |
| 17 | **Non-determinism** | Two runs "should" match but do not, with no way to tell why. | All RNGs seeded; seed, dtype, device, revision, and library versions stamped in the manifest; opt-in strict determinism. | `utils/seeding.py`, `RunManifest` |
| 18 | **Gated / private / missing models** | A bare `401` or `404` from the Hub is useless in a notebook. | Translated into numbered instructions naming the secret to add and the licence to accept. | `model_service._explain_load_failure` |
| 19 | **Token leaking into `.git/config`** | The clone URL carries the PAT and persists on the runtime's disk. | `git remote set-url` scrubs it immediately after cloning; git errors are filtered before printing. | notebook cell 1 |
| 20 | **Token leaking into output/logs** | A manifest or log line containing a PAT is a live credential in a file you might share. | `RunConfig.to_dict()` nulls the token; only presence is logged, via `redact()`. Covered by a test. | `config/settings.py`, `utils/env.redact`, `test_storage.py::test_no_token_appears_in_output` |
| 21 | **Re-running notebook cells** | Duplicate log lines; a clone that fails because the directory exists. | Logging handler attached once with `propagate=False`; bootstrap pulls when the repo is already present; `ModelService.load()` is idempotent. | `utils/logging.py`, notebook cell 1 |
| 22 | **Losing results on disconnect** | Colab wipes `/content` when the runtime is recycled. | Drive-mount cell writes the same run to `MyDrive`; a download cell pulls files locally. | notebook §9 |
| 23 | **torch reinstall breaking CUDA** | Pinning torch makes pip replace Colab's CUDA-matched build — minutes lost, GPU possibly broken. | torch left unpinned; install uses `--upgrade-strategy only-if-needed`. | `requirements.txt`, notebook cell 2 |
| 24 | **Per-call overrides misrecorded** | `batch(texts, pooling="none")` saved under a manifest claiming `last_token` — a file that lies about itself. | The effective config travels on `BatchResult.extraction` and builds the manifest. | `container.Analyzer.save` |
| 25 | **`max_length` beyond the model's context** | Requesting 8192 on a 2048-context model produces a position-embedding error deep in the forward pass. | Clamped to the model's context, then to `max_length_cap`; both clamps logged. | `model_service.effective_max_length` |
| 26 | **Layer index out of range** | `IndexError` inside the forward pass, with no clue what is valid. | Validated at resolution time with the valid range and the model's depth in the message. | `model_service.resolve_layers` |
| 27 | **`trust_remote_code` models** | Loading fails with an opaque message, or executes untrusted code without the user realising. | Off by default; the error says exactly which flag to set and what it means. | `model_service._explain_load_failure` |
| 28 | **Unsafe output filenames** | A name with `/` or `:` breaks on Drive/Windows or writes to the wrong directory. | Sanitised to `[A-Za-z0-9-_.]`. | `storage_service._sanitize` |
| 29 | **Missing sidecar after moving files** | Copying only the `.json` gives a `KeyError` or silent `None` values. | `load_run()` raises a message saying to keep the `.json` and `.npz` together. | `storage_service.load_run` |
| 31 | **Notebook `source` lines without trailing newlines** | `nbformat` validates and the JSON parses, but Colab concatenates `source` elements verbatim — every cell renders as one enormous line and no code cell runs. | Enforced by a test that checks each element (bar the last) ends in `\n`, and that code cells compile when joined with `""`, the way a reader joins them. | `tests/test_notebook.py` |
| 30 | **Padding rows in per-token output** | A `[seq_len, hidden]` matrix that includes PAD rows makes token counts wrong downstream. | `unpad_sequence` trims each item to its real tokens; one row per real token, asserted in tests. | `utils/pooling.unpad_sequence` |

## Known limits (not handled — by choice)

| Limit | Consequence | See |
|---|---|---|
| Instruct models get **raw text**, not a chat template | Hidden states differ from what the model sees in chat use. | `design_decisions.md` → "Deliberately not built yet" |
| No long-document chunking | Text beyond the context is truncated (and flagged), never split and stitched. | same |
| No attention weights | Only hidden states are captured. | same |
| No layer *aggregation* | You can select layers 8–16, but combining them into one vector is downstream work. | same |
| Single device only | No `device_map="auto"` sharding; a model must fit one GPU (or CPU RAM). | same |
| Whole run held in memory | A very large corpus should be run in slices; there is no streaming sink yet. | same |
| Bit-exact GPU reproducibility not guaranteed | Same seed + same GPU + same versions is reproducible; across GPU models it may not be. | `design_decisions.md` → "Strict determinism is opt-in" |
