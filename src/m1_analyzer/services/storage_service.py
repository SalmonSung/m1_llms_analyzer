"""Storage service: persist a run as JSON (+ optional lossless .npy sidecar).

Format contract
---------------
* The JSON is always written and is always the index: it carries the manifest,
  every record's metadata, and (usually) the numbers themselves.
* Numbers in JSON are rounded to ``float_precision`` decimals. Text floats are
  bulky and 17 significant digits is noise for this purpose.
* When a run is large, ``npy_mode="auto"`` moves the raw float32 arrays into a
  ``.npz`` sidecar and leaves ``values: null`` plus an ``npy_ref`` pointer in the
  JSON. A per-token dump on a 4k-hidden model is millions of floats -- as text
  that is tens of MB that no editor will open, and JSON parsing becomes the
  bottleneck. The sidecar is lossless and loads in milliseconds via numpy.
* Writes are atomic (temp file + ``os.replace``): a crashed or interrupted Colab
  runtime can never leave a half-written JSON that looks valid.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..config.settings import StorageConfig
from ..domain.records import BatchResult, RunManifest, WrittenPaths
from ..utils.logging import get_logger

log = get_logger("storage")


class StorageService:
    """Writes and reads run files."""

    def __init__(self, config: StorageConfig | None = None):
        self.config = config or StorageConfig()

    # ----------------------------------------------------------------- write

    def write(
        self, manifest: RunManifest, result: BatchResult, name: str | None = None
    ) -> WrittenPaths:
        cfg = self.config
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = _sanitize(name or manifest.run_id)
        json_path = out_dir / f"{stem}.json"
        npy_path = out_dir / f"{stem}.npz"

        use_sidecar = self._should_write_sidecar(result)
        arrays: dict[str, np.ndarray] = {}

        records_json = []
        for record in result.records:
            layers_json: dict[str, Any] = {}
            for label, state in record.layers.items():
                entry: dict[str, Any] = {
                    "index": state.index,
                    "requested": state.requested,
                    "shape": list(state.shape),
                }
                if use_sidecar:
                    key = f"{record.id}::{label}"
                    arrays[key] = np.asarray(state.values, dtype=np.float32)
                    entry["values"] = None
                    entry["npy_ref"] = key
                else:
                    entry["values"] = _round_nested(state.values, cfg.float_precision)
                    entry["npy_ref"] = None
                layers_json[label] = entry

            record_json: dict[str, Any] = {
                "id": record.id,
                "token_count": record.token_count,
                "original_token_count": record.original_token_count,
                "truncated": record.truncated,
                "layers": layers_json,
            }
            if cfg.include_input_text:
                record_json["text"] = record.text
            if record.tokens is not None:
                record_json["tokens"] = record.tokens
            records_json.append(record_json)

        payload = {
            "schema_version": manifest.schema_version,
            "run": asdict(manifest),
            "counts": {
                "records": len(result.records),
                "failures": len(result.failures),
                "total_floats": result.total_floats,
            },
            "storage": {
                "float_precision": cfg.float_precision,
                "values_in_sidecar": use_sidecar,
                "sidecar_file": npy_path.name if use_sidecar else None,
                "include_input_text": cfg.include_input_text,
            },
            "records": records_json,
            "failures": [asdict(f) for f in result.failures],
        }

        written_npy = None
        if use_sidecar:
            _atomic_write(npy_path, lambda p: np.savez_compressed(p, **arrays), binary=True)
            written_npy = str(npy_path)
            log.info("Wrote %d array(s) to sidecar %s", len(arrays), npy_path)

        _atomic_write(
            json_path,
            lambda p: p.write_text(
                json.dumps(payload, indent=cfg.indent, ensure_ascii=False), encoding="utf-8"
            ),
        )
        size_mb = json_path.stat().st_size / 1024**2
        log.info("Wrote %s (%.2f MB, %d record(s))", json_path, size_mb, len(result.records))
        return WrittenPaths(json_path=str(json_path), npy_path=written_npy)

    def _should_write_sidecar(self, result: BatchResult) -> bool:
        mode = self.config.npy_mode
        if mode == "always":
            return True
        if mode == "never":
            return False
        return result.total_floats > self.config.npy_threshold_floats


# ------------------------------------------------------------------- reading


def load_run(json_path: str | os.PathLike) -> dict[str, Any]:
    """Read a run file back, rehydrating arrays from the sidecar when present.

    Returns the parsed JSON with every ``layers[...]["values"]`` populated as a
    numpy array, so callers never have to care whether the run used a sidecar.
    """
    path = Path(json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    sidecar_name = payload.get("storage", {}).get("sidecar_file")
    arrays = None
    if sidecar_name:
        sidecar = path.parent / sidecar_name
        if not sidecar.exists():
            raise FileNotFoundError(
                f"{path.name} stores its values in '{sidecar_name}', which is missing from "
                f"{path.parent}. Keep the .json and .npz together when moving results."
            )
        arrays = np.load(sidecar)

    for record in payload.get("records", []):
        for entry in record.get("layers", {}).values():
            if entry.get("values") is not None:
                entry["values"] = np.asarray(entry["values"], dtype=np.float32)
            elif arrays is not None and entry.get("npy_ref"):
                entry["values"] = arrays[entry["npy_ref"]]
    return payload


# ------------------------------------------------------------------- helpers


def make_run_id(prefix: str = "run") -> str:
    """Timestamped, sortable, filesystem-safe run identifier."""
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _round_nested(values: np.ndarray, precision: int) -> Any:
    """Round to `precision` decimals and convert to plain Python lists.

    ``np.round(...).tolist()`` still yields binary floats whose repr can carry
    17 digits, so the rounding is applied again per element via `round`, which
    is what actually keeps the file small and readable.
    """
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        log.warning(
            "Non-finite values (inf/nan) in hidden states -- serialising them as null. "
            "This usually means an overflow in a low-precision dtype; try dtype='float32'."
        )
        arr = np.where(np.isfinite(arr), arr, np.nan)
    rounded = np.round(arr, precision).tolist()
    return _clean(rounded, precision)


def _clean(obj: Any, precision: int) -> Any:
    if isinstance(obj, list):
        return [_clean(x, precision) for x in obj]
    if obj != obj:  # NaN -- JSON has no NaN literal, so null it is
        return None
    return round(float(obj), precision)


def _atomic_write(path: Path, writer, binary: bool = False) -> None:
    """Write via a temp file in the same directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.stem}.", suffix=suffix)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        if binary:
            # np.savez_compressed appends .npz unless the path already ends in it.
            target = tmp if tmp.suffix == ".npz" else tmp.with_suffix(".npz")
            writer(target)
            tmp = target
        else:
            writer(tmp)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _sanitize(name: str) -> str:
    """Keep filenames portable across Colab, Drive, Windows, and Linux."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in name).strip("-.")
    return safe or "run"
