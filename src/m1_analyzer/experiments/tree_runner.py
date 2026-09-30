"""Tree recovery on the runner: jobs C (T1 endpoint distances) and A (T4 frame-only re-scoring).

Both jobs read the Task 1b span-cost cache of Qwen3-0.6B-Base (``words``, ``tree``
and the span inventory ``spans["it"]`` of every sentence) and call the
**reference code** `tree_runner_ref` unmodified -- its functions take our loaded
model and tokenizer. This module adds only what the runner needs around them:

* the model, loaded through `Analyzer` in **float32** at a pinned revision
  (`load_model_f32`); BOS is ``<|endoftext|>`` as an id, as in the 1b run;
* a resumable working cache per job (`run_t1`, `run_t4`): one fsynced JSONL line
  per sentence, header checked on resume (`jsonl_cache`), so a Colab
  pre-emption loses at most one sentence;
* the pre-registered checks, as pure functions returning numbers and ``pass``
  (`check_t1_anchor`, `check_t4_anchor`, `check_t4_cache`, `check_t4_causality`);
* the deliverable (`finalize`): a meta block line (model, revision, dtype,
  versions, GPU, wall time, reference sha, whether it was modified, the check
  numbers), then one line per sentence -- ``{"id", "t1"}`` for C, and
  ``{"id", "end", "orig", "rows"}`` for A, gzipped.

Job B (8a with Jensen-Shannon) lives in `task_8a_js`.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..utils.batching import maybe_progress
from ..utils.logging import get_logger
from . import tree_runner_ref as ref
from .jsonl_cache import append_row, check_header, mirror, read_jsonl, rewrite, write_header

log = get_logger("tree_runner")

#: sha256 of `tree_runner_ref.py` as sent in the runner package.
REFERENCE_SHA256 = "8d7c67289c705bd235c432e6f6e931888513c85565080117e28db35485f77074"
REFERENCE_PATH = Path(ref.__file__)

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
MODEL_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
BOS_TOKEN = "<|endoftext|>"
T1_OUTPUT = "tree_t1_qwen3_0.6b.jsonl"
T4_OUTPUT = "tree_t4_qwen3_0.6b.jsonl.gz"

#: Thresholds of the guideline's checks.
T1_MIN_SPEARMAN = 0.999
T1_MAX_MEDIAN_REL = 0.01
T4_ANCHOR_TOL_NATS = 0.05
T4_CACHE_MIN_SPEARMAN = 0.999
T4_MAX_PRE_CHECK = 1e-3

#: Row layout of `tree_runner_ref.t4_rows`.
T4_FIELDS = ("total", "n_tok", "n_pre", "n_suf", "suf_sub", "end_sub", "pre_check")
_UNCHECKED_HEADER_KEYS = frozenset({"kind", "date", "device", "library_versions", "gpu"})


# ------------------------------------------------------------------ reference


def reference_sha256(path: str | os.PathLike = REFERENCE_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_unmodified() -> bool:
    """Is the imported reference byte-identical to the one sent?"""
    return reference_sha256() == REFERENCE_SHA256


# ------------------------------------------------------------------ inputs


def load_1b_cache(path: str | os.PathLike, *, fillers: Sequence[str] = ref.FILLERS) -> tuple[dict, list[dict]]:
    """``(header, sentences)`` from the 1b span-cost cache.

    Each sentence is ``{"id", "words", "tree", "keys", "spans"}``; ``keys`` is the
    span inventory (the keys of ``spans["it"]``, in cache order). The header's
    proforms must be exactly the reference's 17 fillers, ``<del>`` included.
    """
    header, rows, _ = read_jsonl(Path(path))
    if header is None:
        raise ValueError(f"{path} has no header line; it is not a span-cost cache.")
    proforms = list(header.get("proforms", []))
    if sorted(proforms) != sorted(fillers):
        raise ValueError(f"{path}: the header's proforms {proforms} are not the reference's fillers {list(fillers)}.")
    out = []
    for row in rows:
        if "tree" not in row:
            raise ValueError(f"{path}: sentence {row.get('id')!r} carries no tree; job A needs its end mark.")
        out.append({"id": row["id"], "words": list(row["words"]), "tree": row["tree"],
                    "keys": list(row["spans"]["it"]), "spans": row["spans"], "base": row.get("base")})
    return header, out


def load_anchor(path: str | os.PathLike) -> dict:
    """The anchor file sent with the package (never written)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("sentences", "t4"):
        if key not in payload:
            raise ValueError(f"{path}: expected a {key!r} block (keys: {sorted(payload)}).")
    return payload


def count_spans(sentences: Iterable[Mapping[str, Any]]) -> int:
    return sum(len(s["keys"]) for s in sentences)


# ------------------------------------------------------------------- model


def load_model_f32(
    model_id: str = MODEL_ID,
    revision: str | None = MODEL_REVISION,
    *,
    device: str = "auto",
    hf_token: str | None = None,
    cache_dir: str | None = None,
    bos_token: str | None = BOS_TOKEN,
) -> tuple[Any, Any, int, str, Any]:
    """``(tokenizer, model, bos_id, device, analyzer)``: float32 weights, LM head, pinned revision.

    Loaded through `Analyzer` rather than the reference's ``load`` (whose
    ``dtype=`` keyword needs a newer transformers); the objects handed to the
    reference functions are the same. ``bos_token=None`` uses the tokenizer's
    EOS (a test model without ``<|endoftext|>``).
    """
    from ..config.settings import ModelConfig, RunConfig
    from ..container import Analyzer

    analyzer = Analyzer(RunConfig(model=ModelConfig(
        model_id=model_id, revision=revision, device=device, dtype="float32", head="causal_lm",
        hf_token=hf_token, cache_dir=cache_dir,
    )))
    tok, mdl = analyzer.models.tokenizer, analyzer.models.model
    if bos_token is None:
        bos = int(tok.eos_token_id)
    else:
        bos = tok.convert_tokens_to_ids(bos_token)
        if bos is None or bos == tok.unk_token_id:
            raise ValueError(f"{bos_token!r} is not a token of {model_id}.")
    return tok, mdl, int(bos), str(analyzer.models.device), analyzer


# ---------------------------------------------------------------- the loops


def _header(job: str, model_meta: Mapping[str, Any], input_name: str, extra: Mapping[str, Any] | None) -> dict:
    return {
        "kind": "header", "job": job, "model_id": model_meta.get("model_id"), "revision": model_meta.get("revision"),
        "dtype": model_meta.get("dtype"), "bos_id": model_meta.get("bos_id"),
        "reference_code_sha": reference_sha256(), "input_cache": input_name,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        **(dict(extra) if extra else {}),
    }


def _run(
    job: str,
    compute,
    sentences: Sequence[Mapping[str, Any]],
    *,
    cache_path: str | os.PathLike | None,
    model_meta: Mapping[str, Any],
    input_name: str,
    limit: int | None,
    ids: Sequence[str] | None,
    show_progress: bool,
    mirror_path: str | os.PathLike | None,
    mirror_every: int,
    header_extra: Mapping[str, Any] | None = None,
) -> list[dict]:
    """Resumable sentence loop shared by T1 and T4; returns the rows in input order."""
    header = _header(job, model_meta, input_name, header_extra)
    done: dict[str, dict] = {}
    path = Path(cache_path) if cache_path else None
    if path is not None and path.exists():
        existing, rows, truncated = read_jsonl(path)
        if existing is not None:
            check_header(existing, header, path, unchecked=_UNCHECKED_HEADER_KEYS)
            header = existing
        for row in rows:
            done[row["id"]] = row
        if truncated:
            rewrite(path, header, rows)
        log.info("Resuming %s: %d sentences already in %s.", job, len(done), path)
    elif path is not None:
        write_header(path, header)

    wanted = list(sentences)
    if ids is not None:
        keep = set(ids)
        wanted = [s for s in wanted if s["id"] in keep]
    if limit is not None:
        wanted = wanted[: int(limit)]
    pending = [s for s in wanted if s["id"] not in done]
    fh = path.open("a", encoding="utf-8") if path is not None else None
    since_mirror = 0
    try:
        for sentence in maybe_progress(pending, show_progress, desc=job):
            started = time.perf_counter()
            row = compute(sentence)
            row["seconds"] = round(time.perf_counter() - started, 3)
            done[sentence["id"]] = row
            if fh is not None:
                append_row(fh, row)
                since_mirror += 1
                if mirror_path and since_mirror >= mirror_every:
                    mirror(path, mirror_path)
                    since_mirror = 0
    finally:
        if fh is not None:
            fh.close()
        if mirror_path and path is not None and since_mirror:
            mirror(path, mirror_path)
    return [done[s["id"]] for s in wanted if s["id"] in done]


def run_t1(
    tok: Any, mdl: Any, bos: int, sentences: Sequence[Mapping[str, Any]], *,
    device: str = "cpu", cache_path: str | os.PathLike | None = None, model_meta: Mapping[str, Any] | None = None,
    input_name: str = "", limit: int | None = None, ids: Sequence[str] | None = None, show_progress: bool = True,
    mirror_path: str | os.PathLike | None = None, mirror_every: int = 25,
) -> list[dict]:
    """Job C: ``{"id", "t1": {"i,j": [l2, js]}}`` per sentence, via `tree_runner_ref.t1_rows`."""
    def compute(s):
        return {"id": s["id"], "t1": ref.t1_rows(tok, mdl, bos, s["words"], s["keys"], device=device)}
    return _run("t1", compute, sentences, cache_path=cache_path, model_meta=model_meta or {}, input_name=input_name,
                limit=limit, ids=ids, show_progress=show_progress, mirror_path=mirror_path, mirror_every=mirror_every)


def run_t4(
    tok: Any, mdl: Any, bos: int, sentences: Sequence[Mapping[str, Any]], *,
    device: str = "cpu", batch: int = 64, cache_path: str | os.PathLike | None = None,
    model_meta: Mapping[str, Any] | None = None, input_name: str = "", limit: int | None = None,
    ids: Sequence[str] | None = None, show_progress: bool = True,
    mirror_path: str | os.PathLike | None = None, mirror_every: int = 5,
) -> list[dict]:
    """Job A: ``{"id", "end", "orig", "rows"}`` per sentence, via `tree_runner_ref.t4_rows`.

    `batch` only changes how many substitutions share a forward pass; beyond
    float rounding it changes no number (and is not part of the resume header).
    """
    def compute(s):
        end = ref.end_string(s["tree"])
        orig, rows = ref.t4_rows(tok, mdl, bos, s["words"], s["keys"], end, batch=batch, device=device)
        return {"id": s["id"], "end": end, "orig": orig, "rows": rows}
    return _run("t4", compute, sentences, cache_path=cache_path, model_meta=model_meta or {}, input_name=input_name,
                limit=limit, ids=ids, show_progress=show_progress, mirror_path=mirror_path, mirror_every=mirror_every)


def load_rows(path: str | os.PathLike) -> tuple[dict | None, list[dict]]:
    """A working cache or a finished deliverable (``.gz`` too): ``(header or meta, rows)``."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        head = lines[0] if lines and ("meta" in lines[0] or lines[0].get("kind") == "header") else None
        return (head.get("meta", head) if head else None), lines[1:] if head else lines
    header, rows, _ = read_jsonl(path)
    if header is None and rows and "meta" in rows[0]:
        return rows[0]["meta"], rows[1:]
    return header, rows


# ------------------------------------------------------------------ checks


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    if len(a) < 2:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def _rel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b != 0, np.abs(a - b) / np.abs(b), np.where(a == b, 0.0, np.inf))


def check_t1_anchor(rows: Sequence[Mapping[str, Any]], anchor: Mapping[str, Any]) -> dict[str, Any]:
    """Our T1 values against the anchor's (first 20 sentences): Spearman and median relative diff per distance."""
    mine = {r["id"]: r["t1"] for r in rows}
    missing_sent = [s["id"] for s in anchor["sentences"] if s["id"] not in mine]
    if missing_sent:
        raise ValueError(f"{len(missing_sent)} anchor sentence(s) not scored (first: {missing_sent[0]!r}).")
    ours, theirs, missing = [], [], 0
    for s in anchor["sentences"]:
        for key, value in s["t1"].items():
            if key not in mine[s["id"]]:
                missing += 1
                continue
            ours.append(mine[s["id"]][key])
            theirs.append(value)
    if missing:
        raise ValueError(f"{missing} anchor span(s) are missing from our rows.")
    ours_a, theirs_a = np.asarray(ours, dtype=np.float64), np.asarray(theirs, dtype=np.float64)
    out: dict[str, Any] = {"n_sentences": len(anchor["sentences"]), "n_spans": len(ours)}
    passed = True
    for k, name in enumerate(("l2", "js")):
        rho = _spearman(ours_a[:, k], theirs_a[:, k])
        rel = _rel(ours_a[:, k], theirs_a[:, k])
        med, worst = float(np.median(rel)), float(np.max(rel))
        ok = rho >= T1_MIN_SPEARMAN and med <= T1_MAX_MEDIAN_REL
        passed &= ok
        out[name] = {"spearman": rho, "median_rel_diff": med, "max_rel_diff": worst, "pass": bool(ok)}
    out["thresholds"] = {"min_spearman": T1_MIN_SPEARMAN, "max_median_rel_diff": T1_MAX_MEDIAN_REL}
    out["pass"] = bool(passed)
    return out


def _iter_t4(rows: Sequence[Mapping[str, Any]]):
    for r in rows:
        for filler, table in r["rows"].items():
            for key, v in table.items():
                yield r["id"], filler, key, v


def check_t4_anchor(rows: Sequence[Mapping[str, Any]], anchor_t4: Sequence[Mapping[str, Any]],
                    *, tol: float = T4_ANCHOR_TOL_NATS) -> dict[str, Any]:
    """Check 1: token counts identical, ``total`` / ``suf_sub`` / ``end_sub`` within `tol` nats."""
    mine = {r["id"]: r for r in rows}
    count_mismatch, missing, n = [], 0, 0
    worst = {"total": 0.0, "suf_sub": 0.0, "end_sub": 0.0}
    orig_problems = []
    for a in anchor_t4:
        if a["id"] not in mine:
            raise ValueError(f"anchor sentence {a['id']!r} was not scored.")
        m = mine[a["id"]]
        if m["end"] != a["end"]:
            orig_problems.append(f"{a['id']}: end {m['end']!r} != {a['end']!r}")
        if m["orig"]["ids"] != a["orig"]["ids"] or m["orig"]["end_ids"] != a["orig"]["end_ids"]:
            orig_problems.append(f"{a['id']}: original ids differ")
        for filler, table in a["rows"].items():
            for key, v in table.items():
                ours = m["rows"].get(filler, {}).get(key)
                if ours is None:
                    missing += 1
                    continue
                n += 1
                if list(ours[1:4]) != list(v[1:4]):
                    count_mismatch.append(f"{a['id']} {filler} {key}: {ours[1:4]} != {v[1:4]}")
                for name, k in (("total", 0), ("suf_sub", 4), ("end_sub", 5)):
                    worst[name] = max(worst[name], abs(float(ours[k]) - float(v[k])))
    passed = not count_mismatch and not missing and not orig_problems and all(w <= tol for w in worst.values())
    return {
        "n_sentences": len(anchor_t4), "n_rows": n, "missing": missing,
        "count_mismatches": len(count_mismatch), "count_mismatch_examples": count_mismatch[:5],
        "orig_problems": orig_problems, "max_abs_diff": worst, "tol_nats": tol, "pass": bool(passed),
    }


def check_t4_cache(rows: Sequence[Mapping[str, Any]], cache_sentences: Sequence[Mapping[str, Any]],
                   *, min_spearman: float = T4_CACHE_MIN_SPEARMAN,
                   fillers: Sequence[str] | None = None) -> dict[str, Any]:
    """Check 2: ``n_tok`` equals the 1b cache's count on every row; ``total`` rank-correlates with its total.

    `fillers` restricts the check to those strings (a 1b cache that scored only some of them)."""
    cache = {s["id"]: s["spans"] for s in cache_sentences}
    ours, theirs, mismatches, missing = [], [], [], 0
    for sid, filler, key, v in _iter_t4(rows):
        if fillers is not None and filler not in fillers:
            continue
        c = cache.get(sid, {}).get(filler, {}).get(key)
        if c is None:
            missing += 1
            continue
        if int(v[1]) != int(c[1]):
            mismatches.append(f"{sid} {filler} {key}: n_tok {v[1]} != cache {c[1]}")
        ours.append(float(v[0]))
        theirs.append(float(c[0]))
    rho = _spearman(ours, theirs)
    diff = np.abs(np.asarray(ours) - np.asarray(theirs)) if ours else np.zeros(0)
    passed = not mismatches and not missing and rho >= min_spearman
    return {
        "n_sentences": len(rows), "n_rows": len(ours), "missing": missing, "n_tok_mismatches": len(mismatches),
        "n_tok_mismatch_examples": mismatches[:5], "spearman_total": rho,
        "max_abs_diff_total": float(diff.max()) if diff.size else 0.0,
        "median_abs_diff_total": float(np.median(diff)) if diff.size else 0.0,
        "min_spearman": min_spearman, "pass": bool(passed),
    }


def check_t4_causality(rows: Sequence[Mapping[str, Any]], *, tol: float = T4_MAX_PRE_CHECK) -> dict[str, Any]:
    """Check 3: ``|pre_check|`` at most `tol` on every row."""
    values = [abs(float(v[6])) for _, _, _, v in _iter_t4(rows)]
    worst = max(values, default=0.0)
    over = sum(1 for x in values if x > tol)
    return {"n_rows": len(values), "max_abs_pre_check": worst, "n_over": over, "tol": tol, "pass": over == 0}


# -------------------------------------------------------------------- meta


def environment_meta() -> dict[str, Any]:
    """Library versions and the GPU, for every meta block."""
    versions: dict[str, Any] = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("torch", "transformers", "nltk", "scipy"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 - a missing optional library is reported, not fatal
            versions[name] = None
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return {"versions": versions, "gpu": gpu}


def run_meta(
    job: str,
    *,
    model_id: str,
    revision: str | None,
    dtype: str = "float32",
    wall_seconds: float | None = None,
    checks: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The deliverable's meta block."""
    env = environment_meta()
    return {
        "job": job, "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "model_id": model_id, "revision": revision, "dtype": dtype, "logits_dtype": "float32",
        "torch": env["versions"].get("torch"), "transformers": env["versions"].get("transformers"),
        "versions": env["versions"], "gpu": env["gpu"],
        "wall_seconds": None if wall_seconds is None else round(float(wall_seconds), 1),
        "reference_code_sha": reference_sha256(), "reference_code_modified": not reference_unmodified(),
        "checks": dict(checks) if checks else {},
        **(dict(extra) if extra else {}),
    }


def _strip(row: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {k: row[k] for k in keys}


def finalize(
    rows: Sequence[Mapping[str, Any]],
    out_path: str | os.PathLike,
    meta: Mapping[str, Any],
    *,
    job: str,
    expected: int | None = None,
) -> Path:
    """Write the deliverable: ``{"meta": ...}``, then one line per sentence (gzipped if `out_path` ends in .gz).

    `expected` (the number of sentences) refuses a partial file. The row keys
    are the requested ones: ``id, t1`` (C) or ``id, end, orig, rows`` (A).
    """
    keys = {"t1": ("id", "t1"), "t4": ("id", "end", "orig", "rows")}[job]
    if expected is not None and len(rows) != expected:
        raise ValueError(f"{len(rows)} of {expected} sentences are scored; finish the run before finalising.")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    opener = (lambda p: gzip.open(p, "wt", encoding="utf-8")) if out.suffix == ".gz" else (
        lambda p: open(p, "w", encoding="utf-8"))
    with opener(tmp) as fh:
        fh.write(json.dumps({"meta": dict(meta)}) + "\n")
        for row in rows:
            fh.write(json.dumps(_strip(row, keys)) + "\n")
    os.replace(tmp, out)
    return out


def wall_seconds(rows: Sequence[Mapping[str, Any]]) -> float:
    """Summed per-sentence compute time recorded in the working cache (survives resumes)."""
    return float(sum(float(r.get("seconds", 0.0)) for r in rows))


__all__ = [
    "REFERENCE_SHA256", "MODEL_ID", "MODEL_REVISION", "BOS_TOKEN", "T1_OUTPUT", "T4_OUTPUT", "T4_FIELDS",
    "reference_sha256", "reference_unmodified", "load_1b_cache", "load_anchor", "count_spans", "load_model_f32",
    "run_t1", "run_t4", "load_rows", "check_t1_anchor", "check_t4_anchor", "check_t4_cache", "check_t4_causality",
    "environment_meta", "run_meta", "finalize", "wall_seconds",
]
