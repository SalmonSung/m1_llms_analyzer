"""Task 8a with a second distance (runner job B): Jensen-Shannon and the head share on every row.

The Task 8a L2 distance turned out to be carried almost entirely by the
improbable words. This job re-runs **exactly** the 8a passes (same frames file,
``tok(prefix)`` with the tokenizer's default special tokens, state = float32
``log_softmax`` at the last token, `NextTokenStateService` with
``bos_policy="none"``) on the five models of the 8a run and adds, per
(frame, structure, control) row::

    js_ret,   js_ctrl    tree_runner_ref.jsd_distance(s_E, s_REF), (s_C, s_REF)   sqrt(JSD) in nats, float64
    head_ret, head_ctrl  share of ||a - b||^2 carried by the union of both states' top-100 words

and repeats ``ret`` / ``ctrl`` (L2, computed exactly as `task_8a.frame_scalars`)
as the check: they must match the delivered ``record_8a_multimodel.json``
within 2 % relative on every row (`l2_match`), or the record is refused.

Phase A (`score_model_8a_js`) writes one resumable JSONL cache per model;
phase B (`build_record_8a_js`) merges them into ``record_8a_js.json``.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..utils.batching import maybe_progress
from ..utils.logging import get_logger
from . import tree_runner_ref as ref
from .frames_8a import (
    CODES, PAIRS, REF, frame_prefixes, frames_sha256, last_tok, load_frames_8a, n_tok, token_assertions,
)
from .jsonl_cache import append_row, check_header, mirror, read_jsonl, rewrite, write_header
from .task_8a import ModelSpec, model_meta_8a, prefix_ids
from .tree_runner import environment_meta, reference_sha256, reference_unmodified

log = get_logger("task_8a_js")

TASK = "8a_js"
SCHEMA = 1
HEAD_K = 100
L2_MATCH_TOL = 0.02
#: The five models of the delivered 8a run.
MODELS_8A_JS: tuple[str, ...] = ("qwen25_0.5b", "qwen3_0.6b", "qwen3_1.7b", "qwen3_8b", "llama31_8b")
JS_MEASURE = "tree_runner_ref.jsd_distance: sqrt of the Jensen-Shannon divergence in nats between the two states, float64"
HEAD_MEASURE = (f"top = union(argsort(-a)[:{HEAD_K}], argsort(-b)[:{HEAD_K}]); "
                "sum(((a - b)**2)[top]) / sum((a - b)**2), float64")
ROW_KEYS = ("frame", "structure", "control", "ret", "ctrl", "js_ret", "js_ctrl", "head_ret", "head_ctrl")
_UNCHECKED_HEADER_KEYS = frozenset({"kind", "date", "device", "library_versions", "n_params_b", "gpu", "versions"})


# ---------------------------------------------------------------- measures


def _top(s: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), s.shape[0])
    return np.argpartition(-s, k - 1)[:k]


def head_share(a: np.ndarray, b: np.ndarray, k: int = HEAD_K) -> float:
    """Share of the squared L2 distance carried by the union of the two states' top-`k` words."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    d2 = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    total = float(d2.sum())
    if total == 0.0:
        return 0.0
    top = np.union1d(_top(a, k), _top(b, k))
    return float(d2[top].sum() / total)


def pair_measures(s_e: np.ndarray, s_c: np.ndarray, s_ref: np.ndarray, *, k: int = HEAD_K) -> dict[str, float]:
    """The six numbers of one row; ``ret``/``ctrl`` exactly as `task_8a.frame_scalars` computes them."""
    e = np.asarray(s_e, dtype=np.float32).ravel()
    c = np.asarray(s_c, dtype=np.float32).ravel()
    r = np.asarray(s_ref, dtype=np.float32).ravel()
    return {
        "ret": float(np.linalg.norm(e - r)), "ctrl": float(np.linalg.norm(c - r)),
        "js_ret": ref.jsd_distance(e, r), "js_ctrl": ref.jsd_distance(c, r),
        "head_ret": head_share(e, r, k), "head_ctrl": head_share(c, r, k),
    }


# ------------------------------------------------------------------ scoring


def score_frames_js_batch(
    states: Any,
    tokenizer: Any,
    frames: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    batch_size: int | None = None,
    k: int = HEAD_K,
) -> list[dict[str, Any]]:
    """Score ``[(frame_index, frame), ...]`` with the 8a passes; one cache row per frame.

    Mirrors `task_8a.score_frames_batch` (same ids, same assertions, same state
    read-out) and reduces each frame's fourteen states to its seven rows.
    """
    if states.bos_id() is not None:
        raise ValueError("Task 8a scores tok(prefix) as the tokenizer returns it; use ScoringConfig(bos_policy='none').")
    if not frames:
        return []
    started = time.perf_counter()
    seqs: list[list[int]] = []
    owners: list[tuple[int, str]] = []
    for index, frame in frames:
        prefixes = frame_prefixes(frame)
        counts = {code: n_tok(tokenizer, p) for code, p in prefixes.items()}
        lasts = {code: last_tok(tokenizer, p) for code, p in prefixes.items()}
        failures = token_assertions(counts, lasts)
        if failures:
            raise ValueError(f"frame {index} fails the token assertions under this tokenizer: {failures}")
        for code in CODES:
            seqs.append(prefix_ids(tokenizer, prefixes[code]))
            owners.append((index, code))
    results = states.states(seqs, [[len(ids) - 1] for ids in seqs], batch_size=batch_size or len(seqs))
    by_frame: dict[int, dict[str, np.ndarray]] = {index: {} for index, _ in frames}
    for (index, code), result in zip(owners, results):
        by_frame[index][code] = result.states[0]
    seconds = (time.perf_counter() - started) / len(frames)
    out = []
    for index, _ in frames:
        st = by_frame[index]
        rows = []
        for e, c in PAIRS:
            rows.append({"structure": e, "control": c, **pair_measures(st[e], st[c], st[REF], k=k)})
        out.append({"kind": "frame", "frame": int(index), "rows": rows, "seconds": round(seconds, 4)})
    return out


def score_model_8a_js(
    analyzer: Any,
    spec: ModelSpec,
    frames_path: str | os.PathLike,
    *,
    cache_path: str | os.PathLike | None = None,
    frames_per_batch: int = 8,
    provenance: Mapping[str, Any] | None = None,
    show_progress: bool = True,
    mirror_path: str | os.PathLike | None = None,
    mirror_every: int = 10,
    limit: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Phase A for one model: every frame, cached and resumable. ``(header, rows)`` in frame order."""
    if frames_per_batch < 1:
        raise ValueError("frames_per_batch must be >= 1.")
    states = analyzer.states
    if states is None:
        raise RuntimeError("Job B needs the LM head: build the Analyzer with head='causal_lm'.")
    if states.config.bos_policy != "none":
        raise ValueError("Job B needs ScoringConfig(bos_policy='none'), exactly as the 8a run.")
    tokenizer = analyzer.models.tokenizer
    frames = load_frames_8a(frames_path)["frames"]
    meta = model_meta_8a(analyzer, spec)
    header = {
        "kind": "header", "schema": SCHEMA, "task": TASK, "model_key": spec.key, "hf_id": meta["hf_id"],
        "revision": meta.get("revision"), "weight_dtype": meta.get("weight_dtype"), "logits_dtype": "float32",
        "bos_added": bool(meta.get("bos_added")), "vocab_size": int(meta["vocab_size"]), "n_params_b": meta.get("n_params_b"),
        "frames_file": Path(frames_path).name, "frames_sha256": frames_sha256(frames_path), "n_frames": len(frames),
        "reference_code_sha": reference_sha256(), "js_measure": JS_MEASURE, "head_measure": HEAD_MEASURE,
        "head_k": HEAD_K, "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        **(dict(provenance) if provenance else {}),
    }
    done: dict[int, dict] = {}
    path = Path(cache_path) if cache_path else None
    if path is not None and path.exists():
        existing, rows, truncated = read_jsonl(path)
        if existing is not None:
            check_header(existing, header, path, unchecked=_UNCHECKED_HEADER_KEYS)
            header = existing
        for row in rows:
            done[int(row["frame"])] = row
        if truncated:
            rewrite(path, header, rows)
        log.info("Resuming: %d of %d frames already scored in %s.", len(done), len(frames), path)
    elif path is not None:
        write_header(path, header)

    wanted = list(enumerate(frames)) if limit is None else list(enumerate(frames))[: int(limit)]
    pending = [(i, f) for i, f in wanted if i not in done]
    chunks = [pending[k: k + frames_per_batch] for k in range(0, len(pending), frames_per_batch)]
    fh = path.open("a", encoding="utf-8") if path is not None else None
    since_mirror = 0
    try:
        for chunk in maybe_progress(chunks, show_progress, desc="frames"):
            for row in score_frames_js_batch(states, tokenizer, chunk, batch_size=len(chunk) * len(CODES)):
                done[int(row["frame"])] = row
                if fh is not None:
                    append_row(fh, row)
                    since_mirror += 1
            if fh is not None and mirror_path and since_mirror >= mirror_every:
                mirror(path, mirror_path)
                since_mirror = 0
    finally:
        if fh is not None:
            fh.close()
        if mirror_path and path is not None and since_mirror:
            mirror(path, mirror_path)
    return header, [done[i] for i, _ in enumerate(frames) if i in done]


def load_scores_8a_js(path: str | os.PathLike) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header, rows, _ = read_jsonl(Path(path))
    if header is None or header.get("task") != TASK:
        raise ValueError(f"{path} has no job-B header line; it is not a scores_8a_js cache.")
    return header, sorted(rows, key=lambda r: int(r["frame"]))


def rows_from_cache(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The record's rows, in frame then `PAIRS` order."""
    out = []
    for frame_row in rows:
        for r in frame_row["rows"]:
            out.append({"frame": int(frame_row["frame"]), **{k: r[k] for k in ROW_KEYS if k != "frame"}})
    return out


# -------------------------------------------------------------------- check


def load_record_8a(path: str | os.PathLike) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def l2_match(rows: Sequence[Mapping[str, Any]], record_8a: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Our ``ret``/``ctrl`` against the delivered 8a record's, on every row we have: the largest relative diff."""
    if key not in record_8a.get("models", {}):
        raise ValueError(f"record_8a_multimodel has no model {key!r} (has {sorted(record_8a.get('models', {}))}).")
    theirs = {(int(r["frame"]), r["structure"], r["control"]): r for r in record_8a["models"][key]["rows"]}
    worst, worst_row, missing = 0.0, None, 0
    for r in rows:
        t = theirs.get((int(r["frame"]), r["structure"], r["control"]))
        if t is None:
            missing += 1
            continue
        for name in ("ret", "ctrl"):
            b = float(t[name])
            rel = abs(float(r[name]) - b) / abs(b) if b else (0.0 if float(r[name]) == b else float("inf"))
            if rel > worst:
                worst, worst_row = rel, {"frame": int(r["frame"]), "structure": r["structure"], "control": r["control"],
                                         "field": name, "ours": float(r[name]), "record": b}
    return {"model": key, "n_rows": len(rows) - missing, "missing": missing, "max_rel": worst, "worst": worst_row,
            "tol": L2_MATCH_TOL, "pass": missing == 0 and worst <= L2_MATCH_TOL}


# ------------------------------------------------------------------- record


def build_record_8a_js(
    caches: Mapping[str, str | os.PathLike],
    record_8a: Mapping[str, Any] | str | os.PathLike,
    frames_path: str | os.PathLike,
    *,
    keys: Sequence[str] | None = None,
    allow_partial: bool = False,
    wall_seconds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Phase B: ``record_8a_js.json`` from every cache present. Raises if a model's L2 is off by more than 2 %."""
    if not isinstance(record_8a, Mapping):
        record_8a = load_record_8a(record_8a)
    sha = frames_sha256(frames_path)
    n = len(load_frames_8a(frames_path)["frames"])
    keys = list(keys) if keys is not None else [k for k in MODELS_8A_JS if k in caches] + [
        k for k in caches if k not in MODELS_8A_JS]
    models, match, model_meta = {}, {}, {}
    for key in keys:
        if key not in caches or not Path(caches[key]).exists():
            log.warning("no job-B cache for %s; it will be missing from the record.", key)
            continue
        header, frame_rows = load_scores_8a_js(caches[key])
        problems = []
        if header.get("model_key") != key:
            problems.append(f"cache is for model_key {header.get('model_key')!r}, not {key!r}")
        if header.get("frames_sha256") != sha:
            problems.append("cache was scored on a different frames file (sha256 differs)")
        if sorted(int(r["frame"]) for r in frame_rows) != list(range(n)):
            problems.append(f"cache holds {len(frame_rows)} of {n} frames")
        if problems:
            message = f"{caches[key]}: " + "; ".join(problems)
            if not allow_partial:
                raise ValueError(message)
            log.warning("%s -- skipped.", message)
            continue
        rows = rows_from_cache(frame_rows)
        m = l2_match(rows, record_8a, key)
        if not m["pass"]:
            raise ValueError(f"{key}: L2 differs from record_8a_multimodel.json by {m['max_rel']:.4f} relative "
                             f"(tol {L2_MATCH_TOL}; worst {m['worst']}, missing {m['missing']}). Stop and report.")
        match[key] = m["max_rel"]
        models[key] = {"rows": rows}
        model_meta[key] = {
            "hf_id": header.get("hf_id"), "revision": header.get("revision"), "weight_dtype": header.get("weight_dtype"),
            "logits_dtype": header.get("logits_dtype"), "bos_added": header.get("bos_added"),
            "wall_seconds": round(float((wall_seconds or {}).get(key, sum(float(r.get("seconds", 0)) for r in frame_rows))), 1),
        }
    env = environment_meta()
    record = {
        "meta": {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "reference_code_sha": reference_sha256(),
            "reference_code_modified": not reference_unmodified(),
            "l2_match_max_rel": match, "l2_match_tol": L2_MATCH_TOL, "versions": env["versions"], "gpu": env["gpu"],
            "frames_sha256": sha, "n_frames": n, "js_measure": JS_MEASURE, "head_measure": HEAD_MEASURE,
            "models": model_meta,
        },
        "models": models,
    }
    validate_record_8a_js(record, n_frames=n)
    return record


def validate_record_8a_js(record: Mapping[str, Any], *, n_frames: int | None = None) -> None:
    """Fail loudly if the record lacks a requested field or holds an impossible number."""
    for key in ("meta", "models"):
        if key not in record:
            raise ValueError(f"record is missing {key!r}")
    for key in ("date", "reference_code_sha", "l2_match_max_rel", "versions"):
        if key not in record["meta"]:
            raise ValueError(f"meta is missing {key!r}")
    for key, block in record["models"].items():
        rows = block.get("rows")
        if rows is None:
            raise ValueError(f"models.{key} is missing 'rows'")
        if n_frames is not None and len(rows) != len(PAIRS) * n_frames:
            raise ValueError(f"models.{key}: {len(rows)} rows, expected {len(PAIRS) * n_frames}")
        if key not in record["meta"]["l2_match_max_rel"]:
            raise ValueError(f"meta.l2_match_max_rel lacks {key!r}")
        for k, row in enumerate(rows):
            missing = [name for name in ROW_KEYS if name not in row]
            if missing:
                raise ValueError(f"models.{key}: row {k} lacks {missing}")
            expected = PAIRS[k % len(PAIRS)]
            if (row["structure"], row["control"]) != tuple(expected) or int(row["frame"]) != k // len(PAIRS):
                raise ValueError(f"models.{key}: row {k} is out of frame/PAIRS order")
            for name in ("ret", "ctrl"):
                if not (np.isfinite(row[name]) and row[name] >= 0):
                    raise ValueError(f"models.{key}: row {k} {name} is {row[name]}")
            for name in ("js_ret", "js_ctrl"):
                if not (0 <= float(row[name]) <= np.sqrt(np.log(2)) + 1e-9):
                    raise ValueError(f"models.{key}: row {k} {name} is {row[name]}, outside [0, sqrt(ln 2)]")
            for name in ("head_ret", "head_ctrl"):
                if not (0 <= float(row[name]) <= 1 + 1e-9):
                    raise ValueError(f"models.{key}: row {k} {name} is {row[name]}, outside [0, 1]")


__all__ = [
    "TASK", "HEAD_K", "L2_MATCH_TOL", "MODELS_8A_JS", "JS_MEASURE", "HEAD_MEASURE", "ROW_KEYS", "head_share",
    "pair_measures", "score_frames_js_batch", "score_model_8a_js", "load_scores_8a_js", "rows_from_cache",
    "load_record_8a", "l2_match", "build_record_8a_js", "validate_record_8a_js",
]
