"""Task 9b: does the deletion change what the model *writes*, not only what it predicts?

Task 9a compares next-token *distributions* after a splice (its divergence
``div``). The claim it stands in for is about the *text* that follows. One
greedy continuation cannot measure that -- after a sentence boundary the next
sentence is nearly free, and greedy decodes from the original and the spliced
context part within a few tokens either way -- so this task uses two text
measures that do not depend on a single decode, every parameter fixed before
anything is generated (`GenerationProtocol`, written to the cache header).

Inputs are the 9a cache (paragraph text and per-token ``surprisal``) and the
9a record (its ``cuts`` are exactly the labelled close / far cuts, in order);
both are read and never written. For each labelled cut ``(i, j)`` with
original ids ``ids`` and spliced ids ``ids[:i+1] + ids[j+1:]``:

**Measure A -- the author's actual continuation.** The ``W`` tokens after the
join are the same tokens in both sequences (``ids[j+1+k]`` is
``spliced[i+1+k]``; admissibility guarantees they exist). ::

    lp_orig[k] = log p(ids[j+1+k]     | ids[:j+1+k])        original context
    lp_spl[k]  = log p(spliced[i+1+k] | spliced[:i+1+k])    spliced context, same target token
    true_dlogp = mean_k (lp_orig[k] - lp_spl[k])             nats/token; > 0: the deletion made the real text less likely

``lp_spl`` is one scored pass of the spliced ids. ``lp_orig`` is a fresh pass
of the original ids in the same session, so both sides share one set of
numerics; the cache's ``surprisal`` (``-lp_orig`` from the 9a pass, 6 dp) is
the *invariant* it is checked against, not the input.

**Measure B -- sampled continuations, compared as distributions.** ``K``
continuations of ``L`` new tokens are sampled from context O = ``ids[:j+1]``
and from context S = ``spliced[:i+1]`` (the same token ends both; only the
deleted sentences differ) under nucleus sampling with a per-cut seed, the same
seed for O and S so the two sets are paired by random state. Token F1 over
multisets of ids gives ``within`` (mean over the O-O pairs), ``cross`` (mean
over the O-S pairs) and ``sample_overlap = cross / within``; every sampled
continuation is stored so another overlap function can be applied later.

**Descriptive only.** ``first_diff_greedy``: the first position at which greedy
decodes from O and S differ, capped. It is in the record because the theory
document quotes it; nothing is judged on it.

Phase A (`compute_generation_9b`) writes one JSONL line per cut, fsynced and
resumable, under a header that pins the protocol *and* the sha256 of both 9a
inputs; phase A' (`build_record_9b`) assembles the record and reports the
three invariants (`check_invariants_9b`): the cached surprisal reproduces
``lp_orig``; the cut keys match the 9a record exactly; ``within > 0`` for every
cut (a collapsed sample set leaves measure B undefined there and is reported,
never adjusted). The pre-registered analysis (stratified far vs close on both
measures, Spearman with ``state_div``) is deliberately done outside this repo,
from the record.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..utils.batching import maybe_progress
from ..utils.logging import get_logger
from .decoding import first_diff, greedy_continuation, overlap_stats, sample_continuations
from .jsonl_cache import append_row, check_header, mirror, read_jsonl, rewrite, write_header
from .splice import load_splices, splice_ids
from .task_9a import CLOSE, FAR, cut_key

log = get_logger("task_9b")

TASK = "9b"
SCHEMA = 1
OVERLAP_MEASURE = "token F1 on ids, cross/within"
EOS_POLICY = "ordinary token: never suppressed, never a stop; every sample is exactly L ids"
#: Header fields that may differ between the writer and a resumer.
_UNCHECKED_HEADER_KEYS = frozenset({"kind", "date", "device", "device_info", "library_versions"})
#: The per-cut fields the request asks for, in this order; everything else in a record cut is additive.
REQUESTED_CUT_FIELDS = (
    "paragraph", "i", "j", "pair", "stratum", "seg_len", "state_div", "true_dlogp", "true_dlogp_k",
    "sample_overlap", "within", "cross", "samples_orig", "samples_spliced", "first_diff_greedy",
)
ADDITIVE_CUT_FIELDS = (
    "key", "cut_index", "seed", "grammatical", "n_eos_orig", "n_eos_spliced", "within_spliced",
    "greedy_orig", "greedy_spliced", "cached_lp_orig_max_abs_diff",
)


@dataclass(frozen=True)
class GenerationProtocol:
    """Everything fixed before a token is generated. Written to the cache header verbatim."""

    W_true: int = 20          #: tokens of the author's continuation scored (measure A)
    K: int = 16               #: samples per context (measure B)
    L: int = 30               #: new tokens per sample
    top_p: float = 0.95
    temperature: float = 1.0
    seed_base: int = 20250914  #: seed for cut_index c is seed_base + c, the same for O and S
    greedy_cap: int = 30      #: greedy tokens decoded, and the cap on first_diff_greedy

    def __post_init__(self) -> None:
        if self.W_true < 1 or self.K < 2 or self.L < 1 or self.greedy_cap < 1:
            raise ValueError("need W_true >= 1, K >= 2 (within needs a pair), L >= 1, greedy_cap >= 1.")
        if not 0 < self.top_p <= 1 or self.temperature <= 0:
            raise ValueError("need 0 < top_p <= 1 and temperature > 0.")

    @property
    def sampling(self) -> str:
        return f"nucleus p={self.top_p}, T={self.temperature}, seed {self.seed_base}+idx, paired O/S"

    def seed_for(self, cut_index: int) -> int:
        return int(self.seed_base) + int(cut_index)

    def as_header(self) -> dict[str, Any]:
        return {
            "W_true": self.W_true, "K": self.K, "L": self.L, "top_p": self.top_p, "temperature": self.temperature,
            "seed_base": self.seed_base, "greedy_cap": self.greedy_cap,
            "sampling": self.sampling, "overlap": OVERLAP_MEASURE, "eos": EOS_POLICY,
            "top_k": None, "repetition_penalty": 1.0,
        }


# ------------------------------------------------------------------ inputs


@dataclass
class Inputs9a:
    """The two 9a files, read once, with their identity for the header."""

    record: dict[str, Any]
    header: dict[str, Any]
    rows: list[dict[str, Any]]
    source_record: dict[str, Any]
    source_cache: dict[str, Any]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_9a_inputs(record_path: str | os.PathLike, cache_path: str | os.PathLike) -> Inputs9a:
    """Read the 9a record and cache (read-only) and check they fit each other."""
    record_path, cache_path = Path(record_path), Path(cache_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    cuts = record.get("cuts")
    if not cuts:
        raise ValueError(f"{record_path} has no cuts; it is not a Task 9a record.")
    bad = [c for c in cuts if c.get("pair") not in (CLOSE, FAR)]
    if bad:
        raise ValueError(f"{record_path}: {len(bad)} cut(s) are not labelled close/far; a 9a record only lists labelled cuts.")
    for name in ("paragraph", "i", "j", "seg_len", "divergence", "stratum"):
        if any(name not in c for c in cuts):
            raise ValueError(f"{record_path}: every cut needs {name!r}.")
    header, rows = load_splices(cache_path)
    by_id = {r["id"]: r for r in rows}
    missing = sorted({c["paragraph"] for c in cuts} - set(by_id))
    if missing:
        raise ValueError(f"{len(missing)} paragraph(s) of {record_path.name} are not in {cache_path.name} (first: {missing[0]!r}).")
    lacking = sum(1 for r in rows if "surprisal" not in r)
    if lacking:
        log.warning("%s: %d paragraph(s) lack `surprisal`; invariant (1) cannot be checked there.", cache_path, lacking)
    meta = record.get("meta", {})
    source_record = {
        "name": record_path.name, "sha256": _sha256(record_path), "n_cuts": len(cuts),
        "model": meta.get("model"), "date": meta.get("date"), "window": record.get("window"),
    }
    source_cache = {
        "name": cache_path.name, "sha256": _sha256(cache_path), "schema": header.get("schema"),
        "window": header.get("window"), "model_id": header.get("model_id"), "revision": header.get("revision"),
        "dtype": header.get("dtype"), "bos_token": header.get("bos_token"), "vocab_size": header.get("vocab_size"),
    }
    return Inputs9a(record, header, rows, source_record, source_cache)


def cut_specs(record_9a: Mapping[str, Any], rows_9a: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One spec per record cut, in record order: identity, labels, and its paragraph's text and surprisal."""
    by_id = {r["id"]: r for r in rows_9a}
    specs = []
    for index, c in enumerate(record_9a["cuts"]):
        row = by_id[c["paragraph"]]
        specs.append({
            "key": cut_key(c["paragraph"], int(c["i"]), int(c["j"])), "cut_index": index,
            "paragraph": c["paragraph"], "i": int(c["i"]), "j": int(c["j"]), "pair": c["pair"],
            "stratum": c["stratum"], "seg_len": int(c["seg_len"]), "state_div": float(c["divergence"]),
            "grammatical": bool(c.get("grammatical", True)),
            "text": row["text"], "n_tokens": int(row["n_tokens"]), "surprisal": row.get("surprisal"),
        })
    return specs


# ---------------------------------------------------------------- measures


def _all_token_logprobs(states, ids: Sequence[int]) -> np.ndarray:
    """log p of every token of `ids` from one pass (needs a BOS so token 0 is scored)."""
    [result] = states.states([list(ids)], [[]], batch_size=1, return_token_logprobs=True)
    lp = np.asarray(result.token_logprobs, dtype=np.float64).ravel()
    if len(lp) != len(ids):
        raise ValueError(
            f"got {len(lp)} token log-probabilities for {len(ids)} tokens: token 0 is unscored. Task 9b needs "
            "bos_policy='auto' (the policy the 9a cache was scored under), not 'none'."
        )
    return lp


def measure_a(
    states,
    ids: Sequence[int],
    i: int,
    j: int,
    W: int,
    *,
    orig_token_logprobs: np.ndarray,
    cached_surprisal: Sequence[float | None] | None = None,
) -> dict[str, Any]:
    """Measure A for one cut: the true continuation's log-probability under both contexts."""
    n = len(ids)
    if n - (j + 1) < W:
        raise ValueError(f"cut ({i}, {j}): only {n - (j + 1)} tokens follow j, {W} needed.")
    spliced = splice_ids(ids, i, j)
    lp_spl_all = _all_token_logprobs(states, spliced)
    lp_orig = np.asarray(orig_token_logprobs, dtype=np.float64)[j + 1: j + 1 + W]
    lp_spl = lp_spl_all[i + 1: i + 1 + W]
    if not (spliced[i + 1: i + 1 + W] == list(ids[j + 1: j + 1 + W])):
        raise AssertionError("spliced[i+1:i+1+W] is not ids[j+1:j+1+W]; the splice is wrong.")
    dk = lp_orig - lp_spl
    cached_diff = None
    if cached_surprisal is not None:
        cached = cached_surprisal[j + 1: j + 1 + W]
        if len(cached) == W and all(x is not None for x in cached):
            cached_diff = float(np.max(np.abs(lp_orig + np.asarray(cached, dtype=np.float64))))
    return {
        "lp_orig": [round(float(x), 6) for x in lp_orig], "lp_spl": [round(float(x), 6) for x in lp_spl],
        "true_dlogp_k": [round(float(x), 6) for x in dk], "true_dlogp": round(float(np.mean(dk)), 6),
        "cached_lp_orig_max_abs_diff": None if cached_diff is None else round(cached_diff, 8),
    }


def measure_b(
    provider,
    ids: Sequence[int],
    i: int,
    j: int,
    protocol: GenerationProtocol,
    *,
    seed: int,
    bos_id: int | None,
    kv_cache: bool = True,
    max_length: int | None = None,
) -> dict[str, Any]:
    """Measure B (and the greedy descriptive) for one cut: O = ids[:j+1], S = ids[:i+1]."""
    context_o = list(ids[: j + 1])
    context_s = list(ids[: i + 1])  # == spliced[:i+1]
    common = dict(K=protocol.K, L=protocol.L, top_p=protocol.top_p, temperature=protocol.temperature,
                  bos_id=bos_id, kv_cache=kv_cache, max_length=max_length)
    samples_o = sample_continuations(provider, context_o, seed=seed, **common)
    samples_s = sample_continuations(provider, context_s, seed=seed, **common)
    greedy_o = greedy_continuation(provider, context_o, L=protocol.greedy_cap, bos_id=bos_id, kv_cache=kv_cache, max_length=max_length)
    greedy_s = greedy_continuation(provider, context_s, L=protocol.greedy_cap, bos_id=bos_id, kv_cache=kv_cache, max_length=max_length)
    stats = overlap_stats(samples_o, samples_s)
    eos = getattr(provider.tokenizer, "eos_token_id", None)
    n_eos = lambda s: int((s == eos).any(axis=1).sum()) if eos is not None else 0  # noqa: E731
    return {
        "samples_orig": samples_o.astype(int).tolist(), "samples_spliced": samples_s.astype(int).tolist(),
        "within": round(stats["within"], 6), "cross": round(stats["cross"], 6),
        "within_spliced": round(stats["within_spliced"], 6),
        "sample_overlap": None if stats["collapsed"] else round(stats["sample_overlap"], 6),
        "n_eos_orig": n_eos(samples_o), "n_eos_spliced": n_eos(samples_s),
        "greedy_orig": greedy_o.astype(int).tolist(), "greedy_spliced": greedy_s.astype(int).tolist(),
        "first_diff_greedy": int(first_diff(greedy_o, greedy_s, protocol.greedy_cap)),
    }


def score_cut(
    states,
    provider,
    spec: Mapping[str, Any],
    protocol: GenerationProtocol,
    *,
    orig_logprobs_cache: dict[str, np.ndarray],
    kv_cache: bool = True,
    max_length: int | None = None,
) -> dict[str, Any]:
    """Both measures for one cut; the cache row. One original pass per paragraph, memoised."""
    started = time.perf_counter()
    ids = states.encode(spec["text"])
    if len(ids) != int(spec["n_tokens"]):
        raise ValueError(
            f"{spec['paragraph']}: the text re-encodes to {len(ids)} tokens but the 9a cache says {spec['n_tokens']}. "
            "This is not the model / tokenizer the cache was scored with."
        )
    if spec["paragraph"] not in orig_logprobs_cache:
        orig_logprobs_cache[spec["paragraph"]] = _all_token_logprobs(states, ids)
    a = measure_a(states, ids, spec["i"], spec["j"], protocol.W_true,
                  orig_token_logprobs=orig_logprobs_cache[spec["paragraph"]], cached_surprisal=spec.get("surprisal"))
    seed = protocol.seed_for(spec["cut_index"])
    b = measure_b(provider, ids, spec["i"], spec["j"], protocol, seed=seed, bos_id=states.bos_id(),
                  kv_cache=kv_cache, max_length=max_length)
    row = {
        "kind": "cut", "key": spec["key"], "cut_index": int(spec["cut_index"]), "paragraph": spec["paragraph"],
        "i": int(spec["i"]), "j": int(spec["j"]), "pair": spec["pair"], "stratum": spec["stratum"],
        "seg_len": int(spec["seg_len"]), "state_div": float(spec["state_div"]), "grammatical": bool(spec["grammatical"]),
        "seed": seed, "n_tokens": len(ids),
    }
    row.update(a)
    row.update(b)
    row["seconds"] = round(time.perf_counter() - started, 3)
    return row


# ------------------------------------------------------------------ phase A


def preregistration_9b(
    protocol: GenerationProtocol,
    inputs: Inputs9a,
    *,
    provenance: Mapping[str, Any] | None = None,
    kv_cache: bool = True,
    bos_id: int | None = None,
    eos_id: int | None = None,
    pad_id: int | None = None,
) -> dict[str, Any]:
    """The cache header: the protocol, the identity of both 9a inputs, and the model."""
    return {
        "kind": "header", "schema": SCHEMA, "task": TASK,
        "protocol": {**protocol.as_header(), "kv_cache": bool(kv_cache)},
        "source_record": dict(inputs.source_record), "source_cache": dict(inputs.source_cache),
        "n_cuts_expected": len(inputs.record["cuts"]),
        "bos_id": bos_id, "eos_id": eos_id, "pad_id": pad_id,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        **(dict(provenance) if provenance else {}),
    }


def compute_generation_9b(
    states,
    provider,
    record_9a_path: str | os.PathLike,
    cache_9a_path: str | os.PathLike,
    *,
    cache_path: str | os.PathLike | None = None,
    protocol: GenerationProtocol = GenerationProtocol(),
    provenance: Mapping[str, Any] | None = None,
    kv_cache: bool = True,
    show_progress: bool = True,
    mirror_path: str | os.PathLike | None = None,
    mirror_every: int = 10,
    limit: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measures A and B for every labelled cut of the 9a record; cache and resume.

    Returns ``(header, rows)`` in record order. The header is written before
    any generation; on resume it must match (the differing field is named --
    a different protocol, or a different 9a record or cache by sha256, refuses)
    and finished cuts are skipped. `limit` scores only the first `limit`
    record cuts (a dry run). Neither 9a file is ever opened for writing.
    """
    if states.bos_id() is None:
        raise ValueError("Task 9b needs bos_policy='auto' (token 0 scored, as in the 9a cache), not 'none'.")
    inputs = load_9a_inputs(record_9a_path, cache_9a_path)
    specs = cut_specs(inputs.record, inputs.rows)
    tokenizer = provider.tokenizer
    header = preregistration_9b(
        protocol, inputs, provenance=provenance, kv_cache=kv_cache, bos_id=states.bos_id(),
        eos_id=getattr(tokenizer, "eos_token_id", None), pad_id=getattr(tokenizer, "pad_token_id", None),
    )
    max_length = provider.effective_max_length(states.config.max_length, states.config.max_length_cap)

    done: dict[str, dict] = {}
    path = Path(cache_path) if cache_path else None
    if path is not None and path.exists():
        existing, rows, truncated = read_jsonl(path)
        if existing is not None:
            check_header(existing, header, path, unchecked=_UNCHECKED_HEADER_KEYS)
            header = existing
        for row in rows:
            done[row["key"]] = row
        if truncated:
            rewrite(path, header, rows)
        log.info("Resuming: %d of %d cuts already generated in %s.", len(done), len(specs), path)
    elif path is not None:
        write_header(path, header)

    wanted = specs if limit is None else specs[: int(limit)]
    pending = [s for s in wanted if s["key"] not in done]
    orig_cache: dict[str, np.ndarray] = {}
    fh = path.open("a", encoding="utf-8") if path is not None else None
    since_mirror = 0
    try:
        for spec in maybe_progress(pending, show_progress, desc="cuts"):
            row = score_cut(states, provider, spec, protocol, orig_logprobs_cache=orig_cache,
                            kv_cache=kv_cache, max_length=max_length)
            done[spec["key"]] = row
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
    return header, [done[s["key"]] for s in specs if s["key"] in done]


def load_generation_9b(path: str | os.PathLike) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a generation cache back: ``(header, rows)``."""
    header, rows, _ = read_jsonl(Path(path))
    if header is None or header.get("task") != TASK:
        raise ValueError(f"{path} has no Task 9b header line; it is not a generation cache.")
    return header, rows


# --------------------------------------------------------------- invariants


def check_invariants_9b(rows: Sequence[Mapping[str, Any]], record_9a: Mapping[str, Any], *, tol: float = 1e-5) -> dict[str, Any]:
    """The three invariants to report, as numbers.

    (1) the cached 9a ``surprisal`` reproduces ``lp_orig`` (max abs difference
    over every cut, and how many cuts are within `tol`); (2) the cut keys are
    exactly the 9a record's, in order; (3) ``within > 0`` everywhere (the
    collapsed cuts are listed; measure B is undefined there). ``ok`` covers
    (1) and (2) plus ``state_div`` matching the record; a collapsed cut is
    reported, not a failure.
    """
    expected = [cut_key(c["paragraph"], int(c["i"]), int(c["j"])) for c in record_9a["cuts"]]
    got = [r["key"] for r in rows]
    missing = [k for k in expected if k not in set(got)]
    extra = [k for k in got if k not in set(expected)]
    keys_match = not missing and not extra and len(got) == len(set(got))
    diffs = [r["cached_lp_orig_max_abs_diff"] for r in rows if r.get("cached_lp_orig_max_abs_diff") is not None]
    max_diff = float(max(diffs)) if diffs else None
    div_of = {cut_key(c["paragraph"], int(c["i"]), int(c["j"])): float(c["divergence"]) for c in record_9a["cuts"]}
    div_ok = all(r["key"] in div_of and abs(float(r["state_div"]) - div_of[r["key"]]) <= 1e-9 for r in rows)
    collapsed = [r["key"] for r in rows if not (float(r["within"]) > 0)]
    lp_ok = max_diff is None or max_diff <= tol
    return {
        "n_expected": len(expected), "n_rows": len(rows),
        "keys_match": bool(keys_match), "order_ok": bool(got == expected), "missing": missing, "extra": extra,
        "max_abs_cached_lp_diff": max_diff, "n_cuts_within_tol": int(sum(1 for d in diffs if d <= tol)),
        "n_without_cached_surprisal": int(len(rows) - len(diffs)), "tol": tol, "lp_orig_ok": bool(lp_ok),
        "n_collapsed": len(collapsed), "collapsed": collapsed,
        "n_with_eos_orig": int(sum(1 for r in rows if r["n_eos_orig"])),
        "n_with_eos_spliced": int(sum(1 for r in rows if r["n_eos_spliced"])),
        "state_div_matches": bool(div_ok),
        "ok": bool(keys_match and got == expected and lp_ok and div_ok),
    }


# ------------------------------------------------------------------ record


def build_record_9b(
    header: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    record_9a: Mapping[str, Any],
    *,
    model: str | None = None,
    revision: str | None = None,
    dtype: str | None = None,
    notes: str = "",
    tol: float = 1e-5,
) -> dict[str, Any]:
    """The Task 9b record: the requested fields per cut, the protocol, the meta, the invariants."""
    invariants = check_invariants_9b(rows, record_9a, tol=tol)
    cuts = []
    for r in rows:
        cut = {name: r[name] for name in REQUESTED_CUT_FIELDS}
        cut.update({name: r.get(name) for name in ADDITIVE_CUT_FIELDS})
        cuts.append(cut)
    protocol = {k: v for k, v in header["protocol"].items() if k != "kv_cache"}
    protocol["bos_policy"] = "auto"
    note_parts = [
        f"n_cuts={len(cuts)}/{invariants['n_expected']}",
        "keys match the 9a record" if invariants["keys_match"] and invariants["order_ok"] else "KEYS DO NOT MATCH THE 9A RECORD",
        (f"max |lp_orig + cached surprisal| = {invariants['max_abs_cached_lp_diff']:.2e}"
         if invariants["max_abs_cached_lp_diff"] is not None else "cached surprisal unavailable"),
        f"{invariants['n_collapsed']} collapsed sample set(s)",
        f"source record sha256 {str(header.get('source_record', {}).get('sha256'))[:12]}",
    ]
    if notes:
        note_parts.append(notes)
    record = {
        "cuts": cuts,
        "protocol": protocol,
        "meta": {
            "model": model or header.get("model_id") or "unknown",
            "revision": revision if revision is not None else header.get("revision"),
            "dtype": dtype if dtype is not None else header.get("dtype"),
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "notes": "; ".join(note_parts),
        },
        "invariants": invariants,
        "source": {"record_9a": dict(header.get("source_record", {})), "cache_9a": dict(header.get("source_cache", {}))},
    }
    validate_record_9b(record)
    return record


def validate_record_9b(record: Mapping[str, Any]) -> None:
    """Fail loudly if the record does not have the shape the request specifies."""
    for key in ("cuts", "protocol", "meta"):
        if key not in record:
            raise ValueError(f"record is missing {key!r}")
    protocol = record["protocol"]
    for key in ("W_true", "K", "L", "sampling", "overlap"):
        if key not in protocol:
            raise ValueError(f"protocol is missing {key!r}")
    for key in ("model", "revision", "dtype", "date", "notes"):
        if key not in record["meta"]:
            raise ValueError(f"meta is missing {key!r}")
    W, K, L = int(protocol["W_true"]), int(protocol["K"]), int(protocol["L"])
    cap = int(protocol.get("greedy_cap", L))
    for n, c in enumerate(record["cuts"]):
        for name in REQUESTED_CUT_FIELDS:
            if name not in c:
                raise ValueError(f"cut {n} is missing {name!r}")
        if c["pair"] not in (CLOSE, FAR):
            raise ValueError(f"cut {n}: pair {c['pair']!r} is not close/far")
        if len(c["true_dlogp_k"]) != W:
            raise ValueError(f"cut {n}: true_dlogp_k has {len(c['true_dlogp_k'])} values, W_true is {W}")
        for name in ("samples_orig", "samples_spliced"):
            if len(c[name]) != K or any(len(s) != L for s in c[name]):
                raise ValueError(f"cut {n}: {name} is not {K} samples of {L} ids")
        if not (float(c["within"]) >= 0):
            raise ValueError(f"cut {n}: within is {c['within']}")
        if (c["sample_overlap"] is None) != (float(c["within"]) == 0):
            raise ValueError(f"cut {n}: sample_overlap must be null exactly when within == 0")
        if not 0 <= int(c["first_diff_greedy"]) <= cap:
            raise ValueError(f"cut {n}: first_diff_greedy {c['first_diff_greedy']} is outside 0..{cap}")


__all__ = [
    "TASK", "SCHEMA", "OVERLAP_MEASURE", "EOS_POLICY", "REQUESTED_CUT_FIELDS", "ADDITIVE_CUT_FIELDS",
    "GenerationProtocol", "Inputs9a", "load_9a_inputs", "cut_specs", "measure_a", "measure_b", "score_cut",
    "preregistration_9b", "compute_generation_9b", "load_generation_9b", "check_invariants_9b",
    "build_record_9b", "validate_record_9b",
]
