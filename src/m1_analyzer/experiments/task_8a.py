"""Task 8a: the clause-return test at scale, across models.

For every frame (`frames_8a`), fourteen prefixes are scored: the bare subject
``REF``, one embedded pass per clause type, a token-matched control per
embedded pass, and the legacy control ``ORC_A``. At each prefix the model's
**state** is ``log_softmax(logits[-1])`` over its vocabulary, computed in
float32 whatever the weight dtype (`NextTokenStateService` does the cast).
Per (embedded E, control C) pair::

    ret  = || s_E - s_REF ||_2        the embedded state's distance back to the bare subject
    ctrl = || s_C - s_REF ||_2        the control's, same token count, same last token

Ratios, intervals and tests are computed outside this repository from the raw
rows; this module writes ``ret`` and ``ctrl`` (nothing normalised), the
**verb mass** at every stop (``sum exp(s[id(" w")])`` over the fixed verb pool,
restricted per model to the words whose leading-space form is one token), the
log-probability of the frame's main verb ``V`` at every stop, and the ten
most probable next tokens after every pass.

Tokenisation is the tokenizer's own: ``tok(prefix)`` with its default
``add_special_tokens`` (Llama and Gemma prepend a BOS, Qwen and GPT-2 do not),
the state read at the last position. That is exactly what the frame filter
counted, and what the local run did (the ``anchor`` check reproduces its
Qwen2.5-0.5B numbers to 2 % before anything else is scored). The state
service is therefore used with ``bos_policy="none"`` -- it must not add a
second BOS -- and refuses otherwise.

Phase 0 (tokenizers only) generates the frames once; phase A
(`score_model_8a`) scores one model per run into a resumable JSONL cache
(one line per frame, all fourteen passes); phase B (`build_record_8a`,
CPU) merges the frames, every model cache present and the anchor block into
``record_8a_multimodel.json`` in the requested schema, and
`validate_record_8a` re-checks every token assertion and count.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..utils.batching import maybe_progress
from ..utils.logging import get_logger
from .frames_8a import (
    CODES, PAIRS, REF, SLOTS, VERB_POOL_8A, frame_prefixes, frames_sha256, last_tok, load_frames_8a, n_tok,
    single, split_sentence, token_assertions, verb_pool_kept,
)
from .jsonl_cache import append_row, check_header, mirror, read_jsonl, rewrite, write_header

log = get_logger("task_8a")

TASK = "8a"
SCHEMA = 1
TOP_K = 10
DISTANCE_MEASURE = (
    "L2 norm of the difference of two float32 log_softmax next-token vectors, each read at the last "
    "token of its prefix; ret = ||s_E - s_REF||, ctrl = ||s_C - s_REF||; nothing normalised"
)
VERBMASS_MEASURE = "sum of exp(logp) over the ids of ' w' for every verb-pool word w that is a single token"
LOGP_V_MEASURE = "log_softmax value at the id of ' V' (the frame's main verb), single token by the pool filter"
TOP10_MEASURE = "the ten highest-probability next tokens after the prefix, as [decoded string, probability]"
BOS_POLICY = "tokenizer default: tok(prefix) with its own add_special_tokens; the state service adds nothing"
#: Header fields that may differ between the writer and a resumer.
_UNCHECKED_HEADER_KEYS = frozenset({"kind", "date", "device", "device_info", "library_versions", "n_params_b"})


# ------------------------------------------------------------------ models


@dataclass(frozen=True)
class ModelSpec:
    """One row of the model table: the key the record uses and how to load it."""

    key: str
    hf_id: str
    role: str = ""
    revision: str | None = None
    trust_remote_code: bool = False
    gated: bool = False
    base: bool = True           #: False for a post-trained checkpoint (reported, outside the pass rule)
    notes: str = ""

    def with_overrides(self, **changes: Any) -> "ModelSpec":
        """A copy with e.g. a different ``hf_id`` or ``revision`` (the notebook's overrides)."""
        return dataclasses.replace(self, **changes)


#: The seven models, base checkpoints where a base release exists. The ids for
#: gpt-oss and Gemma 4 are best guesses to be confirmed against the Hub; the
#: notebook lets every id and revision be overridden, and the record stores what
#: was actually loaded (``hf_id``, resolved ``revision``, ``weight_dtype``).
MODELS_8A: dict[str, ModelSpec] = {
    "qwen25_0.5b": ModelSpec("qwen25_0.5b", "Qwen/Qwen2.5-0.5B", role="anchor: must reproduce the local record first"),
    "qwen3_0.6b": ModelSpec("qwen3_0.6b", "Qwen/Qwen3-0.6B-Base", role="reference model of the 9-series and 1b"),
    "qwen3_1.7b": ModelSpec("qwen3_1.7b", "Qwen/Qwen3-1.7B-Base", role="size ladder"),
    "qwen3_8b": ModelSpec("qwen3_8b", "Qwen/Qwen3-8B-Base", role="size ladder"),
    "llama31_8b": ModelSpec("llama31_8b", "meta-llama/Llama-3.1-8B", role="same size, other family", gated=True),
    "gptoss_20b": ModelSpec(
        "gptoss_20b", "openai/gpt-oss-20b", role="post-trained reasoning MoE, no base release", base=False,
        notes="no base checkpoint exists; reported with that caveat and outside the pass rule. Weights ship in MXFP4 "
              "and are dequantised to bf16 on GPUs without native support (~42 GB).",
    ),
    "gemma4_31b": ModelSpec(
        "gemma4_31b", "google/gemma-4-31b", role="largest", gated=True,
        notes="hf_id is a best guess: confirm the base (pre-trained) checkpoint's id on the Hub before loading. "
              "Multimodal wrapper: loaded in text-only mode (ModelService reads text_config). ~62 GB in bf16.",
    ),
}
ANCHOR_KEY = "qwen25_0.5b"
#: The models a free Colab GPU can hold; the notebook's ``all-small`` choice.
SMALL_KEYS: tuple[str, ...] = ("qwen25_0.5b", "qwen3_0.6b", "qwen3_1.7b")


def load_tokenizers(
    keys: Sequence[str],
    *,
    hf_ids: Mapping[str, str] | None = None,
    token: str | None = None,
    cache_dir: str | None = None,
    trust_remote_code: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Phase 0: the tokenizers alone (a few MB each), keyed like `MODELS_8A`.

    This is the one place outside `ModelService` that touches the Hub, because
    the frame filter needs every tokenizer before any model is loaded.
    """
    from transformers import AutoTokenizer

    out = {}
    for key in keys:
        spec = MODELS_8A.get(key)
        hf_id = (hf_ids or {}).get(key) or (spec.hf_id if spec else key)
        kwargs: dict[str, Any] = {"cache_dir": cache_dir}
        if spec is not None:
            kwargs["revision"] = spec.revision
            kwargs["trust_remote_code"] = spec.trust_remote_code
        if trust_remote_code and key in trust_remote_code:
            kwargs["trust_remote_code"] = bool(trust_remote_code[key])
        if token:
            kwargs["token"] = token
        log.info("Loading tokenizer %s (%s)", key, hf_id)
        out[key] = AutoTokenizer.from_pretrained(hf_id, **kwargs)
    return out


# ---------------------------------------------------------------- tokens


def prefix_ids(tokenizer: Any, prefix: str) -> list[int]:
    """The ids the model sees: the tokenizer's default special tokens, nothing else."""
    return [int(t) for t in tokenizer(prefix).input_ids]


def bos_added(tokenizer: Any) -> bool:
    """Does ``tok(text)`` prepend a token that ``add_special_tokens=False`` does not?"""
    probe = "The soldier"
    return len(tokenizer(probe).input_ids) > len(tokenizer(probe, add_special_tokens=False).input_ids)


def word_id(tokenizer: Any, word: str) -> int:
    """The single id of ``" word"``; raises if the word is not one token here."""
    ids = list(tokenizer(" " + word, add_special_tokens=False).input_ids)
    if len(ids) != 1:
        raise ValueError(f"{word!r} is {len(ids)} tokens with a leading space in this tokenizer, not 1.")
    return int(ids[0])


def verb_ids(tokenizer: Any, pool: Sequence[str] = VERB_POOL_8A) -> tuple[list[str], list[int]]:
    """``(kept words, their ids)`` for the verb-mass readout under this tokenizer."""
    kept = verb_pool_kept(tokenizer, pool)
    return kept, [word_id(tokenizer, w) for w in kept]


def model_meta_8a(analyzer: Any, spec: ModelSpec) -> dict[str, Any]:
    """The record's per-model ``meta`` block, from the loaded model."""
    meta = analyzer.models.metadata()
    model = analyzer.models.model
    n_params = sum(int(p.numel()) for p in model.parameters())
    return {
        "hf_id": spec.hf_id,
        "revision": meta.get("revision"),
        "weight_dtype": meta.get("dtype"),
        "logits_dtype": "float32",
        "bos_added": bos_added(analyzer.models.tokenizer),
        "n_params_b": round(n_params / 1e9, 3),
        "vocab_size": int(analyzer.states.vocab_size),
    }


# --------------------------------------------------------------- scalars


def _decode_one(tokenizer: Any, index: int) -> str:
    try:
        return tokenizer.decode([int(index)])
    except Exception:  # noqa: BLE001 - an id past the tokenizer's vocabulary (padded LM head)
        return f"<id {int(index)}>"


def top_k_tokens(state: np.ndarray, tokenizer: Any, k: int = TOP_K) -> list[list[Any]]:
    """The `k` most probable next tokens of one state as ``[[decoded, prob], ...]``, best first."""
    s = np.asarray(state, dtype=np.float64).ravel()
    k = min(int(k), s.shape[0])
    top = np.argpartition(-s, k - 1)[:k]
    top = top[np.argsort(-s[top], kind="stable")]
    return [[_decode_one(tokenizer, i), float(np.exp(s[i]))] for i in top]


def frame_scalars(
    states_by_code: Mapping[str, np.ndarray],
    *,
    verb_token_ids: Sequence[int],
    v_id: int,
    tokenizer: Any,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """Everything the record needs from one frame's fourteen states; the vectors are then dropped.

    ``dist_to_ref[code]`` for every code but ``REF``; ``verbmass``, ``logp_V`` and
    ``top10`` for every code. Floats are kept at full precision: the anchor
    tolerances are tight and the user's statistics are computed from these.
    """
    missing = [code for code in CODES if code not in states_by_code]
    if missing:
        raise ValueError(f"states missing for {missing}")
    ref = np.asarray(states_by_code[REF], dtype=np.float32).ravel()
    ids = np.asarray(list(verb_token_ids), dtype=np.int64)
    out: dict[str, Any] = {"dist_to_ref": {}, "verbmass": {}, "logp_V": {}, "top10": {}}
    for code in CODES:
        s = np.asarray(states_by_code[code], dtype=np.float32).ravel()
        if s.shape != ref.shape:
            raise ValueError(f"{code}: state has shape {s.shape}, REF has {ref.shape}")
        if code != REF:
            out["dist_to_ref"][code] = float(np.linalg.norm(s - ref))
        out["verbmass"][code] = float(np.exp(s[ids].astype(np.float64)).sum()) if ids.size else 0.0
        out["logp_V"][code] = float(s[int(v_id)])
        out["top10"][code] = top_k_tokens(s, tokenizer, top_k)
    return out


def score_frames_batch(
    states: Any,
    tokenizer: Any,
    frames: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    verb_token_ids: Sequence[int],
    batch_size: int | None = None,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    """Score ``[(frame_index, frame), ...]``: one forward batch, one cache row per frame.

    Every prefix is ``tok(prefix)`` under the tokenizer's default special tokens
    and its state is read at its last token. The row records the token count
    and last token of every pass so the assertions can be re-checked later;
    they are also checked here, so a tokenizer that drifted since the frames
    were filtered fails loudly instead of scoring a mismatched pair.
    """
    if states.bos_id() is not None:
        raise ValueError(
            "Task 8a scores tok(prefix) as the tokenizer returns it; the state service must not prepend a BOS. "
            "Use ScoringConfig(bos_policy='none')."
        )
    if not frames:
        return []
    started = time.perf_counter()
    seqs: list[list[int]] = []
    owners: list[tuple[int, str]] = []
    per_frame: dict[int, dict[str, Any]] = {}
    for index, frame in frames:
        prefixes = frame_prefixes(frame)
        counts = {code: n_tok(tokenizer, p) for code, p in prefixes.items()}
        lasts = {code: last_tok(tokenizer, p) for code, p in prefixes.items()}
        failures = token_assertions(counts, lasts)
        if failures:
            raise ValueError(f"frame {index} fails the token assertions under this tokenizer: {failures}")
        per_frame[index] = {"n_tok": counts, "last_tok": lasts, "v_id": word_id(tokenizer, frame["V"])}
        for code in CODES:
            ids = prefix_ids(tokenizer, prefixes[code])
            if len(ids) != counts[code]:
                raise AssertionError(f"frame {index} {code}: prefix_ids gave {len(ids)} ids, n_tok {counts[code]}")
            seqs.append(ids)
            owners.append((index, code))
    positions = [[len(ids) - 1] for ids in seqs]
    results = states.states(seqs, positions, batch_size=batch_size or len(seqs))
    by_frame: dict[int, dict[str, np.ndarray]] = {index: {} for index, _ in frames}
    for (index, code), result in zip(owners, results):
        by_frame[index][code] = result.states[0]
    seconds = (time.perf_counter() - started) / len(frames)
    rows = []
    for index, _frame in frames:
        scal = frame_scalars(by_frame[index], verb_token_ids=verb_token_ids, v_id=per_frame[index]["v_id"],
                             tokenizer=tokenizer, top_k=top_k)
        rows.append({
            "kind": "frame", "frame": int(index),
            "n_tok": per_frame[index]["n_tok"], "last_tok": per_frame[index]["last_tok"],
            "v_id": per_frame[index]["v_id"], **scal, "seconds": round(seconds, 4),
        })
    by_frame.clear()
    return rows


# ------------------------------------------------------------------ phase A


def preregistration_8a(
    spec: ModelSpec,
    meta: Mapping[str, Any],
    *,
    frames_path: str | os.PathLike,
    n_frames: int,
    seed: int,
    verb_kept: Sequence[str],
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The cache header: the model's identity, the frames file's, and every measure definition."""
    return {
        "kind": "header", "schema": SCHEMA, "task": TASK,
        "model_key": spec.key, "hf_id": meta["hf_id"], "revision": meta.get("revision"),
        "weight_dtype": meta.get("weight_dtype"), "logits_dtype": "float32",
        "bos_added": bool(meta.get("bos_added")), "vocab_size": int(meta["vocab_size"]),
        "n_params_b": meta.get("n_params_b"), "base": spec.base, "role": spec.role, "notes": spec.notes,
        "frames_file": Path(frames_path).name, "frames_sha256": frames_sha256(frames_path),
        "n_frames": int(n_frames), "seed": int(seed),
        "verb_pool": list(VERB_POOL_8A), "verb_pool_kept": list(verb_kept),
        "distance_measure": DISTANCE_MEASURE, "verbmass_measure": VERBMASS_MEASURE,
        "logp_V_measure": LOGP_V_MEASURE, "top10_measure": TOP10_MEASURE, "top_k": TOP_K,
        "bos_policy": BOS_POLICY,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        **(dict(provenance) if provenance else {}),
    }


def score_model_8a(
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
    """Phase A for one model: every pass of every frame, cached and resumable.

    Returns ``(header, rows)`` in frame order. The header is written before any
    forward pass; on resume it must match (a different model, revision, dtype
    or frames file -- by sha256 -- refuses, naming the field) and finished
    frames are skipped. `limit` scores only the first `limit` frames (a dry run).
    """
    if frames_per_batch < 1:
        raise ValueError("frames_per_batch must be >= 1.")
    states = analyzer.states
    if states is None:
        raise RuntimeError("Task 8a needs the LM head: build the Analyzer with head='causal_lm'.")
    if states.config.bos_policy != "none":
        raise ValueError("Task 8a needs ScoringConfig(bos_policy='none'): the prefix ids already carry the tokenizer's BOS.")
    tokenizer = analyzer.models.tokenizer
    payload = load_frames_8a(frames_path)
    frames = payload["frames"]
    kept, ids = verb_ids(tokenizer)
    meta = model_meta_8a(analyzer, spec)
    header = preregistration_8a(spec, meta, frames_path=frames_path, n_frames=len(frames),
                                seed=int(payload["meta"].get("seed", 0)), verb_kept=kept, provenance=provenance)

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
            rows = score_frames_batch(states, tokenizer, chunk, verb_token_ids=ids,
                                      batch_size=len(chunk) * len(CODES))
            for row in rows:
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


def load_scores_8a(path: str | os.PathLike) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a model's score cache back: ``(header, rows)`` sorted by frame."""
    header, rows, _ = read_jsonl(Path(path))
    if header is None or header.get("task") != TASK:
        raise ValueError(f"{path} has no Task 8a header line; it is not a score cache.")
    return header, sorted(rows, key=lambda r: int(r["frame"]))


# ------------------------------------------------------------------- rows


def rows_from_frame(row: Mapping[str, Any], pairs: Sequence[tuple[str, str]] = tuple(PAIRS)) -> list[dict[str, Any]]:
    """The record's seven rows for one cached frame, in `PAIRS` order."""
    out = []
    for e, c in pairs:
        out.append({
            "frame": int(row["frame"]), "structure": e, "control": c,
            "ret": float(row["dist_to_ref"][e]), "ctrl": float(row["dist_to_ref"][c]),
            "verbmass_ref": float(row["verbmass"][REF]), "verbmass_emb": float(row["verbmass"][e]),
            "verbmass_ctrl": float(row["verbmass"][c]),
            "logp_V_ref": float(row["logp_V"][REF]), "logp_V_emb": float(row["logp_V"][e]),
            "logp_V_ctrl": float(row["logp_V"][c]),
        })
    return out


def model_block(header: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One entry of the record's ``models``: meta, kept pool, token table, rows, top-10s."""
    tokens, record_rows, top10 = [], [], {}
    for row in rows:
        index = int(row["frame"])
        for code in CODES:
            tokens.append({"frame": index, "code": code, "n_tok": int(row["n_tok"][code]),
                           "last_tok": row["last_tok"][code]})
            top10[f"{index}:{code}"] = [[str(t), float(p)] for t, p in row["top10"][code]]
        record_rows.extend(rows_from_frame(row))
    return {
        "meta": {
            "hf_id": header["hf_id"], "revision": header.get("revision"), "weight_dtype": header.get("weight_dtype"),
            "logits_dtype": header.get("logits_dtype", "float32"), "bos_added": bool(header.get("bos_added")),
            "n_params_b": header.get("n_params_b"), "vocab_size": header.get("vocab_size"),
        },
        "verb_pool_kept": list(header.get("verb_pool_kept", [])),
        "tokens": tokens,
        "rows": record_rows,
        "top10": top10,
    }


# ----------------------------------------------------------------- anchor


@dataclass
class AnchorSet:
    """The local run's 30 frames and its Qwen2.5-0.5B rows, read from ``frames_8a_local30.json``."""

    frames: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    path: str
    sha256: str
    meta: dict[str, Any] = field(default_factory=dict)


_ANCHOR_ROW_KEYS = ("frame", "structure", "control", "ret", "ctrl", "verbmass_ref", "verbmass_emb", "verbmass_ctrl")


def load_anchor_file(path: str | os.PathLike) -> AnchorSet:
    """Read the anchor file (never written): frames with formatted sentences, reference rows."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    rows = payload.get("qwen25_reference_rows", payload.get("rows"))
    if not frames or not rows:
        raise ValueError(f"{path}: expected 'frames' and 'qwen25_reference_rows' (keys: {sorted(payload)}).")
    for index, frame in enumerate(frames):
        missing = [c for c in CODES if c not in frame.get("sentences", {})]
        if missing or any(s not in frame for s in SLOTS):
            raise ValueError(f"{path}: frame {index} lacks {missing or 'a slot'}; keys: {sorted(frame)}")
    for k, row in enumerate(rows):
        missing = [key for key in _ANCHOR_ROW_KEYS if key not in row]
        if missing:
            raise ValueError(f"{path}: row {k} lacks {missing}; keys: {sorted(row)}")
        if (row["structure"], row["control"]) not in PAIRS:
            raise ValueError(f"{path}: row {k} pair ({row['structure']}, {row['control']}) is not in PAIRS")
        if not 0 <= int(row["frame"]) < len(frames):
            raise ValueError(f"{path}: row {k} names frame {row['frame']}, out of range 0..{len(frames) - 1}")
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {k: v for k, v in payload.items() if k not in ("frames", "qwen25_reference_rows", "rows")}
    return AnchorSet(frames=list(frames), rows=list(rows), path=str(path), sha256=h, meta=meta)


def anchor_tokens_check(tokenizer: Any, anchor: AnchorSet) -> list[str]:
    """Before any forward pass: does our tokenisation reproduce the rows' ``n_tok`` / ``stop_tok``?

    Returns the mismatches as text (empty = every embedded prefix agrees). A BOS
    that the local run did not add shows up here as ``n_tok`` off by one on every
    row; a trailing space as a different ``stop_tok``.
    """
    problems = []
    for row in anchor.rows:
        frame = anchor.frames[int(row["frame"])]
        prefix = split_sentence(frame["sentences"][row["structure"]])[0]
        if "n_tok" in row and n_tok(tokenizer, prefix) != int(row["n_tok"]):
            problems.append(f"frame {row['frame']} {row['structure']}: n_tok {n_tok(tokenizer, prefix)} != {row['n_tok']}")
        if "stop_tok" in row and last_tok(tokenizer, prefix) != row["stop_tok"]:
            problems.append(f"frame {row['frame']} {row['structure']}: last_tok {last_tok(tokenizer, prefix)!r} != {row['stop_tok']!r}")
    return problems


def compare_anchor_rows(ours: Sequence[Mapping[str, Any]], theirs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Join our rows with the anchor's on ``(frame, structure, control)``; the maxima and the worst row."""
    mine = {(int(r["frame"]), r["structure"], r["control"]): r for r in ours}
    per_row, missing = [], []
    for row in theirs:
        key = (int(row["frame"]), row["structure"], row["control"])
        if key not in mine:
            missing.append(key)
            continue
        m = mine[key]
        rel = lambda a, b: abs(a - b) / abs(b) if b else (0.0 if a == b else float("inf"))  # noqa: E731
        entry = {
            "frame": key[0], "structure": key[1], "control": key[2],
            "ret": m["ret"], "ret_ref": float(row["ret"]), "rel_diff_ret": rel(m["ret"], float(row["ret"])),
            "ctrl": m["ctrl"], "ctrl_ref": float(row["ctrl"]), "rel_diff_ctrl": rel(m["ctrl"], float(row["ctrl"])),
        }
        vm = []
        for name in ("verbmass_ref", "verbmass_emb", "verbmass_ctrl"):
            entry[name] = m[name]
            entry[name + "_ref"] = float(row[name])
            vm.append(abs(m[name] - float(row[name])))
        entry["abs_diff_verbmass"] = max(vm)
        per_row.append(entry)
    if missing:
        raise ValueError(f"{len(missing)} anchor row(s) were not scored (first: {missing[0]}).")
    worst = max(per_row, key=lambda e: max(e["rel_diff_ret"], e["rel_diff_ctrl"]), default=None)
    return {
        "n_rows": len(per_row),
        "max_rel_diff_ret": max((e["rel_diff_ret"] for e in per_row), default=0.0),
        "max_rel_diff_ctrl": max((e["rel_diff_ctrl"] for e in per_row), default=0.0),
        "max_abs_diff_verbmass": max((e["abs_diff_verbmass"] for e in per_row), default=0.0),
        "worst": worst, "per_row": per_row,
    }


def anchor_check(
    analyzer: Any,
    spec: ModelSpec,
    anchor_path: str | os.PathLike,
    *,
    tol_rel: float = 0.02,
    tol_verbmass: float = 0.01,
    frames_per_batch: int = 8,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Score the anchor frames with this pipeline and compare with the local run's rows.

    ``pass`` needs every row within `tol_rel` on ``ret`` and ``ctrl`` and within
    `tol_verbmass` on all three verb masses, and no token mismatch. The block
    is written to ``anchor_8a.json`` in full; the record keeps its summary.
    """
    anchor = load_anchor_file(anchor_path)
    tokenizer = analyzer.models.tokenizer
    if analyzer.states.config.bos_policy != "none":
        raise ValueError("anchor_check needs ScoringConfig(bos_policy='none').")
    mismatches = anchor_tokens_check(tokenizer, anchor)
    kept, ids = verb_ids(tokenizer)
    meta = model_meta_8a(analyzer, spec)
    ours: list[dict[str, Any]] = []
    indexed = list(enumerate(anchor.frames))
    chunks = [indexed[k: k + frames_per_batch] for k in range(0, len(indexed), frames_per_batch)]
    for chunk in maybe_progress(chunks, show_progress, desc="anchor"):
        for row in score_frames_batch(analyzer.states, tokenizer, chunk, verb_token_ids=ids,
                                      batch_size=len(chunk) * len(CODES)):
            ours.extend(rows_from_frame(row))
    cmp = compare_anchor_rows(ours, anchor.rows)
    passed = (
        not mismatches and cmp["n_rows"] == len(anchor.rows)
        and cmp["max_rel_diff_ret"] < tol_rel and cmp["max_rel_diff_ctrl"] < tol_rel
        and cmp["max_abs_diff_verbmass"] < tol_verbmass
    )
    return {
        "model": spec.key, "frames_file": Path(anchor_path).name, "sha256": anchor.sha256,
        "max_rel_diff_ret": float(cmp["max_rel_diff_ret"]), "max_rel_diff_ctrl": float(cmp["max_rel_diff_ctrl"]),
        "max_abs_diff_verbmass": float(cmp["max_abs_diff_verbmass"]), "n_rows": int(cmp["n_rows"]),
        "pass": bool(passed), "tol_rel": tol_rel, "tol_verbmass": tol_verbmass,
        "token_mismatches": mismatches, "worst": cmp["worst"], "per_row": cmp["per_row"],
        "hf_id": meta["hf_id"], "revision": meta["revision"], "weight_dtype": meta["weight_dtype"],
        "bos_added": meta["bos_added"], "verb_pool_kept": kept, "n_frames": len(anchor.frames),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }


def anchor_summary(block: Mapping[str, Any]) -> dict[str, Any]:
    """The record's ``anchor`` block: the requested fields, plus the file's sha256."""
    return {
        "model": block["model"], "frames_file": block["frames_file"],
        "max_rel_diff_ret": block["max_rel_diff_ret"], "max_rel_diff_ctrl": block["max_rel_diff_ctrl"],
        "max_abs_diff_verbmass": block["max_abs_diff_verbmass"], "n_rows": block["n_rows"], "pass": block["pass"],
        "sha256": block.get("sha256"),
    }


def _variant_distance(analyzer: Any, prefix_e: str, prefix_ref: str, *, extra: Sequence[int] = (), suffix: str = "") -> float:
    """``||s_E - s_REF||`` with the same change applied to both prefixes (a diagnosis probe)."""
    tokenizer = analyzer.models.tokenizer
    seqs = [list(extra) + prefix_ids(tokenizer, prefix_e + suffix), list(extra) + prefix_ids(tokenizer, prefix_ref + suffix)]
    results = analyzer.states.states(seqs, [[len(s) - 1] for s in seqs], batch_size=2)
    return float(np.linalg.norm(results[0].states[0] - results[1].states[0]))


def anchor_diagnosis(block: Mapping[str, Any], *, analyzer: Any = None, anchor_path: str | os.PathLike | None = None) -> str:
    """Why the anchor failed, in words; with a model, which variant reproduces the local value.

    Re-scores the worst row's embedded prefix (a) with a trailing space and (b)
    with the tokenizer's BOS / EOS prepended, and reports whether either lands
    within the tolerance of the local ``ret``. The dtype and BOS facts of the
    run are stated in any case.
    """
    lines = []
    if block.get("pass"):
        return "anchor passed: the pipeline reproduces the local Qwen2.5-0.5B numbers."
    lines.append(
        f"anchor FAILED on {block['model']}: max rel diff ret {block['max_rel_diff_ret']:.4f}, "
        f"ctrl {block['max_rel_diff_ctrl']:.4f} (tol {block.get('tol_rel', 0.02)}), "
        f"max abs diff verbmass {block['max_abs_diff_verbmass']:.4f} (tol {block.get('tol_verbmass', 0.01)})."
    )
    if block.get("token_mismatches"):
        n = len(block["token_mismatches"])
        lines.append(f"{n} token mismatch(es) before any forward pass, e.g. {block['token_mismatches'][0]}: "
                     "our tokenisation differs from the local run's (a BOS added or missing, or a trailing space).")
    lines.append(f"this run: weight_dtype={block.get('weight_dtype')}, bos_added={block.get('bos_added')}, "
                 f"hf_id={block.get('hf_id')}, revision={block.get('revision')}.")
    worst = block.get("worst")
    if worst:
        lines.append(f"worst row: frame {worst['frame']} {worst['structure']}/{worst['control']}: "
                     f"ret {worst['ret']:.3f} vs local {worst['ret_ref']:.3f}, ctrl {worst['ctrl']:.3f} vs {worst['ctrl_ref']:.3f}.")
    if analyzer is not None and anchor_path is not None and worst:
        anchor = load_anchor_file(anchor_path)
        frame = anchor.frames[int(worst["frame"])]
        prefix_e = split_sentence(frame["sentences"][worst["structure"]])[0]
        prefix_ref = split_sentence(frame["sentences"][REF])[0]
        tol = float(block.get("tol_rel", 0.02))
        target = float(worst["ret_ref"])
        probes = {"trailing space on the prefix": dict(suffix=" ")}
        tokenizer = analyzer.models.tokenizer
        for attr, name in (("bos_token_id", "BOS"), ("eos_token_id", "EOS")):
            tid = getattr(tokenizer, attr, None)
            if tid is not None:
                probes[f"{name} prepended"] = dict(extra=[int(tid)])
                break
        for name, kwargs in probes.items():
            try:
                d = _variant_distance(analyzer, prefix_e, prefix_ref, **kwargs)
            except Exception as exc:  # noqa: BLE001 - a probe must never mask the diagnosis
                lines.append(f"probe '{name}' failed: {exc}")
                continue
            hit = abs(d - target) / target < tol if target else False
            lines.append(f"probe '{name}': ret {d:.3f} -> {'REPRODUCES the local value' if hit else 'does not reproduce it'}.")
    lines.append("most likely causes, in order: BOS handling, weight dtype, a trailing space in the prefix. "
                 "Do not proceed to the 200 frames until the anchor passes.")
    return "\n".join(lines)


# ------------------------------------------------------------------ record


def missing_models(caches: Mapping[str, Any], keys: Sequence[str] = tuple(MODELS_8A)) -> list[str]:
    """The model keys with no cache file on disk."""
    return [k for k in keys if k not in caches or not Path(caches[k]).exists()]


def build_record_8a(
    frames_path: str | os.PathLike,
    caches: Mapping[str, str | os.PathLike],
    *,
    anchor: Mapping[str, Any] | None = None,
    allow_partial: bool = False,
    keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Phase B: the frames, every model cache present, and the anchor, in the requested schema.

    A cache is accepted only if its header names this frames file (by sha256)
    and it holds every frame; otherwise the build raises, or under
    `allow_partial` skips that model with a warning. `keys` defaults to the
    seven `MODELS_8A` keys followed by any other key in `caches` (a smoke
    model); the registry models with no cache are warned about, and the
    record lists what is there, in that order.
    """
    payload = load_frames_8a(frames_path)
    sha = frames_sha256(frames_path)
    n = len(payload["frames"])
    if keys is None:
        keys = list(MODELS_8A) + [k for k in caches if k not in MODELS_8A]
    models: dict[str, Any] = {}
    for key in keys:
        if key not in caches or not Path(caches[key]).exists():
            log.warning("no cache for %s; it will be missing from the record.", key)
            continue
        header, rows = load_scores_8a(caches[key])
        problems = []
        if header.get("model_key") != key:
            problems.append(f"cache is for model_key {header.get('model_key')!r}, not {key!r}")
        if header.get("frames_sha256") != sha:
            problems.append("cache was scored on a different frames file (sha256 differs)")
        frames_seen = sorted({int(r["frame"]) for r in rows})
        if frames_seen != list(range(n)):
            problems.append(f"cache holds {len(frames_seen)} of {n} frames")
        if problems:
            message = f"{caches[key]}: " + "; ".join(problems)
            if not allow_partial:
                raise ValueError(message)
            log.warning("%s -- skipped.", message)
            continue
        models[key] = model_block(header, rows)
    record = {
        "meta": {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), **payload["meta"]},
        "verb_pool": list(payload["verb_pool"]),
        "frames": payload["frames"],
        "models": models,
        "anchor": anchor_summary(anchor) if anchor else None,
    }
    validate_record_8a(record)
    return record


def validate_record_8a(record: Mapping[str, Any]) -> None:
    """Fail loudly if the record does not have the requested shape, or a number is impossible."""
    for key in ("meta", "verb_pool", "frames", "models", "anchor"):
        if key not in record:
            raise ValueError(f"record is missing {key!r}")
    meta = record["meta"]
    for key in ("date", "seed", "n_frames_requested", "n_frames", "pools", "tokenizers_filtered_on", "generator"):
        if key not in meta:
            raise ValueError(f"meta is missing {key!r}")
    for key in ("pool_nouns", "pool_verbs", "pool_mains", "pool_cvs", "draws"):
        if key not in meta["pools"]:
            raise ValueError(f"meta.pools is missing {key!r}")
    frames = record["frames"]
    n = len(frames)
    if int(meta["n_frames"]) != n:
        raise ValueError(f"meta.n_frames is {meta['n_frames']} but there are {n} frames")
    seen = set()
    for index, frame in enumerate(frames):
        for slot in SLOTS:
            if slot not in frame:
                raise ValueError(f"frame {index} lacks slot {slot!r}")
        pair = (frame["N1"], frame["V"])
        if pair in seen:
            raise ValueError(f"frame {index} repeats the (N1, V) pair {pair}")
        seen.add(pair)
        sentences = frame.get("sentences", {})
        for code in CODES:
            if code not in sentences:
                raise ValueError(f"frame {index} lacks sentence {code!r}")
            prefix, suffix = split_sentence(sentences[code])
            if prefix.endswith(" ") or not prefix:
                raise ValueError(f"frame {index} {code}: prefix {prefix!r} is empty or ends with a space")
    if record["anchor"] is not None:
        for key in ("model", "frames_file", "max_rel_diff_ret", "max_rel_diff_ctrl", "max_abs_diff_verbmass", "n_rows", "pass"):
            if key not in record["anchor"]:
                raise ValueError(f"anchor block is missing {key!r}")
    for key, block in record["models"].items():
        for name in ("meta", "verb_pool_kept", "tokens", "rows", "top10"):
            if name not in block:
                raise ValueError(f"models.{key} is missing {name!r}")
        for name in ("hf_id", "revision", "weight_dtype", "logits_dtype", "bos_added", "n_params_b", "vocab_size"):
            if name not in block["meta"]:
                raise ValueError(f"models.{key}.meta is missing {name!r}")
        if block["meta"]["logits_dtype"] != "float32":
            raise ValueError(f"models.{key}: logits_dtype is {block['meta']['logits_dtype']!r}, not float32")
        extra = set(block["verb_pool_kept"]) - set(record["verb_pool"])
        if extra:
            raise ValueError(f"models.{key}: verb_pool_kept has words outside verb_pool: {sorted(extra)}")
        if len(block["tokens"]) != len(CODES) * n:
            raise ValueError(f"models.{key}: {len(block['tokens'])} token entries, expected {len(CODES) * n}")
        if len(block["rows"]) != len(PAIRS) * n:
            raise ValueError(f"models.{key}: {len(block['rows'])} rows, expected {len(PAIRS) * n}")
        if len(block["top10"]) != len(CODES) * n:
            raise ValueError(f"models.{key}: {len(block['top10'])} top10 entries, expected {len(CODES) * n}")
        counts: dict[int, dict[str, int]] = {}
        lasts: dict[int, dict[str, str]] = {}
        for t in block["tokens"]:
            counts.setdefault(int(t["frame"]), {})[t["code"]] = int(t["n_tok"])
            lasts.setdefault(int(t["frame"]), {})[t["code"]] = t["last_tok"]
        for index in range(n):
            if set(counts.get(index, {})) != set(CODES):
                raise ValueError(f"models.{key}: frame {index} lacks a token entry for some pass")
            failures = token_assertions(counts[index], lasts[index])
            if failures:
                raise ValueError(f"models.{key}: frame {index} fails the token assertions: {failures}")
            for code in CODES:
                top = block["top10"].get(f"{index}:{code}")
                if top is None or len(top) != TOP_K:
                    raise ValueError(f"models.{key}: top10 for {index}:{code} does not have {TOP_K} entries")
                for entry in top:
                    if len(entry) != 2 or not (0 < float(entry[1]) <= 1):
                        raise ValueError(f"models.{key}: top10 {index}:{code} has a bad entry {entry!r}")
        for k, row in enumerate(block["rows"]):
            expected = PAIRS[k % len(PAIRS)]
            if (row["structure"], row["control"]) != expected or int(row["frame"]) != k // len(PAIRS):
                raise ValueError(f"models.{key}: row {k} is {row['frame']}:{row['structure']}/{row['control']}, "
                                 f"expected {k // len(PAIRS)}:{expected[0]}/{expected[1]}")
            for name in ("ret", "ctrl"):
                if not (float(row[name]) >= 0) or not np.isfinite(row[name]):
                    raise ValueError(f"models.{key}: row {k} {name} is {row[name]}")
            for name in ("verbmass_ref", "verbmass_emb", "verbmass_ctrl"):
                if not (0 <= float(row[name]) <= 1.0 + 1e-6):
                    raise ValueError(f"models.{key}: row {k} {name} is {row[name]}, outside [0, 1]")
            for name in ("logp_V_ref", "logp_V_emb", "logp_V_ctrl"):
                if not (float(row[name]) <= 1e-6) or not np.isfinite(row[name]):
                    raise ValueError(f"models.{key}: row {k} {name} is {row[name]}, not a log-probability")


__all__ = [
    "TASK", "SCHEMA", "TOP_K", "DISTANCE_MEASURE", "VERBMASS_MEASURE", "LOGP_V_MEASURE", "TOP10_MEASURE",
    "BOS_POLICY", "ModelSpec", "MODELS_8A", "ANCHOR_KEY", "SMALL_KEYS", "load_tokenizers", "prefix_ids",
    "bos_added", "word_id", "verb_ids", "model_meta_8a", "top_k_tokens", "frame_scalars", "score_frames_batch",
    "preregistration_8a", "score_model_8a", "load_scores_8a", "rows_from_frame", "model_block", "AnchorSet",
    "load_anchor_file", "anchor_tokens_check", "compare_anchor_rows", "anchor_check", "anchor_summary",
    "anchor_diagnosis", "missing_models", "build_record_8a", "validate_record_8a",
]
