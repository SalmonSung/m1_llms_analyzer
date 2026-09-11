"""Phase A of Task 9a: admissible cuts, spliced texts, endpoint distances and
downstream divergences, cached per paragraph.

A **cut** is a pair of token positions ``(i, j)`` in one paragraph. The spliced
sequence is ``ids[:i + 1] + ids[j + 1:]``: everything strictly after endpoint
``i`` up to and including endpoint ``j`` is deleted (``j - i`` tokens) and the
text rejoined. The **state** at a position is the model's next-token
log-probability vector there (`NextTokenStateService`).

Admissibility, applied by construction rather than by labelling:
  (a) both endpoints are the same boundary type (`boundaries.py`), so a
      sentence end is joined to a sentence start and whole sentences go;
  (b) at least ``window`` tokens follow ``j``, so the downstream comparison
      has somewhere to live.
Cuts that land mid-word or mid-clause are never generated.

Per cut, the cache stores:
  ``d``      L2 distance between the two endpoint states of the ORIGINAL text
             (the close/far axis);
  ``div_k``  for ``k = 0..window-1``, the L2 distance between the spliced
             state at ``i + 1 + k`` and the original state at ``j + 1 + k``
             (the same token, with and without the deleted sentences before it);
  ``div``    the median of ``div_k`` -- the pre-registered divergence;
  ``fl``     the spliced text's fluency in nats per token (exploratory: Act 3's
             number, deliberately NOT the divergence);
  ``retokenises``  whether encoding the decoded splice reproduces the spliced
             ids (the ids are what the model saw either way);
  ``del_surp``     the total surprisal, in nats, of the deleted tokens
             ``ids[i+1 : j+1]`` in their ORIGINAL context -- how much
             information the span carried -- and ``del_surp_mean`` that sum
             over ``seg_len``. A second matching variable beside length: far
             cuts may simply delete more information per token.

Per paragraph, ``surprisal[t] = -log p(ids[t] | ids[:t])`` for every token, read
from the same original-text pass that produces the endpoint and downstream
states: ``surprisal[t]`` comes from the state at position ``t - 1`` (the BOS row
for ``t = 0``), the float32 log-softmax the distances are taken in. ``fluency``
is the mean of ``surprisal`` over ALL scored tokens, token 0 included. The
fields are additive: `add_surprisal` fills them into an existing cache without
moving any other byte, so an audit sheet keyed on that cache still lines up.

Splicing happens on token ids so the alignment ``i + 1 + k <-> j + 1 + k`` is
exact; the decoded text is only for the human audit. The cache header is the
pre-registration: measure, window, boundary kinds, strata, decile and the
length-matching bin width are written before the first cut is scored, and phase B reads them back.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..utils.batching import maybe_progress
from ..utils.logging import get_logger
from .boundaries import BOUNDARY_KINDS, SENTENCE, boundary_positions
from .jsonl_cache import append_row, check_header, mirror, read_jsonl, rewrite, write_header
from .paragraphs import Paragraph

log = get_logger("splice")

SCHEMA = 2
TASK = "9a"
DEFAULT_WINDOW = 20
#: Deleted-segment length strata, ``[lo, hi)`` in tokens. Pre-registered in the header.
DEFAULT_STRATA: tuple[tuple[int, int], ...] = ((20, 40), (40, 80), (80, 160))
#: Fraction of each matching bin's cuts (by endpoint distance) that count as close / far.
DEFAULT_DECILE = 0.1
#: Width in tokens of the length-matching bins inside each stratum. Labels are
#: assigned within these bins, so close and far cuts delete the same amount of
#: text to within a bin; a stratum of doubling width alone leaves a residual
#: length gradient (the far decile of 80-160 deletes more than its close decile).
DEFAULT_MATCH_WIDTH = 10

DIVERGENCE_MEASURE = (
    "median over k = 0..window-1 of the L2 distance between the spliced text's next-token "
    "log-probability vector at position i+1+k and the original text's at position j+1+k"
)
DISTANCE_MEASURE = (
    "L2 distance between the original text's next-token log-probability vectors at the two "
    "endpoint positions i and j"
)
SURPRISAL_MEASURE = (
    "-log p of the actual next token from the original-text pass, float32 log-softmax, "
    "same BOS/indexing as states"
)
FLUENCY_DEFINITION = (
    "fluency = mean of surprisal over ALL scored tokens: with a BOS prepended (bos_policy='auto') "
    "that is every token including token 0 predicted from BOS, so mean(surprisal) reproduces "
    "fluency and mean(surprisal[1:]) does not; under bos_policy='none' token 0 is unscored (null) "
    "and the two agree"
)
#: Header fields that may differ between the writer and a resumer. `schema` is
#: deliberately NOT among them: schema 2 tightened the boundary rule, so resuming
#: a schema-1 cache would mix cuts generated under two different definitions of
#: an admissible endpoint.
_UNCHECKED_HEADER_KEYS = frozenset({"kind", "date", "n_paragraphs", "corpus"})

#: What makes a position an admissible endpoint. Recorded in the header and
#: compared on resume, so the rule a cache was built under is never in doubt.
BOUNDARY_RULE = (
    "sentence-final: terminal punctuation (+ closers), the token ends exactly there, whitespace "
    "follows, AND the text after it begins a new sentence (right-side check, schema 2)"
)


@dataclass(frozen=True)
class Cut:
    i: int
    j: int
    boundary: str = SENTENCE

    @property
    def seg_len(self) -> int:
        return self.j - self.i


def admissible_cuts(boundaries: Mapping[str, Sequence[int]], n_tokens: int, window: int) -> list[Cut]:
    """Every same-type endpoint pair with at least `window` tokens after `j`."""
    if window < 1:
        raise ValueError("window must be >= 1.")
    cuts: list[Cut] = []
    for kind, positions in boundaries.items():
        pos = sorted(set(int(p) for p in positions))
        for a, i in enumerate(pos):
            for j in pos[a + 1:]:
                if n_tokens - (j + 1) >= window:
                    cuts.append(Cut(i, j, kind))
    return cuts


def splice_ids(ids: Sequence[int], i: int, j: int) -> list[int]:
    """``ids[:i+1] + ids[j+1:]``: delete the ``j - i`` tokens strictly after `i` through `j`."""
    if not 0 <= i < j < len(ids):
        raise ValueError(f"need 0 <= i < j < n, got i={i}, j={j}, n={len(ids)}")
    return list(ids[: i + 1]) + list(ids[j + 1:])


def splice_text(text: str, offsets: Sequence[tuple[int, int]], i: int, j: int, marker: str = "") -> str:
    """The rejoined *text*: everything up to the end of token `i`, then from the token after `j`."""
    head = text[: offsets[i][1]]
    tail = text[offsets[j][1]:]
    return head + marker + (tail if tail.startswith(" ") else " " + tail)


def preregistration(
    *,
    window: int = DEFAULT_WINDOW,
    splitter: str = "punkt",
    boundary_kinds: Sequence[str] = BOUNDARY_KINDS,
    primary_boundary: str = SENTENCE,
    strata: Sequence[Sequence[int]] = DEFAULT_STRATA,
    decile: float = DEFAULT_DECILE,
    match_width: int = DEFAULT_MATCH_WIDTH,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The cache header: everything fixed before a cut is scored."""
    if match_width < 1:
        raise ValueError("match_width must be >= 1 token.")
    if primary_boundary not in boundary_kinds:
        raise ValueError(f"primary_boundary {primary_boundary!r} is not among boundary_kinds {tuple(boundary_kinds)}")
    if not 0 < decile <= 0.5:
        raise ValueError("decile must be in (0, 0.5].")
    edges = [[int(lo), int(hi)] for lo, hi in strata]
    for lo, hi in edges:
        if not 0 < lo < hi:
            raise ValueError(f"bad stratum [{lo}, {hi})")
    return {
        "kind": "header", "schema": SCHEMA, "task": TASK,
        "window": int(window),
        "divergence_measure": DIVERGENCE_MEASURE,
        "distance_measure": DISTANCE_MEASURE,
        "splitter": splitter,
        "boundary_rule": BOUNDARY_RULE,
        "boundary_kinds": list(boundary_kinds),
        "primary_boundary": primary_boundary,
        "strata": edges,
        "decile": float(decile),
        "match_width": int(match_width),
        "splice": "token ids: ids[:i+1] + ids[j+1:]; states are float32 log-softmax vectors",
        "surprisal_measure": SURPRISAL_MEASURE,
        **(dict(provenance) if provenance else {}),
    }


# ----------------------------------------------------------------- surprisal


def surprisal_from_logprobs(token_logprobs, n_tokens: int) -> list[float | None]:
    """Per-token surprisal in nats, ``n_tokens`` long, rounded like ``fluency``.

    The state service scores every token when a BOS is prepended (``n_tokens``
    log-probabilities) and all but the first under ``bos_policy="none"``
    (``n_tokens - 1``); in the latter case entry 0 is ``None``, since nothing
    precedes token 0 to predict it from. Any other length is an error.
    """
    values = [round(-float(x), 6) for x in np.asarray(token_logprobs, dtype=float).ravel()]
    if len(values) == n_tokens:
        return values
    if len(values) == n_tokens - 1:
        return [None] + values
    raise ValueError(f"got {len(values)} token log-probabilities for {n_tokens} tokens; expected n or n-1.")


def deleted_surprisal(surprisal: Sequence[float | None], i: int, j: int) -> tuple[float, float]:
    """``(del_surp, del_surp_mean)``: the surprisal summed over the deleted tokens
    ``i+1 .. j`` (positions ``surprisal[i+1:j+1]``, ``j - i`` of them) and per token."""
    if not 0 <= i < j < len(surprisal):
        raise ValueError(f"need 0 <= i < j < n, got i={i}, j={j}, n={len(surprisal)}")
    span = surprisal[i + 1: j + 1]
    if len(span) != j - i or any(x is None for x in span):
        raise ValueError(f"cut ({i}, {j}): the deleted span has no complete surprisal.")
    total = float(sum(span))
    return round(total, 6), round(total / (j - i), 6)


def check_surprisal(rows: Sequence[Mapping[str, Any]], tol: float = 1e-3) -> dict[str, Any]:
    """The two invariants of the surprisal field, as numbers to report.

    (1) ``len(surprisal) == n_tokens`` for every paragraph and
    ``len(surprisal[i+1:j+1]) == seg_len`` for every cut; (2) how closely the
    mean of ``surprisal`` reproduces the paragraph's ``fluency`` -- over all
    tokens (it should, to rounding) and over ``surprisal[1:]`` (it should not,
    with a BOS: token 0 is scored from the BOS row and counted in ``fluency``).
    Rows without the field are counted in ``n_missing`` and skipped.
    """
    with_field = [r for r in rows if "surprisal" in r]
    n_bad_par = n_bad_cut = n_null0 = n_cuts = 0
    n_ok_all = n_ok_tail = 0
    max_all = max_tail = max_del = 0.0
    for r in with_field:
        surp = r["surprisal"]
        if len(surp) != int(r["n_tokens"]):
            n_bad_par += 1
        if surp and surp[0] is None:
            n_null0 += 1
        scored = [x for x in surp if x is not None]
        tail = [x for x in surp[1:] if x is not None]
        fluency = r.get("fluency")
        if fluency is not None and scored:
            diff_all = abs(float(np.mean(scored)) - float(fluency))
            max_all = max(max_all, diff_all)
            n_ok_all += diff_all <= tol
            if tail:
                diff_tail = abs(float(np.mean(tail)) - float(fluency))
                max_tail = max(max_tail, diff_tail)
                n_ok_tail += diff_tail <= tol
        for c in r["cuts"]:
            n_cuts += 1
            i, j = int(c["i"]), int(c["j"])
            span = surp[i + 1: j + 1]
            if len(span) != int(c["seg_len"]) or any(x is None for x in span):
                n_bad_cut += 1
            elif "del_surp" in c:
                max_del = max(max_del, abs(float(c["del_surp"]) - float(sum(span))))
    n = len(with_field)
    return {
        "n_paragraphs": len(rows), "n_with_surprisal": n, "n_missing": len(rows) - n, "n_cuts": n_cuts,
        "n_bad_paragraph_length": n_bad_par, "n_bad_cut_length": n_bad_cut, "n_null_token0": n_null0,
        "max_abs_mean_all_minus_fluency": max_all, "n_within_tol_all": n_ok_all,
        "max_abs_mean_from_1_minus_fluency": max_tail, "n_within_tol_from_1": n_ok_tail,
        "max_abs_del_surp_minus_sum": max_del, "tol": tol,
        "fluency_definition": FLUENCY_DEFINITION,
        "ok": bool(n and not n_bad_par and not n_bad_cut and n_ok_all == n),
    }


# ------------------------------------------------------------ one paragraph


def paragraph_boundaries(states, paragraph: Paragraph, *, splitter: str = "punkt",
                         boundary_kinds: Sequence[str] = BOUNDARY_KINDS):
    """``(ids, offsets, {kind: positions}, n_rejected)`` without touching the model."""
    ids, offsets = states.encode_with_offsets(paragraph.text)
    boundaries, rejected = boundary_positions(paragraph.text, offsets, splitter=splitter, kinds=boundary_kinds)
    return ids, offsets, boundaries, rejected


def score_paragraph(
    states,
    paragraph: Paragraph,
    *,
    window: int = DEFAULT_WINDOW,
    splitter: str = "punkt",
    boundary_kinds: Sequence[str] = BOUNDARY_KINDS,
    batch_size: int = 4,
    context_tokens: int = 12,
) -> dict[str, Any]:
    """Score every admissible cut of one paragraph; returns the cache row."""
    ids, offsets, boundaries, rejected = paragraph_boundaries(
        states, paragraph, splitter=splitter, boundary_kinds=boundary_kinds,
    )
    n = len(ids)
    cuts = admissible_cuts(boundaries, n, window)
    row: dict[str, Any] = {
        "kind": "paragraph", "id": paragraph.id, "text": paragraph.text, "source": paragraph.source,
        "info": dict(paragraph.info), "n_tokens": n,
        "boundaries": {k: [int(p) for p in v] for k, v in boundaries.items()},
        "n_rejected_boundaries": int(rejected), "cuts": [],
    }

    endpoint_positions = sorted({p for v in boundaries.values() for p in v})
    downstream = sorted({c.j + 1 + k for c in cuts for k in range(window)})
    positions = sorted(set(endpoint_positions) | set(downstream))
    [original] = states.states([ids], [positions], batch_size=1, return_token_logprobs=True)
    row["fluency"] = round(-original.mean_logprob, 6)
    surprisal = surprisal_from_logprobs(original.token_logprobs, n)
    row["surprisal"] = surprisal
    if not cuts:
        return row
    index = {p: k for k, p in enumerate(positions)}
    S = original.states

    spliced = [splice_ids(ids, c.i, c.j) for c in cuts]
    spliced_positions = [[c.i + 1 + k for k in range(window)] for c in cuts]
    results = states.states(spliced, spliced_positions, batch_size=batch_size, return_token_logprobs=True)

    for cut, seq, result in zip(cuts, spliced, results):
        target = np.stack([S[index[cut.j + 1 + k]] for k in range(window)])
        div_k = np.linalg.norm(result.states - target, axis=1)
        d = float(np.linalg.norm(S[index[cut.i]] - S[index[cut.j]]))
        decoded = states.decode(seq)
        del_surp, del_surp_mean = deleted_surprisal(surprisal, cut.i, cut.j)
        row["cuts"].append({
            "i": cut.i, "j": cut.j, "boundary": cut.boundary, "seg_len": cut.seg_len,
            "char_i": int(offsets[cut.i][1]), "char_j": int(offsets[cut.j][1]),
            "d": round(d, 5),
            "div": round(float(np.median(div_k)), 5),
            "div_k": [round(float(x), 4) for x in div_k],
            "fl": round(-result.mean_logprob, 6),
            "retokenises": states.encode(decoded) == seq,
            "join": (states.decode(ids[max(0, cut.i - context_tokens): cut.i + 1]) + " ⟦cut⟧ "
                     + states.decode(ids[cut.j + 1: cut.j + 1 + context_tokens])),
            # Last, so a cache scored with these fields and one augmented by
            # `add_surprisal` serialise identically.
            "del_surp": del_surp, "del_surp_mean": del_surp_mean,
        })
    return row


# ------------------------------------------------------------------ corpus


def compute_splices(
    states,
    paragraphs: Sequence[Paragraph],
    *,
    window: int = DEFAULT_WINDOW,
    cache_path: str | os.PathLike | None = None,
    provenance: Mapping[str, Any] | None = None,
    splitter: str = "punkt",
    boundary_kinds: Sequence[str] = BOUNDARY_KINDS,
    primary_boundary: str = SENTENCE,
    strata: Sequence[Sequence[int]] = DEFAULT_STRATA,
    decile: float = DEFAULT_DECILE,
    match_width: int = DEFAULT_MATCH_WIDTH,
    batch_size: int = 4,
    show_progress: bool = True,
    mirror_path: str | os.PathLike | None = None,
    mirror_every: int = 10,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score every admissible cut of every paragraph; cache and resume.

    Returns ``(header, rows)`` in `paragraphs` order. The header is written
    before any scoring; on resume it must match (the differing field is named)
    and finished paragraphs are skipped.
    """
    header = preregistration(
        window=window, splitter=splitter, boundary_kinds=boundary_kinds, primary_boundary=primary_boundary,
        strata=strata, decile=decile, match_width=match_width, provenance=provenance,
    )
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
        log.info("Resuming: %d of %d paragraphs already scored in %s.", len(done), len(paragraphs), path)
        lacking = sum(1 for row in rows if "surprisal" not in row)
        if lacking:
            log.warning(
                "%s predates the surprisal field: %d scored paragraph(s) lack `surprisal` / `del_surp`. "
                "Run add_surprisal(states, path) afterwards to fill them in place and stamp the header.",
                path, lacking,
            )
    elif path is not None:
        write_header(path, header)

    fh = path.open("a", encoding="utf-8") if path is not None else None
    since_mirror = 0
    try:
        pending = [p for p in paragraphs if p.id not in done]
        for paragraph in maybe_progress(pending, show_progress, desc="paragraphs"):
            row = score_paragraph(states, paragraph, window=window, splitter=splitter,
                                  boundary_kinds=boundary_kinds, batch_size=batch_size)
            done[paragraph.id] = row
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
    return header, [done[p.id] for p in paragraphs if p.id in done]


def load_splices(path: str | os.PathLike) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a splice cache back: ``(header, rows)``."""
    header, rows, _ = read_jsonl(Path(path))
    if header is None or header.get("task") != TASK:
        raise ValueError(f"{path} has no Task 9a header line; it is not a splice cache.")
    return header, rows


def add_surprisal(
    states,
    cache_path: str | os.PathLike,
    *,
    out_path: str | os.PathLike | None = None,
    batch_size: int = 1,
    show_progress: bool = True,
    mirror_path: str | os.PathLike | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Fill ``surprisal`` / ``del_surp`` / ``del_surp_mean`` into an existing cache, additively.

    One forward pass per paragraph on the ORIGINAL ids -- no splicing, no
    positions requested, so no vocabulary vector leaves the device -- with the
    same BOS policy the states were cached under. Rows that already carry
    ``surprisal`` are skipped; every pre-existing key and value is re-serialised
    byte for byte (new keys go last); the header gains ``surprisal_measure``;
    the file is rewritten atomically, in place unless `out_path` is given, and
    not at all when nothing changed. ``batch_size=1`` reproduces the original
    pass exactly (no padding); a larger batch is faster and equal to rounding.

    Returns ``(header, rows, report)`` with `check_surprisal`'s report plus
    ``n_augmented``, ``n_skipped`` and ``written``.
    """
    path = Path(cache_path)
    header, rows = load_splices(path)
    stamped = header.get("surprisal_measure")
    if stamped is not None and stamped != SURPRISAL_MEASURE:
        raise ValueError(
            f"{path} carries surprisal_measure={stamped!r}, not {SURPRISAL_MEASURE!r}; refusing to mix "
            "two definitions in one cache. Use a different out_path."
        )
    pending = [r for r in rows if "surprisal" not in r]
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    ids_of: dict[str, list[int]] = {}
    for r in pending:
        ids = states.encode(r["text"])
        if len(ids) != int(r["n_tokens"]):
            raise ValueError(
                f"{r['id']}: the text re-encodes to {len(ids)} tokens but the cache says {r['n_tokens']}. "
                "This is not the model / tokenizer the cache was scored with."
            )
        ids_of[r["id"]] = ids
    chunks = [pending[k: k + batch_size] for k in range(0, len(pending), batch_size)]
    for chunk in maybe_progress(chunks, show_progress, desc="surprisal"):
        results = states.states([ids_of[r["id"]] for r in chunk], [[] for _ in chunk],
                                batch_size=batch_size, return_token_logprobs=True)
        for r, result in zip(chunk, results):
            surprisal = surprisal_from_logprobs(result.token_logprobs, int(r["n_tokens"]))
            r["surprisal"] = surprisal
            for c in r["cuts"]:
                if int(c["seg_len"]) != int(c["j"]) - int(c["i"]):
                    raise ValueError(f"{r['id']}: cut ({c['i']}, {c['j']}) has seg_len {c['seg_len']}.")
                c["del_surp"], c["del_surp_mean"] = deleted_surprisal(surprisal, int(c["i"]), int(c["j"]))
    changed = bool(pending) or stamped is None
    header.setdefault("surprisal_measure", SURPRISAL_MEASURE)

    written = None
    target = Path(out_path) if out_path is not None else path
    if out_path is not None or changed:
        target.parent.mkdir(parents=True, exist_ok=True)
        rewrite(target, header, rows)
        written = target
        if mirror_path:
            mirror(target, mirror_path)
    report = check_surprisal(rows)
    report.update({"n_augmented": len(pending), "n_skipped": len(rows) - len(pending),
                   "written": str(written) if written else None})
    log.info("Surprisal: %d paragraph(s) augmented, %d already had it; lengths ok=%s; "
             "max |mean(surprisal) - fluency| = %.2e; written to %s.",
             len(pending), len(rows) - len(pending), not (report["n_bad_paragraph_length"] or report["n_bad_cut_length"]),
             report["max_abs_mean_all_minus_fluency"], written)
    return header, rows, report


def cut_count(rows: Sequence[Mapping[str, Any]], boundary: str | None = None) -> int:
    return sum(1 for r in rows for c in r["cuts"] if boundary is None or c["boundary"] == boundary)
