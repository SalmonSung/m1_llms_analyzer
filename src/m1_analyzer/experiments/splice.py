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
             ids (the ids are what the model saw either way).

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

SCHEMA = 1
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
#: Header fields that may differ between the writer and a resumer.
_UNCHECKED_HEADER_KEYS = frozenset({"kind", "schema", "date", "n_paragraphs", "corpus"})


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
        "boundary_kinds": list(boundary_kinds),
        "primary_boundary": primary_boundary,
        "strata": edges,
        "decile": float(decile),
        "match_width": int(match_width),
        "splice": "token ids: ids[:i+1] + ids[j+1:]; states are float32 log-softmax vectors",
        **(dict(provenance) if provenance else {}),
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


def cut_count(rows: Sequence[Mapping[str, Any]], boundary: str | None = None) -> int:
    return sum(1 for r in rows for c in r["cuts"] if boundary is None or c["boundary"] == boundary)
