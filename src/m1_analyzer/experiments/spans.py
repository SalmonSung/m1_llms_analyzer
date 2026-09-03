"""Span algebra for bracketing experiments: baselines, induction, scoring.

A *span* is ``(i, j)`` -- the 0-based indices of its first and last word,
inclusive, with ``j > i``. Single words and the whole sentence are never
candidates: every bracketing contains them, so they carry no information and
would hand every method free points. Everything here is pure Python on plain
tuples, so it is testable without a model and usable by any task that scores
brackets (0c, 1a, 1b, 3a).

Formulas (from the theory doc's appendix):

* crossing: ``(i, j)`` and ``(k, l)`` cross iff ``i < k <= j < l`` or
  ``k < i <= l < j``. A set of pairwise non-crossing spans is a bracketing; a
  maximal one over N words has N - 2 non-trivial spans (a full binary tree).
* unlabelled bracket F1: ``P = |A n G| / |A|``, ``R = |A n G| / |G|``,
  ``F1 = 2PR / (P + R)``.
* rank curve: sort spans by cost ascending; for the cheapest fraction ``x``,
  record the fraction of gold spans recovered. The diagonal is chance.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np

Span = tuple[int, int]


# ------------------------------------------------------------------ basics


def enumerate_spans(n: int) -> list[Span]:
    """Every contiguous run of 2..n-1 words, in a fixed (i, j) order."""
    return [(i, j) for i in range(n) for j in range(i + 1, n) if not (i == 0 and j == n - 1)]


def crosses(a: Span, b: Span) -> bool:
    (i, j), (k, l) = a, b
    return (i < k <= j < l) or (k < i <= l < j)


def is_trivial(span: Span, n: int) -> bool:
    i, j = span
    return j <= i or (i == 0 and j == n - 1)


def non_trivial(spans: Iterable[Span], n: int) -> set[Span]:
    return {s for s in spans if not is_trivial(s, n)}


# --------------------------------------------------------------- baselines


def right_branching(n: int) -> set[Span]:
    """Every suffix: (1, n-1), (2, n-1), ... (n-2, n-1)."""
    return {(i, n - 1) for i in range(1, n - 1)}


def left_branching(n: int) -> set[Span]:
    """Every prefix: (0, 1), (0, 2), ... (0, n-2)."""
    return {(0, j) for j in range(1, n - 1)}


def random_binary(n: int, rng: np.random.Generator) -> set[Span]:
    """A random full binary bracketing: split uniformly at random, recurse.

    This is the standard "random tree" baseline (uniform over split points at
    each node, not uniform over the Catalan set of trees).
    """
    out: set[Span] = set()

    def rec(i: int, j: int) -> None:
        if j - i < 1:
            return
        if not (i == 0 and j == n - 1):
            out.add((i, j))
        if j - i == 1:
            return
        k = int(rng.integers(i, j))  # left half is i..k, right half k+1..j
        rec(i, k)
        rec(k + 1, j)

    rec(0, n - 1)
    return out


# --------------------------------------------------------------- induction


def rank_spans(costs: Mapping[Span, float]) -> list[Span]:
    """Spans cheapest first, with a deterministic tie-break (shorter, then leftmost)."""
    return sorted(costs, key=lambda s: (costs[s], s[1] - s[0], s[0]))


def greedy_induce(costs: Mapping[Span, float], n: int, *, trace: bool = False):
    """Cheapest-first, never cross: the theory doc's induction.

    Walks every scored span in rank order and keeps it unless it crosses a span
    already kept. Nothing is filtered by cost. With all non-trivial spans scored
    the result is a full binary bracketing of n - 2 spans. Returns the kept
    spans in acceptance order; with ``trace=True`` also returns
    ``[(span, cost, blocked_by_or_None), ...]`` for every span visited.
    """
    kept: list[Span] = []
    visited = []
    for span in rank_spans(costs):
        if is_trivial(span, n):
            continue
        blocker = next((k for k in kept if crosses(span, k)), None)
        if blocker is None:
            kept.append(span)
        if trace:
            visited.append((span, float(costs[span]), blocker))
    return (kept, visited) if trace else kept


def cky_induce(costs: Mapping[Span, float], n: int) -> list[Span]:
    """The non-crossing set with the smallest *total* cost (exact, O(n^3)).

    Only spans present in `costs` may be used; a missing span is treated as
    unavailable (infinite cost), so a full binary tree is returned only when
    every non-trivial span was scored. Trivial spans cost nothing.
    """
    inf = float("inf")
    best = [[inf] * n for _ in range(n)]
    split = [[-1] * n for _ in range(n)]
    for i in range(n):
        best[i][i] = 0.0
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            own = 0.0 if (i == 0 and j == n - 1) else costs.get((i, j), inf)
            if own == inf:
                continue
            for k in range(i, j):
                total = best[i][k] + best[k + 1][j] + own
                if total < best[i][j]:
                    best[i][j], split[i][j] = total, k
    out: list[Span] = []

    def collect(i: int, j: int) -> None:
        if j <= i or split[i][j] < 0:
            return
        if not (i == 0 and j == n - 1):
            out.append((i, j))
        k = split[i][j]
        collect(i, k)
        collect(k + 1, j)

    if best[0][n - 1] < inf:
        collect(0, n - 1)
    return out


INDUCERS = {"greedy": greedy_induce, "cky": cky_induce}


# ----------------------------------------------------------------- scoring


def bracket_prf(pred: Iterable[Span], gold: Iterable[Span]) -> tuple[float, float, float, int]:
    """``(precision, recall, f1, n_match)`` on unlabelled spans.

    Two empty sets agree perfectly; one empty set against a non-empty one is a
    complete miss. Callers that want to skip empty-gold sentences do so upstream.
    """
    pred_set, gold_set = set(pred), set(gold)
    match = len(pred_set & gold_set)
    if not pred_set and not gold_set:
        return 1.0, 1.0, 1.0, 0
    p = match / len(pred_set) if pred_set else 0.0
    r = match / len(gold_set) if gold_set else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1, match


def rank_curve(costs: Mapping[Span, float], gold: Iterable[Span], grid: Sequence[float]) -> list[float]:
    """Fraction of gold recovered when the cheapest ``x`` of all spans is accepted.

    Accepted count is ``ceil(x * n_spans)``; the curve is 0 at x = 0 and, since
    every gold span is a scored span, 1 at x = 1.
    """
    gold_set = set(gold)
    ordered = rank_spans(costs)
    total = len(ordered)
    if not gold_set or total == 0:
        return [0.0 for _ in grid]
    hits = np.cumsum([1 if s in gold_set else 0 for s in ordered])
    out = []
    for x in grid:
        k = int(math.ceil(x * total))
        out.append(0.0 if k <= 0 else float(hits[min(k, total) - 1]) / len(gold_set))
    return out


def spans_to_brackets(words: Sequence[str], spans: Iterable[Span]) -> str:
    """Render spans as a bracketed string, for printing."""
    opens: dict[int, int] = {}
    closes: dict[int, int] = {}
    for i, j in sorted(spans, key=lambda s: (s[0], -s[1])):
        opens[i] = opens.get(i, 0) + 1
        closes[j] = closes.get(j, 0) + 1
    parts = []
    for k, w in enumerate(words):
        parts.append("[" * opens.get(k, 0) + w + "]" * closes.get(k, 0))
    return " ".join(parts)
