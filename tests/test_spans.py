"""Span algebra against hand-computed values from the theory doc's 13-word sentence."""

import numpy as np
import pytest

from m1_analyzer.experiments.spans import (
    bracket_prf,
    cky_induce,
    crosses,
    enumerate_spans,
    greedy_induce,
    left_branching,
    random_binary,
    rank_curve,
    rank_spans,
    right_branching,
    spans_to_brackets,
)
from m1_analyzer.experiments.stats import trapezoid_area
from m1_analyzer.experiments.treebank import theory_example

N = 13
GOLD = theory_example().gold_spans


def _non_crossing(spans):
    spans = list(spans)
    return all(not crosses(a, b) for a in spans for b in spans)


def test_enumerate_spans_counts_every_non_trivial_run():
    spans = enumerate_spans(N)
    assert len(spans) == 77  # the theory doc's triangle
    assert (0, N - 1) not in spans
    assert all(j > i for i, j in spans)
    assert len(enumerate_spans(3)) == 2


@pytest.mark.parametrize("a,b,expected", [
    ((0, 2), (1, 4), True), ((1, 4), (0, 2), True),
    ((0, 2), (3, 5), False), ((0, 5), (1, 2), False), ((1, 2), (0, 5), False),
    ((0, 2), (0, 2), False), ((0, 2), (2, 4), True),
])
def test_crossing_rule(a, b, expected):
    assert crosses(a, b) is expected


def test_baselines_are_full_binary_bracketings():
    for spans in (right_branching(N), left_branching(N), random_binary(N, np.random.default_rng(0))):
        assert len(spans) == N - 2
        assert _non_crossing(spans)
        assert (0, N - 1) not in spans


def test_baseline_f1_matches_theory_doc():
    assert bracket_prf(right_branching(N), GOLD)[2] == pytest.approx(0.333, abs=1e-3)
    assert bracket_prf(left_branching(N), GOLD)[2] == pytest.approx(0.222, abs=1e-3)
    rng = np.random.default_rng(0)
    mean = np.mean([bracket_prf(random_binary(N, rng), GOLD)[2] for _ in range(2000)])
    assert 0.12 < mean < 0.20  # the doc reports 0.16 over 2000 draws


def test_prf_hand_computation():
    pred = {(0, 1), (0, 2), (4, 5), (4, 6), (3, 6), (0, 6), (10, 11), (10, 12), (9, 12), (8, 12), (7, 12)}
    p, r, f1, match = bracket_prf(pred, GOLD)
    assert match == 7 and p == pytest.approx(7 / 11) and r == pytest.approx(1.0)
    assert f1 == pytest.approx(2 * (7 / 11) / (1 + 7 / 11))
    assert bracket_prf(set(), set()) == (1.0, 1.0, 1.0, 0)
    assert bracket_prf({(0, 1)}, set())[2] == 0.0
    assert bracket_prf(set(), {(0, 1)})[2] == 0.0


def _costs(gold_cheap=True):
    costs = {}
    for s in enumerate_spans(N):
        costs[s] = 0.3 if (s in GOLD and gold_cheap) else (2.0 if any(crosses(s, g) for g in GOLD) else 1.0)
    return costs


def test_greedy_yields_a_full_binary_tree_and_recovers_cheap_gold():
    kept = greedy_induce(_costs(), N)
    assert len(kept) == N - 2
    assert _non_crossing(kept)
    assert GOLD <= set(kept)


def test_greedy_is_deterministic_under_ties():
    costs = {s: 1.0 for s in enumerate_spans(N)}
    a = greedy_induce(costs, N)
    b = greedy_induce(dict(reversed(list(costs.items()))), N)
    assert a == b
    assert rank_spans(costs)[0] == (0, 1)  # shortest, leftmost first


def test_greedy_trace_names_the_blocker():
    costs = {(0, 2): 0.1, (1, 3): 0.2, (0, 1): 0.3, (2, 3): 0.4}
    kept, trace = greedy_induce(costs, 5, trace=True)
    assert kept[0] == (0, 2)
    blocked = {span: blocker for span, _, blocker in trace if blocker}
    assert blocked[(1, 3)] == (0, 2)


def test_greedy_ignores_trivial_spans_in_costs():
    costs = _costs()
    costs[(0, N - 1)] = -5.0
    costs[(3, 3)] = -5.0
    kept = greedy_induce(costs, N)
    assert (0, N - 1) not in kept and (3, 3) not in kept


def test_cky_finds_the_minimum_total_and_is_a_tree():
    costs = _costs()
    kept = cky_induce(costs, N)
    assert len(kept) == N - 2 and _non_crossing(kept)
    assert sum(costs[s] for s in kept) <= sum(costs[s] for s in greedy_induce(costs, N)) + 1e-9
    assert cky_induce({}, 4) == []


def test_rank_curve_pinned_and_gold_first():
    curve = rank_curve(_costs(), GOLD, [0.0, 0.1, 0.5, 1.0])
    assert curve[0] == 0.0 and curve[-1] == 1.0
    assert curve[1] == 1.0  # 7 gold spans are the 7 cheapest of 77 -> recovered by 10%
    assert rank_curve({}, GOLD, [0.5]) == [0.0]
    grid = np.linspace(0, 1, 26)
    rng = np.random.default_rng(0)
    areas = []
    for _ in range(300):  # costs that know nothing: area averages to chance
        random_costs = {s: float(rng.random()) for s in enumerate_spans(N)}
        areas.append(trapezoid_area(grid, rank_curve(random_costs, GOLD, grid)))
    assert abs(np.mean(areas) - 0.5) < 0.05


def test_spans_to_brackets_renders():
    text = spans_to_brackets("a b c d".split(), {(0, 1), (2, 3)})
    assert text == "[a b] [c d]"
