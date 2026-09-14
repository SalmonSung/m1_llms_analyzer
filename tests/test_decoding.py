"""Sampling, greedy decoding and bag-of-token overlap on the tiny model."""

import numpy as np
import pytest

from m1_analyzer import Analyzer, ModelConfig, RunConfig, ScoringConfig
from m1_analyzer.experiments.decoding import (
    first_diff,
    greedy_continuation,
    nucleus_filter,
    overlap_stats,
    sample_continuations,
    token_f1,
)
from m1_analyzer.testing import build_tiny_local_model


@pytest.fixture(scope="module")
def analyzer(tmp_path_factory) -> Analyzer:
    path = build_tiny_local_model(tmp_path_factory.mktemp("tiny_decode"), max_positions=256)
    return Analyzer(RunConfig(model=ModelConfig(model_id=path, head="causal_lm"), scoring=ScoringConfig(batch_size=4)))


def test_token_f1_is_multiset_overlap():
    assert token_f1([1, 2, 3], [3, 1, 2]) == 1.0
    assert token_f1([1, 2, 3], [4, 5, 6]) == 0.0
    assert token_f1([1, 1, 2], [1, 2, 2]) == pytest.approx(2 / 3)     # multiplicity counts: {1, 2} shared
    assert token_f1([1, 1, 2], [1, 2, 2]) == token_f1([1, 2, 2], [1, 1, 2])
    assert token_f1([1, 1, 1], [1]) == pytest.approx(0.5)             # 2 * 1 / (3 + 1)
    assert token_f1([], []) == 0.0


def test_nucleus_keeps_the_smallest_set_that_crosses_p():
    torch = pytest.importorskip("torch")
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    kept = nucleus_filter(probs, 0.95)
    assert kept[0, 3] == 0 and (kept[0, :3] > 0).all()                # 0.5 + 0.3 = 0.8 < 0.95 -> the 0.15 crosses and is kept
    assert float(kept.sum()) == pytest.approx(1.0)
    assert torch.allclose(kept[0, :3], probs[0, :3] / 0.95)
    assert torch.equal(nucleus_filter(probs, 1.0), probs)
    tiny = nucleus_filter(probs, 1e-6)
    assert tiny[0, 0] == 1.0 and float(tiny[0, 1:].sum()) == 0.0        # the top-1 token always survives
    unsorted = torch.tensor([[0.05, 0.15, 0.5, 0.3], [0.25, 0.25, 0.25, 0.25]])
    out = nucleus_filter(unsorted, 0.6)
    assert out[0, 2] > 0 and out[0, 3] > 0 and out[0, 0] == 0 and out[0, 1] == 0
    assert (out[1] > 0).sum() == 3 and float(out[1].sum()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        nucleus_filter(probs, 0.0)


def test_greedy_equals_full_recompute_and_the_state_service(analyzer):
    states = analyzer.states
    context = states.encode("The quick brown fox jumps over the lazy dog. The dog")
    bos = states.bos_id()
    cached = greedy_continuation(analyzer.models, context, L=6, bos_id=bos, kv_cache=True)
    plain = greedy_continuation(analyzer.models, context, L=6, bos_id=bos, kv_cache=False)
    assert cached.shape == (6,) and np.array_equal(cached, plain)
    # Token t is the argmax of the state at the last position of context + greedy[:t].
    for t in range(6):
        seq = context + [int(x) for x in cached[:t]]
        [res] = states.states([seq], [[len(seq) - 1]])
        assert int(np.argmax(res.states[0])) == int(cached[t])


def test_sampling_is_seeded_paired_and_bounded(analyzer):
    torch = pytest.importorskip("torch")
    states = analyzer.states
    context = states.encode("The quick brown fox jumps over the lazy dog.")
    bos = states.bos_id()
    torch_before = torch.get_rng_state().clone()
    np_before = np.random.get_state()[1].copy()
    kwargs = dict(K=4, L=6, top_p=0.95, temperature=1.0, bos_id=bos)
    a = sample_continuations(analyzer.models, context, seed=7, **kwargs)
    b = sample_continuations(analyzer.models, context, seed=7, **kwargs)
    c = sample_continuations(analyzer.models, context, seed=8, **kwargs)
    assert a.shape == (4, 6) and np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert (a >= 0).all() and (a < states.vocab_size).all()
    assert torch.equal(torch.get_rng_state(), torch_before)            # the global RNGs are untouched
    assert np.array_equal(np.random.get_state()[1], np_before)
    greedy = greedy_continuation(analyzer.models, context, L=6, bos_id=bos)
    near_greedy = sample_continuations(analyzer.models, context, seed=1, K=2, L=6, top_p=1e-6, temperature=1.0, bos_id=bos)
    assert np.array_equal(near_greedy[0], greedy) and np.array_equal(near_greedy[1], greedy)
    no_cache = sample_continuations(analyzer.models, context, seed=7, kv_cache=False, **kwargs)
    assert np.array_equal(no_cache, a)


def test_context_too_long_raises(analyzer):
    states = analyzer.states
    context = states.encode("the dog " * 20)
    with pytest.raises(ValueError, match="exceeds the length limit 30"):
        greedy_continuation(analyzer.models, context, L=5, bos_id=states.bos_id(), max_length=30)
    with pytest.raises(ValueError, match="empty"):
        greedy_continuation(analyzer.models, [], L=5, bos_id=None)


def test_first_diff_and_overlap_stats_by_hand():
    assert first_diff([1, 2, 3], [1, 2, 4], cap=30) == 2
    assert first_diff([1, 2, 3], [9, 2, 3], cap=30) == 0
    assert first_diff([1, 2, 3], [1, 2, 3], cap=30) == 3
    assert first_diff(list(range(40)), list(range(40)), cap=30) == 30
    orig = np.array([[1, 2, 3], [1, 2, 4], [1, 5, 6], [7, 8, 9]])
    spliced = np.array([[1, 2, 3], [4, 4, 4], [1, 2, 9], [9, 9, 9]])
    stats = overlap_stats(orig, spliced)
    within = np.mean([token_f1(orig[a], orig[b]) for a in range(4) for b in range(a + 1, 4)])
    cross = np.mean([token_f1(o, s) for o in orig for s in spliced])
    assert stats["n_within_pairs"] == 6 and stats["n_cross_pairs"] == 16
    assert stats["within"] == pytest.approx(within) and stats["cross"] == pytest.approx(cross)
    assert stats["sample_overlap"] == pytest.approx(cross / within) and not stats["collapsed"]
    same = np.array([[1, 2, 3]] * 4)
    assert overlap_stats(same, same)["within"] == 1.0
    disjoint = np.array([[1, 2], [3, 4], [5, 6]])
    collapsed = overlap_stats(disjoint, disjoint)
    assert collapsed["within"] == 0.0 and collapsed["collapsed"] and np.isnan(collapsed["sample_overlap"])
