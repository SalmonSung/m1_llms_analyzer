"""The adaptive batch size: sticky halving, growth after clean batches, ceilings."""

import pytest

from m1_analyzer.utils.batching import AdaptiveBatchSize, run_with_oom_halving


def test_shrink_is_sticky_and_caps_growth_below_the_failed_size():
    sizer = AdaptiveBatchSize(64, maximum=512, grow_after=3)
    assert sizer.current == 64 and sizer.ceiling == 512
    assert sizer.shrink(64) == 32 and sizer.ceiling == 32 and sizer.shrinks == 1
    assert list(map(len, sizer.chunks(list(range(70))))) == [32, 32, 6]
    for _ in range(10):
        sizer.success()
    assert sizer.current == 32 and sizer.grows == 0  # never retries 64
    assert sizer.shrink(32) == 16
    for _ in range(3):
        sizer.success()
    assert sizer.current == 16  # ceiling is now 16 as well


def test_growth_doubles_after_a_streak_up_to_the_maximum():
    sizer = AdaptiveBatchSize(8, maximum=64, grow_after=2)
    seen = []
    for _ in range(12):
        seen.append(sizer.current)
        sizer.success()
    assert seen == [8, 8, 16, 16, 32, 32, 64, 64, 64, 64, 64, 64]
    assert sizer.grows == 3
    sizer.shrink()  # default: the current size failed
    assert sizer.current == 32 and sizer.ceiling == 32


def test_pinned_size_never_grows_and_chunks_follow_the_current_size():
    sizer = AdaptiveBatchSize(4)
    assert sizer.maximum == 4
    chunks = []
    for chunk in sizer.chunks(list(range(10))):
        chunks.append(chunk)
        if len(chunks) == 1:
            sizer.shrink(4)
    assert chunks == [[0, 1, 2, 3], [4, 5], [6, 7], [8, 9]]
    with pytest.raises(ValueError):
        AdaptiveBatchSize(0)


def test_run_with_oom_halving_reports_every_oom_to_the_sizer():
    sizer = AdaptiveBatchSize(8, maximum=8)
    results, errors = {}, []

    def forward(indices):
        if len(indices) > 2:
            raise MemoryError("simulated")
        return [i * 10 for i in indices]

    run_with_oom_halving(list(range(8)), forward, results.__setitem__,
                         lambda i, exc, oom: errors.append(i), sizer=sizer)
    assert results == {i: i * 10 for i in range(8)} and errors == []
    assert sizer.current == 2 and sizer.shrinks == 2  # 8 -> 4 -> 2, both remembered
