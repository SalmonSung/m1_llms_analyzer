"""invoke/batch behaviour: equivalence, ordering, truncation, failure isolation."""

import numpy as np
import pytest

from m1_analyzer.domain.records import make_item_id
from tests.conftest import TINY_HIDDEN, TINY_LAYERS


def test_invoke_returns_pooled_vector(analyzer):
    record = analyzer.invoke("hello world")
    assert record.layer(-1).shape == (TINY_HIDDEN,)
    assert record.token_count == 2
    assert record.truncated is False
    assert record.id == make_item_id("hello world")


def test_invoke_and_batch_agree_exactly(analyzer):
    """The two APIs share one code path, so their numbers must be identical."""
    text = "the quick brown fox"
    single = analyzer.invoke(text)
    batched = analyzer.batch(["a b c", text, "hello"])
    match = next(r for r in batched if r.text == text)
    np.testing.assert_allclose(single.layer(-1), match.layer(-1), rtol=1e-5, atol=1e-6)


def test_batch_preserves_input_order_despite_length_sorting(analyzer, sample_texts):
    result = analyzer.batch(sample_texts)
    assert [r.text for r in result.records] == sample_texts


def test_batch_size_one_matches_large_batch(make_analyzer, sample_texts):
    small = make_analyzer(batch_size=1).batch(sample_texts)
    large = make_analyzer(batch_size=16).batch(sample_texts)
    for a, b in zip(small.records, large.records):
        np.testing.assert_allclose(a.layer(-1), b.layer(-1), rtol=1e-4, atol=1e-5)


def test_multiple_layers_are_distinct(make_analyzer):
    record = make_analyzer(layers=[0, "middle", -1]).invoke("hello world")
    assert set(record.layers) == {"0", "middle", "-1"}
    assert record.layers["-1"].index == TINY_LAYERS
    assert not np.allclose(record.layer(0), record.layer(-1))


def test_pooling_none_returns_per_token_matrix(make_analyzer):
    record = make_analyzer(pooling="none").invoke("the quick brown fox")
    assert record.layer(-1).shape == (record.token_count, TINY_HIDDEN)


def test_pooling_none_strips_padding_rows(make_analyzer):
    """Per-token output must have exactly one row per real token, never PAD rows."""
    result = make_analyzer(pooling="none").batch(["hello", "the quick brown fox jumps"])
    for record in result.records:
        assert record.layer(-1).shape[0] == record.token_count


def test_include_tokens_lines_up_with_rows(make_analyzer):
    record = make_analyzer(pooling="none", include_tokens=True).invoke("hello world")
    assert record.tokens is not None
    assert len(record.tokens) == record.layer(-1).shape[0]


def test_truncation_is_flagged_not_silent(make_analyzer):
    long_text = " ".join(["hello world"] * 30)
    record = make_analyzer(max_length=8).invoke(long_text)
    assert record.truncated is True
    assert record.token_count == 8
    assert record.original_token_count > 8


def test_per_call_overrides_do_not_mutate_config(analyzer):
    assert analyzer.config.extraction.pooling == "last_token"
    record = analyzer.invoke("hello world", pooling="mean")
    assert record.layer(-1).shape == (TINY_HIDDEN,)
    assert analyzer.config.extraction.pooling == "last_token"


def test_unknown_override_rejected(analyzer):
    with pytest.raises(TypeError, match="Unknown extraction option"):
        analyzer.invoke("hello world", poolingg="mean")


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_empty_input_raises_clear_error(analyzer, bad):
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        analyzer.invoke(bad)


def test_batch_rejects_a_bare_string(analyzer):
    with pytest.raises(TypeError, match="sequence of strings"):
        analyzer.batch("hello world")


def test_non_string_input_rejected(analyzer):
    with pytest.raises(TypeError, match="expected str"):
        analyzer.batch(["ok", 42])


def test_empty_batch_rejected(analyzer):
    with pytest.raises(ValueError, match="No inputs"):
        analyzer.batch([])


def test_one_bad_item_does_not_kill_the_run(analyzer, monkeypatch):
    """A failure is recorded and the other inputs still produce records."""
    original = analyzer.inference._forward
    calls = {"n": 0}

    def flaky(texts, *args, **kwargs):
        calls["n"] += 1
        if len(texts) == 1 and texts[0] == "boom":
            raise RuntimeError("synthetic failure")
        if any(t == "boom" for t in texts):
            raise RuntimeError("synthetic failure")
        return original(texts, *args, **kwargs)

    monkeypatch.setattr(analyzer.inference, "_forward", flaky)
    result = analyzer.batch(["hello world", "boom", "a b c"])
    assert len(result.records) == 2
    assert len(result.failures) == 1
    assert result.failures[0].error_type == "RuntimeError"
    assert result.failures[0].text_preview == "boom"
    assert result.ok is False


def test_oom_falls_back_to_smaller_batches(make_analyzer, monkeypatch, sample_texts):
    """A memory error halves the batch instead of aborting the run."""
    analyzer = make_analyzer(batch_size=4)
    original = analyzer.inference._forward
    seen = []

    def oom_on_big_batches(texts, *args, **kwargs):
        seen.append(len(texts))
        if len(texts) > 1:
            raise MemoryError("simulated OOM")
        return original(texts, *args, **kwargs)

    monkeypatch.setattr(analyzer.inference, "_forward", oom_on_big_batches)
    result = analyzer.batch(sample_texts)
    assert len(result.records) == len(sample_texts)
    assert result.ok
    assert max(seen) > 1 and min(seen) == 1


def test_duplicate_inputs_share_an_id(analyzer):
    result = analyzer.batch(["hello world", "hello world", "a b c"])
    assert result.records[0].id == result.records[1].id
    assert result.records[0].id != result.records[2].id


def test_continue_on_error_false_raises(analyzer, monkeypatch):
    """Opting out of failure isolation must actually propagate the exception."""
    original = analyzer.inference._forward

    def always_fails(texts, *args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(analyzer.inference, "_forward", always_fails)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        analyzer.batch(["hello world"], continue_on_error=False)


def test_invoke_raises_rather_than_recording_a_failure(analyzer, monkeypatch):
    def always_fails(texts, *args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(analyzer.inference, "_forward", always_fails)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        analyzer.invoke("hello world")


def test_result_carries_the_effective_config(analyzer):
    result = analyzer.batch(["hello world"], pooling="mean")
    assert result.extraction["pooling"] == "mean"
    assert analyzer.config.extraction.pooling == "last_token"
