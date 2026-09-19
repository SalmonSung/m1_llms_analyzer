"""LogProbService on the tiny model: exact numbers, BOS, padding, failures, OOM."""

import numpy as np
import pytest
import torch

from m1_analyzer import Analyzer, ModelConfig, RunConfig, ScoringConfig, StorageConfig
from m1_analyzer.services.model_service import ModelService, UnsupportedArchitectureError
from m1_analyzer.services.scoring_service import LogProbService
from m1_analyzer.testing import TINY_HIDDEN


@pytest.fixture(scope="module")
def lm(tiny_model_path) -> ModelService:
    return ModelService(ModelConfig(model_id=tiny_model_path, head="causal_lm")).load()


@pytest.fixture
def scorer(lm) -> LogProbService:
    return LogProbService(lm, ScoringConfig(batch_size=4))


def _manual(lm, text, bos=True):
    tok, model = lm.tokenizer, lm.model
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if bos:
        ids = [tok.eos_token_id] + ids
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0].float()
    lp = torch.log_softmax(logits[:-1], dim=-1).gather(-1, torch.tensor(ids[1:]).unsqueeze(-1))
    return float(lp.sum()), len(ids) - 1


def test_causal_head_loads_and_exposes_logits(lm):
    assert type(lm.model).__name__.endswith("LMHeadModel")
    assert lm.metadata()["head"] == "causal_lm"


def test_score_matches_manual_log_softmax(scorer, lm):
    text = "the quick brown fox"
    got = scorer.score_one(text)
    total, n = _manual(lm, text)
    assert got.n_tokens == n == 4  # every real token predicted, thanks to the BOS
    assert got.sum_logprob == pytest.approx(total, abs=1e-4)
    assert got.mean_logprob == pytest.approx(total / n, abs=1e-4)


def test_bos_policy_none_predicts_from_the_second_token(scorer, lm):
    text = "the quick brown fox"
    got = scorer.score_one(text, bos_policy="none")
    total, n = _manual(lm, text, bos=False)
    assert got.n_tokens == 3
    assert got.sum_logprob == pytest.approx(total, abs=1e-4)


def test_batched_scores_equal_single_scores(scorer):
    texts = ["hello", "the quick brown fox jumps over the lazy dog", "a b c", "model layer"]
    batched = scorer.score(texts)
    assert [s.text for s in batched] == texts  # input order despite length sorting
    for text, item in zip(texts, batched):
        assert item.sum_logprob == pytest.approx(scorer.score_one(text).sum_logprob, abs=1e-4)


def test_token_logprobs_sum_to_the_total(scorer):
    item = scorer.score(["the quick brown fox"], return_token_logprobs=True)[0]
    assert item.token_logprobs.shape == (item.n_tokens,)
    assert float(item.token_logprobs.sum()) == pytest.approx(item.sum_logprob, abs=1e-5)


def test_non_finite_scores_become_failures(scorer, monkeypatch):
    original = scorer._forward

    def poisoned(texts, *args, **kwargs):
        out = original(texts, *args, **kwargs)
        for item in out:
            if item.text == "hello world":
                item.sum_logprob = float("nan")
        return out

    monkeypatch.setattr(scorer, "_forward", poisoned)
    result = scorer.score(["hello world", "a b c"])
    assert len(result.scores) == 1 and result.scores[0].text == "a b c"
    assert len(result.failures) == 1 and "float32" in result.failures[0].error
    with pytest.raises(ValueError, match="non-finite"):
        scorer.score(["hello world"], continue_on_error=False)


def test_oom_halves_the_batch(scorer, monkeypatch):
    original = scorer._forward
    seen = []

    def oom_on_big(texts, *args, **kwargs):
        seen.append(len(texts))
        if len(texts) > 1:
            raise MemoryError("simulated OOM")
        return original(texts, *args, **kwargs)

    monkeypatch.setattr(scorer, "_forward", oom_on_big)
    result = scorer.score(["hello world", "a b c", "one two", "three four five"])
    assert result.ok and len(result) == 4
    assert max(seen) > 1 and min(seen) == 1


def test_single_token_without_bos_is_a_clear_failure(scorer):
    result = scorer.score(["hello"], bos_policy="none")
    assert not result.ok and "at least one token" in result.failures[0].error


def test_input_validation(scorer):
    with pytest.raises(TypeError):
        scorer.score("hello")
    with pytest.raises(ValueError):
        scorer.score([])
    with pytest.raises(ValueError, match="empty"):
        scorer.score(["  "])
    with pytest.raises(TypeError, match="Unknown scoring option"):
        scorer.score(["hello"], batch_sizee=2)


def test_base_head_cannot_score(loaded_model):
    with pytest.raises(RuntimeError, match="causal_lm"):
        LogProbService(loaded_model).score(["hello"])


def test_analyzer_for_scoring_and_hidden_states_coexist(tiny_model_path, tmp_path):
    analyzer = Analyzer(RunConfig(
        model=ModelConfig(model_id=tiny_model_path, head="causal_lm"),
        storage=StorageConfig(output_dir=str(tmp_path)),
    ))
    assert analyzer.describe()["head"] == "causal_lm"
    assert analyzer.score_one("hello world").n_tokens == 2
    assert analyzer.invoke("hello world").layer(-1).shape == (TINY_HIDDEN,)
    assert analyzer.manifest().extra["head"] == "causal_lm"


def test_base_analyzer_explains_how_to_score(analyzer):
    with pytest.raises(RuntimeError, match="for_scoring"):
        analyzer.score(["hello"])


def test_causal_head_rejects_models_without_one(tiny_model_path, monkeypatch):
    service = ModelService(ModelConfig(model_id=tiny_model_path, head="causal_lm"))

    class NotCausal:
        model_type = "not-a-decoder"

    with pytest.raises(UnsupportedArchitectureError, match="no causal language-model head"):
        service._require_causal_lm(NotCausal())


def test_causal_head_accepts_multimodal_wrapper_via_text_config(tiny_model_path):
    """A vision+text wrapper's own model_type may be unknown; its text_config decides."""
    service = ModelService(ModelConfig(model_id=tiny_model_path, head="causal_lm"))

    class TextPart:
        model_type = "gpt2"

    class Wrapper:
        model_type = "some-new-multimodal-wrapper"
        text_config = TextPart()

    service._require_causal_lm(Wrapper())  # must not raise


def test_halved_batch_size_sticks_across_calls_and_grows_back(lm):
    scorer = LogProbService(lm, ScoringConfig(batch_size="auto", max_batch_size=8))
    scorer.batch_sizer = None
    original = scorer._forward
    seen = []

    def oom_above_two(texts, *args, **kwargs):
        seen.append(len(texts))
        if len(texts) > 2:
            raise MemoryError("simulated OOM")
        return original(texts, *args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(scorer, "_forward", oom_above_two)
    try:
        texts = ["one two", "three four five", "six seven", "eight nine ten", "eleven twelve"]
        assert scorer.score(texts).ok
        assert scorer.batch_sizer.current == 2 and seen[0] > 2  # the CPU default was tried once
        seen.clear()
        assert scorer.score(texts).ok
        assert max(seen) == 2  # the second call starts at the learned size: no OOM at all
        for _ in range(scorer.batch_sizer.grow_after * 2):
            scorer.batch_sizer.success()
        assert scorer.batch_sizer.current == 2  # growth never retries a size that failed
    finally:
        monkeypatch.undo()


def test_calibration_finds_the_largest_batch_that_fits(lm):
    scorer = LogProbService(lm, ScoringConfig(max_batch_size=64))
    original = scorer._forward
    tried = []

    def oom_at_32(texts, *args, **kwargs):
        tried.append(len(texts))
        if len(texts) >= 32:
            raise MemoryError("simulated OOM")
        return original(texts, *args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(scorer, "_forward", oom_at_32)
    try:
        assert scorer.calibrate_batch_size(["a b c", "one two three four five"], start=8) == 16
        assert tried == [8, 16, 32]
        assert scorer.batch_sizer.current == 16 and scorer.batch_sizer.ceiling == 16
        tried.clear()
        assert scorer.score(["a b", "c d e", "f g"]).ok and max(tried) <= 16
    finally:
        monkeypatch.undo()
    # Without any OOM the probe stops at max_batch_size.
    scorer = LogProbService(lm, ScoringConfig(max_batch_size=16))
    assert scorer.calibrate_batch_size(["a b c"], start=4) == 16
    assert scorer.batch_sizer.ceiling == 16


def test_pinned_batch_size_overrides_reset_the_sizer(scorer):
    scorer.score(["a b", "c d", "e f", "g h", "i j"])
    assert scorer.batch_sizer.current == 4 and scorer.batch_sizer.maximum == 4
    scorer.score(["a b", "c d", "e f"], batch_size=2)
    assert scorer.batch_sizer.maximum == 2
