"""NextTokenStateService on the tiny model: exact vectors, BOS, batching, guards."""

import numpy as np
import pytest
import torch

from m1_analyzer import Analyzer, ModelConfig, RunConfig, ScoringConfig
from m1_analyzer.services.model_service import ModelService
from m1_analyzer.services.state_service import NextTokenStateService


@pytest.fixture(scope="module")
def lm(tiny_model_path) -> ModelService:
    return ModelService(ModelConfig(model_id=tiny_model_path, head="causal_lm")).load()


@pytest.fixture
def svc(lm) -> NextTokenStateService:
    return NextTokenStateService(lm, ScoringConfig(batch_size=4))


def _manual(lm, ids, bos=True):
    full = ([lm.tokenizer.eos_token_id] if bos else []) + list(ids)
    with torch.no_grad():
        logits = lm.model(input_ids=torch.tensor([full])).logits[0].float()
    return torch.log_softmax(logits, dim=-1).numpy()


def test_states_match_manual_log_softmax(svc, lm):
    ids = svc.encode("hello world. the quick fox jumps.")
    [res] = svc.states([ids], [[0, 2, len(ids) - 1]], return_token_logprobs=True)
    ref = _manual(lm, ids)
    assert res.states.shape == (3, svc.vocab_size)
    assert np.allclose(np.exp(res.states).sum(axis=1), 1.0, atol=1e-5)
    for k, p in enumerate([0, 2, len(ids) - 1]):
        assert np.allclose(res.states[k], ref[p + 1], atol=1e-4)  # BOS shifts the logit row by one
    manual_lp = ref[:-1][np.arange(len(ids)), ids]
    assert res.n_tokens == len(ids)
    assert np.allclose(res.token_logprobs, manual_lp, atol=1e-4)


def test_mean_logprob_agrees_with_the_scorer(tiny_model_path):
    a = Analyzer(RunConfig(model=ModelConfig(model_id=tiny_model_path, head="causal_lm")))
    text = "the quick brown fox jumps over the lazy dog."
    ids = a.states.encode(text)
    [res] = a.next_token_states([ids], [[1]], return_token_logprobs=True)
    assert res.mean_logprob == pytest.approx(a.score_one(text).mean_logprob, abs=1e-5)


def test_bos_policy_none_has_no_state_at_position_zero(lm):
    svc = NextTokenStateService(lm, ScoringConfig(bos_policy="none"))
    ids = svc.encode("the quick brown fox")
    [res] = svc.states([ids], [[1, 3]], return_token_logprobs=True)
    ref = _manual(lm, ids, bos=False)
    assert np.allclose(res.states[0], ref[1], atol=1e-4)
    assert len(res.token_logprobs) == len(ids) - 1
    with pytest.raises(ValueError, match="position 0"):
        svc.states([ids], [[0]])


def test_batched_states_equal_single_states_and_prefixes_agree(svc):
    a = svc.encode("hello world. the quick fox jumps over the dog.")
    b = svc.encode("a b c.")
    single = [svc.states([a], [[0, 4]])[0], svc.states([b], [[1]])[0]]
    batched = svc.states([a, b], [[0, 4], [1]], batch_size=2)
    for s, t in zip(single, batched):
        assert np.allclose(s.states, t.states, atol=1e-5)
    # Causal attention: a prefix's states do not depend on what follows it.
    [prefix] = svc.states([a[:5]], [[0, 4]])
    assert np.allclose(prefix.states, batched[0].states, atol=1e-5)


def test_encode_with_offsets_and_decode_round_trip(svc):
    text = "hello world. the quick fox jumps."
    ids, offsets = svc.encode_with_offsets(text)
    assert len(ids) == len(offsets)
    assert text[offsets[2][0]:offsets[2][1]] == "."
    assert svc.encode(svc.decode(ids)) == ids


def test_guards(svc):
    ids = svc.encode("hello world")
    with pytest.raises(ValueError, match="same length"):
        svc.states([ids], [[0], [1]])
    with pytest.raises(ValueError, match="outside"):
        svc.states([ids], [[5]])
    with pytest.raises(ValueError, match="empty"):
        svc.states([[]], [[]])
    with pytest.raises(ValueError, match="max_length"):
        svc.states([ids * 40], [[0]])


def test_base_head_refuses(tiny_model_path):
    a = Analyzer(RunConfig(model=ModelConfig(model_id=tiny_model_path, head="base")))
    assert a.states is None
    with pytest.raises(RuntimeError, match="LM head"):
        a.next_token_states([[1, 2]], [[0]])
