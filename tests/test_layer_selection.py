"""Layer resolution: the indexing convention must be exact and loudly validated."""

import pytest

from m1_analyzer.testing import TINY_LAYERS, TINY_MAX_POSITIONS


def test_last_layer_is_the_final_block(loaded_model):
    assert loaded_model.resolve_layers(-1) == [("-1", TINY_LAYERS)]


def test_positive_and_negative_indices_agree(loaded_model):
    assert loaded_model.resolve_layers(TINY_LAYERS)[0][1] == loaded_model.resolve_layers(-1)[0][1]


def test_index_zero_is_the_embedding_output(loaded_model):
    assert loaded_model.resolve_layers(0) == [("0", 0)]


def test_all_covers_embeddings_plus_every_block(loaded_model):
    resolved = loaded_model.resolve_layers("all")
    assert [i for _, i in resolved] == list(range(TINY_LAYERS + 1))


def test_middle_keyword(loaded_model):
    assert loaded_model.resolve_layers("middle") == [("middle", TINY_LAYERS // 2)]


def test_mixed_sequence_preserves_order(loaded_model):
    resolved = loaded_model.resolve_layers([-1, "middle", 1])
    assert [label for label, _ in resolved] == ["-1", "middle", "1"]
    assert [i for _, i in resolved] == [TINY_LAYERS, TINY_LAYERS // 2, 1]


def test_duplicate_indices_are_collapsed(loaded_model):
    resolved = loaded_model.resolve_layers([-1, TINY_LAYERS, "last"])
    assert len(resolved) == 1


@pytest.mark.parametrize("spec", [TINY_LAYERS + 1, -(TINY_LAYERS + 2), 999])
def test_out_of_range_raises_with_range_in_message(loaded_model, spec):
    with pytest.raises(ValueError, match="outside the valid range"):
        loaded_model.resolve_layers(spec)


def test_unknown_keyword_raises(loaded_model):
    with pytest.raises(ValueError, match="Unknown layer keyword"):
        loaded_model.resolve_layers("penultimate")


def test_effective_max_length_respects_model_and_cap(loaded_model):
    from m1_analyzer.testing import TINY_MAX_POSITIONS

    assert loaded_model.effective_max_length(None, 4096) == TINY_MAX_POSITIONS
    assert loaded_model.effective_max_length(8, 4096) == 8
    # A request beyond the model's context is clamped, not honoured.
    assert loaded_model.effective_max_length(10_000, 4096) == TINY_MAX_POSITIONS
    # The cap wins over everything.
    assert loaded_model.effective_max_length(None, 4) == 4


def test_metadata_reports_provenance(loaded_model):
    meta = loaded_model.metadata()
    assert meta["num_hidden_layers"] == TINY_LAYERS
    assert meta["device"] == "cpu"
    assert meta["dtype"] == "float32"
    assert meta["pad_token_substituted"] is True  # tiny tokenizer ships no PAD
    assert "torch" in meta["library_versions"]


class _FakeConfig:
    """Minimal stand-in for a transformers config object."""

    def __init__(self, **kwargs):
        self.architectures = kwargs.pop("architectures", ["FakeModel"])
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, name):  # unset attributes behave like absent config keys
        return None


def test_encoder_decoder_models_are_rejected(loaded_model):
    """T5/BART hidden_states are the encoder's -- saving them as 'the last layer' would lie."""
    from m1_analyzer.services.model_service import UnsupportedArchitectureError

    with pytest.raises(UnsupportedArchitectureError, match="encoder-decoder"):
        loaded_model._reject_unsupported(_FakeConfig(is_encoder_decoder=True, architectures=["T5Model"]))


def test_multimodal_models_are_accepted_in_text_only_mode(loaded_model, caplog):
    """Gemma 3/4, Qwen-VL, ...: called with text only, the wrapper *is* its language model."""
    import logging

    loaded_model._multimodal = False
    with caplog.at_level(logging.WARNING):
        loaded_model._reject_unsupported(
            _FakeConfig(vision_config={"hidden_size": 8}, text_config=_FakeConfig(model_type="gemma4_text"))
        )
    assert loaded_model._multimodal is True
    assert any("multimodal" in r.getMessage() and "vision_config" in r.getMessage() for r in caplog.records)
    loaded_model._multimodal = False


def test_vision_encoder_decoder_models_are_still_rejected(loaded_model):
    """TrOCR/Donut are encoder-decoder in disguise; the vision flag must not let them through."""
    from m1_analyzer.services.model_service import UnsupportedArchitectureError

    with pytest.raises(UnsupportedArchitectureError, match="encoder-decoder"):
        loaded_model._reject_unsupported(_FakeConfig(is_vision_encoder_decoder=True, vision_config={}))


def test_nested_encoder_decoder_text_config_is_rejected(loaded_model):
    from m1_analyzer.services.model_service import UnsupportedArchitectureError

    with pytest.raises(UnsupportedArchitectureError, match="encoder-decoder"):
        loaded_model._reject_unsupported(_FakeConfig(text_config=_FakeConfig(is_encoder_decoder=True)))


def test_layer_and_size_attributes_come_from_text_config(loaded_model, monkeypatch):
    """A composite config carries no top-level layer count; it lives under text_config."""
    from m1_analyzer.services import model_service as ms

    wrapper = _FakeConfig(
        model_type="gemma4",
        vision_config={"hidden_size": 8},
        text_config=_FakeConfig(model_type="gemma4_text", num_hidden_layers=3, hidden_size=16, max_position_embeddings=64),
    )
    monkeypatch.setattr(type(loaded_model), "hf_config", property(lambda self: wrapper))
    assert loaded_model.num_hidden_layers == 3
    assert loaded_model.hidden_size == 16
    # Context length is the smaller of tokenizer and text_config limits; the tiny
    # tokenizer's limit is TINY_MAX_POSITIONS, so 64 must win only if it is smaller.
    assert loaded_model.effective_max_length(None, 4096) == min(64, TINY_MAX_POSITIONS)
    assert loaded_model.resolve_layers("all") == [(str(i), i) for i in range(4)]
    meta = loaded_model.metadata()
    assert meta["text_model_type"] == "gemma4_text"
    assert meta["model_type"] == "gemma4"


def test_text_config_as_plain_dict_is_readable(loaded_model):
    from m1_analyzer.services.model_service import ModelService, _cfg_get

    cfg = _FakeConfig(text_config={"num_hidden_layers": 5, "model_type": "llama"})
    assert _cfg_get(ModelService._text_config(cfg), "num_hidden_layers") == 5
    # No text_config: the config itself is the text config.
    plain = _FakeConfig(num_hidden_layers=7)
    assert ModelService._text_config(plain) is plain


def test_decoder_only_models_are_accepted(loaded_model):
    loaded_model._reject_unsupported(_FakeConfig(is_encoder_decoder=False))


def test_gated_repo_error_names_the_colab_secret(loaded_model):
    """The most likely first-run failure gets the most actionable message."""
    explained = loaded_model._explain_load_failure(RuntimeError("401 Client Error: gated repo"))
    assert "HF_TOKEN" in str(explained)
    assert "huggingface.co/settings/tokens" in str(explained)


def test_missing_repo_error_is_explained(loaded_model):
    explained = loaded_model._explain_load_failure(RuntimeError("404 Client Error"))
    assert "not found on the Hugging Face Hub" in str(explained)


def test_trust_remote_code_error_is_explained(loaded_model):
    explained = loaded_model._explain_load_failure(
        RuntimeError("Loading this model requires trust_remote_code=True")
    )
    assert "trust_remote_code=True" in str(explained)
