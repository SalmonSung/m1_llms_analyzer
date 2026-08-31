"""Layer resolution: the indexing convention must be exact and loudly validated."""

import pytest

from tests.conftest import TINY_LAYERS


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
    from tests.conftest import TINY_MAX_POSITIONS

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


def test_vision_models_are_rejected(loaded_model):
    from m1_analyzer.services.model_service import UnsupportedArchitectureError

    with pytest.raises(UnsupportedArchitectureError, match="multimodal/vision"):
        loaded_model._reject_unsupported(_FakeConfig(vision_config={"hidden_size": 8}))


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
