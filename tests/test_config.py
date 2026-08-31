"""Config validation: bad settings must fail immediately, not deep in a forward pass."""

import pytest

from m1_analyzer import ExtractionConfig, ModelConfig, RunConfig, StorageConfig


def test_defaults_are_valid():
    config = RunConfig()
    assert config.extraction.pooling == "last_token"
    assert config.extraction.layers == -1
    assert config.storage.npy_mode == "auto"


@pytest.mark.parametrize("pooling", ["last_token", "mean", "cls", "none"])
def test_all_pooling_modes_accepted(pooling):
    assert ExtractionConfig(pooling=pooling).pooling == pooling


def test_unknown_pooling_rejected():
    with pytest.raises(ValueError, match="pooling must be one of"):
        ExtractionConfig(pooling="max")


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="batch_size"):
        ExtractionConfig(batch_size=0)


def test_empty_model_id_rejected():
    with pytest.raises(ValueError, match="model_id"):
        ModelConfig(model_id="   ")


def test_unknown_npy_mode_rejected():
    with pytest.raises(ValueError, match="npy_mode"):
        StorageConfig(npy_mode="maybe")


@pytest.mark.parametrize("spec", [-1, 0, 5, "all", "last", "middle", [-1, "middle"], (0, 1)])
def test_valid_layer_specs(spec):
    assert ExtractionConfig(layers=spec).layers == spec


@pytest.mark.parametrize("spec", ["penultimate", [], True, 3.5])
def test_invalid_layer_specs(spec):
    with pytest.raises((ValueError, TypeError)):
        ExtractionConfig(layers=spec)


def test_fingerprint_is_stable_and_sensitive():
    a, b = RunConfig(), RunConfig()
    assert a.fingerprint() == b.fingerprint()
    c = RunConfig(extraction=ExtractionConfig(pooling="mean"))
    assert c.fingerprint() != a.fingerprint()


def test_token_never_leaves_the_config():
    config = RunConfig(model=ModelConfig(hf_token="hf_supersecret"))
    assert config.to_dict()["model"]["hf_token"] is None
    assert "hf_supersecret" not in str(config.to_dict())
