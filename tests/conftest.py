"""Shared fixtures.

Everything here is offline: `build_tiny_local_model` writes a randomly-initialised
GPT-2-shaped model to a temp dir, so the suite exercises the real
`from_pretrained` path without touching the Hugging Face Hub.
"""

from __future__ import annotations

import pytest

from m1_analyzer import Analyzer, ExtractionConfig, ModelConfig, RunConfig, StorageConfig
from m1_analyzer.services.model_service import ModelService
from m1_analyzer.testing import build_tiny_local_model

TINY_LAYERS = 4
TINY_HIDDEN = 16
TINY_MAX_POSITIONS = 32


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory) -> str:
    """A tiny local model, built once for the whole session."""
    directory = tmp_path_factory.mktemp("tiny_model")
    return build_tiny_local_model(
        directory, num_layers=TINY_LAYERS, hidden_size=TINY_HIDDEN,
        max_positions=TINY_MAX_POSITIONS,
    )


@pytest.fixture(scope="session")
def loaded_model(tiny_model_path) -> ModelService:
    return ModelService(ModelConfig(model_id=tiny_model_path)).load()


@pytest.fixture
def make_analyzer(tiny_model_path, tmp_path):
    """Factory for an Analyzer over the tiny model, writing into a temp dir."""

    def _make(**extraction_kwargs) -> Analyzer:
        storage_kwargs = {
            key: extraction_kwargs.pop(key)
            for key in list(extraction_kwargs)
            if key in vars(StorageConfig())
        }
        return Analyzer(
            RunConfig(
                model=ModelConfig(model_id=tiny_model_path),
                extraction=ExtractionConfig(**extraction_kwargs),
                storage=StorageConfig(output_dir=str(tmp_path / "outputs"), **storage_kwargs),
            )
        )

    return _make


@pytest.fixture
def analyzer(make_analyzer) -> Analyzer:
    return make_analyzer()


@pytest.fixture
def sample_texts() -> list[str]:
    return ["hello world", "the quick brown fox jumps over the lazy dog", "a b c"]
