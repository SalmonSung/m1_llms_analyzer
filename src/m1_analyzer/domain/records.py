"""The data passed between services.

These types are the contract: the inference service produces them, the storage
service consumes them, and neither imports the other. They hold plain numpy
arrays (not torch tensors) so nothing downstream needs a live CUDA context or a
particular torch version to read a result.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SCHEMA_VERSION = "1.0"


def make_item_id(text: str) -> str:
    """Stable, content-derived id.

    Content hashing (rather than a positional index) means the same input gets
    the same id across runs and across files, which is what makes two result
    files joinable and duplicates detectable.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class LayerState:
    """One layer's hidden state for one input."""

    #: Absolute index into the model's hidden_states tuple (0 == embeddings).
    index: int
    #: The spec that selected it, e.g. "-1" or "12" -- kept so the JSON is
    #: readable in the caller's own terms as well as absolute terms.
    requested: str
    #: (hidden,) when pooled, (tokens, hidden) when pooling="none".
    values: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.values.shape)


@dataclass
class ExtractionRecord:
    """Everything extracted for a single input string."""

    id: str
    text: str
    token_count: int
    layers: dict[str, LayerState]
    truncated: bool = False
    #: Token count before truncation; equals token_count when not truncated.
    original_token_count: int | None = None
    tokens: list[str] | None = None

    def layer(self, key: str | int) -> np.ndarray:
        """Convenience accessor: `record.layer(-1)` -> the array."""
        return self.layers[str(key)].values

    @property
    def total_floats(self) -> int:
        return sum(int(np.prod(s.values.shape)) for s in self.layers.values())


@dataclass
class ExtractionFailure:
    """A single input that could not be processed, kept instead of raising.

    A long batch must not lose 9,999 good results because item 5,000 was bad, so
    failures travel alongside the successes and land in the output JSON.
    """

    id: str
    text_preview: str
    error_type: str
    error: str
    index: int | None = None


@dataclass
class BatchResult:
    """Successes and failures from one invoke/batch call, in input order."""

    records: list[ExtractionRecord] = field(default_factory=list)
    failures: list[ExtractionFailure] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    #: The *effective* extraction settings for this call, including any per-call
    #: overrides. Carried here so `save()` stamps the manifest with what actually
    #: ran rather than with the analyzer's default config.
    extraction: dict[str, Any] | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, i: int) -> ExtractionRecord:
        return self.records[i]

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def total_floats(self) -> int:
        return sum(r.total_floats for r in self.records)


@dataclass
class RunManifest:
    """Provenance stamped into every output file.

    Without this, two JSON files are indistinguishable a week later. It answers:
    which model + revision, on what hardware, in which dtype, which layers, which
    pooling, which seed, which library versions, and when.
    """

    run_id: str
    created_at: str
    model_id: str
    revision: str | None
    architecture: str | None
    device: str
    dtype: str
    layers_requested: Any
    layers_resolved: list[int]
    layer_index_convention: str
    pooling: str
    max_length: int
    seed: int
    num_hidden_layers: int
    hidden_size: int
    library_versions: dict[str, str]
    config_fingerprint: str
    schema_version: str = SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class WrittenPaths:
    """What `save()` produced on disk."""

    json_path: str
    npy_path: str | None = None

    def __str__(self) -> str:
        return self.json_path if self.npy_path is None else f"{self.json_path} (+ {self.npy_path})"
