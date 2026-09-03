"""Service contracts.

Each service depends on these Protocols, never on another service's concrete
class. That is what makes the "microservice" boundary real in-process: a fake
model provider is enough to unit-test inference, and any service could be moved
behind HTTP later by writing a client that satisfies the same Protocol -- no
pipeline code changes.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from ..domain.records import BatchResult, ExtractionRecord, RunManifest, ScoreResult, WrittenPaths


@runtime_checkable
class ModelProvider(Protocol):
    """Owns the tokenizer + model and everything hardware-related."""

    @property
    def model(self) -> Any: ...

    @property
    def tokenizer(self) -> Any: ...

    @property
    def num_hidden_layers(self) -> int: ...

    @property
    def hidden_size(self) -> int: ...

    @property
    def device(self) -> str: ...

    def load(self) -> "ModelProvider": ...

    def resolve_layers(self, spec: Any) -> list[tuple[str, int]]:
        """Map a layer spec to ``[(requested_label, absolute_index), ...]``."""

    def effective_max_length(self, requested: int | None, cap: int) -> int: ...

    def metadata(self) -> dict[str, Any]: ...


@runtime_checkable
class InferenceEngine(Protocol):
    """Turns text into hidden-state records."""

    def invoke(self, text: str, **overrides: Any) -> ExtractionRecord: ...

    def batch(self, texts: Sequence[str], **overrides: Any) -> BatchResult: ...


@runtime_checkable
class SequenceScorer(Protocol):
    """Turns text into log-probabilities (needs a model loaded with the LM head).

    Experiments depend on this seam only, so a dict-backed fake can drive the
    whole analysis in a test, and a remote scorer could replace the local one.
    """

    def score(self, texts: Sequence[str], **overrides: Any) -> ScoreResult: ...


@runtime_checkable
class ResultSink(Protocol):
    """Persists a run."""

    def write(self, manifest: RunManifest, result: BatchResult, name: str | None = None) -> WrittenPaths: ...
