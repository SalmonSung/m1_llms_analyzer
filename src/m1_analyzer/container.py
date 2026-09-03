"""Service container and the `Analyzer` facade.

`Analyzer` is the only object the notebook needs. It wires the services
together from one `RunConfig`, owns the loaded model for the session, and
exposes `invoke` / `batch` / `save` (hidden states) and `score` (log-probabilities,
when the model was loaded with its LM head). Services stay independently constructible --
the container just does the wiring in one place, so swapping an implementation
(a different sink, a remote model provider) is a one-line change here.
"""

from __future__ import annotations

from typing import Any, Sequence

from .config.settings import ExtractionConfig, ModelConfig, RunConfig, StorageConfig
from .domain.records import BatchResult, ExtractionRecord, RunManifest, ScoreResult, SentenceScore, WrittenPaths
from .services.inference_service import InferenceService
from .services.model_service import ModelService, library_versions
from .services.scoring_service import LogProbService
from .services.storage_service import StorageService, make_run_id
from .utils.logging import get_logger
from .utils.seeding import seed_everything

log = get_logger("container")


class Analyzer:
    """Facade over model + inference + storage services."""

    def __init__(
        self,
        config: RunConfig | None = None,
        *,
        model_service: ModelService | None = None,
        inference_service: InferenceService | None = None,
        storage_service: StorageService | None = None,
        scoring_service: LogProbService | None = None,
    ):
        self.config = config or RunConfig()
        seed_everything(self.config.seed, strict=self.config.strict_determinism)

        self.models = model_service or ModelService(self.config.model)
        self.models.load()
        self.inference = inference_service or InferenceService(self.models, self.config.extraction)
        self.storage = storage_service or StorageService(self.config.storage)
        #: Only available when the model carries its LM head (head="causal_lm").
        self.scoring: LogProbService | None = scoring_service
        if self.scoring is None and self.config.model.head == "causal_lm":
            self.scoring = LogProbService(self.models, self.config.scoring)

    # ------------------------------------------------------------ constructors

    @classmethod
    def from_config(cls, config: RunConfig) -> "Analyzer":
        return cls(config)

    @classmethod
    def quick(
        cls,
        model_id: str,
        *,
        layers: Any = -1,
        pooling: str = "last_token",
        output_dir: str = "outputs",
        head: str = "base",
        **model_kwargs: Any,
    ) -> "Analyzer":
        """Shorthand for the common notebook case."""
        return cls(
            RunConfig(
                model=ModelConfig(model_id=model_id, head=head, **model_kwargs),
                extraction=ExtractionConfig(layers=layers, pooling=pooling),
                storage=StorageConfig(output_dir=output_dir),
            )
        )

    @classmethod
    def for_scoring(cls, model_id: str, *, output_dir: str = "outputs", **model_kwargs: Any) -> "Analyzer":
        """Shorthand for an experiment run: loads the LM head so `score()` works."""
        return cls.quick(model_id, output_dir=output_dir, head="causal_lm", **model_kwargs)

    # ---------------------------------------------------------------- pipeline

    def invoke(self, text: str, **overrides: Any) -> ExtractionRecord:
        """Extract hidden states for a single input."""
        return self.inference.invoke(text, **overrides)

    def batch(self, texts: Sequence[str], **overrides: Any) -> BatchResult:
        """Extract hidden states for many inputs (input order preserved)."""
        return self.inference.batch(texts, **overrides)

    def save(
        self,
        result: BatchResult | ExtractionRecord,
        name: str | None = None,
        **overrides: Any,
    ) -> WrittenPaths:
        """Persist a result as JSON (+ sidecar when large). Accepts a single record too."""
        if isinstance(result, ExtractionRecord):
            result = BatchResult(records=[result])
        # The settings that actually produced the result win over the analyzer's
        # defaults; an explicit override here still wins over both.
        effective = {**(result.extraction or {}), **overrides}
        return self.storage.write(self.manifest(**effective), result, name=name)

    def run(self, texts: Sequence[str], name: str | None = None, **overrides: Any):
        """Extract then save in one call. Returns ``(BatchResult, WrittenPaths)``."""
        result = self.batch(texts, **overrides)
        return result, self.save(result, name=name)

    # ----------------------------------------------------------------- scoring

    def score(self, texts: Sequence[str], **overrides: Any) -> ScoreResult:
        """Log-probability of each text (needs ``ModelConfig(head="causal_lm")``)."""
        return self._scorer().score(texts, **overrides)

    def score_one(self, text: str, **overrides: Any) -> SentenceScore:
        return self._scorer().score_one(text, **overrides)

    def _scorer(self) -> LogProbService:
        if self.scoring is None:
            raise RuntimeError(
                "This Analyzer was built with head='base' (hidden states only). Scoring "
                "log-probabilities needs the LM head: use Analyzer.for_scoring(model_id) or "
                "ModelConfig(head='causal_lm')."
            )
        return self.scoring

    # ---------------------------------------------------------------- manifest

    def manifest(self, **overrides: Any) -> RunManifest:
        """Provenance for the current configuration."""
        from datetime import datetime, timezone

        extraction = self.config.extraction
        meta = self.models.metadata()
        resolved = self.models.resolve_layers(overrides.get("layers", extraction.layers))
        max_length = self.models.effective_max_length(
            overrides.get("max_length", extraction.max_length),
            overrides.get("max_length_cap", extraction.max_length_cap),
        )
        return RunManifest(
            run_id=make_run_id(),
            created_at=datetime.now(timezone.utc).isoformat(),
            model_id=meta["model_id"],
            revision=meta["revision"],
            architecture=meta["architecture"],
            device=meta["device"],
            dtype=meta["dtype"],
            layers_requested=_jsonable(overrides.get("layers", extraction.layers)),
            layers_resolved=[index for _, index in resolved],
            layer_index_convention=(
                "hidden_states[0] is the embedding output; hidden_states[i] is the output of "
                "transformer block i for i in 1..num_hidden_layers"
            ),
            pooling=overrides.get("pooling", extraction.pooling),
            max_length=max_length,
            seed=self.config.seed,
            num_hidden_layers=meta["num_hidden_layers"],
            hidden_size=meta["hidden_size"],
            library_versions=meta["library_versions"],
            config_fingerprint=self.config.fingerprint(),
            extra={
                "layer_labels": [label for label, _ in resolved],
                "device_info": meta["device_info"],
                "pad_token_substituted": meta["pad_token_substituted"],
                "head": meta["head"],
                "model_context_length": meta["context_length"],
                "strict_determinism": self.config.strict_determinism,
            },
        )

    # ----------------------------------------------------------------- teardown

    def unload(self) -> None:
        """Free the model. Useful in Colab before loading a second, larger model."""
        self.models._model = None  # noqa: SLF001 - deliberate teardown
        self.models._tokenizer = None  # noqa: SLF001
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass
        log.info("Model unloaded and caches cleared.")

    def describe(self) -> dict[str, Any]:
        """One-glance summary for printing in a notebook cell."""
        meta = self.models.metadata()
        return {
            "model_id": meta["model_id"],
            "head": meta["head"],
            "architecture": meta["architecture"],
            "num_hidden_layers": meta["num_hidden_layers"],
            "hidden_size": meta["hidden_size"],
            "device": meta["device"],
            "dtype": meta["dtype"],
            "context_length": meta["context_length"],
            "layers": [
                {"requested": label, "index": index}
                for label, index in self.models.resolve_layers(self.config.extraction.layers)
            ],
            "pooling": self.config.extraction.pooling,
            "versions": library_versions(),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Analyzer(model={self.config.model.model_id!r}, "
            f"layers={self.config.extraction.layers!r}, "
            f"pooling={self.config.extraction.pooling!r}, device={self.models.device!r})"
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value
