"""Typed configuration for a run.

Plain dataclasses (no YAML/Hydra) because the notebook *is* the config surface:
one cell constructs a `RunConfig` and gets IDE/`?` introspection, defaults, and
`__post_init__` validation that fails fast with an explicit message instead of
surfacing as a shape error deep inside a forward pass.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence, Union

#: How a per-token hidden-state matrix is reduced to what gets saved.
POOLING_MODES = ("last_token", "mean", "cls", "none")

#: When to write the lossless float32 sidecar next to the JSON.
NPY_MODES = ("auto", "always", "never")

#: A layer spec is an int, a list of ints, or one of these keywords.
LAYER_KEYWORDS = ("all", "last", "middle")

LayerSpec = Union[int, str, Sequence[Union[int, str]]]


@dataclass
class ModelConfig:
    """Which model to load and how to place it on hardware."""

    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    #: Pin a commit SHA / tag for reproducibility. None = repo default branch.
    revision: str | None = None
    device: str = "auto"          # auto | cpu | cuda | mps
    dtype: str = "auto"           # auto | float32 | float16 | bfloat16
    #: Off by default: enabling it executes arbitrary code from the model repo.
    trust_remote_code: bool = False
    #: Explicit token wins over Colab secrets / env vars. Leave None normally.
    hf_token: str | None = None
    #: Local HF cache dir (Colab default is fine; set to a Drive path to persist).
    cache_dir: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string.")
        self.model_id = self.model_id.strip()


@dataclass
class ExtractionConfig:
    """What to pull out of the forward pass."""

    #: -1 == last transformer block. See ModelService.resolve_layers for the
    #: full indexing convention (index 0 is the embedding output).
    layers: LayerSpec = -1
    pooling: str = "last_token"   # last_token | mean | cls | none
    #: None -> use the model's own context length (capped by max_length_cap).
    max_length: int | None = None
    #: Hard ceiling so a 128k-context model cannot silently allocate a huge batch.
    max_length_cap: int = 4096
    truncation: bool = True
    batch_size: int = 8
    #: Sort inputs by token length before batching to cut padding waste.
    sort_by_length: bool = True
    #: Keep going (and record the failure) when a single item blows up.
    continue_on_error: bool = True
    #: Store the tokenized text alongside per-token states (pooling="none").
    include_tokens: bool = False

    def __post_init__(self) -> None:
        if self.pooling not in POOLING_MODES:
            raise ValueError(f"pooling must be one of {POOLING_MODES}, got {self.pooling!r}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self.max_length is not None and self.max_length < 1:
            raise ValueError("max_length must be >= 1 when set.")
        if self.max_length_cap < 1:
            raise ValueError("max_length_cap must be >= 1.")
        _validate_layer_spec(self.layers)


@dataclass
class StorageConfig:
    """Where and how results are persisted."""

    output_dir: str = "outputs"
    #: Decimal places kept in JSON. The .npy sidecar is always full float32.
    float_precision: int = 6
    npy_mode: str = "auto"        # auto | always | never
    #: In "auto" mode, write the sidecar once a run exceeds this many floats.
    npy_threshold_floats: int = 1_000_000
    #: Include the input text in the JSON (turn off for sensitive corpora).
    include_input_text: bool = True
    indent: int | None = 2

    def __post_init__(self) -> None:
        if self.npy_mode not in NPY_MODES:
            raise ValueError(f"npy_mode must be one of {NPY_MODES}, got {self.npy_mode!r}")
        if not 0 <= self.float_precision <= 17:
            raise ValueError("float_precision must be between 0 and 17.")


@dataclass
class RunConfig:
    """The single object the notebook edits: model + extraction + storage + seed."""

    model: ModelConfig = field(default_factory=ModelConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    seed: int = 42
    #: Opt into torch deterministic algorithms (slower; may raise on some ops).
    strict_determinism: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Never let a token reach a manifest, a log, or a JSON file.
        d["model"]["hf_token"] = None
        return d

    def fingerprint(self) -> str:
        """Stable hash of the run settings, used to tell two output files apart."""
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _validate_layer_spec(spec: LayerSpec) -> None:
    if isinstance(spec, str):
        if spec not in LAYER_KEYWORDS:
            raise ValueError(f"layer keyword must be one of {LAYER_KEYWORDS}, got {spec!r}")
        return
    if isinstance(spec, bool):  # bool is an int subclass; almost surely a mistake
        raise ValueError("layers must be an int, a keyword, or a sequence -- not a bool.")
    if isinstance(spec, int):
        return
    if isinstance(spec, (list, tuple)):
        if not spec:
            raise ValueError("layers sequence must not be empty.")
        for item in spec:
            _validate_layer_spec(item)
        return
    raise TypeError(f"Unsupported layers spec type: {type(spec).__name__}")
