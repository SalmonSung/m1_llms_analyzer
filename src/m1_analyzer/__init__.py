"""m1_analyzer -- hidden-state extraction for Hugging Face models.

Typical use (see notebooks/colab_entrypoint.ipynb)::

    from m1_analyzer import Analyzer, RunConfig, ModelConfig, ExtractionConfig

    analyzer = Analyzer.quick("Qwen/Qwen2.5-0.5B-Instruct", layers=[-1, "middle"])
    record   = analyzer.invoke("hello world")
    result   = analyzer.batch(["one", "two", "three"])
    paths    = analyzer.save(result, name="my_run")
"""

from .config.settings import ExtractionConfig, ModelConfig, RunConfig, StorageConfig
from .container import Analyzer
from .domain.records import (
    BatchResult,
    ExtractionFailure,
    ExtractionRecord,
    LayerState,
    RunManifest,
    WrittenPaths,
)
from .services.inference_service import InferenceService
from .services.model_service import ModelLoadError, ModelService, UnsupportedArchitectureError
from .services.storage_service import StorageService, load_run
from .utils.env import in_colab, resolve_hf_token
from .utils.logging import configure_logging, get_logger

__version__ = "0.1.0"

__all__ = [
    "Analyzer",
    "RunConfig",
    "ModelConfig",
    "ExtractionConfig",
    "StorageConfig",
    "BatchResult",
    "ExtractionRecord",
    "ExtractionFailure",
    "LayerState",
    "RunManifest",
    "WrittenPaths",
    "ModelService",
    "InferenceService",
    "StorageService",
    "ModelLoadError",
    "UnsupportedArchitectureError",
    "load_run",
    "in_colab",
    "resolve_hf_token",
    "configure_logging",
    "get_logger",
    "__version__",
]
