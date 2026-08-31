"""Device and dtype selection.

Picking these automatically is what makes one notebook work on a free CPU
runtime, a T4, and an A100 without edits:

* CPU + float16 is a trap -- many kernels have no fp16 CPU implementation and the
  ones that do are slower than fp32. CPU therefore always gets float32.
* bfloat16 is preferred on GPUs that support it (Ampere+): same speed as fp16
  with fp32's exponent range, so hidden states cannot overflow to inf.
* Hidden states are always cast to float32 before leaving the GPU, so the saved
  numbers do not depend on the compute dtype's precision.
"""

from __future__ import annotations

from typing import Any

from .logging import get_logger

log = get_logger("device")

VALID_DEVICES = ("auto", "cpu", "cuda", "mps")
VALID_DTYPES = ("auto", "float32", "float16", "bfloat16")


def resolve_device(preference: str = "auto") -> str:
    import torch

    if preference not in VALID_DEVICES:
        raise ValueError(f"device must be one of {VALID_DEVICES}, got {preference!r}")

    if preference != "auto":
        if preference == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but no CUDA device is available.")
        return preference

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device: str, preference: str = "auto") -> Any:
    import torch

    if preference not in VALID_DTYPES:
        raise ValueError(f"dtype must be one of {VALID_DTYPES}, got {preference!r}")

    if preference != "auto":
        dtype = getattr(torch, preference)
        if device == "cpu" and dtype in (torch.float16,):
            log.warning("float16 on CPU is unsupported by many kernels; using float32 instead.")
            return torch.float32
        return dtype

    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float32
    return torch.float32


def describe_device(device: str) -> dict:
    """Human-readable device facts for the run manifest / environment report."""
    import torch

    info: dict[str, Any] = {"device": device}
    if device == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_total_memory_gb"] = round(props.total_memory / 1024**3, 2)
        info["cuda_version"] = torch.version.cuda
    return info
