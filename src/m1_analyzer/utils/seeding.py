"""Run-level determinism.

Seeding every RNG makes tokenization/sampling-adjacent code reproducible, and the
seed is stamped into the run manifest. Full bit-exact determinism on CUDA is *not*
guaranteed (see docs/design_decisions.md): cuBLAS/cuDNN kernel selection and
reduction order can differ between runs and between GPUs. `strict=True` opts into
torch's deterministic algorithms, which is slower and raises for ops with no
deterministic implementation -- so it is off by default.
"""

from __future__ import annotations

import os
import random

from .logging import get_logger

log = get_logger("seeding")


def seed_everything(seed: int, strict: bool = False) -> int:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy is a hard dep in practice
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if strict:
            # Required by torch for deterministic cuBLAS reductions.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except Exception:  # pragma: no cover
        pass

    log.debug("Seeded RNGs with %s (strict=%s).", seed, strict)
    return seed
