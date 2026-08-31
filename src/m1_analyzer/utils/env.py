"""Secret and environment resolution.

Secrets are looked up in a fixed order so the same code runs unchanged in Colab,
in CI, and on a laptop:

    1. Colab's secret manager (``google.colab.userdata``)
    2. Process environment variables (the name itself, then any aliases)
    3. ``None``

``None`` is a legitimate result, not an error: ungated Hugging Face models load
fine without a token, so the token-less path must stay on the happy path.
"""

from __future__ import annotations

import os
from typing import Sequence

from .logging import get_logger

log = get_logger("env")

#: Alternative environment-variable names checked after the primary name.
SECRET_ALIASES = {
    "HF_TOKEN": ("HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"),
    "GITHUB_TOKEN": ("GH_TOKEN",),
}


def in_colab() -> bool:
    """True when running inside a Google Colab runtime."""
    try:
        import google.colab  # noqa: F401
    except Exception:
        return False
    return True


def _from_colab_secrets(name: str) -> str | None:
    if not in_colab():
        return None
    try:
        from google.colab import userdata  # type: ignore

        value = userdata.get(name)
    except Exception:
        # Not configured, access not granted, or non-Colab shim: fall through
        # to environment variables rather than failing the run.
        return None
    return value or None


def resolve_secret(name: str, aliases: Sequence[str] | None = None) -> str | None:
    """Return the secret's value, or ``None`` when it is not configured anywhere.

    The value is never logged; only *where* it was found is logged, and only at
    debug level.
    """
    value = _from_colab_secrets(name)
    if value:
        log.debug("Resolved %s from Colab secrets.", name)
        return value.strip()

    candidates = [name, *(aliases if aliases is not None else SECRET_ALIASES.get(name, ()))]
    for key in candidates:
        value = os.environ.get(key)
        if value:
            log.debug("Resolved %s from environment variable %s.", name, key)
            return value.strip()

    log.debug("%s is not configured (checked Colab secrets and %s).", name, candidates)
    return None


def resolve_hf_token(explicit: str | None = None) -> str | None:
    """Hugging Face token for gated repos, or ``None`` for the ungated path."""
    if explicit:
        return explicit
    return resolve_secret("HF_TOKEN")


def redact(secret: str | None, keep: int = 4) -> str:
    """Render a secret safely for logs/printing: never reveals the full value."""
    if not secret:
        return "<unset>"
    if len(secret) <= keep:
        return "*" * len(secret)
    return f"{'*' * (len(secret) - keep)}{secret[-keep:]}"
