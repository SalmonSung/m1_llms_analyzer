"""Mask-aware pooling of a padded hidden-state batch.

Every function takes ``hidden`` of shape ``(batch, seq, hidden)`` and an
``attention_mask`` of shape ``(batch, seq)``, and returns ``(batch, hidden)``.

The important property: the *position* of padding is never assumed. `last_token`
derives the last real index from ``attention_mask.sum(-1) - 1`` rather than
taking ``hidden[:, -1]``, so a right-padded batch and a left-padded batch give
identical numbers. Taking ``[:, -1]`` is the single most common bug in this kind
of code -- on a right-padded batch it silently pools a PAD token.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch


def last_token_pool(hidden: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
    """Hidden state at each sequence's last non-padding token.

    Found by scanning the mask from the right (``flip`` + ``argmax``) rather than
    with the common ``attention_mask.sum(1) - 1`` shortcut. That shortcut is only
    correct for right padding: on a left-padded batch it lands on the *first*
    real token instead of the last. Scanning from the right is correct for both.
    """
    import torch

    mask = attention_mask.to(torch.long)
    _assert_non_empty(mask.sum(dim=1))
    seq_len = mask.size(1)
    # argmax returns the first maximal entry, so on the flipped mask it is the
    # distance from the end to the last real token.
    from_end = mask.flip(dims=[1]).argmax(dim=1)
    idx = seq_len - 1 - from_end
    batch_idx = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_idx, idx]


def mean_pool(hidden: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
    """Mean over real tokens only; padding contributes nothing to numerator or denominator."""
    lengths = attention_mask.sum(dim=1)
    _assert_non_empty(lengths)
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    return summed / lengths.unsqueeze(-1).to(hidden.dtype)


def cls_pool(hidden: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
    """First real token. With left padding the first real token is not index 0."""
    import torch

    first = attention_mask.to(torch.long).argmax(dim=1)
    _assert_non_empty(attention_mask.sum(dim=1))
    batch_idx = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_idx, first]


POOLERS = {
    "last_token": last_token_pool,
    "mean": mean_pool,
    "cls": cls_pool,
}


def apply_pooling(mode: str, hidden: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
    """Dispatch to a pooler. ``mode='none'`` is handled by the caller (no reduction)."""
    try:
        pooler = POOLERS[mode]
    except KeyError:
        raise ValueError(
            f"Unknown pooling mode {mode!r}. Expected one of {sorted(POOLERS)} or 'none'."
        ) from None
    return pooler(hidden, attention_mask)


def unpad_sequence(hidden: "torch.Tensor", attention_mask: "torch.Tensor") -> list["torch.Tensor"]:
    """Split a padded batch into per-item ``(tokens, hidden)`` tensors with padding removed.

    Used for ``pooling='none'`` so a saved per-token matrix never contains rows
    that correspond to PAD.
    """
    out = []
    for row, mask in zip(hidden, attention_mask):
        keep = mask.to(bool)
        out.append(row[keep])
    return out


def _assert_non_empty(lengths: "torch.Tensor") -> None:
    if bool((lengths == 0).any()):
        raise ValueError(
            "An input produced zero tokens (empty attention mask), so there is nothing to pool. "
            "Empty or whitespace-only inputs must be filtered out before extraction."
        )
