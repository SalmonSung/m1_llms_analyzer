"""Batch plumbing shared by the inference and scoring services.

Both services walk a list of item indices in chunks, halve a chunk on CUDA
out-of-memory, and isolate per-item failures. Keeping that loop in one place
means the two cannot drift in how they recover from a bad batch.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence, TypeVar

from .logging import get_logger

log = get_logger("batching")

T = TypeVar("T")


def chunk(items: Sequence[int], size: int) -> Iterable[list[int]]:
    """Yield consecutive slices of `items` of at most `size`."""
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def maybe_progress(items: Sequence, enabled: bool, desc: str = "batches"):
    """Wrap in tqdm when asked and available; tqdm is a soft dependency."""
    if not enabled:
        return items
    try:
        from tqdm.auto import tqdm

        return tqdm(items, desc=desc, unit="batch")
    except Exception:  # never block a run on a progress bar
        return items


def oom_errors() -> tuple:
    """CUDA OOM classes, tolerant of torch versions and torch being absent."""
    errors: list[type] = []
    try:
        import torch

        errors.append(torch.cuda.OutOfMemoryError)
    except Exception:  # pragma: no cover
        pass
    errors.append(MemoryError)
    return tuple(errors)


OOM_ERRORS = oom_errors()


def empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


def run_with_oom_halving(
    indices: list[int],
    forward: Callable[[list[int]], Sequence[T]],
    on_result: Callable[[int, T], None],
    on_item_error: Callable[[int, Exception, bool], None],
    *,
    raise_on_error: bool = False,
) -> None:
    """Run `forward` over `indices`, halving on OOM and isolating item errors.

    ``forward(chunk_indices)`` must return one result per index, in order.
    ``on_result(index, result)`` receives each success. ``on_item_error(index,
    exc, is_oom)`` receives each single-item failure; it decides whether to
    record or raise. A non-OOM exception on a multi-item chunk is retried one
    item at a time so the failing item is identified and the rest survive
    (unless ``raise_on_error``, in which case it propagates immediately).
    """
    try:
        for index, result in zip(indices, forward(indices)):
            on_result(index, result)
        return
    except OOM_ERRORS as exc:
        if len(indices) == 1:
            on_item_error(indices[0], exc, True)
            return
        half = max(1, len(indices) // 2)
        log.warning("Out of memory at batch size %d; retrying in halves of %d.", len(indices), half)
        empty_cuda_cache()
        for sub in chunk(indices, half):
            run_with_oom_halving(sub, forward, on_result, on_item_error, raise_on_error=raise_on_error)
        return
    except Exception as exc:  # noqa: BLE001 - isolate the bad item, keep the run
        if len(indices) == 1:
            on_item_error(indices[0], exc, False)
            return
        if raise_on_error:
            raise
        log.debug("Chunk of %d failed (%s); retrying one by one.", len(indices), type(exc).__name__)
        for i in indices:
            run_with_oom_halving([i], forward, on_result, on_item_error, raise_on_error=raise_on_error)
