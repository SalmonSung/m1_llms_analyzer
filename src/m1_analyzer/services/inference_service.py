"""Inference service: text in, hidden-state records out.

Design points worth knowing before editing:

* ``invoke`` is literally ``batch`` of one. There is a single code path, so the
  two APIs can never drift apart in pooling, truncation, or dtype handling.
* Batching is OOM-safe. A CUDA OOM halves the batch and retries down to size 1
  rather than killing a long run, and a per-item exception is recorded as a
  failure instead of raised (unless ``continue_on_error=False``).
* Inputs are sorted by token length before batching so each batch pads to a
  similar length. Original order is restored before returning.
* Hidden states are cast to float32 on the CPU before becoming numpy, so a
  bf16/fp16 forward pass still yields values that round-trip through JSON.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

import numpy as np

from ..config.settings import ExtractionConfig
from ..domain.records import (
    BatchResult,
    ExtractionFailure,
    ExtractionRecord,
    LayerState,
    make_item_id,
)
from ..utils.logging import get_logger
from ..utils.pooling import apply_pooling, unpad_sequence
from .interfaces import ModelProvider

log = get_logger("inference")

#: Characters of the offending input kept in a failure record.
_PREVIEW_CHARS = 200


class InferenceService:
    """Runs forward passes and extracts the configured layers."""

    def __init__(self, provider: ModelProvider, config: ExtractionConfig | None = None):
        self.provider = provider
        self.config = config or ExtractionConfig()

    # -------------------------------------------------------------- public API

    def invoke(self, text: str, **overrides: Any) -> ExtractionRecord:
        """Extract hidden states for one input.

        Raises on failure (a single call has no partial-success story to tell);
        use :meth:`batch` when you want failures collected instead.
        """
        if not isinstance(text, str):
            raise TypeError(f"invoke() expects a string, got {type(text).__name__}.")
        result = self.batch([text], _raise_on_error=True, **overrides)
        if not result.records:
            failure = result.failures[0]
            raise RuntimeError(f"{failure.error_type}: {failure.error}")
        return result.records[0]

    def batch(
        self,
        texts: Sequence[str],
        *,
        show_progress: bool = False,
        _raise_on_error: bool = False,
        **overrides: Any,
    ) -> BatchResult:
        """Extract hidden states for many inputs, preserving input order."""
        config = self._with_overrides(overrides)
        texts = self._validate_inputs(texts)
        started = time.perf_counter()

        layers = self.provider.resolve_layers(config.layers)
        max_length = self.provider.effective_max_length(config.max_length, config.max_length_cap)

        order = self._batch_order(texts, config)
        records: dict[int, ExtractionRecord] = {}
        failures: list[ExtractionFailure] = []

        chunks = list(_chunk(order, config.batch_size))
        for chunk in _maybe_progress(chunks, show_progress):
            self._run_chunk(
                chunk, texts, layers, max_length, config, records, failures,
                batch_size=config.batch_size, raise_on_error=_raise_on_error,
            )

        ordered = [records[i] for i in sorted(records)]
        elapsed = time.perf_counter() - started
        if failures:
            log.warning("%d/%d inputs failed; see BatchResult.failures.", len(failures), len(texts))
        log.info("Extracted %d record(s) in %.2fs.", len(ordered), elapsed)
        return BatchResult(
            records=ordered,
            failures=failures,
            elapsed_seconds=elapsed,
            extraction=dict(vars(config)),
        )

    # ------------------------------------------------------------- batch plumbing

    def _run_chunk(
        self,
        chunk: list[int],
        texts: Sequence[str],
        layers: list[tuple[str, int]],
        max_length: int,
        config: ExtractionConfig,
        records: dict[int, ExtractionRecord],
        failures: list[ExtractionFailure],
        *,
        batch_size: int,
        raise_on_error: bool,
    ) -> None:
        """Process one chunk, splitting it on OOM and isolating per-item errors."""
        try:
            for idx, record in zip(chunk, self._forward(
                [texts[i] for i in chunk], layers, max_length, config
            )):
                records[idx] = record
            return
        except _OOM_ERRORS as exc:
            if len(chunk) == 1:
                self._record_failure(
                    failures, chunk[0], texts[chunk[0]], exc, raise_on_error, config,
                    hint=(
                        "A single input exhausted device memory. Lower "
                        "ExtractionConfig.max_length, use pooling != 'none', or a smaller model."
                    ),
                )
                return
            half = max(1, len(chunk) // 2)
            log.warning(
                "Out of memory at batch size %d; retrying in halves of %d.", len(chunk), half
            )
            self._empty_cache()
            for sub in _chunk(chunk, half):
                self._run_chunk(
                    sub, texts, layers, max_length, config, records, failures,
                    batch_size=half, raise_on_error=raise_on_error,
                )
            return
        except Exception as exc:  # noqa: BLE001 - isolate the bad item, keep the run
            if len(chunk) == 1:
                self._record_failure(
                    failures, chunk[0], texts[chunk[0]], exc, raise_on_error, config
                )
                return
            if not config.continue_on_error and raise_on_error:
                raise
            log.debug("Chunk of %d failed (%s); retrying one by one.", len(chunk), type(exc).__name__)
            for i in chunk:
                self._run_chunk(
                    [i], texts, layers, max_length, config, records, failures,
                    batch_size=1, raise_on_error=raise_on_error,
                )

    def _record_failure(
        self,
        failures: list[ExtractionFailure],
        index: int,
        text: str,
        exc: Exception,
        raise_on_error: bool,
        config: ExtractionConfig,
        hint: str | None = None,
    ) -> None:
        message = str(exc)
        if hint:
            message = f"{message}\nHint: {hint}"
        if raise_on_error or not config.continue_on_error:
            raise type(exc)(message) if not hint else RuntimeError(message)
        log.error("Input %d failed: %s: %s", index, type(exc).__name__, message)
        failures.append(
            ExtractionFailure(
                id=make_item_id(text),
                text_preview=text[:_PREVIEW_CHARS],
                error_type=type(exc).__name__,
                error=message,
                index=index,
            )
        )

    # ------------------------------------------------------------ forward pass

    def _forward(
        self,
        texts: list[str],
        layers: list[tuple[str, int]],
        max_length: int,
        config: ExtractionConfig,
    ) -> list[ExtractionRecord]:
        import torch

        tokenizer = self.provider.tokenizer
        model = self.provider.model
        device = self.provider.device

        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=config.truncation,
            max_length=max_length,
        )
        # Length before truncation, so a clipped prompt is never analysed silently.
        raw_lengths = [len(tokenizer(t, truncation=False)["input_ids"]) for t in texts]

        encoded = {k: v.to(device) for k, v in encoded.items() if hasattr(v, "to")}
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(encoded["input_ids"])
            encoded["attention_mask"] = attention_mask

        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True, return_dict=True)

        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError(
                f"{type(model).__name__} returned no hidden_states even with "
                "output_hidden_states=True. This architecture is not supported."
            )
        expected = self.provider.num_hidden_layers + 1
        if len(hidden_states) != expected:
            log.warning(
                "Model returned %d hidden states, expected %d (num_hidden_layers + 1). "
                "Layer indices are taken against the returned tuple.",
                len(hidden_states), expected,
            )

        per_layer: dict[str, list[np.ndarray]] = {}
        for label, index in layers:
            if index >= len(hidden_states):
                raise ValueError(
                    f"Layer {label!r} -> index {index} but the model returned only "
                    f"{len(hidden_states)} hidden states (valid: 0..{len(hidden_states) - 1})."
                )
            # float32 on CPU: hidden states must not depend on the compute dtype.
            hidden = hidden_states[index].to(torch.float32)
            if config.pooling == "none":
                per_layer[label] = [t.cpu().numpy() for t in unpad_sequence(hidden, attention_mask)]
            else:
                pooled = apply_pooling(config.pooling, hidden, attention_mask)
                per_layer[label] = list(pooled.cpu().numpy())

        token_counts = attention_mask.sum(dim=1).tolist()
        token_lists = self._decode_tokens(encoded["input_ids"], attention_mask) if (
            config.include_tokens and config.pooling == "none"
        ) else None

        records = []
        for i, text in enumerate(texts):
            layer_states = {
                label: LayerState(
                    index=dict(layers)[label],
                    requested=label,
                    values=np.ascontiguousarray(per_layer[label][i]),
                )
                for label, _ in layers
            }
            token_count = int(token_counts[i])
            records.append(
                ExtractionRecord(
                    id=make_item_id(text),
                    text=text,
                    token_count=token_count,
                    layers=layer_states,
                    truncated=raw_lengths[i] > token_count,
                    original_token_count=raw_lengths[i],
                    tokens=token_lists[i] if token_lists else None,
                )
            )
            if records[-1].truncated:
                log.warning(
                    "Input %s truncated: %d tokens -> %d (max_length=%d). The saved state "
                    "describes only the kept prefix.",
                    records[-1].id, raw_lengths[i], token_count, max_length,
                )
        return records

    def _decode_tokens(self, input_ids, attention_mask) -> list[list[str]]:
        tokenizer = self.provider.tokenizer
        out = []
        for ids, mask in zip(input_ids, attention_mask):
            kept = ids[mask.to(bool)]
            out.append(tokenizer.convert_ids_to_tokens(kept.tolist()))
        return out

    # ---------------------------------------------------------------- helpers

    def _with_overrides(self, overrides: dict[str, Any]) -> ExtractionConfig:
        """Per-call overrides without mutating the service's own config."""
        if not overrides:
            return self.config
        unknown = set(overrides) - set(vars(self.config))
        if unknown:
            raise TypeError(f"Unknown extraction option(s): {sorted(unknown)}")
        merged = {**vars(self.config), **overrides}
        return ExtractionConfig(**merged)

    @staticmethod
    def _validate_inputs(texts: Sequence[str]) -> list[str]:
        if isinstance(texts, str):
            raise TypeError("batch() expects a sequence of strings; pass [text] or use invoke().")
        items = list(texts)
        if not items:
            raise ValueError("No inputs provided.")
        for i, text in enumerate(items):
            if not isinstance(text, str):
                raise TypeError(f"Input {i} is {type(text).__name__}, expected str.")
            if not text.strip():
                raise ValueError(
                    f"Input {i} is empty or whitespace-only. It would tokenize to zero real "
                    "tokens, leaving nothing to pool. Filter empty inputs before extraction."
                )
        return items

    def _batch_order(self, texts: Sequence[str], config: ExtractionConfig) -> list[int]:
        """Indices in processing order -- length-sorted to minimise padding."""
        if not config.sort_by_length or len(texts) < 3:
            return list(range(len(texts)))
        return sorted(range(len(texts)), key=lambda i: len(texts[i]))

    @staticmethod
    def _empty_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass


def _chunk(items: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _maybe_progress(items: list, enabled: bool):
    if not enabled:
        return items
    try:
        from tqdm.auto import tqdm

        return tqdm(items, desc="batches", unit="batch")
    except Exception:  # tqdm is a soft dependency; never block a run on it
        return items


def _oom_errors() -> tuple:
    """CUDA OOM classes, tolerant of torch versions and torch being absent."""
    errors: list[type] = []
    try:
        import torch

        errors.append(torch.cuda.OutOfMemoryError)
    except Exception:  # pragma: no cover
        pass
    errors.append(MemoryError)
    return tuple(errors)


_OOM_ERRORS = _oom_errors()
