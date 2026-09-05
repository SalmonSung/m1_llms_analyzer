"""Scoring service: text in, log-probability out.

This is the second thing a language model can tell you about a sentence
besides its hidden states -- how likely it is. The teaching doc's "fluency" is
the mean log-probability per predicted token, and every substitution test is a
difference of two such means. Design points:

* The model must carry its LM head (``ModelConfig(head="causal_lm")``); the
  service refuses a bare transformer with a message that says so.
* A BOS token is prepended uniformly (``bos_policy="auto"``) so the first real
  token is predicted too, whatever the tokenizer's own habit. GPT-2 and Qwen
  have no BOS and use ``<|endoftext|>`` as the document separator; Llama-style
  tokenizers do add one. Tokenising with ``add_special_tokens=False`` and
  prepending by hand gives every model the same treatment.
* Padding is forced to the right. Some models (Qwen3 among them) derive
  position ids from ``arange`` when none are passed, so left padding would
  shift every real token's position and change its score.
* Per-token log-probabilities are ``gather(logit) - logsumexp(logits)`` in
  float32 over small sub-chunks. A full ``log_softmax`` over a 150k-word
  vocabulary for a 64-sequence batch is more than a gigabyte and is exactly
  what runs a T4 out of memory. ``labels=`` is never used: HF's loss is a batch
  mean, not a per-sequence sum.
* A non-finite score is a *failure*, never a number. fp16 can overflow inside a
  model; the fix is ``dtype="float32"``, and the record says so.
* Batching is OOM-safe and failure-isolating via `utils.batching`, the same
  loop the inference service uses.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from ..config.settings import ScoringConfig
from ..domain.records import ExtractionFailure, ScoreResult, SentenceScore, make_item_id
from ..utils.batching import (
    OOM_ERRORS,
    AdaptiveBatchSize,
    empty_cuda_cache,
    maybe_progress,
    run_with_oom_halving,
)
from ..utils.logging import get_logger
from .interfaces import ModelProvider

log = get_logger("scoring")

_PREVIEW_CHARS = 200


class LogProbService:
    """Scores texts under a causal language model."""

    #: Starting size for ``batch_size="auto"`` when there is no CUDA device to probe.
    CPU_AUTO_BATCH_SIZE = 32

    def __init__(self, provider: ModelProvider, config: ScoringConfig | None = None):
        self.provider = provider
        self.config = config or ScoringConfig()
        #: The batch size learned so far; shared across score() calls so a halving
        #: (or a calibration) on one sentence carries over to the next.
        self.batch_sizer: AdaptiveBatchSize | None = None
        self._sizer_key: tuple[Any, int] | None = None

    # -------------------------------------------------------------- public API

    def score(
        self,
        texts: Sequence[str],
        *,
        show_progress: bool = False,
        return_token_logprobs: bool = False,
        **overrides: Any,
    ) -> ScoreResult:
        """Log-probability of each text, in input order."""
        config = self._with_overrides(overrides)
        texts = self._validate_inputs(texts)
        self._require_lm_head()
        started = time.perf_counter()

        max_length = self.provider.effective_max_length(config.max_length, config.max_length_cap)
        order = self._batch_order(texts, config)
        scores: dict[int, SentenceScore] = {}
        failures: list[ExtractionFailure] = []

        def forward(indices: list[int]) -> list[SentenceScore]:
            return self._forward([texts[i] for i in indices], max_length, config, return_token_logprobs)

        def on_result(index: int, result: SentenceScore) -> None:
            if not np.isfinite(result.sum_logprob):
                self._fail(failures, index, texts[index], ValueError(
                    "non-finite log-probability (inf/nan). This is usually a float16 overflow "
                    "inside the model; re-run with ModelConfig(dtype='float32')."
                ), config)
                return
            scores[index] = result

        def on_item_error(index: int, exc: Exception, is_oom: bool) -> None:
            if is_oom:
                exc = RuntimeError(
                    f"{exc}\nHint: a single text exhausted device memory. Lower "
                    "ScoringConfig.max_length or logit_chunk, or use a smaller model."
                )
            self._fail(failures, index, texts[index], exc, config)

        sizer = self._sizer_for(config, texts)
        for indices in maybe_progress(sizer.chunks(order), show_progress, desc="scoring"):
            before = sizer.current
            run_with_oom_halving(
                indices, forward, on_result, on_item_error,
                raise_on_error=not config.continue_on_error, sizer=sizer,
            )
            if sizer.current == before:
                sizer.success()

        ordered = [scores[i] for i in sorted(scores)]
        elapsed = time.perf_counter() - started
        if failures:
            log.warning("%d/%d texts failed to score; see ScoreResult.failures.", len(failures), len(texts))
        log.debug("Scored %d text(s) in %.2fs.", len(ordered), elapsed)
        return ScoreResult(
            scores=ordered, failures=failures, elapsed_seconds=elapsed,
            scoring=dict(vars(config)),
        )

    def score_one(self, text: str, **overrides: Any) -> SentenceScore:
        """Score a single text; raises on failure."""
        result = self.score([text], **overrides)
        if not result.scores:
            failure = result.failures[0]
            raise RuntimeError(f"{failure.error_type}: {failure.error}")
        return result.scores[0]

    # ------------------------------------------------------------- batch size

    def calibrate_batch_size(self, texts: Sequence[str], *, start: int = 8, **overrides: Any) -> int:
        """Probe the device: the largest batch of copies of the longest text that fits.

        Doubles from `start` until a forward pass runs out of memory or
        ``max_batch_size`` is reached, then keeps the last size that worked. The
        result seeds the adaptive sizer used by later `score()` calls. Call it
        once, on the longest texts of the run, so the size is chosen against the
        worst case rather than the first sentence.
        """
        config = self._with_overrides(overrides)
        texts = self._validate_inputs(texts)
        self._require_lm_head()
        tokenizer = self.provider.tokenizer
        longest = max(texts, key=lambda t: len(tokenizer(t, add_special_tokens=False)["input_ids"]))
        max_length = self.provider.effective_max_length(config.max_length, config.max_length_cap)
        maximum = config.max_batch_size

        fits, size = 0, max(1, min(start, maximum))
        while True:
            try:
                self._forward([longest] * size, max_length, config, False)
            except OOM_ERRORS:
                empty_cuda_cache()
                if fits:
                    break
                if size == 1:
                    raise RuntimeError(
                        "One copy of the longest text exhausts device memory. Lower "
                        "ScoringConfig.max_length or logit_chunk, or use a smaller model."
                    )
                size = max(1, size // 2)
                continue
            fits = size
            if size >= maximum:
                break
            size = min(size * 2, maximum)

        self.batch_sizer = AdaptiveBatchSize(fits, maximum=maximum)
        if fits < maximum:
            self.batch_sizer.ceiling = fits  # growth must not retry the size that failed
        self._sizer_key = self._key(config)
        log.info("Calibrated batch size: %d (%d-token longest text, ceiling %d).", fits,
                 len(tokenizer(longest, add_special_tokens=False)["input_ids"]), self.batch_sizer.ceiling)
        return fits

    @staticmethod
    def _key(config: ScoringConfig) -> tuple[Any, int]:
        return (config.batch_size, config.max_batch_size)

    def _sizer_for(self, config: ScoringConfig, texts: Sequence[str]) -> AdaptiveBatchSize:
        """The sizer for this config; built (or calibrated) on first use."""
        if self.batch_sizer is not None and self._sizer_key == self._key(config):
            return self.batch_sizer
        if config.batch_size == "auto":
            if str(self.provider.device).startswith("cuda"):
                self.calibrate_batch_size(texts, **{k: v for k, v in vars(config).items()})
                return self.batch_sizer
            log.info("batch_size='auto' without a CUDA device: starting at %d, growing to at most %d.",
                     self.CPU_AUTO_BATCH_SIZE, config.max_batch_size)
            self.batch_sizer = AdaptiveBatchSize(min(self.CPU_AUTO_BATCH_SIZE, config.max_batch_size),
                                                 maximum=config.max_batch_size)
        else:
            self.batch_sizer = AdaptiveBatchSize(int(config.batch_size), maximum=int(config.batch_size))
        self._sizer_key = self._key(config)
        return self.batch_sizer

    # ----------------------------------------------------------- forward pass

    def _forward(
        self,
        texts: list[str],
        max_length: int,
        config: ScoringConfig,
        return_token_logprobs: bool,
    ) -> list[SentenceScore]:
        import torch

        tokenizer = self.provider.tokenizer
        model = self.provider.model
        device = self.provider.device

        bos = self._bos_id(config)
        encoded_ids = []
        for text in texts:
            ids = tokenizer(text, add_special_tokens=False, truncation=True,
                            max_length=max_length - (1 if bos is not None else 0))["input_ids"]
            if bos is not None:
                ids = [bos] + ids
            if len(ids) < 2:
                raise ValueError(
                    f"{text[:60]!r} tokenises to {len(ids)} token(s); at least one token must be "
                    "predicted. With bos_policy='none' a text needs two or more tokens."
                )
            encoded_ids.append(ids)

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        width = max(len(ids) for ids in encoded_ids)
        input_ids = torch.full((len(texts), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(texts), width), dtype=torch.long)
        for row, ids in enumerate(encoded_ids):
            input_ids[row, : len(ids)] = torch.tensor(ids)
            attention_mask[row, : len(ids)] = 1  # right padding, by construction
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise RuntimeError(
                f"{type(model).__name__} returned no logits. Load the model with "
                "ModelConfig(head='causal_lm') to score log-probabilities."
            )

        # Position t predicts token t+1: drop the last logit row and the first target.
        targets = input_ids[:, 1:]
        target_mask = attention_mask[:, 1:].to(torch.bool)
        token_lp = torch.empty(targets.shape, dtype=torch.float32, device=device)
        for start in range(0, len(texts), config.logit_chunk):
            stop = start + config.logit_chunk
            block = logits[start:stop, :-1, :].to(torch.float32)
            picked = block.gather(-1, targets[start:stop].unsqueeze(-1)).squeeze(-1)
            token_lp[start:stop] = picked - torch.logsumexp(block, dim=-1)
            del block
        token_lp = torch.where(target_mask, token_lp, torch.zeros_like(token_lp)).cpu().numpy()
        counts = target_mask.sum(dim=1).cpu().numpy()

        records = []
        for i, text in enumerate(texts):
            n = int(counts[i])
            records.append(
                SentenceScore(
                    id=make_item_id(text),
                    text=text,
                    n_tokens=n,
                    sum_logprob=float(token_lp[i, :n].sum()),
                    token_logprobs=token_lp[i, :n].copy() if return_token_logprobs else None,
                )
            )
        return records

    def _bos_id(self, config: ScoringConfig) -> int | None:
        if config.bos_policy == "none":
            return None
        tokenizer = self.provider.tokenizer
        for attr in ("bos_token_id", "eos_token_id"):
            value = getattr(tokenizer, attr, None)
            if value is not None:
                return int(value)
        raise RuntimeError(
            "bos_policy='auto' needs a BOS or EOS token, and this tokenizer has neither. "
            "Use bos_policy='none'."
        )

    def bos_token(self, config: ScoringConfig | None = None) -> str | None:
        """The literal token prepended under the current policy (for records)."""
        bos = self._bos_id(config or self.config)
        return None if bos is None else self.provider.tokenizer.convert_ids_to_tokens(bos)

    # ---------------------------------------------------------------- helpers

    def _require_lm_head(self) -> None:
        model = self.provider.model
        if not hasattr(model, "get_output_embeddings") or model.get_output_embeddings() is None:
            raise RuntimeError(
                f"{type(model).__name__} has no language-model head, so it cannot score "
                "log-probabilities. Load it with ModelConfig(head='causal_lm')."
            )

    def _fail(self, failures, index, text, exc, config) -> None:
        if not config.continue_on_error:
            raise exc
        log.error("Text %d failed to score: %s: %s", index, type(exc).__name__, exc)
        failures.append(
            ExtractionFailure(
                id=make_item_id(text),
                text_preview=text[:_PREVIEW_CHARS],
                error_type=type(exc).__name__,
                error=str(exc),
                index=index,
            )
        )

    def _with_overrides(self, overrides: dict[str, Any]) -> ScoringConfig:
        if not overrides:
            return self.config
        unknown = set(overrides) - set(vars(self.config))
        if unknown:
            raise TypeError(f"Unknown scoring option(s): {sorted(unknown)}")
        return ScoringConfig(**{**vars(self.config), **overrides})

    @staticmethod
    def _validate_inputs(texts: Sequence[str]) -> list[str]:
        if isinstance(texts, str):
            raise TypeError("score() expects a sequence of strings; pass [text] or use score_one().")
        items = list(texts)
        if not items:
            raise ValueError("No inputs provided.")
        for i, text in enumerate(items):
            if not isinstance(text, str):
                raise TypeError(f"Input {i} is {type(text).__name__}, expected str.")
            if not text.strip():
                raise ValueError(f"Input {i} is empty or whitespace-only; there is nothing to score.")
        return items

    def _batch_order(self, texts: Sequence[str], config: ScoringConfig) -> list[int]:
        if not config.sort_by_length or len(texts) < 3:
            return list(range(len(texts)))
        return sorted(range(len(texts)), key=lambda i: len(texts[i]))
