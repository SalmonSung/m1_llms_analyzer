"""Continuations from a loaded causal LM, and bag-of-token overlap between them.

Task 9b asks whether a deletion changes what the model *writes*, not only what
it predicts. That needs text after a context, generated under a protocol fixed
in advance, and a way to compare two sets of generated texts. This module is
the model-facing half: it knows nothing about cuts or caches.

* `sample_continuations` draws ``K`` continuations of ``L`` new tokens from one
  context with **nucleus sampling** (top-p, temperature; no top-k, no repetition
  penalty). The randomness comes from a `torch.Generator` seeded per call, so
  two calls with the same seed and the same shapes consume the same random
  stream -- which is what makes samples from an original and a spliced context
  *paired*. The global RNGs are never touched. The EOS token is an ordinary
  token: it is never suppressed and never stops decoding, so every sample has
  exactly ``L`` ids.
* `greedy_continuation` is the same loop with ``argmax``.
* Both run the prefix once and then one token per step with the model's KV
  cache (``kv_cache=False`` re-feeds the whole sequence every step -- slower,
  identical, and the oracle the tests compare against). Probabilities are taken
  from float32 logits, as the state service does.
* `token_f1` is the F1 between two continuations read as **multisets** of token
  ids (bag overlap with multiplicity), and `overlap_stats` turns two sample
  sets into the pre-registered numbers: the mean F1 within one set, the mean
  F1 across the two, and their ratio.
"""

from __future__ import annotations

import inspect
from collections import Counter
from itertools import combinations
from typing import Any, Callable, Sequence

import numpy as np

from ..utils.logging import get_logger

log = get_logger("decoding")


# ------------------------------------------------------------------ overlap


def token_f1(a: Sequence[int], b: Sequence[int]) -> float:
    """F1 over the *multisets* of token ids: ``2 * sum_t min(cnt_a[t], cnt_b[t]) / (|a| + |b|)``.

    Order is ignored, multiplicity is not; identical bags give 1.0, disjoint
    ones 0.0, two empty sequences 0.0.
    """
    if not len(a) and not len(b):
        return 0.0
    ca, cb = Counter(int(t) for t in a), Counter(int(t) for t in b)
    common = sum((ca & cb).values())
    return 2.0 * common / (len(a) + len(b))


def first_diff(a: Sequence[int], b: Sequence[int], cap: int) -> int:
    """The first index at which `a` and `b` differ, capped at `cap` (returned when they agree that far)."""
    n = min(len(a), len(b), cap)
    for k in range(n):
        if int(a[k]) != int(b[k]):
            return k
    return n


def overlap_stats(orig: np.ndarray, spliced: np.ndarray) -> dict[str, Any]:
    """The pre-registered overlap numbers for two sample sets of shape ``(K, L)``.

    ``within``  mean token F1 over the ``K (K - 1) / 2`` unordered pairs of `orig`;
    ``cross``   mean token F1 over the ``K * K`` (orig, spliced) pairs;
    ``sample_overlap = cross / within`` -- 1.0 means the spliced context
    generates the same bag-of-token distribution as the original. ``within``
    of 0 (every original sample disjoint from every other) leaves the ratio
    undefined: it is returned as ``nan`` and ``collapsed`` is set, never
    adjusted. ``within_spliced`` is the same statistic on the spliced set,
    kept as a diagnostic.
    """
    orig = np.asarray(orig)
    spliced = np.asarray(spliced)
    if orig.ndim != 2 or spliced.ndim != 2:
        raise ValueError("expected two (K, L) arrays of token ids.")
    within_pairs = [token_f1(orig[a], orig[b]) for a, b in combinations(range(len(orig)), 2)]
    within_spliced_pairs = [token_f1(spliced[a], spliced[b]) for a, b in combinations(range(len(spliced)), 2)]
    cross_pairs = [token_f1(o, s) for o in orig for s in spliced]
    within = float(np.mean(within_pairs)) if within_pairs else float("nan")
    within_spliced = float(np.mean(within_spliced_pairs)) if within_spliced_pairs else float("nan")
    cross = float(np.mean(cross_pairs)) if cross_pairs else float("nan")
    collapsed = not (within > 0)
    return {
        "within": within,
        "cross": cross,
        "within_spliced": within_spliced,
        "sample_overlap": float("nan") if collapsed else cross / within,
        "n_within_pairs": len(within_pairs),
        "n_cross_pairs": len(cross_pairs),
        "collapsed": bool(collapsed),
    }


# ----------------------------------------------------------------- sampling


def nucleus_filter(probs: Any, top_p: float) -> Any:
    """Keep the smallest set of highest-probability tokens whose mass reaches `top_p`.

    `probs` is a ``(B, V)`` tensor of probabilities. Tokens are sorted
    descending; a token is kept while the cumulative mass *before* it is below
    `top_p`, so the token that crosses the threshold is included and the top-1
    token always survives. The kept mass is renormalised to 1; ``top_p >= 1``
    returns `probs` unchanged.
    """
    import torch

    if not 0.0 < top_p:
        raise ValueError("top_p must be > 0.")
    if top_p >= 1.0:
        return probs
    sorted_probs, order = torch.sort(probs, dim=-1, descending=True)
    before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
    kept = torch.where(before < top_p, sorted_probs, torch.zeros_like(sorted_probs))
    kept = kept / kept.sum(dim=-1, keepdim=True)
    out = torch.zeros_like(probs)
    out.scatter_(-1, order, kept)
    return out


def _accepts_logits_to_keep(model: Any) -> bool:
    try:
        return "logits_to_keep" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic forward signatures
        return False


def _step(model: Any, input_ids: Any, attention_mask: Any, past: Any, use_cache: bool, keep_last: bool) -> tuple[Any, Any]:
    """One forward call; returns ``(float32 logits of the last position, past_key_values)``."""
    import torch

    kwargs: dict[str, Any] = dict(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache, return_dict=True)
    if past is not None:
        kwargs["past_key_values"] = past
    if keep_last:
        kwargs["logits_to_keep"] = 1  # only the last position's logits leave the model (Qwen3 & co.)
    out = model(**kwargs)
    logits = out.logits[:, -1, :].to(torch.float32)
    if not torch.isfinite(logits).all():
        raise ValueError(
            "non-finite logits while decoding (inf/nan). This is usually a float16 overflow inside "
            "the model; re-run with ModelConfig(dtype='float32')."
        )
    return logits, getattr(out, "past_key_values", None) if use_cache else None


def _continue(
    provider: Any,
    context: Sequence[int],
    *,
    n_rows: int,
    n_new: int,
    bos_id: int | None,
    kv_cache: bool,
    max_length: int | None,
    choose: Callable[[Any], Any],
) -> np.ndarray:
    """``(n_rows, n_new)`` ids chosen step by step by `choose(logits) -> (n_rows,) ids`."""
    import torch

    if n_rows < 1 or n_new < 1:
        raise ValueError("need at least one row and one new token.")
    prefix = ([int(bos_id)] if bos_id is not None else []) + [int(t) for t in context]
    if not prefix:
        raise ValueError("the context is empty and no BOS is prepended: nothing to continue from.")
    limit = max_length if max_length is not None else provider.effective_max_length(None, 1 << 30)
    if len(prefix) + n_new > limit:
        raise ValueError(
            f"a prefix of {len(prefix)} tokens (BOS included) plus {n_new} new tokens exceeds the "
            f"length limit {limit}. Shorten the context or raise ScoringConfig.max_length_cap."
        )

    model = provider.model
    device = provider.device
    keep_last = _accepts_logits_to_keep(model)
    input_ids = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0).repeat(n_rows, 1)
    generated = torch.empty((n_rows, n_new), dtype=torch.long, device=device)
    past = None
    step_ids = input_ids
    with torch.inference_mode():
        for t in range(n_new):
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
            if kv_cache:
                logits, past = _step(model, step_ids, attention_mask, past, True, keep_last)
            else:
                logits, _ = _step(model, input_ids, attention_mask, None, False, keep_last)
            next_ids = choose(logits).to(torch.long).reshape(n_rows)
            generated[:, t] = next_ids
            input_ids = torch.cat([input_ids, next_ids[:, None]], dim=1)
            step_ids = next_ids[:, None]
    return generated.cpu().numpy()


def sample_continuations(
    provider: Any,
    context: Sequence[int],
    *,
    K: int,
    L: int,
    top_p: float,
    temperature: float,
    seed: int,
    bos_id: int | None,
    kv_cache: bool = True,
    max_length: int | None = None,
) -> np.ndarray:
    """``(K, L)`` sampled token ids: nucleus sampling from ``[bos] + context``.

    All ``K`` rows are drawn together, one `torch.multinomial` per step from a
    `torch.Generator` on the model's device seeded with `seed` -- so a second
    call with the same seed, ``K``, ``L`` and vocabulary consumes the identical
    random stream whatever its context. No top-k, no repetition penalty; the
    EOS token is sampled like any other and does not stop the loop.
    """
    import torch

    if temperature <= 0:
        raise ValueError("temperature must be > 0.")
    device = torch.device(provider.device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    def choose(logits: Any) -> Any:
        probs = torch.softmax(logits / temperature, dim=-1)
        probs = nucleus_filter(probs, top_p)
        return torch.multinomial(probs, 1, generator=generator).squeeze(1)

    return _continue(provider, context, n_rows=K, n_new=L, bos_id=bos_id, kv_cache=kv_cache,
                     max_length=max_length, choose=choose)


def greedy_continuation(
    provider: Any,
    context: Sequence[int],
    *,
    L: int,
    bos_id: int | None,
    kv_cache: bool = True,
    max_length: int | None = None,
) -> np.ndarray:
    """``(L,)`` argmax token ids from ``[bos] + context``; the same loop, no randomness."""
    import torch

    def choose(logits: Any) -> Any:
        return torch.argmax(logits, dim=-1)

    return _continue(provider, context, n_rows=1, n_new=L, bos_id=bos_id, kv_cache=kv_cache,
                     max_length=max_length, choose=choose)[0]
