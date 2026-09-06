"""State service: token ids in, full next-token log-probability vectors out.

The splice experiment (Task 9a) needs the model's *state* at a position: the
complete log-probability vector over the vocabulary for the next token, the
same object the teaching doc's Acts 6-8 measure distances between. The scoring
service deliberately never materialises that vector (it takes one target's
log-probability by ``gather - logsumexp``), so this is a separate, narrow
service:

* Input is **token ids**, not text. The experiment splices sequences at the
  token level so that a position in the spliced sequence is provably the same
  token as a position in the original; re-tokenising a rejoined string could
  move the join by a token. ``encode`` / ``decode`` are here for callers that
  start from text.
* A BOS token is prepended under the scorer's ``bos_policy`` so position 0 has
  a state too. Positions are always counted over the *text* tokens: the state
  at position ``p`` is the distribution over token ``p + 1`` given tokens
  ``0..p`` (and BOS).
* Only the requested positions are reduced to float32 log-softmax vectors and
  moved to the CPU. A 150k-vocabulary model's logits for 300 positions are
  180 MB in float32; twenty of them are 12 MB.
* Batches halve on CUDA out-of-memory through the shared `utils.batching`
  loop; a non-finite state is an error naming the dtype fix, never a number.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..config.settings import ScoringConfig
from ..domain.records import StateResult
from ..utils.batching import AdaptiveBatchSize, run_with_oom_halving
from ..utils.logging import get_logger
from .interfaces import ModelProvider

log = get_logger("states")


class NextTokenStateService:
    """Full next-token log-probability vectors at chosen positions of a token sequence."""

    def __init__(self, provider: ModelProvider, config: ScoringConfig | None = None):
        self.provider = provider
        self.config = config or ScoringConfig()

    # --------------------------------------------------------------- tokens

    @property
    def vocab_size(self) -> int:
        model = self.provider.model
        head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if head is None:
            raise RuntimeError(
                f"{type(model).__name__} has no language-model head, so it has no next-token "
                "states. Load it with ModelConfig(head='causal_lm')."
            )
        return int(head.weight.shape[0])

    def encode(self, text: str) -> list[int]:
        """Token ids of `text`, no special tokens (BOS is added inside `states`)."""
        return list(self.provider.tokenizer(text, add_special_tokens=False)["input_ids"])

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """``(ids, [(char_start, char_end), ...])`` so text spans map to token positions."""
        encoded = self.provider.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(encoded["input_ids"]), [tuple(o) for o in encoded["offset_mapping"]]

    def decode(self, ids: Sequence[int]) -> str:
        return self.provider.tokenizer.decode(list(ids), skip_special_tokens=False)

    def bos_id(self) -> int | None:
        """The token prepended under the current BOS policy (None for ``"none"``)."""
        if self.config.bos_policy == "none":
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

    def bos_token(self) -> str | None:
        bos = self.bos_id()
        return None if bos is None else self.provider.tokenizer.convert_ids_to_tokens(bos)

    # --------------------------------------------------------------- states

    def states(
        self,
        sequences: Sequence[Sequence[int]],
        positions: Sequence[Sequence[int]],
        *,
        batch_size: int = 4,
        return_token_logprobs: bool = False,
        show_progress: bool = False,
    ) -> list[StateResult]:
        """One `StateResult` per sequence, in input order.

        ``positions[k]`` are text-token positions of ``sequences[k]``; the
        result's ``states`` has one float32 row per position, in the order given.
        With ``return_token_logprobs`` the log-probability of every *predicted*
        token is returned as well (every token when a BOS is prepended).
        """
        if len(sequences) != len(positions):
            raise ValueError("sequences and positions must have the same length.")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        self.vocab_size  # raises early on a model without a head
        max_length = self.provider.effective_max_length(self.config.max_length, self.config.max_length_cap)
        bos = self.bos_id()
        offset = 1 if bos is not None else 0

        seqs = [list(int(t) for t in s) for s in sequences]
        poss = [list(int(p) for p in ps) for ps in positions]
        for k, (ids, ps) in enumerate(zip(seqs, poss)):
            if not ids:
                raise ValueError(f"sequence {k} is empty.")
            if len(ids) + offset > max_length:
                raise ValueError(
                    f"sequence {k} has {len(ids)} tokens (+{offset} BOS), above the effective "
                    f"max_length {max_length}. Raise ScoringConfig.max_length_cap or shorten the text."
                )
            bad = [p for p in ps if not 0 <= p < len(ids)]
            if bad:
                raise ValueError(f"sequence {k}: positions {bad} are outside 0..{len(ids) - 1}.")
            if offset == 0 and 0 in ps:
                raise ValueError(
                    f"sequence {k}: position 0 has no state under bos_policy='none' (nothing "
                    "precedes it). Use bos_policy='auto' or start at position 1."
                )

        results: dict[int, StateResult] = {}

        def forward(indices: list[int]) -> list[StateResult]:
            return self._forward([seqs[i] for i in indices], [poss[i] for i in indices],
                                 bos, return_token_logprobs)

        def on_result(index: int, result: StateResult) -> None:
            results[index] = result

        def on_item_error(index: int, exc: Exception, is_oom: bool) -> None:
            if is_oom:
                raise RuntimeError(
                    f"{exc}\nHint: one sequence of {len(seqs[index])} tokens exhausted device "
                    "memory even alone. Use a smaller model or fewer positions per sequence."
                ) from exc
            raise exc

        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        sizer = AdaptiveBatchSize(batch_size, maximum=batch_size)
        chunks = list(sizer.chunks(order))
        if show_progress:
            from ..utils.batching import maybe_progress
            chunks = maybe_progress(chunks, True, desc="states")
        for indices in chunks:
            run_with_oom_halving(indices, forward, on_result, on_item_error, raise_on_error=True, sizer=sizer)
        return [results[i] for i in range(len(seqs))]

    def _forward(
        self,
        seqs: list[list[int]],
        poss: list[list[int]],
        bos: int | None,
        return_token_logprobs: bool,
    ) -> list[StateResult]:
        import torch

        model = self.provider.model
        device = self.provider.device
        tokenizer = self.provider.tokenizer
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id

        offset = 1 if bos is not None else 0
        full = [([bos] if bos is not None else []) + ids for ids in seqs]
        width = max(len(ids) for ids in full)
        input_ids = torch.full((len(full), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(full), width), dtype=torch.long)
        for row, ids in enumerate(full):
            input_ids[row, : len(ids)] = torch.tensor(ids)
            attention_mask[row, : len(ids)] = 1  # right padding, as the scorer does
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise RuntimeError(
                f"{type(model).__name__} returned no logits. Load the model with "
                "ModelConfig(head='causal_lm') to read next-token states."
            )

        results = []
        for row, (ids, ps) in enumerate(zip(seqs, poss)):
            picked = logits[row, [p + offset for p in ps], :].to(torch.float32)
            states = torch.log_softmax(picked, dim=-1) if len(ps) else picked
            states_np = states.cpu().numpy()
            if not np.isfinite(states_np).all():
                raise ValueError(
                    "non-finite next-token state (inf/nan). This is usually a float16 overflow "
                    "inside the model; re-run with ModelConfig(dtype='float32')."
                )
            token_lp = None
            if return_token_logprobs:
                n = len(ids)
                # Position t (in the padded input) predicts input token t + 1.
                block = logits[row, : n, :] if offset else logits[row, : n - 1, :]
                block = block.to(torch.float32)
                targets = torch.tensor(ids if offset else ids[1:], device=block.device)
                lp = block.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(block, dim=-1)
                token_lp = lp.cpu().numpy()
            results.append(StateResult(
                positions=list(ps), states=states_np, n_tokens=len(ids), token_logprobs=token_lp,
            ))
        return results
