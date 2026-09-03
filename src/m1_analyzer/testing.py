"""Offline test fixtures.

Builds a randomly-initialised, few-hundred-KB GPT-2-shaped model and a tiny
word-level tokenizer entirely in-process, then saves them in Hugging Face layout
so `ModelService` can load them through the exact same `from_pretrained` path a
real model uses. No network, no Hub, no cached weights.

This exists so the test suite and `scripts/smoke_test.py --offline` can verify
the whole pipeline (loading, layer selection, pooling, batching, serialisation)
in a sandbox with no internet, and so CI never depends on the Hub being up. The
outputs are meaningless numbers -- it tests plumbing, not model quality.
"""

from __future__ import annotations

from pathlib import Path

#: Shape of the model `build_tiny_local_model` produces. Tests assert against these, so
#: they live here rather than in `tests/conftest.py`: a test module importing
#: `tests.conftest` only resolves when the repo root happens to be on `sys.path`, which is
#: true for `python -m pytest` but not for the `pytest` console script.
TINY_LAYERS = 4
TINY_HIDDEN = 16
TINY_MAX_POSITIONS = 32

DEFAULT_VOCAB = [
    "[UNK]", "[PAD]", "[EOS]",
    "hello", "world", "the", "quick", "brown", "fox", "jumps", "over", "lazy",
    "dog", "a", "b", "c", "model", "layer", "hidden", "state", "token", "test",
    "one", "two", "three", "four", "five", "colab", "vector", "text",
]


def build_tiny_local_model(
    directory: str | Path,
    *,
    num_layers: int = TINY_LAYERS,
    hidden_size: int = TINY_HIDDEN,
    max_positions: int = TINY_MAX_POSITIONS,
) -> str:
    """Create a tiny GPT-2 model + tokenizer on disk. Returns the directory path.

    The directory is a drop-in ``model_id`` for `ModelConfig`, because
    `from_pretrained` accepts a local path exactly like a Hub repo name.
    """
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2Model, PreTrainedTokenizerFast

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    vocab = {token: i for i, token in enumerate(DEFAULT_VOCAB)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        eos_token="[EOS]",
        # Deliberately no pad_token: exercises ModelService's eos-as-pad fallback.
    )
    tokenizer.model_max_length = max_positions

    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=max_positions,
        n_embd=hidden_size,
        n_layer=num_layers,
        n_head=2,
        n_inner=hidden_size * 2,
        # Defaults point at GPT-2's real vocab (50256), which is out of range here.
        bos_token_id=vocab["[EOS]"],
        eos_token_id=vocab["[EOS]"],
    )
    torch.manual_seed(0)
    model = GPT2Model(config)

    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return str(directory)


# ------------------------------------------------------------- fake scorer

#: Costs the fake scorer assigns to a span by its relation to the gold tree.
FAKE_COST_GOLD = 0.3
FAKE_COST_NEUTRAL = 1.0
FAKE_COST_CROSSING = 2.0


class FakeSpanScorer:
    """A `SequenceScorer` that knows the answer: gold spans are cheap, crossing
    spans dear, everything else in between, with a little seeded noise.

    The tiny local model cannot test replacement policies -- its word-level
    vocabulary maps ``it``/``there``/``did``/``then`` all to ``[UNK]`` -- so the
    experiment analyses are tested against this fake instead. It reproduces
    the *shape* of a passing Task 1b run (substitution beats right-branching),
    which is what the plumbing tests need; it says nothing about any model.
    """

    def __init__(self, sentences, proforms=("it", "there", "did", "then"), *, noise=0.15, seed=1,
                 base_per_token=-3.0):
        import numpy as np

        from .domain.records import make_item_id
        from .experiments.proforms import substitute
        from .experiments.spans import crosses, enumerate_spans
        from .experiments.treebank import detokenize_ptb

        self._make_id = make_item_id
        rng = np.random.default_rng(seed)
        self.table: dict[str, tuple[float, int]] = {}
        self.calls = 0
        for s in sentences:
            self.table[s.text] = (base_per_token * s.n, s.n)
            for (i, j) in enumerate_spans(s.n):
                if (i, j) in s.gold_spans:
                    cost = FAKE_COST_GOLD
                elif any(crosses((i, j), g) for g in s.gold_spans):
                    cost = FAKE_COST_CROSSING
                else:
                    cost = FAKE_COST_NEUTRAL
                for k, proform in enumerate(proforms):
                    text = detokenize_ptb(substitute(s.words, i, j, proform))
                    n = s.n - (j - i)
                    jitter = float(rng.normal(0, noise)) + 0.05 * k
                    self.table.setdefault(text, ((base_per_token - cost - jitter) * n, n))

    def score(self, texts, **overrides):
        from .domain.records import ExtractionFailure, ScoreResult, SentenceScore

        self.calls += 1
        scores, failures = [], []
        for index, text in enumerate(texts):
            if text not in self.table:
                failures.append(ExtractionFailure(
                    id=self._make_id(text), text_preview=text, error_type="KeyError",
                    error="fake scorer has no entry for this text", index=index,
                ))
                continue
            total, n = self.table[text]
            scores.append(SentenceScore(id=self._make_id(text), text=text, n_tokens=n, sum_logprob=total))
        return ScoreResult(scores=scores, failures=failures)
