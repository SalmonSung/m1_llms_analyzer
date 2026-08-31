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

DEFAULT_VOCAB = [
    "[UNK]", "[PAD]", "[EOS]",
    "hello", "world", "the", "quick", "brown", "fox", "jumps", "over", "lazy",
    "dog", "a", "b", "c", "model", "layer", "hidden", "state", "token", "test",
    "one", "two", "three", "four", "five", "colab", "vector", "text",
]


def build_tiny_local_model(
    directory: str | Path,
    *,
    num_layers: int = 4,
    hidden_size: int = 16,
    max_positions: int = 32,
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
