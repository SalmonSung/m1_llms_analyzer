"""Paragraphs for the splice experiment (Task 9a).

The unit is a paragraph of 150-300 tokens with several sentences, because a
cut removes whole sentences and needs `window` tokens of text after the rejoin.
Two sources:

* ``load_wikipedia_paragraphs``: the ``wikimedia/wikipedia`` dump streamed
  through the `datasets` library. It is untokenised prose with real paragraph
  breaks, so no detokeniser stands between the corpus and the model (wikitext,
  even its "raw" variant, is Moses-tokenised: ``word ,`` and ``@-@``). Streaming
  reads only the articles it consumes; nothing is downloaded in bulk.
* ``load_paragraph_file``: a plain-text file with blank-line-separated
  paragraphs, or a JSONL file with a ``text`` field, so any corpus can be
  dropped in.

Selection is deterministic: paragraphs are taken in source order into a pool
of ``pool_factor * n`` qualifying candidates, then a seeded sample of ``n`` is
drawn and returned in source order. The filter needs the model's own token
count, so callers pass ``count_tokens`` (``analyzer.states.encode`` works).
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from ..domain.records import make_item_id
from ..utils.logging import get_logger
from .boundaries import CLOSERS, END_PUNCT, sentence_char_spans

log = get_logger("paragraphs")

WIKIPEDIA_DATASET = "wikimedia/wikipedia"
WIKIPEDIA_CONFIG = "20231101.en"
WIKIPEDIA_NAME = "Wikipedia (wikimedia/wikipedia, 20231101.en, streamed)"

_WS = re.compile(r"\s+")


@dataclass
class Paragraph:
    """One paragraph and where it came from."""

    id: str
    text: str
    source: str = "file"
    info: dict = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "source": self.source, "info": dict(self.info)}

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "Paragraph":
        return cls(id=str(obj["id"]), text=str(obj["text"]), source=str(obj.get("source", "file")),
                   info=dict(obj.get("info", {})))


def clean_paragraph(text: str) -> str | None:
    """Collapse whitespace; None unless the text ends like a sentence."""
    text = _WS.sub(" ", text).strip()
    if not text:
        return None
    last = len(text) - 1
    while last >= 0 and text[last] in CLOSERS:
        last -= 1
    if last < 0 or text[last] not in END_PUNCT:
        return None
    return text


def count_sentences(text: str, splitter: str = "punkt") -> int:
    return len(sentence_char_spans(text, splitter))


def select_paragraphs(
    candidates: Iterable[tuple[str, str, dict]],
    count_tokens: Callable[[str], int],
    *,
    n: int = 200,
    min_tokens: int = 150,
    max_tokens: int = 300,
    min_sentences: int = 5,
    splitter: str = "punkt",
    seed: int = 42,
    pool_factor: int = 3,
    source: str = "file",
) -> list[Paragraph]:
    """Filter ``(id, text, info)`` candidates and draw a seeded sample of `n`.

    ``count_tokens(text)`` is the model's own token count. The pool holds the
    first ``pool_factor * n`` qualifying paragraphs in source order; the sample
    is drawn from it with `seed` and returned in source order. Rejections are
    counted by reason and logged.
    """
    if n < 1:
        raise ValueError("n must be >= 1.")
    pool: list[Paragraph] = []
    reasons = {"not_prose": 0, "tokens": 0, "sentences": 0}
    target = max(n, pool_factor * n)
    for pid, raw, info in candidates:
        text = clean_paragraph(raw)
        if text is None:
            reasons["not_prose"] += 1
            continue
        n_tok = count_tokens(text)
        if not min_tokens <= n_tok <= max_tokens:
            reasons["tokens"] += 1
            continue
        n_sent = count_sentences(text, splitter)
        if n_sent < min_sentences:
            reasons["sentences"] += 1
            continue
        pool.append(Paragraph(id=pid, text=text, source=source,
                              info={**info, "n_tokens": n_tok, "n_sentences": n_sent, "pool_index": len(pool)}))
        if len(pool) >= target:
            break
    if not pool:
        raise ValueError(
            f"No paragraph qualified (rejected: {reasons}). Loosen min_tokens/max_tokens/min_sentences."
        )
    if len(pool) < n:
        log.warning("Only %d paragraph(s) qualified, fewer than the %d asked for (rejected: %s).",
                    len(pool), n, reasons)
        chosen = list(pool)
    else:
        rng = random.Random(seed)
        chosen = sorted(rng.sample(pool, n), key=lambda p: p.info["pool_index"])
    log.info("Selected %d paragraphs from a pool of %d (rejected: %s).", len(chosen), len(pool), reasons)
    return chosen


# ------------------------------------------------------------------ sources


def iter_text_file(path: str | os.PathLike) -> Iterator[tuple[str, str, dict]]:
    """Blank-line-separated paragraphs of a text file, or the ``text`` of each JSONL line."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for k, line in enumerate(fh):
                if not line.strip():
                    continue
                obj = json.loads(line)
                yield str(obj.get("id", f"{path.stem}:{k}")), str(obj["text"]), {"line": k}
        return
    text = path.read_text(encoding="utf-8")
    for k, block in enumerate(re.split(r"\n\s*\n", text)):
        if block.strip():
            yield f"{path.stem}:{k}", block, {"block": k}


def load_paragraph_file(path: str | os.PathLike, count_tokens: Callable[[str], int], **kwargs: Any) -> list[Paragraph]:
    """`select_paragraphs` over a text or JSONL file."""
    kwargs.setdefault("source", str(path))
    return select_paragraphs(iter_text_file(path), count_tokens, **kwargs)


def iter_wikipedia_paragraphs(
    config: str = WIKIPEDIA_CONFIG,
    *,
    max_articles: int | None = None,
    max_per_article: int = 2,
    min_chars: int = 200,
    dataset: str = WIKIPEDIA_DATASET,
) -> Iterator[tuple[str, str, dict]]:
    """Stream Wikipedia articles and yield their paragraphs (lines of the text).

    Headings and list items are short or unpunctuated and fall to
    `clean_paragraph` / the length filter downstream. At most `max_per_article`
    paragraphs per article keep one long article from dominating the corpus.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency message
        raise ImportError("Streaming Wikipedia needs the `datasets` package: pip install datasets") from exc
    stream = load_dataset(dataset, config, split="train", streaming=True)
    for a, article in enumerate(stream):
        if max_articles is not None and a >= max_articles:
            break
        taken = 0
        for k, line in enumerate(str(article.get("text", "")).split("\n")):
            if len(line.strip()) < min_chars:
                continue
            yield f"wiki:{article.get('id', a)}:{k}", line, {"title": article.get("title"), "article": a, "line": k}
            taken += 1
            if taken >= max_per_article:
                break


def load_wikipedia_paragraphs(count_tokens: Callable[[str], int], *, config: str = WIKIPEDIA_CONFIG,
                              max_articles: int | None = 20_000, max_per_article: int = 2, **kwargs: Any) -> list[Paragraph]:
    """`select_paragraphs` over the streamed Wikipedia dump."""
    kwargs.setdefault("source", WIKIPEDIA_NAME)
    return select_paragraphs(
        iter_wikipedia_paragraphs(config, max_articles=max_articles, max_per_article=max_per_article),
        count_tokens, **kwargs,
    )


# ------------------------------------------------------------- hand examples


def hand_paragraphs() -> list[Paragraph]:
    """Three small paragraphs for smoke tests: several short sentences each,
    spelled with the tiny test model's vocabulary so its word-level tokenizer
    can read them (any real tokenizer reads them too)."""
    texts = [
        "the quick brown fox jumps over the lazy dog. the dog jumps over the fox. "
        "the fox jumps over the model. the model jumps over the layer. the layer jumps over the state. "
        "the state jumps over the token. the token jumps over the text. the text jumps over the dog.",
        "hello world. the quick fox jumps. the brown dog jumps. the lazy fox jumps over the dog. "
        "one two three four five. the model jumps over the hidden state. the token jumps over the vector. "
        "the vector jumps over the text. the text jumps over the world.",
        "a b c. one two three. the quick brown fox jumps over the lazy dog. the lazy dog jumps over the quick fox. "
        "four five one two. the model jumps over the layer. the hidden state jumps over the token. "
        "the colab text jumps over the vector. the vector jumps over the model.",
    ]
    return [Paragraph(id=f"hand-{k}", text=t, source="hand", info={"index": k}) for k, t in enumerate(texts)]


def paragraph_id(text: str) -> str:
    return make_item_id(text)
