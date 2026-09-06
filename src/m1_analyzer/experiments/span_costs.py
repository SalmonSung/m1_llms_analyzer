"""Phase A of a substitution experiment: score every span, cache the raw numbers.

For each sentence, every non-trivial span is replaced by each proform the
policy asks for, the variant is scored, and the *raw* log-probability sum and
predicted-token count are stored -- not the derived cost. That way phase B can
choose mean-per-token or total normalisation, or a different policy over the
same proforms, without touching the GPU.

The cache is JSONL, one line per finished sentence after a header line, written
with ``flush + fsync``. Colab pre-empts free sessions; a re-run with the same
cache path skips finished sentences and refuses (naming the field) if the
header disagrees on model, proforms, or BOS policy.

Each sentence row also carries the answer key (gold spans, source, provenance
and, when given, the original bracketed tree), so one file is a complete record
of the run: `load_span_costs_with_gold` rebuilds both the tables and the
sentences anywhere, with no treebank installed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..utils.batching import maybe_progress
from ..utils.logging import get_logger
from .jsonl_cache import append_row, check_header, mirror, read_jsonl, rewrite, write_header
from .proforms import ReplacementPolicy, substitute
from .spans import Span, enumerate_spans
from .treebank import (
    TreebankSentence,
    detokenize_ptb,
    gold_header_fields,
    sentence_from_json,
    sentence_to_json,
)

log = get_logger("span_costs")

SCHEMA = 2
#: Header fields that may legitimately differ between the writer and a resumer.
_UNCHECKED_HEADER_KEYS = frozenset({"kind", "includes_trees"})
NORMALISATIONS = ("mean", "sum")


@dataclass
class SpanCostTable:
    """Raw scores for one sentence: the base text and every (proform, span) variant."""

    sentence_id: str
    words: list[str]
    text: str
    base_sum: float
    base_n: int
    #: proform -> span -> (sum_logprob, n_tokens)
    spans: dict[str, dict[Span, tuple[float, int]]] = field(default_factory=dict)
    failed: int = 0

    @property
    def n(self) -> int:
        return len(self.words)

    @property
    def base_mean(self) -> float:
        return self.base_sum / self.base_n

    def cost(self, span: Span, proform: str, normalisation: str = "mean") -> float:
        """``L(x) - L(x')``: how much fluency drops when `span` becomes `proform`."""
        s, n = self.spans[proform][span]
        if normalisation == "mean":
            return self.base_mean - s / n
        if normalisation == "sum":
            return self.base_sum - s
        raise ValueError(f"normalisation must be one of {NORMALISATIONS}, got {normalisation!r}")

    def costs_by_proform(self, span: Span, normalisation: str = "mean") -> dict[str, float]:
        return {p: self.cost(span, p, normalisation) for p, table in self.spans.items() if span in table}

    def costs_for(self, policy: ReplacementPolicy, normalisation: str = "mean") -> dict[Span, float]:
        """One cost per span under `policy` (spans with no scored proform are omitted)."""
        out: dict[Span, float] = {}
        for span in enumerate_spans(self.n):
            wanted = policy.candidates(self.words, span[0], span[1])
            available = {p: c for p, c in self.costs_by_proform(span, normalisation).items() if p in wanted}
            if available:
                out[span] = policy.choose(available)
        return out

    # ------------------------------------------------------------- (de)serialise

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "sentence",
            "id": self.sentence_id,
            "words": self.words,
            "text": self.text,
            "base": [self.base_sum, self.base_n],
            "spans": {p: {f"{i},{j}": [s, n] for (i, j), (s, n) in t.items()} for p, t in self.spans.items()},
            "failed": self.failed,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "SpanCostTable":
        spans = {
            p: {tuple(int(x) for x in key.split(",")): (float(v[0]), int(v[1])) for key, v in t.items()}
            for p, t in d["spans"].items()
        }
        return cls(
            sentence_id=d["id"], words=list(d["words"]), text=d["text"],
            base_sum=float(d["base"][0]), base_n=int(d["base"][1]),
            spans=spans, failed=int(d.get("failed", 0)),
        )


# ------------------------------------------------------------------ the cache


def load_span_costs(path: str | os.PathLike) -> tuple[dict, list[SpanCostTable]]:
    """Read a cache back: ``(header, tables)``."""
    header, rows, _ = read_jsonl(Path(path))
    if header is None:
        raise ValueError(f"{path} has no header line; it is not a span-cost cache.")
    return header, [SpanCostTable.from_json(r) for r in rows]


def load_span_costs_with_gold(
    path: str | os.PathLike,
) -> tuple[dict, list[SpanCostTable], list[TreebankSentence]]:
    """Read a cache that embeds its answer key: ``(header, tables, sentences)``.

    Tables and sentences are index-aligned. Needs no NLTK: this is everything
    phase B (`analyse_1b`) takes.
    """
    header, rows, _ = read_jsonl(Path(path))
    if header is None:
        raise ValueError(f"{path} has no header line; it is not a span-cost cache.")
    missing = [r.get("id", "?") for r in rows if "gold_spans" not in r]
    if missing:
        raise ValueError(
            f"{path}: {len(missing)} row(s) carry no gold spans (first: {missing[0]!r}). The cache "
            "was written before the answer key was embedded (schema < 2); pair it with its "
            "gold_*.jsonl via load_gold_jsonl instead."
        )
    return header, [SpanCostTable.from_json(r) for r in rows], [sentence_from_json(r) for r in rows]


def compute_span_costs(
    scorer,
    sentences: Sequence[TreebankSentence],
    policy: ReplacementPolicy,
    *,
    cache_path: str | os.PathLike | None = None,
    provenance: dict[str, Any] | None = None,
    show_progress: bool = True,
    mirror_path: str | os.PathLike | None = None,
    mirror_every: int = 25,
    trees: Mapping[str, str] | None = None,
    **score_kwargs: Any,
) -> list[SpanCostTable]:
    """Score every (span, proform) variant of every sentence; cache and resume.

    `scorer` is anything with ``score(texts, **kwargs) -> ScoreResult``
    (`Analyzer`, `LogProbService`, or a fake). `provenance` goes into the
    header (model id, dtype, BOS token, treebank ...) and is compared on resume.
    `trees` (sentence id -> bracketed parse, from `ptb_tree_strings`) is stored
    with each row so gold can be re-derived later without the corpus.
    Sentences whose base text fails to score are skipped and logged; a variant
    that fails is dropped and counted in ``table.failed``.
    """
    header = {
        "kind": "header", "schema": SCHEMA,
        "policy": policy.name, "proforms": list(policy.proforms),
        **gold_header_fields(trees),
        **(provenance or {}),
    }
    done: dict[str, SpanCostTable] = {}
    path = Path(cache_path) if cache_path else None
    if path is not None and path.exists():
        existing, rows, truncated = read_jsonl(path)
        if existing is not None:
            check_header(existing, header, path, unchecked=_UNCHECKED_HEADER_KEYS)
        for row in rows:
            done[row["id"]] = SpanCostTable.from_json(row)
        if truncated:
            rewrite(path, existing or header, rows)
        log.info("Resuming: %d of %d sentences already scored in %s.", len(done), len(sentences), path)
    elif path is not None:
        write_header(path, header)

    tables: list[SpanCostTable] = []
    fh = path.open("a", encoding="utf-8") if path is not None else None
    since_mirror = 0
    try:
        pending = [s for s in sentences if s.id not in done]
        for sentence in maybe_progress(pending, show_progress, desc="sentences"):
            table = _score_sentence(scorer, sentence, policy, score_kwargs)
            if table is None:
                continue
            done[sentence.id] = table
            if fh is not None:
                append_row(fh, _row(table, sentence, trees))
                since_mirror += 1
                if mirror_path and since_mirror >= mirror_every:
                    mirror(path, mirror_path)
                    since_mirror = 0
    finally:
        if fh is not None:
            fh.close()
        if mirror_path and path is not None and since_mirror:
            mirror(path, mirror_path)
    for sentence in sentences:
        if sentence.id in done:
            tables.append(done[sentence.id])
    return tables


def _row(table: SpanCostTable, sentence: TreebankSentence, trees: Mapping[str, str] | None) -> dict[str, Any]:
    """One cache line: the raw scores plus the sentence's answer key."""
    row = table.to_json()
    for key, value in sentence_to_json(sentence, trees).items():
        row.setdefault(key, value)
    return row


def _score_sentence(scorer, sentence: TreebankSentence, policy: ReplacementPolicy, score_kwargs) -> SpanCostTable | None:
    words = sentence.words
    base_text = sentence.text or detokenize_ptb(words)
    variants: dict[str, list[tuple[str, Span]]] = {}
    for i, j in enumerate_spans(len(words)):
        for proform in policy.candidates(words, i, j):
            text = detokenize_ptb(substitute(words, i, j, proform))
            variants.setdefault(text, []).append((proform, (i, j)))
    texts = [base_text] + list(variants)
    result = scorer.score(texts, **score_kwargs)
    by_text = {s.text: s for s in result.scores}
    if base_text not in by_text:
        log.error("Sentence %s: base text failed to score; skipping it.", sentence.id)
        return None
    base = by_text[base_text]
    table = SpanCostTable(
        sentence_id=sentence.id, words=list(words), text=base_text,
        base_sum=base.sum_logprob, base_n=base.n_tokens,
    )
    for text, owners in variants.items():
        score = by_text.get(text)
        for proform, span in owners:
            if score is None:
                table.failed += 1
                continue
            table.spans.setdefault(proform, {})[span] = (score.sum_logprob, score.n_tokens)
    if table.failed:
        log.warning("Sentence %s: %d variant(s) failed to score.", sentence.id, table.failed)
    return table


def variant_count(sentences: Iterable[TreebankSentence], policy: ReplacementPolicy) -> int:
    """How many texts phase A will score (base + variants), for time estimates."""
    total = 0
    for s in sentences:
        total += 1 + sum(len(policy.candidates(s.words, i, j)) for i, j in enumerate_spans(s.n))
    return total
