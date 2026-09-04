"""Gold brackets from a treebank, in the shape the span algebra scores.

A treebank sentence becomes ``TreebankSentence``: its words (the units spans
are defined over), the detokenised text a language model reads, and the set of
gold spans. Conventions, all of them stated in the record's ``treebank`` field
and in ``docs/design_decisions.md``:

* **Traces are removed.** PTB marks empty elements (``*T*-1``, ``0``, ``*U*``)
  with the POS tag ``-NONE-``. They are not words; a node whose yield becomes
  empty after removing them is dropped too, recursively.
* **Punctuation is removed.** The unsupervised-parsing literature (ON-LSTM,
  Compound PCFG, Kim et al. 2020) scores on punctuation-free sentences; the
  tag set dropped here is theirs: ``, . : `` '' -LRB- -RRB- # $``. ``$`` and
  ``#`` are currency tags in PTB and are dropped with the rest, so "$ 5 million"
  scores as "5 million". Configurable via ``drop_tags``.
* **Unaries collapse and labels vanish.** Gold is a *set* of ``(i, j)`` spans,
  so ``(S (NP ...))`` over the same words counts once and NP/VP/PP play no role
  -- the substitution test yields no category labels, so scoring is unlabelled.
* **Single words and the whole sentence are excluded** (see `spans.py`).

Only the free NLTK sample of the Penn Treebank is wired up today: 10% of the
WSJ portion (3,914 sentences, sections 00-01) with real annotator trees.
Universal Dependencies needs an arc-to-span conversion with conventions of its
own; `load_ud_conllu` is a documented stub until a task needs it.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..utils.logging import get_logger
from .spans import Span, is_trivial

log = get_logger("treebank")

#: POS tags removed before scoring (the ON-LSTM / Kim et al. evaluation convention).
PUNCT_TAGS: frozenset[str] = frozenset({",", ".", ":", "``", "''", "-LRB-", "-RRB-", "#", "$"})
TRACE_TAG = "-NONE-"

PTB_NLTK_NAME = "PTB (NLTK WSJ 10% sample)"

#: Version of the gold-export format written by `save_gold_jsonl`.
GOLD_SCHEMA = 1

#: What the gold spans mean, copied into every export so a file explains itself.
CONVENTIONS = (
    "spans are (first_word, last_word) inclusive, 0-based, over `words`; traces (-NONE-) and "
    "punctuation tags removed with any node they empty; unaries collapsed (gold is a set); "
    "labels dropped (unlabelled scoring); single words and the whole sentence excluded"
)


@dataclass
class TreebankSentence:
    """One sentence with its answer key."""

    id: str
    words: list[str]
    gold_spans: set[Span]
    source: str = "hand"
    text: str | None = None
    #: Free-form provenance (file id, sentence index, original length ...).
    info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.gold_spans = {tuple(s) for s in self.gold_spans}
        bad = [s for s in self.gold_spans if is_trivial(s, self.n)]
        if bad:
            raise ValueError(f"{self.id}: trivial gold spans {sorted(bad)} (single words / whole sentence)")
        if self.text is None:
            self.text = detokenize_ptb(self.words)

    @property
    def n(self) -> int:
        return len(self.words)


# --------------------------------------------------------------- tree -> spans


def gold_spans_from_tree(tree, drop_tags: Iterable[str] = PUNCT_TAGS) -> tuple[list[str], set[Span]]:
    """Words and non-trivial gold spans of an ``nltk.Tree``.

    Leaves under ``-NONE-`` or a dropped tag contribute no word; a node whose
    yield is then empty contributes no span (so emptied parents vanish with
    their children, however deep). Spans are over the surviving words.
    """
    from nltk import Tree

    drop = set(drop_tags) | {TRACE_TAG}
    words: list[str] = []
    spans: set[Span] = set()

    def walk(node) -> int:
        """Append this node's words; return how many it contributed."""
        if not isinstance(node, Tree):
            words.append(str(node))
            return 1
        if len(node) == 1 and not isinstance(node[0], Tree):  # preterminal (POS word)
            if node.label() in drop:
                return 0
            words.append(str(node[0]))
            return 1
        start = len(words)
        count = sum(walk(child) for child in node)
        if count >= 2:
            spans.add((start, start + count - 1))
        return count

    walk(tree)
    n = len(words)
    return words, {s for s in spans if not is_trivial(s, n)}


# ----------------------------------------------------------------- detokenise

_UNESCAPE = (("\\/", "/"), ("\\*", "*"))


def detokenize_ptb(words: Sequence[str]) -> str:
    """PTB tokens -> the string a modern LM reads.

    NLTK's Treebank detokenizer undoes the PTB conventions (``-LRB-``/``-RRB-``,
    ``` `` ``/``''`` quotes, ``do n't``, ``'s``, ``$ 5``, ``5 %``); the escapes
    ``\\/`` and ``\\*`` are unescaped here. Spans are still defined over the
    tokens, so a span boundary can fall inside a contraction ("do | n't"); the
    resulting string is then slightly odd, which is documented and accepted.
    """
    from nltk.tokenize.treebank import TreebankWordDetokenizer

    text = TreebankWordDetokenizer().detokenize(list(words), convert_parentheses=True)
    for old, new in _UNESCAPE:
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------------ loaders


def ensure_nltk_treebank(download: bool = True) -> None:
    """Make ``nltk.corpus.treebank`` available, fetching it once if allowed."""
    import nltk

    try:
        nltk.data.find("corpora/treebank")
    except LookupError:
        if not download:
            raise
        nltk.download("treebank", quiet=True)
        nltk.data.find("corpora/treebank")


def iter_ptb_nltk(drop_tags: Iterable[str] = PUNCT_TAGS, download: bool = True) -> Iterable[TreebankSentence]:
    """Every sentence of the NLTK PTB sample, converted; ids are ``fileid:index``."""
    ensure_nltk_treebank(download)
    from nltk.corpus import treebank

    drop = frozenset(drop_tags)
    for fileid in treebank.fileids():
        for index, tree in enumerate(treebank.parsed_sents(fileid)):
            words, gold = gold_spans_from_tree(tree, drop)
            if not words:
                continue
            yield TreebankSentence(
                id=f"{fileid}:{index}",
                words=words,
                gold_spans=gold,
                source=PTB_NLTK_NAME,
                info={"fileid": fileid, "index": index, "n_leaves": len(tree.leaves())},
            )


def load_ptb_nltk(
    n: int | None = None,
    *,
    min_len: int = 5,
    max_len: int = 30,
    seed: int = 0,
    drop_tags: Iterable[str] = PUNCT_TAGS,
    require_gold: bool = True,
    download: bool = True,
) -> list[TreebankSentence]:
    """A seeded sample of `n` PTB sentences with `min_len..max_len` words.

    ``require_gold`` skips flat sentences that have no non-trivial gold span
    after punctuation removal (nothing to score). The sample is drawn with
    ``random.Random(seed)`` over the eligible ids, then returned in corpus
    order, so the same seed gives the same sentences on every machine.
    """
    eligible = [
        s for s in iter_ptb_nltk(drop_tags, download)
        if min_len <= s.n <= max_len and (s.gold_spans or not require_gold)
    ]
    if n is not None and n < len(eligible):
        chosen = set(random.Random(seed).sample(range(len(eligible)), n))
        eligible = [s for k, s in enumerate(eligible) if k in chosen]
    return eligible


def load_ud_conllu(path: str, **kwargs):  # pragma: no cover - documented stub
    """Universal Dependencies -> spans (subtree yields). Not implemented yet.

    UD draws arcs, not boxes: every word's subtree becomes a span, non-projective
    yields must be dropped, there is no VP node and prepositions hang under
    their noun, so the gold is sparser and *not comparable* with PTB numbers.
    Implement when a task needs it, and put the conversion in the record.
    """
    raise NotImplementedError("UD support is planned; see the docstring for the conversion rules.")


# --------------------------------------------------------------- gold export


def ptb_tree_strings(sentences: Sequence[TreebankSentence], download: bool = True) -> dict[str, str]:
    """Raw bracketed tree per sentence id, for sentences loaded from the NLTK PTB.

    Each file is parsed once, not once per sentence. Sentences without a
    ``fileid`` in their ``info`` (hand examples, other corpora) are skipped.
    """
    wanted: dict[str, list[TreebankSentence]] = {}
    for sentence in sentences:
        fileid = sentence.info.get("fileid")
        if fileid is not None:
            wanted.setdefault(fileid, []).append(sentence)
    if not wanted:
        return {}
    ensure_nltk_treebank(download)
    from nltk.corpus import treebank

    out: dict[str, str] = {}
    for fileid, group in wanted.items():
        parsed = treebank.parsed_sents(fileid)
        for sentence in group:
            index = sentence.info.get("index")
            if index is not None and index < len(parsed):
                out[sentence.id] = parsed[index].pformat(margin=1_000_000)
    return out


def save_gold_jsonl(
    sentences: Sequence[TreebankSentence],
    path: str | os.PathLike,
    *,
    provenance: Mapping[str, Any] | None = None,
    trees: Mapping[str, str] | None = None,
) -> str:
    """Write the answer key as JSONL: a header, then one line per sentence.

    This is what makes a span-cost cache re-analysable somewhere else. The cache
    identifies sentences by id but carries no gold, and rebuilding gold needs
    NLTK, the corpus, and the same conventions. Exporting it pairs the two files
    into a self-contained record of the run. Pass `trees` (from
    `ptb_tree_strings`) to include the original bracketed parse, so gold can be
    re-derived under different conventions without the corpus.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "kind": "header",
        "schema": GOLD_SCHEMA,
        "n_sentences": len(sentences),
        "conventions": CONVENTIONS,
        "punct_tags": sorted(PUNCT_TAGS),
        "trace_tag": TRACE_TAG,
        "includes_trees": bool(trees),
        **(provenance or {}),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        for sentence in sentences:
            row: dict[str, Any] = {
                "kind": "sentence",
                "id": sentence.id,
                "n": sentence.n,
                "words": list(sentence.words),
                "text": sentence.text,
                "gold_spans": [list(sp) for sp in sorted(sentence.gold_spans)],
                "source": sentence.source,
                "info": sentence.info,
            }
            if trees and sentence.id in trees:
                row["tree"] = trees[sentence.id]
            fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    log.info("Wrote %d gold sentence(s) to %s", len(sentences), path)
    return str(path)


def load_gold_jsonl(path: str | os.PathLike) -> tuple[dict[str, Any], list[TreebankSentence]]:
    """Read a gold export back: ``(header, sentences)``. Needs no NLTK."""
    header: dict[str, Any] | None = None
    sentences: list[TreebankSentence] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("kind") == "header":
                header = obj
                continue
            sentences.append(
                TreebankSentence(
                    id=obj["id"],
                    words=list(obj["words"]),
                    gold_spans={tuple(sp) for sp in obj["gold_spans"]},
                    source=obj.get("source", "unknown"),
                    text=obj.get("text"),
                    info=dict(obj.get("info", {}), **({"tree": obj["tree"]} if "tree" in obj else {})),
                )
            )
    if header is None:
        raise ValueError(f"{path} has no header line; it is not a gold export.")
    return header, sentences


# ------------------------------------------------------------- hand examples


def theory_example() -> TreebankSentence:
    """The teaching doc's sentence with its seven hand-written PTB-style gold spans."""
    words = "The tall man with the red hat quickly opened the heavy wooden door".split()
    gold = {(0, 2), (4, 6), (3, 6), (0, 6), (9, 12), (8, 12), (7, 12)}
    return TreebankSentence(id="theory-doc", words=words, gold_spans=gold, source="hand parse (theory doc)")


def hand_examples() -> list[TreebankSentence]:
    """Three short hand-parsed sentences for smoke tests (no treebank needed)."""
    return [
        theory_example(),
        TreebankSentence(
            id="hand-1", words="The old dog slept on the warm porch".split(),
            gold_spans={(0, 2), (5, 7), (4, 7), (3, 7)}, source="hand parse",
        ),
        TreebankSentence(
            id="hand-2", words="My sister quickly finished the long book".split(),
            gold_spans={(0, 1), (4, 6), (3, 6), (2, 6)}, source="hand parse",
        ),
    ]
