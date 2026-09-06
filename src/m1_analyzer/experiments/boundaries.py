"""Where a sentence or a clause ends, as token positions.

Task 9a's admissibility rule is applied *by construction*: a cut joins two
endpoints of the same boundary type, so a sentence-final endpoint is spliced to
a sentence start and whole sentences are removed. That rule is only as good as
the boundary finder, so it is deliberately conservative and every rejection is
counted rather than silently dropped.

Sentence-final positions
    The text is split into sentences (``splitter="punkt"``, NLTK's trained
    model, which knows ``Mr.`` and ``U.S.`` are not sentence ends; or
    ``"regex"``, a terminal-punctuation rule for offline tests). Each sentence's
    last non-space character is mapped to the token that contains it through
    the tokenizer's offset mapping. The position is kept only when that token
    ends exactly at the sentence end (a token straddling two sentences is not a
    boundary), the sentence ends in ``. ! ?`` optionally followed by closing
    quotes or brackets, and whitespace separates it from the next token.

Clause-final positions
    Tokens whose text ends in ``, ; :`` or a dash, followed by whitespace, that
    are not already sentence-final. A comma-to-comma splice is *not* grammatical
    by construction (list commas, appositives, sentence adverbs), which is why
    clause cuts are a secondary boundary type in the record, never the primary.
"""

from __future__ import annotations

import re
from typing import Sequence

SENTENCE = "sentence"
CLAUSE = "clause"
BOUNDARY_KINDS = (SENTENCE, CLAUSE)
SPLITTERS = ("punkt", "regex")

END_PUNCT = ".!?"
CLOSERS = "\"'”’)]"
OPENERS = "\"'“‘(["
CLAUSE_PUNCT = (",", ";", ":", "—", "–", "--")

#: A sentence ends at terminal punctuation (plus closers) followed by whitespace,
#: or at the end of the text. No abbreviation knowledge: that is punkt's job.
_REGEX_END = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s+\S|\s*$)")


def sentence_char_spans(text: str, splitter: str = "punkt") -> list[tuple[int, int]]:
    """``[(start, end), ...]`` character spans of the sentences of `text`."""
    if splitter not in SPLITTERS:
        raise ValueError(f"splitter must be one of {SPLITTERS}, got {splitter!r}")
    if splitter == "punkt":
        return _punkt_spans(text)
    return _regex_spans(text)


def _punkt_spans(text: str) -> list[tuple[int, int]]:
    import nltk

    try:
        try:
            from nltk.tokenize import PunktTokenizer  # nltk >= 3.9

            tokenizer = PunktTokenizer("english")
        except ImportError:  # pragma: no cover - older nltk
            tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    except LookupError as exc:
        raise LookupError(
            "NLTK's punkt sentence model is not installed. Run "
            "`python -c \"import nltk; nltk.download('punkt_tab')\"` (or use splitter='regex', "
            "which is what the offline tests do)."
        ) from exc
    return [(int(s), int(e)) for s, e in tokenizer.span_tokenize(text)]


def _regex_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for match in _REGEX_END.finditer(text):
        end = match.end()
        while start < end and text[start].isspace():
            start += 1
        if start < end:
            spans.append((start, end))
        start = end
    while start < len(text) and text[start].isspace():
        start += 1
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _token_at(offsets: Sequence[tuple[int, int]], char: int) -> int | None:
    for t, (s, e) in enumerate(offsets):
        if s <= char < e:
            return t
    return None


def sentence_final_positions(
    text: str, offsets: Sequence[tuple[int, int]], splitter: str = "punkt",
) -> tuple[list[int], int]:
    """``(positions, n_rejected)``: tokens that end a sentence, and how many
    sentence ends the rule refused (no terminal punctuation, a token straddling
    the boundary, no whitespace before the next token, or no covering token)."""
    positions, rejected = [], 0
    for start, end in sentence_char_spans(text, splitter):
        last = end - 1
        while last >= start and text[last].isspace():
            last -= 1
        if last < start:
            continue
        core = last
        while core >= start and text[core] in CLOSERS:
            core -= 1
        if core < start or text[core] not in END_PUNCT:
            rejected += 1
            continue
        t = _token_at(offsets, last)
        if t is None or offsets[t][1] != last + 1:
            rejected += 1
            continue
        if t + 1 < len(offsets) and offsets[t + 1][0] == last + 1:
            rejected += 1  # no whitespace between the sentences: a doubtful split
            continue
        positions.append(t)
    return positions, rejected


def clause_final_positions(
    text: str, offsets: Sequence[tuple[int, int]], exclude: Sequence[int] = (),
) -> list[int]:
    """Tokens ending in clause punctuation followed by whitespace (not sentence-final)."""
    skip = set(exclude)
    positions = []
    for t, (s, e) in enumerate(offsets):
        if t in skip or e <= s:
            continue
        piece = text[s:e].rstrip()
        if not piece or not piece.endswith(CLAUSE_PUNCT):
            continue
        if t + 1 < len(offsets) and offsets[t + 1][0] == s + len(piece):
            continue  # "1,000" or "3:45": not a clause end
        positions.append(t)
    return positions


def boundary_positions(
    text: str, offsets: Sequence[tuple[int, int]], *, splitter: str = "punkt",
    kinds: Sequence[str] = BOUNDARY_KINDS,
) -> tuple[dict[str, list[int]], int]:
    """``({kind: positions}, n_rejected_sentence_ends)`` for the requested kinds."""
    unknown = [k for k in kinds if k not in BOUNDARY_KINDS]
    if unknown:
        raise ValueError(f"unknown boundary kind(s) {unknown}; choose from {BOUNDARY_KINDS}")
    sentence, rejected = sentence_final_positions(text, offsets, splitter)
    out: dict[str, list[int]] = {}
    if SENTENCE in kinds:
        out[SENTENCE] = sentence
    if CLAUSE in kinds:
        out[CLAUSE] = clause_final_positions(text, offsets, exclude=sentence)
    return out, rejected
