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
    quotes or brackets, a whitespace *character* follows in the text, and what
    follows **begins a new sentence**. The whitespace test reads the text, never
    the next token's offset: byte-level BPE tokenizers (Qwen, GPT-2 with
    ``trim_offsets=False``) report ``" The"`` as starting at the space, so an
    offset comparison rejects every boundary.

    The last of those -- the *right-side* test -- exists because a splice uses a
    boundary at both ends: the head must end a sentence AND the tail must resume
    at one. A hand audit of a 200-paragraph Qwen run found three ways the
    left-side tests alone let a mid-sentence position through:

    * ``"... Vol." + "4 (1972) and its successors ..."`` -- punkt split on an
      abbreviation it does not know, stranding a numeral;
    * ``"... a monastery." + "at the court of the Frankish monarchy ..."`` -- the
      corpus extractor dropped an italicised term, so the sentence starts lowercase;
    * ``"... the Gallic Wars." + ", 39 volumes have been released"`` -- a dropped
      ``As of <date>`` template left the next sentence headless.

    Such a position is mid-sentence, so its state is atypical for a sentence end
    and therefore *far* from genuine sentence-end states: in that run 13.5% of the
    "far" group but only 0.5% of the "close" group touched one. The defect
    correlates with the exposure variable, which is why it is rejected by
    construction rather than left to the audit.

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


def starts_sentence(text: str, pos: int) -> bool:
    """Does the text from `pos` begin a new sentence?

    Skips whitespace and opening quotes/brackets, then requires an uppercase
    letter. A lowercase letter (a dropped word), a digit (a stranded ``Vol. 4``)
    or punctuation (a headless ``, 39 volumes``) means `pos` is mid-sentence.
    Running out of text is fine -- that is the end of the paragraph, not a
    broken sentence. A caseless script (CJK, Hebrew, Arabic) is accepted, since
    it carries no capitalisation to test.
    """
    tail = text[pos:].lstrip()
    if not tail:
        return True
    k = 0
    while k < len(tail) and tail[k] in OPENERS:
        k += 1
    if k >= len(tail):
        return False
    char = tail[k]
    if char.isupper():
        return True
    if char.islower() or char.isdigit() or not char.isalnum():
        return False
    return True


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
        if last + 1 < len(text) and not text[last + 1].isspace():
            rejected += 1  # no whitespace between the sentences: a doubtful split
            continue
        if not starts_sentence(text, last + 1):
            rejected += 1  # what follows does not begin a sentence: `last` is mid-sentence
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
        after = s + len(piece)
        if after < len(text) and not text[after].isspace():
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
