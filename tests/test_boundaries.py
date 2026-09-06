"""Sentence and clause boundaries as token positions, with every rejection counted."""

import re

import pytest

from m1_analyzer.experiments.boundaries import (
    boundary_positions,
    clause_final_positions,
    sentence_char_spans,
    sentence_final_positions,
)

TEXT = 'The dog sat. He said "hello." Then, at 3:45, we left; the rest stayed. Prices rose 1,000 units!'


def _offsets(text):
    """Whitespace tokens with punctuation attached, like a BPE tokenizer's leading-space pieces."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def test_regex_spans_cover_every_sentence():
    spans = sentence_char_spans(TEXT, "regex")
    assert [TEXT[s:e] for s, e in spans] == [
        "The dog sat.", 'He said "hello."', "Then, at 3:45, we left; the rest stayed.", "Prices rose 1,000 units!",
    ]


def test_sentence_final_positions_map_to_the_closing_token():
    offsets = _offsets(TEXT)
    positions, rejected = sentence_final_positions(TEXT, offsets, "regex")
    assert [TEXT[offsets[p][0]:offsets[p][1]] for p in positions] == ["sat.", '"hello."', "stayed.", "units!"]
    assert rejected == 0


def test_unpunctuated_and_straddling_ends_are_rejected_not_kept():
    text = "the dog sat. the cat ran"
    offsets = _offsets(text)
    positions, rejected = sentence_final_positions(text, offsets, "regex")
    assert positions == [2] and rejected == 1  # "ran" has no terminal punctuation
    # A token that runs past the sentence end (here: one token covering "sat. the") is not a boundary.
    straddle = [(0, 3), (4, 7), (8, 16), (17, 20), (21, 25)]
    positions, rejected = sentence_final_positions("the dog sat. the cat ran.", straddle, "regex")
    assert positions == [4] and rejected == 1


def test_clause_final_positions_skip_numbers_and_sentence_ends():
    offsets = _offsets(TEXT)
    sentence, _ = sentence_final_positions(TEXT, offsets, "regex")
    clause = clause_final_positions(TEXT, offsets, exclude=sentence)
    assert [TEXT[offsets[p][0]:offsets[p][1]] for p in clause] == ["Then,", "3:45,", "left;"]


def test_boundary_positions_by_kind():
    offsets = _offsets(TEXT)
    out, rejected = boundary_positions(TEXT, offsets, splitter="regex", kinds=("sentence",))
    assert set(out) == {"sentence"} and rejected == 0
    with pytest.raises(ValueError, match="unknown boundary"):
        boundary_positions(TEXT, offsets, splitter="regex", kinds=("word",))
    with pytest.raises(ValueError, match="splitter"):
        sentence_char_spans(TEXT, "spacy")


def test_punkt_is_named_when_missing_or_used_when_present():
    try:
        spans = sentence_char_spans("Dr. Smith went home. He slept.", "punkt")
    except LookupError as exc:
        assert "punkt_tab" in str(exc)
    else:
        assert len(spans) == 2  # punkt knows "Dr." is not a sentence end
