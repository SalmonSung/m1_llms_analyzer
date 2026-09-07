"""Sentence and clause boundaries as token positions, with every rejection counted."""

import re

import pytest

from m1_analyzer.experiments.boundaries import (
    boundary_positions,
    clause_final_positions,
    sentence_char_spans,
    sentence_final_positions,
    starts_sentence,
)

TEXT = 'The dog sat. He said "hello." Then, at 3:45, we left; the rest stayed. Prices rose 1,000 units!'


def _offsets(text):
    """Whitespace tokens with punctuation attached; offsets exclude the space (GPT-2 trims them)."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _offsets_with_spaces(text):
    """The same tokens with the leading space INSIDE the offset, as Qwen's tokenizer reports them."""
    return [(m.start(), m.end()) for m in re.finditer(r"\s*\S+", text)]


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


def test_offsets_that_include_the_leading_space_still_find_every_boundary():
    """Qwen reports ' The' as starting at the space; this rejected 6 of 7 sentence ends in
    the first real run. The whitespace test must read the text, not the next offset."""
    offsets = _offsets_with_spaces(TEXT)
    positions, rejected = sentence_final_positions(TEXT, offsets, "regex")
    assert [TEXT[offsets[p][0]:offsets[p][1]].strip() for p in positions] == ["sat.", '"hello."', "stayed.", "units!"]
    assert rejected == 0
    clause = clause_final_positions(TEXT, offsets, exclude=positions)
    assert [TEXT[offsets[p][0]:offsets[p][1]].strip() for p in clause] == ["Then,", "3:45,", "left;"]


def test_unpunctuated_and_straddling_ends_are_rejected_not_kept():
    text = "The dog sat. The cat ran"
    offsets = _offsets(text)
    positions, rejected = sentence_final_positions(text, offsets, "regex")
    assert positions == [2] and rejected == 1  # "ran" has no terminal punctuation
    # A token that runs past the sentence end (here: one token covering "sat. The") is not a boundary.
    straddle = [(0, 3), (4, 7), (8, 16), (17, 20), (21, 25)]
    positions, rejected = sentence_final_positions("The dog sat. The cat ran.", straddle, "regex")
    assert positions == [4] and rejected == 1


def test_an_all_lowercase_paragraph_yields_no_usable_boundary():
    """The right-side rule needs a capital, so lowercase prose has no usable
    endpoint. Deliberate: a corpus without capitalisation cannot be spliced safely.

    The paragraph-final boundary survives -- nothing follows it to fail the test --
    but it is unusable in a cut: it can never be `i` (no later boundary to pair
    with) and never `j` (no window after it)."""
    text = "the dog sat. the cat ran. the end."
    positions, rejected = sentence_final_positions(text, _offsets(text), "regex")
    assert positions == [7] and rejected == 2
    assert text.__getitem__(slice(*_offsets(text)[7])) == "end."  # the paragraph-final token


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


# --------------------------------------------------------------- right-side rule


def test_starts_sentence_accepts_a_real_start_and_rejects_the_three_audit_shapes():
    assert starts_sentence('The dog sat. He ran.', 13)          # ordinary
    assert starts_sentence('He said. "Yes." Then.', 9)          # opening quote
    assert starts_sentence('It ended.', 9)                      # end of paragraph
    assert starts_sentence('Ends. 日本語の文.', 6)               # caseless script: accepted
    # the three failure shapes the Qwen audit turned up
    assert not starts_sentence('... a monastery. at the court of the Frankish', 17)   # lowercase
    assert not starts_sentence('... Vol. 4 (1972) and its successors', 9)             # stranded digit
    assert not starts_sentence('... the Gallic Wars. , 39 volumes have been', 20)     # headless comma
    assert not starts_sentence('Ends. "', 6)                                          # only an opener left


def test_a_boundary_not_followed_by_a_sentence_start_is_rejected_and_counted():
    """The defect that put 13.5% of the far group on mid-sentence endpoints."""
    for text, kept in (
        # punkt-style split after the abbreviation "Vol." strands a numeral
        ("The band formed. Their albums Vol. 4 (1972) sold well. It ended.", ["formed.", "well.", "ended."]),
        # a dropped "As of <date>" template leaves the next sentence headless
        ("It was written. , 39 volumes have been released. The end.", ["released.", "end."]),
    ):
        offsets = _offsets(text)
        positions, rejected = sentence_final_positions(text, offsets, "regex")
        assert [text[offsets[p][0]:offsets[p][1]] for p in positions] == kept
        assert rejected == 1, f"expected exactly one right-side rejection in {text!r}"


def test_the_rejection_applies_to_both_offset_conventions():
    text = "The band formed. Their albums Vol. 4 (1972) sold well."
    for offsets in (_offsets(text), _offsets_with_spaces(text)):
        positions, rejected = sentence_final_positions(text, offsets, "regex")
        assert [text[offsets[p][0]:offsets[p][1]].strip() for p in positions] == ["formed.", "well."]
        assert rejected == 1


def test_clause_boundaries_are_not_subject_to_the_sentence_start_rule():
    """A clause end is followed by lowercase BY DESIGN; the rule must not touch it."""
    text = "Then, at noon, we left. It ended."
    offsets = _offsets(text)
    sentence, _ = sentence_final_positions(text, offsets, "regex")
    clause = clause_final_positions(text, offsets, exclude=sentence)
    assert [text[offsets[p][0]:offsets[p][1]] for p in clause] == ["Then,", "noon,"]
