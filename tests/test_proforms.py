"""Replacement policies: substitution text, candidates, choice, parsing."""

import pytest

from m1_analyzer.experiments.proforms import (
    DEFAULT_PROFORMS,
    ByLengthClass,
    MinOverSet,
    ReplacementPolicy,
    parse_policy,
    substitute,
)

WORDS = "The tall man opened the door".split()


def test_substitute_replaces_the_run_and_capitalises_at_sentence_start():
    assert substitute(WORDS, 0, 2, "he") == ["He", "opened", "the", "door"]
    assert substitute(WORDS, 4, 5, "it") == ["The", "tall", "man", "opened", "it"]
    assert substitute(WORDS, 1, 1, "x") == ["The", "x", "man", "opened", "the", "door"]
    with pytest.raises(ValueError):
        substitute(WORDS, 3, 9, "it")


def test_min_over_set_scores_every_proform_and_takes_the_cheapest():
    policy = MinOverSet(DEFAULT_PROFORMS)
    assert isinstance(policy, ReplacementPolicy)
    assert policy.candidates(WORDS, 0, 2) == list(DEFAULT_PROFORMS)
    assert policy.choose({"it": 0.9, "there": 0.4, "did": 1.2}) == 0.4
    assert policy.proforms == DEFAULT_PROFORMS
    assert "it" in policy.name
    with pytest.raises(KeyError):
        policy.choose({"he": 0.1})
    with pytest.raises(ValueError):
        MinOverSet([])


def test_by_length_class_picks_one_proform_per_length():
    policy = ByLengthClass({(2, 3): "it", (4, 6): "that", (7, None): "this"})
    assert policy.candidates(WORDS, 0, 1) == ["it"]
    assert policy.candidates(WORDS, 0, 3) == ["that"]
    assert policy.proform_for(40) == "this"
    assert policy.proforms == ("it", "that", "this")
    assert policy.choose({"that": 0.7}) == 0.7
    with pytest.raises(ValueError):
        policy.proform_for(1)


def test_parse_policy_specs():
    assert parse_policy("it,there").proforms == ("it", "there")
    assert parse_policy("min: it, did").proforms == ("it", "did")
    lengths = parse_policy("length:2-3=it,4-6=that,7+=this")
    assert isinstance(lengths, ByLengthClass)
    assert lengths.proform_for(5) == "that"
    with pytest.raises(ValueError):
        parse_policy("length:2-3")


def test_deletion_drops_the_span_and_recapitalises_at_sentence_start():
    from m1_analyzer.experiments.proforms import DELETION
    from m1_analyzer.experiments.treebank import detokenize_ptb

    words = "The tall man opened the door .".split()
    assert substitute(words, 3, 5, DELETION) == ["The", "tall", "man", "."]
    assert substitute(words, 0, 2, DELETION) == ["Opened", "the", "door", "."]
    assert detokenize_ptb(substitute(words, 0, 2, DELETION)) == "Opened the door."
    assert detokenize_ptb(substitute(words, 3, 5, DELETION)) == "The tall man."  # period attaches
    assert substitute(words, 0, 5, DELETION) == ["."]  # only the whole sentence is never asked
    assert substitute(words, 3, 4, "do so") == ["The", "tall", "man", "do so", "door", "."]


def test_min_over_set_scores_controls_but_never_chooses_them():
    from m1_analyzer.experiments.proforms import DELETION

    policy = MinOverSet(["it", "there"], controls=["blorp", DELETION, "it", " "])
    assert policy.proforms == ("it", "there", "blorp", DELETION)  # cache must hold all four
    assert policy.controls == ("blorp", DELETION)  # a control that is also a proform is dropped
    assert policy.candidates(WORDS, 0, 2) == ["it", "there", "blorp", DELETION]
    assert policy.choose({"it": 0.9, "there": 0.6, "blorp": -3.0, DELETION: -9.0}) == 0.6
    assert policy.name == "min over {it, there} + controls {blorp, <del>}"
    assert MinOverSet(["it"]).name == "min over {it}" and MinOverSet(["it"]).controls == ()
    with pytest.raises(ValueError):
        MinOverSet([" "], controls=["blorp"])


def test_parse_policy_controls():
    policy = parse_policy("it, do so", controls="blorp,<del>")
    assert isinstance(policy, MinOverSet)
    assert policy.proforms == ("it", "do so", "blorp", "<del>") and policy.controls == ("blorp", "<del>")
    assert parse_policy("it", controls="").controls == ()
    assert parse_policy("it,<del>").proforms == ("it", "<del>")  # deletion as a real proform is allowed
    with pytest.raises(ValueError, match="controls"):
        parse_policy("length:2-3=it", controls="blorp")
