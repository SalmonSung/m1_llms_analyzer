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
