"""Gold spans from trees: traces, punctuation, unaries, detokenisation, the NLTK loader."""

import pytest

nltk = pytest.importorskip("nltk")
from nltk import Tree  # noqa: E402

from m1_analyzer.experiments.treebank import (  # noqa: E402
    PUNCT_TAGS,
    TreebankSentence,
    detokenize_ptb,
    gold_spans_from_tree,
    hand_examples,
    load_ptb_nltk,
    theory_example,
)


def test_theory_example_has_seven_gold_spans_over_thirteen_words():
    s = theory_example()
    assert s.n == 13 and len(s.gold_spans) == 7
    assert s.text == "The tall man with the red hat quickly opened the heavy wooden door"
    assert all(len(h.gold_spans) >= 3 for h in hand_examples())


def test_trivial_gold_spans_are_rejected():
    with pytest.raises(ValueError, match="trivial"):
        TreebankSentence(id="x", words=["a", "b", "c"], gold_spans={(0, 2)})
    with pytest.raises(ValueError, match="trivial"):
        TreebankSentence(id="x", words=["a", "b", "c"], gold_spans={(1, 1)})


def test_gold_spans_drop_traces_punctuation_and_collapse_unaries():
    tree = Tree.fromstring(
        "(S (NP-SBJ (NP (DT The) (NN dog))) (VP (VBD ran) (PP (IN to) (NP (DT the) (NN park))) "
        "(SBAR (-NONE- *T*-1))) (. .))"
    )
    words, spans = gold_spans_from_tree(tree)
    assert words == ["The", "dog", "ran", "to", "the", "park"]
    # NP-SBJ over NP is one span; the trace-only SBAR and the period vanish;
    # the whole sentence (S) is excluded.
    assert spans == {(0, 1), (2, 5), (4, 5), (3, 5)}


def test_emptied_parents_vanish_recursively():
    tree = Tree.fromstring("(S (NP (DT a) (NN b)) (VP (VBZ c) (S (NP (-NONE- *)) (VP (-NONE- *)))))")
    words, spans = gold_spans_from_tree(tree)
    assert words == ["a", "b", "c"]
    assert spans == {(0, 1)}  # (VP c <empty S>) is a single word now -> trivial


def test_drop_tags_are_configurable():
    tree = Tree.fromstring("(S (NP ($ $) (CD 5)) (VP (VBZ works)))")
    assert gold_spans_from_tree(tree)[0] == ["5", "works"]
    assert gold_spans_from_tree(tree, drop_tags=set())[0] == ["$", "5", "works"]
    assert "$" in PUNCT_TAGS and "-NONE-" not in PUNCT_TAGS


@pytest.mark.parametrize("words,expected", [
    (["I", "do", "n't", "know"], "I don't know"),
    (["-LRB-", "a", "-RRB-"], "(a)"),
    (["``", "hi", "''", "he", "said"], '"hi" he said'),
    (["$", "5", "million"], "$5 million"),
    (["1\\/2", "cup"], "1/2 cup"),
    (["John", "'s", "hat"], "John's hat"),
])
def test_detokenize_ptb(words, expected):
    assert detokenize_ptb(words) == expected


def _treebank_available() -> bool:
    try:
        nltk.data.find("corpora/treebank")
        return True
    except LookupError:
        return False


@pytest.mark.skipif(not _treebank_available(), reason="NLTK treebank corpus not downloaded")
def test_load_ptb_nltk_is_seeded_filtered_and_in_corpus_order():
    a = load_ptb_nltk(20, min_len=5, max_len=12, seed=3, download=False)
    b = load_ptb_nltk(20, min_len=5, max_len=12, seed=3, download=False)
    assert [s.id for s in a] == [s.id for s in b]
    assert len(a) == 20
    assert all(5 <= s.n <= 12 and s.gold_spans for s in a)
    assert all(s.id.startswith("wsj_") for s in a)
    in_order = sorted(a, key=lambda s: (s.info["fileid"], s.info["index"]))
    assert [s.id for s in a] == [s.id for s in in_order]
