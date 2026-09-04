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


# ------------------------------------------------------------- gold export


def test_gold_export_round_trips_without_nltk(tmp_path):
    """The export must rebuild the same answer key on a machine with no corpus."""
    from m1_analyzer.experiments.treebank import (
        CONVENTIONS,
        GOLD_SCHEMA,
        load_gold_jsonl,
        save_gold_jsonl,
    )

    original = hand_examples()
    path = tmp_path / "gold.jsonl"
    save_gold_jsonl(original, path, provenance={"treebank": "hand", "seed": 7})

    header, reloaded = load_gold_jsonl(path)
    assert header["schema"] == GOLD_SCHEMA and header["n_sentences"] == len(original)
    assert header["treebank"] == "hand" and header["seed"] == 7
    assert header["conventions"] == CONVENTIONS
    assert sorted(header["punct_tags"]) == sorted(PUNCT_TAGS)
    assert header["includes_trees"] is False

    assert [s.id for s in reloaded] == [s.id for s in original]
    for got, want in zip(reloaded, original):
        assert got.words == want.words
        assert got.text == want.text
        assert got.gold_spans == want.gold_spans
        assert got.n == want.n


def test_gold_export_is_jsonl_with_one_line_per_sentence(tmp_path):
    import json

    from m1_analyzer.experiments.treebank import save_gold_jsonl

    path = tmp_path / "gold.jsonl"
    save_gold_jsonl(hand_examples(), path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + len(hand_examples())
    assert json.loads(lines[0])["kind"] == "header"
    row = json.loads(lines[1])
    assert row["kind"] == "sentence" and row["id"] == "theory-doc"
    assert row["gold_spans"] == [list(sp) for sp in sorted(theory_example().gold_spans)]
    assert "tree" not in row


def test_gold_export_carries_trees_when_given(tmp_path):
    from m1_analyzer.experiments.treebank import load_gold_jsonl, save_gold_jsonl

    path = tmp_path / "gold.jsonl"
    sentences = hand_examples()
    trees = {sentences[0].id: "(S (NP (DT the) (NN dog)) (VP (VBD ran)))"}
    save_gold_jsonl(sentences, path, trees=trees)
    header, reloaded = load_gold_jsonl(path)
    assert header["includes_trees"] is True
    assert reloaded[0].info["tree"] == trees[sentences[0].id]
    assert "tree" not in reloaded[1].info


def test_load_gold_jsonl_rejects_a_headerless_file(tmp_path):
    from m1_analyzer.experiments.treebank import load_gold_jsonl

    path = tmp_path / "nope.jsonl"
    path.write_text('{"kind": "sentence", "id": "x", "words": ["a", "b"], "gold_spans": []}\n')
    with pytest.raises(ValueError, match="no header line"):
        load_gold_jsonl(path)


def test_ptb_tree_strings_skips_sentences_with_no_fileid():
    from m1_analyzer.experiments.treebank import ptb_tree_strings

    assert ptb_tree_strings(hand_examples(), download=False) == {}


@pytest.mark.skipif(not _treebank_available(), reason="NLTK treebank corpus not downloaded")
def test_ptb_tree_strings_returns_the_original_parse():
    from m1_analyzer.experiments.treebank import ptb_tree_strings

    sentences = load_ptb_nltk(5, min_len=5, max_len=15, seed=1, download=False)
    trees = ptb_tree_strings(sentences, download=False)
    assert set(trees) == {s.id for s in sentences}
    for s in sentences:
        text = trees[s.id]
        assert text.startswith("(") and "\n" not in text
        # every surviving word appears in its own tree
        assert all(word in text for word in s.words)
