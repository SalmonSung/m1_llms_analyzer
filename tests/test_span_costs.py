"""Phase A cache: table build, cost normalisation, JSONL resume, header checks."""

import json

import pytest

from m1_analyzer.experiments.proforms import ByLengthClass, MinOverSet
from m1_analyzer.experiments.span_costs import (
    SpanCostTable,
    compute_span_costs,
    load_span_costs,
    variant_count,
)
from m1_analyzer.experiments.spans import enumerate_spans
from m1_analyzer.experiments.treebank import hand_examples, theory_example
from m1_analyzer.testing import FakeSpanScorer


@pytest.fixture
def sentences():
    return hand_examples()


def test_compute_scores_every_span_and_proform(sentences):
    scorer = FakeSpanScorer(sentences)
    policy = MinOverSet()
    tables = compute_span_costs(scorer, sentences, policy, show_progress=False)
    assert [t.sentence_id for t in tables] == [s.id for s in sentences]
    t = tables[0]
    assert t.n == 13 and t.base_n == 13
    for proform in policy.proforms:
        assert set(t.spans[proform]) == set(enumerate_spans(13))
    costs = t.costs_for(policy)
    assert len(costs) == 77
    gold = theory_example().gold_spans
    assert max(costs[g] for g in gold) < min(c for s, c in costs.items() if s not in gold)
    assert scorer.calls == len(sentences)  # one score() call per sentence
    assert variant_count(sentences, policy) == sum(1 + 4 * len(enumerate_spans(s.n)) for s in sentences)


def test_cost_normalisations_and_policy_subsets():
    t = SpanCostTable("x", ["a", "b", "c", "d"], "a b c d", base_sum=-8.0, base_n=4,
                      spans={"it": {(0, 1): (-9.0, 3)}, "that": {(0, 1): (-6.0, 3)}})
    assert t.cost((0, 1), "it") == pytest.approx(-2.0 - (-3.0))
    assert t.cost((0, 1), "it", "sum") == pytest.approx(1.0)
    assert t.costs_by_proform((0, 1)) == {"it": pytest.approx(1.0), "that": pytest.approx(0.0)}
    assert t.costs_for(MinOverSet(["it", "that"])) == {(0, 1): pytest.approx(0.0)}
    assert t.costs_for(ByLengthClass({(2, None): "it"})) == {(0, 1): pytest.approx(1.0)}
    assert t.costs_for(MinOverSet(["there"])) == {}  # no scored proform -> span omitted
    with pytest.raises(ValueError, match="normalisation"):
        t.cost((0, 1), "it", "median")


def test_round_trip_through_json():
    t = SpanCostTable("x", ["a", "b", "c"], "a b c", -6.0, 3, {"it": {(0, 1): (-4.5, 2)}}, failed=1)
    again = SpanCostTable.from_json(json.loads(json.dumps(t.to_json())))
    assert again == t


def test_cache_resumes_and_skips_finished_sentences(sentences, tmp_path):
    path = tmp_path / "costs.jsonl"
    first = compute_span_costs(FakeSpanScorer(sentences[:2]), sentences[:2], MinOverSet(),
                               cache_path=path, provenance={"model_id": "fake"}, show_progress=False)
    assert len(first) == 2
    header, tables = load_span_costs(path)
    assert header["model_id"] == "fake" and header["proforms"] == list(MinOverSet().proforms)
    assert [t.sentence_id for t in tables] == [s.id for s in sentences[:2]]

    scorer = FakeSpanScorer(sentences)
    tables = compute_span_costs(scorer, sentences, MinOverSet(), cache_path=path,
                                provenance={"model_id": "fake"}, show_progress=False)
    assert len(tables) == 3 and scorer.calls == 1  # only the third sentence was scored

    class NeverScorer:
        def score(self, texts, **kw):
            raise AssertionError("nothing should be scored on a complete cache")

    tables = compute_span_costs(NeverScorer(), sentences, MinOverSet(), cache_path=path,
                                provenance={"model_id": "fake"}, show_progress=False)
    assert len(tables) == 3


def test_header_mismatch_is_refused(sentences, tmp_path):
    path = tmp_path / "costs.jsonl"
    compute_span_costs(FakeSpanScorer(sentences), sentences, MinOverSet(), cache_path=path,
                       provenance={"model_id": "fake"}, show_progress=False)
    with pytest.raises(ValueError, match="model_id"):
        compute_span_costs(FakeSpanScorer(sentences), sentences, MinOverSet(), cache_path=path,
                           provenance={"model_id": "other"}, show_progress=False)
    with pytest.raises(ValueError, match="policy|proforms"):
        compute_span_costs(FakeSpanScorer(sentences), sentences, MinOverSet(["it"]), cache_path=path,
                           provenance={"model_id": "fake"}, show_progress=False)


def test_truncated_last_line_is_rescored(sentences, tmp_path):
    path = tmp_path / "costs.jsonl"
    compute_span_costs(FakeSpanScorer(sentences), sentences, MinOverSet(), cache_path=path,
                       provenance={"model_id": "fake"}, show_progress=False)
    text = path.read_text(encoding="utf-8")
    path.write_text(text[: len(text) - 40], encoding="utf-8")  # cut the last record mid-way
    scorer = FakeSpanScorer(sentences)
    tables = compute_span_costs(scorer, sentences, MinOverSet(), cache_path=path,
                                provenance={"model_id": "fake"}, show_progress=False)
    assert len(tables) == 3 and scorer.calls == 1
    header, reloaded = load_span_costs(path)
    assert len(reloaded) == 3


def test_failed_variants_are_counted_and_failed_base_skips_the_sentence(sentences):
    scorer = FakeSpanScorer(sentences[:1])
    del scorer.table[next(k for k in scorer.table if k.startswith("It "))]  # one variant missing
    tables = compute_span_costs(scorer, sentences[:1], MinOverSet(), show_progress=False)
    assert tables[0].failed >= 1
    scorer = FakeSpanScorer(sentences[:1])
    del scorer.table[sentences[0].text]
    assert compute_span_costs(scorer, sentences[:1], MinOverSet(), show_progress=False) == []


def test_mirror_copies_the_cache(sentences, tmp_path):
    path, mirror = tmp_path / "costs.jsonl", tmp_path / "drive" / "costs.jsonl"
    compute_span_costs(FakeSpanScorer(sentences), sentences, MinOverSet(), cache_path=path,
                       mirror_path=mirror, mirror_every=2, show_progress=False)
    assert mirror.read_text() == path.read_text()
