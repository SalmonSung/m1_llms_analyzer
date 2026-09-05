"""Phase B end to end: record schema, statistics, verdict, and the real figure."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from m1_analyzer.experiments import experiment_figures as EF  # noqa: E402
from m1_analyzer.experiments.proforms import MinOverSet  # noqa: E402
from m1_analyzer.experiments.span_costs import compute_span_costs  # noqa: E402
from m1_analyzer.experiments.task_1b import (  # noqa: E402
    METHOD_RB,
    METHOD_SUB,
    METHODS,
    RANK_GRID,
    analyse_1b,
    evaluate_sentence,
    run_task_1b,
    validate_record,
    verdict,
)
from m1_analyzer.experiments.treebank import TreebankSentence, hand_examples  # noqa: E402
from m1_analyzer.testing import FakeSpanScorer  # noqa: E402


@pytest.fixture(scope="module")
def sentences():
    base = hand_examples()
    extra = []
    for k in range(6):  # enough copies of one length for a by_length row
        s = base[1]
        extra.append(TreebankSentence(id=f"copy-{k}", words=list(s.words), gold_spans=set(s.gold_spans), source=s.source))
    return base + extra


@pytest.fixture(scope="module")
def record_and_tables(sentences):
    scorer = FakeSpanScorer(sentences)
    return run_task_1b(scorer, sentences, policy=MinOverSet(), model="fake", show_progress=False, seed=0)


def test_record_has_the_fig_1b_schema(record_and_tables, sentences):
    record, tables = record_and_tables
    validate_record(record)
    assert record["n_sentences"] == len(sentences) == record["meta"]["n_items"]
    assert [m["name"] for m in record["methods"]] == list(METHODS)
    assert len(record["rank_curve"]) == len(RANK_GRID)
    assert record["rank_curve"][0]["recall_gold"] == 0.0 and record["rank_curve"][-1]["recall_gold"] == 1.0
    assert record["by_length"] == [{"len": 8, "n": 7, "f1_sub": pytest.approx(record["by_length"][0]["f1_sub"]),
                                    "f1_rb": pytest.approx(record["by_length"][0]["f1_rb"])}]
    assert record["meta"]["model"] == "fake" and "inducer=greedy" in record["meta"]["notes"]


def test_fake_run_passes_and_verdict_says_so(record_and_tables):
    record, _ = record_and_tables
    f1 = {m["name"]: m for m in record["methods"]}
    assert f1[METHOD_SUB]["ci"][0] > f1[METHOD_RB]["ci"][1]
    assert record["diagnostics"]["gap_vs_rb"]["ci"][0] > 0
    assert record["diagnostics"]["rank_area"] > 0.8
    assert verdict(record).splitlines()[1].startswith("PASS")


def test_the_same_record_draws_fig_1b_and_fig_0c(record_and_tables, tmp_path):
    record, _ = record_and_tables
    for name, fn in (("fig_1b", EF.fig_1b), ("fig_0c", EF.fig_0c)):
        path = tmp_path / f"{name}.png"
        fig = fn(record, path=str(path))
        plt.close(fig)
        assert path.stat().st_size > 10_000


def test_empty_gold_sentences_are_excluded_and_counted(sentences):
    flat = TreebankSentence(id="flat", words=["a", "b", "c"], gold_spans=set())
    all_sentences = sentences + [flat]
    tables = compute_span_costs(FakeSpanScorer(all_sentences), all_sentences, MinOverSet(), show_progress=False)
    record = analyse_1b(tables, all_sentences, policy=MinOverSet(), model="fake")
    assert record["n_sentences"] == len(sentences)
    assert record["diagnostics"]["n_empty_gold"] == 1
    assert "flat sentence" in record["meta"]["notes"]


def test_cky_inducer_and_sum_normalisation_run(sentences):
    tables = compute_span_costs(FakeSpanScorer(sentences), sentences, MinOverSet(), show_progress=False)
    record = analyse_1b(tables, sentences, policy=MinOverSet(), inducer="cky", normalisation="sum", model="fake")
    validate_record(record)
    assert record["diagnostics"]["inducer"] == "cky"
    with pytest.raises(ValueError, match="inducer"):
        evaluate_sentence(tables[0], sentences[0].gold_spans, MinOverSet(), inducer="magic")


def test_evaluate_sentence_reports_the_theory_doc_baselines(sentences):
    tables = compute_span_costs(FakeSpanScorer(sentences[:1]), sentences[:1], MinOverSet(), show_progress=False)
    row = evaluate_sentence(tables[0], sentences[0].gold_spans, MinOverSet())
    assert row["f1_rb"] == pytest.approx(0.333, abs=1e-3)
    assert row["f1_lb"] == pytest.approx(0.222, abs=1e-3)
    assert row["n_pred"] == 11 and row["n_gold"] == 7


def test_validate_record_catches_missing_methods(record_and_tables):
    record, _ = record_and_tables
    broken = dict(record, methods=[m for m in record["methods"] if m["name"] != METHOD_RB])
    with pytest.raises(ValueError, match="right-branching"):
        validate_record(broken)


def test_verdict_fails_when_substitution_ties_right_branching(record_and_tables):
    record, _ = record_and_tables
    tied = dict(record, diagnostics=dict(record["diagnostics"], gap_vs_rb={"diff": 0.0, "ci": [-0.1, 0.1]}))
    assert "FAIL" in verdict(tied)


def test_mock_figures_still_render(tmp_path):
    """The vendored figure module's own mock data is the schema's ground truth."""
    fig = EF.fig_1b(EF.mock_1b("true"), path=str(tmp_path / "mock.png"))
    plt.close(fig)
    validate_record(EF.mock_1b("false"))


def test_controls_do_not_change_the_induced_record(sentences):
    from m1_analyzer.experiments.proforms import DELETION

    policy = MinOverSet(controls=["blorp", DELETION])
    record, tables = run_task_1b(FakeSpanScorer(sentences, proforms=policy.proforms), sentences,
                                 policy=policy, model="fake", show_progress=False, seed=0)
    assert DELETION in tables[0].spans and "blorp" in tables[0].spans
    assert record["diagnostics"]["policy"].endswith("+ controls {blorp, <del>}")
    # Phase B over the same tables with the controls stripped from the policy: identical.
    plain = analyse_1b(tables, sentences, policy=MinOverSet(), model="fake", seed=0)
    for key in ("methods", "by_length", "rank_curve"):
        assert record[key] == plain[key]
