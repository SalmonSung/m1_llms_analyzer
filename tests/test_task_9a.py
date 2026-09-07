"""Phase B of Task 9a: labels within strata, the stratified test, the audit, the record, the figure."""

import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from m1_analyzer.experiments import experiment_figures as EF  # noqa: E402
from m1_analyzer.experiments.splice import preregistration  # noqa: E402
from m1_analyzer.experiments.task_9a import (  # noqa: E402
    CLOSE,
    boundary_check,
    FAR,
    MID,
    analyse_9a,
    apply_audit,
    assign_pairs,
    audit_sample,
    flatten_cuts,
    load_audit_csv,
    stratified_permutation_test,
    validate_record,
    verdict,
    write_audit_csv,
)


def synthetic_rows(effect: float, *, n_paragraphs: int = 100, cuts_per: int = 15, seed: int = 0):
    """Cache rows shaped like the demo: divergence tracks deleted length; `effect`
    adds a far-minus-close gap that is independent of length."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_paragraphs):
        text = "X. " * 300          # capitalised: the boundary rule requires a sentence start
        cuts = []
        for k in range(cuts_per):
            seg = int(rng.integers(10, 200))
            d = 150 + 0.6 * seg + rng.normal(0, 40)            # endpoint distance grows with length
            base = 60 + 0.9 * seg                                # divergence grows with length
            div = base * (1 + effect * (d - 150 - 0.6 * seg) / 40) + rng.normal(0, 8)
            i = 5 + k
            cuts.append({"i": i, "j": i + seg, "boundary": "sentence" if k % 5 else "clause", "seg_len": seg,
                         "char_i": 3 * i, "char_j": 3 * (i + seg), "d": float(d), "div": float(max(div, 1.0)),
                         "div_k": [float(max(div, 1.0))] * 3, "fl": 3.8 + rng.normal(0, 0.05),
                         "retokenises": True, "join": "X. ⟦cut⟧ X."})
        rows.append({"kind": "paragraph", "id": f"p{p}", "text": text, "n_tokens": 600, "fluency": 3.8,
                     "boundaries": {"sentence": [c["i"] for c in cuts]}, "n_rejected_boundaries": 0, "cuts": cuts})
    return rows


HEADER = preregistration(window=20, splitter="regex", provenance={"model_id": "synthetic"})


def test_pairs_are_deciles_within_each_stratum_of_the_primary_boundary():
    cuts = flatten_cuts(synthetic_rows(0.0))
    assign_pairs(cuts, strata=HEADER["strata"], decile=0.1)
    for k, (lo, hi) in enumerate(HEADER["strata"]):
        group = [c for c in cuts if c["stratum"] == k and c["boundary"] == "sentence"]
        assert all(lo <= c["seg_len"] < hi for c in group)
        n_close = sum(c["pair"] == CLOSE for c in group)
        n_far = sum(c["pair"] == FAR for c in group)
        assert 0.08 * len(group) <= n_close <= 0.16 * len(group) and 0.08 * len(group) <= n_far <= 0.16 * len(group)
        assert sum(c["pair"] == MID for c in group) == len(group) - n_close - n_far
        for b in {c["bin"] for c in group}:  # deciles within each length-matching bin
            in_bin = [c for c in group if c["bin"] == b]
            if any(c["pair"] == CLOSE for c in in_bin) and any(c["pair"] == FAR for c in in_bin):
                assert max(c["d"] for c in in_bin if c["pair"] == CLOSE) < min(c["d"] for c in in_bin if c["pair"] == FAR)
        # Length-matched: close and far delete the same amount of text to within a bin.
        assert abs(np.median([c["seg_len"] for c in group if c["pair"] == CLOSE])
                   - np.median([c["seg_len"] for c in group if c["pair"] == FAR])) <= 12
    assert all(c["pair"] is None for c in cuts if c["boundary"] == "clause")
    assert all(c["pair"] is None and c["stratum"] is None for c in cuts if c["seg_len"] >= 160 or c["seg_len"] < 20)
    assert {c["pair_pooled"] for c in cuts if c["boundary"] == "sentence"} == {CLOSE, FAR, MID}


def test_planted_effect_passes_and_no_effect_fails():
    record = analyse_9a(synthetic_rows(0.5), HEADER, model="synthetic", n_perm=500, n_boot=200)
    validate_record(record)
    st = record["stratified"]
    assert st["ratio"] > 1.1 and st["p"] < 0.05 and st["ci"][0] > 1.0
    assert verdict(record).splitlines()[1].startswith("PASS")
    assert all(s["usable"] for s in record["strata"]) and st["n_paragraphs"] > 1
    assert {c["pair"] for c in record["cuts"]} == {CLOSE, FAR}
    assert all(c["audited"] is False and c["grammatical"] for c in record["cuts"])
    assert "audit not loaded" in record["meta"]["notes"]

    null = analyse_9a(synthetic_rows(0.0, seed=3), HEADER, model="synthetic", n_perm=500, n_boot=200)
    assert null["stratified"]["p"] > 0.05
    assert "FAIL" in verdict(null)
    # The length confound is visible in the diagnostics either way.
    cont = null["diagnostics"]["continuous"]
    assert cont["spearman_seg_div"] > 0.8 and cont["spearman_seg_dist"] > 0.3
    assert "NOT a pass criterion" in verdict(null)


def test_pooled_deciles_are_length_confounded_and_reported_as_such():
    record = analyse_9a(synthetic_rows(0.0, seed=5), HEADER, model="synthetic", n_perm=200, n_boot=100)
    pooled = record["pooled"]
    assert pooled["median_seg_len_far"] > pooled["median_seg_len_close"]
    assert pooled["ratio"] > 1.2 and "NOT the pass criterion" in pooled["note"]
    assert 0.8 < record["stratified"]["ratio"] < 1.2


def test_stratified_permutation_test_shuffles_within_strata():
    rng = np.random.default_rng(0)
    bins = [(0, rng.normal(100, 5, 30), rng.normal(100, 5, 30)),
            (1, rng.normal(200, 5, 15), rng.normal(230, 5, 15)), (1, rng.normal(200, 5, 15), rng.normal(230, 5, 15))]
    stat, p = stratified_permutation_test(bins, 2, min_per_group=10, n_perm=300, seed=0)
    assert stat == pytest.approx(0.5 * np.log(230 / 200) + 0.5 * np.log(1.0), abs=0.05)
    assert p < 0.05
    stat, p = stratified_permutation_test([(0, rng.normal(1, 1, 5), rng.normal(1, 1, 5))], 1, min_per_group=10)
    assert np.isnan(stat) and np.isnan(p)


def test_audit_sheet_round_trip_and_failures_are_excluded(tmp_path):
    rows = synthetic_rows(0.5, seed=1)
    cuts = flatten_cuts(rows)
    assign_pairs(cuts, strata=HEADER["strata"], decile=0.1)
    sample = audit_sample(cuts, n_close=10, n_far=10, n_other=10, seed=0)
    assert [s["audit_group"] for s in sample].count(CLOSE) == 10
    assert [s["audit_group"] for s in sample].count("other") == 10
    assert all(s["boundary"] == "sentence" for s in sample)
    sheet = write_audit_csv(sample, rows, tmp_path / "audit.csv")
    with sheet.open(encoding="utf-8", newline="") as fh:
        lines = list(csv.DictReader(fh))
    assert len(lines) == 30 and "⟦cut⟧" in lines[0]["spliced_text"]
    assert not any("div" in k for k in lines[0])  # the auditor never sees a divergence
    with pytest.raises(ValueError, match="no grammatical judgement"):
        load_audit_csv(sheet)
    # Fill it in: two failures, one of them a far cut.
    failing = [lines[0]["key"], next(r["key"] for r in lines if r["pair"] == FAR)]
    for r in lines:
        r["grammatical"] = "n" if r["key"] in failing else "y"
        r["note"] = "broken join" if r["key"] in failing else ""
    with sheet.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lines[0]))
        w.writeheader(); w.writerows(lines)
    audit = load_audit_csv(sheet)
    assert len(audit) == 30 and audit[failing[0]] == (False, "broken join")

    record = analyse_9a(rows, HEADER, audit=audit, model="synthetic", n_perm=200, n_boot=100)
    assert record["audit"]["n_audited"] == 30 and record["audit"]["n_failed"] == 2
    assert {f["key"] for f in record["audit"]["failed"]} == set(failing)
    drawn = {c["key"] if "key" in c else f"{c['paragraph']}:{c['i']}-{c['j']}": c for c in record["cuts"]}
    assert drawn[failing[1]]["grammatical"] is False and drawn[failing[1]]["audited"] is True
    assert record["meta"]["n_items"] == sum(1 for c in record["cuts"] if c["grammatical"])
    assert "rule failed" in verdict(record)
    with pytest.raises(ValueError, match="not in this cache"):
        apply_audit(cuts, {"nope:1-2": (True, "")})


def test_deviations_from_the_preregistration_are_recorded():
    record = analyse_9a(synthetic_rows(0.5), HEADER, model="synthetic", n_perm=100, n_boot=50,
                        strata=[[10, 60], [60, 200]], decile=0.2)
    assert "DEVIATION" in record["meta"]["notes"]
    assert [s["lo"] for s in record["strata"]] == [10, 60]
    assert record["diagnostics"]["preregistration"]["strata"] == [[20, 40], [40, 80], [80, 160]]


def test_secondary_boundary_is_analysed_separately():
    record = analyse_9a(synthetic_rows(0.5), HEADER, model="synthetic", n_perm=100, n_boot=50)
    assert "clause" in record["diagnostics"]["secondary_boundaries"]
    assert all(c["boundary"] == "sentence" for c in record["cuts"])


def test_the_real_record_draws_fig_9a(tmp_path):
    record = analyse_9a(synthetic_rows(0.5), HEADER, model="synthetic", n_perm=100, n_boot=50)
    path = tmp_path / "fig_9a.png"
    fig = EF.fig_9a(record, path=str(path))
    plt.close(fig)
    assert path.stat().st_size > 10_000


def test_mock_records_render_and_validate(tmp_path):
    for outcome in ("true", "false"):
        mock = EF.mock_9a(outcome)
        validate_record(mock)
        fig = EF.fig_9a(mock, path=str(tmp_path / f"mock_{outcome}.png"))
        plt.close(fig)


def test_validate_record_catches_bad_pairs():
    record = analyse_9a(synthetic_rows(0.5), HEADER, model="synthetic", n_perm=50, n_boot=20)
    broken = dict(record, cuts=[dict(record["cuts"][0], pair="mid")])
    with pytest.raises(ValueError, match="close or far"):
        validate_record(broken)
    with pytest.raises(ValueError, match="no cuts"):
        analyse_9a([{"id": "p", "cuts": [], "text": "", "fluency": 1.0}], HEADER)


# ------------------------------------------------------- boundary diagnostic


def _rows_with_bad_endpoints(n_bad: int):
    """Cache rows whose paragraph text is known, so char offsets can be checked."""
    good = "Alpha ran fast. Bravo sat still. Charlie left town. Delta stayed home. Echo went."
    # index of the character just past each sentence-final token
    ends = [good.index(w) + len(w) for w in ("fast.", "still.", "town.", "home.")]
    rows, cuts = [], []
    for k in range(6):
        # a "bad" paragraph resumes lowercase after its second sentence end
        text = good if k >= n_bad else good.replace("Charlie", "charlie")
        row_cuts = [
            {"i": 2, "j": 5, "boundary": "sentence", "seg_len": 30, "char_i": ends[0], "char_j": ends[1],
             "d": 100.0 + k, "div": 50.0, "div_k": [50.0], "fl": 3.8, "retokenises": True, "join": ""},
            {"i": 5, "j": 8, "boundary": "sentence", "seg_len": 30, "char_i": ends[1], "char_j": ends[2],
             "d": 200.0 + k, "div": 90.0, "div_k": [90.0], "fl": 3.8, "retokenises": True, "join": ""},
        ]
        rows.append({"kind": "paragraph", "id": f"p{k}", "text": text, "n_tokens": 40, "fluency": 3.8,
                     "boundaries": {"sentence": [2, 5, 8]}, "n_rejected_boundaries": 0, "cuts": row_cuts})
        for c in row_cuts:
            cuts.append(dict(c, paragraph=f"p{k}", pair=CLOSE if c["i"] == 2 else FAR))
    return rows, cuts


def test_boundary_check_is_clean_when_every_endpoint_starts_a_sentence():
    rows, cuts = _rows_with_bad_endpoints(0)
    bc = boundary_check(rows, cuts)
    assert bc["clean"] is True
    assert bc["n_endpoints_not_followed_by_a_sentence"] == 0
    assert bc["n_cuts_affected"] == 0 and bc["n_endpoints_checked"] == 18


def test_boundary_check_finds_bad_endpoints_and_splits_them_by_pair():
    """A mid-sentence endpoint lands in one arm, which is what makes it dangerous."""
    rows, cuts = _rows_with_bad_endpoints(3)
    bc = boundary_check(rows, cuts)
    assert bc["clean"] is False
    assert bc["n_endpoints_not_followed_by_a_sentence"] == 3   # one per bad paragraph
    assert bc["n_cuts_affected"] == 6                          # each bad endpoint is in two cuts
    assert bc["by_pair"][CLOSE]["affected"] == 3 and bc["by_pair"][FAR]["affected"] == 3
    assert bc["examples"] and "charlie" in bc["examples"][0]["after"]


def test_a_contaminated_cache_says_so_in_the_verdict():
    record = analyse_9a(synthetic_rows(0.5), HEADER, model="synthetic", n_perm=100, n_boot=50)
    assert record["diagnostics"]["boundary_check"]["clean"] is True
    assert "WARNING" not in verdict(record)
    dirty = dict(record, diagnostics=dict(record["diagnostics"], boundary_check={
        "clean": False, "n_endpoints_checked": 1036, "n_endpoints_not_followed_by_a_sentence": 12,
        "n_cuts_affected": 54, "n_cuts": 2362,
        "by_pair": {CLOSE: {"n": 215, "affected": 1, "rate": 0.005},
                    FAR: {"n": 215, "affected": 29, "rate": 0.135}},
        "examples": []}))
    text = verdict(dirty)
    assert "WARNING" in text and "0.5% of close" in text and "13.5% of far" in text
