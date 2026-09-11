"""Phase A of Task 9a on the tiny model: cuts, splices, states, cache, resume."""

import json

import numpy as np
import pytest

from m1_analyzer import Analyzer, ModelConfig, RunConfig, ScoringConfig
from m1_analyzer.experiments.paragraphs import Paragraph, hand_paragraphs
from m1_analyzer.experiments.splice import (
    DIVERGENCE_MEASURE,
    FLUENCY_DEFINITION,
    SURPRISAL_MEASURE,
    Cut,
    add_surprisal,
    admissible_cuts,
    check_surprisal,
    compute_splices,
    cut_count,
    deleted_surprisal,
    load_splices,
    paragraph_boundaries,
    preregistration,
    score_paragraph,
    splice_ids,
    splice_text,
    surprisal_from_logprobs,
)
from m1_analyzer.testing import build_tiny_local_model


@pytest.fixture(scope="module")
def analyzer(tmp_path_factory) -> Analyzer:
    # Paragraphs are longer than the session-wide tiny model's 32 positions.
    path = build_tiny_local_model(tmp_path_factory.mktemp("tiny_long"), max_positions=256)
    return Analyzer(RunConfig(model=ModelConfig(model_id=path, head="causal_lm"), scoring=ScoringConfig(batch_size=4)))


def test_admissible_cuts_pair_same_type_endpoints_with_room_after_j():
    cuts = admissible_cuts({"sentence": [3, 9, 15, 21], "clause": [6, 12]}, n_tokens=27, window=5)
    sentence = [(c.i, c.j) for c in cuts if c.boundary == "sentence"]
    assert sentence == [(3, 9), (3, 15), (3, 21), (9, 15), (9, 21), (15, 21)]  # j=21 leaves exactly 5
    assert [(c.i, c.j) for c in cuts if c.boundary == "clause"] == [(6, 12)]  # never a sentence-clause pair
    # window=5 needs n - (j + 1) >= 5: with 26 tokens j=21 leaves 4 -> excluded.
    cuts = admissible_cuts({"sentence": [3, 9, 15, 21]}, n_tokens=26, window=5)
    assert [(c.i, c.j) for c in cuts] == [(3, 9), (3, 15), (9, 15)]
    assert all(c.seg_len == c.j - c.i for c in cuts)
    assert [(c.i, c.j) for c in admissible_cuts({"clause": [6, 12]}, 26, 5)] == [(6, 12)]
    with pytest.raises(ValueError):
        admissible_cuts({"sentence": [1, 2]}, 10, 0)


def test_splice_ids_and_text_delete_strictly_between_endpoints():
    ids = list(range(10))
    assert splice_ids(ids, 2, 6) == [0, 1, 2, 7, 8, 9]
    assert Cut(2, 6).seg_len == 4
    with pytest.raises(ValueError):
        splice_ids(ids, 6, 2)
    text = "aa bb. cc dd. ee ff."
    offsets = [(0, 2), (3, 6), (7, 9), (10, 13), (14, 16), (17, 20)]
    assert splice_text(text, offsets, 1, 3) == "aa bb. ee ff."
    assert splice_text(text, offsets, 1, 3, marker=" |") == "aa bb. | ee ff."


def test_preregistration_validates_and_names_everything():
    header = preregistration(window=20, splitter="regex", provenance={"model_id": "m"})
    assert header["divergence_measure"] == DIVERGENCE_MEASURE
    assert header["strata"] == [[20, 40], [40, 80], [80, 160]] and header["decile"] == 0.1
    assert header["primary_boundary"] == "sentence" and header["model_id"] == "m"
    assert header["surprisal_measure"] == SURPRISAL_MEASURE
    with pytest.raises(ValueError, match="primary_boundary"):
        preregistration(boundary_kinds=("clause",))
    with pytest.raises(ValueError, match="decile"):
        preregistration(decile=0.7)
    with pytest.raises(ValueError, match="stratum"):
        preregistration(strata=[[40, 20]])


def test_score_paragraph_measures_what_the_docstring_says(analyzer):
    para = hand_paragraphs()[0]
    ids, offsets, boundaries, rejected = paragraph_boundaries(analyzer.states, para, splitter="regex")
    assert rejected == 0 and len(boundaries["sentence"]) == 8
    row = score_paragraph(analyzer.states, para, window=5, splitter="regex")
    assert row["n_tokens"] == len(ids) and len(row["cuts"]) == len(admissible_cuts(boundaries, len(ids), 5))
    cut = row["cuts"][0]
    assert cut["seg_len"] == cut["j"] - cut["i"] and len(cut["div_k"]) == 5
    assert cut["div"] == pytest.approx(float(np.median(cut["div_k"])), abs=1e-3)
    assert cut["retokenises"] is True and "⟦cut⟧" in cut["join"]
    # Recompute one cut by hand from the state service.
    spliced = splice_ids(ids, cut["i"], cut["j"])
    [orig] = analyzer.states.states([ids], [[cut["i"], cut["j"]] + [cut["j"] + 1 + k for k in range(5)]])
    [spl] = analyzer.states.states([spliced], [[cut["i"] + 1 + k for k in range(5)]])
    d = np.linalg.norm(orig.states[0] - orig.states[1])
    div_k = np.linalg.norm(spl.states - orig.states[2:], axis=1)
    assert cut["d"] == pytest.approx(d, abs=1e-3)
    assert np.allclose(cut["div_k"], div_k, atol=1e-3)
    # The spliced text's own fluency, in nats per token, from the same pass.
    assert cut["fl"] == pytest.approx(-analyzer.score_one(analyzer.states.decode(spliced)).mean_logprob, abs=1e-3)
    # The rejoined TEXT is what the audit sees; the tiny word-level tokenizer decodes
    # with a space before punctuation, so compare modulo whitespace.
    rejoined = splice_text(para.text, offsets, cut["i"], cut["j"])
    assert rejoined.replace(" ", "") == analyzer.states.decode(spliced).replace(" ", "")


def test_a_paragraph_without_cuts_still_gets_a_row(analyzer):
    para = Paragraph(id="one", text="the quick brown fox jumps over the lazy dog.")
    row = score_paragraph(analyzer.states, para, window=5, splitter="regex")
    assert row["cuts"] == [] and row["fluency"] > 0


def test_cache_resumes_and_guards_the_preregistration(analyzer, tmp_path):
    cache = tmp_path / "splices.jsonl"
    paras = hand_paragraphs()
    header, rows = compute_splices(analyzer.states, paras[:2], window=5, splitter="regex", cache_path=cache,
                                   provenance={"model_id": "tiny"}, show_progress=False)
    assert cut_count(rows) > 0 and [r["id"] for r in rows] == [p.id for p in paras[:2]]
    calls = analyzer.states.states
    analyzer.states.states = lambda *a, **k: (_ for _ in ()).throw(AssertionError("scored a finished paragraph"))
    try:
        header2, rows2 = compute_splices(analyzer.states, paras[:2], window=5, splitter="regex", cache_path=cache,
                                         provenance={"model_id": "tiny"}, show_progress=False)
    finally:
        analyzer.states.states = calls
    assert rows2 == rows and header2["window"] == 5
    with pytest.raises(ValueError, match="window=5"):
        compute_splices(analyzer.states, paras, window=6, splitter="regex", cache_path=cache, show_progress=False)
    with pytest.raises(ValueError, match="model_id"):
        compute_splices(analyzer.states, paras, window=5, splitter="regex", cache_path=cache,
                        provenance={"model_id": "other"}, show_progress=False)
    # The third paragraph is added on resume; a truncated last line is repaired first.
    with cache.open("a", encoding="utf-8") as fh:
        fh.write('{"kind": "paragraph", "id": "broken", "cu')
    header3, rows3 = compute_splices(analyzer.states, paras, window=5, splitter="regex", cache_path=cache,
                                     provenance={"model_id": "tiny"}, show_progress=False)
    assert [r["id"] for r in rows3] == [p.id for p in paras]
    loaded_header, loaded_rows = load_splices(cache)
    assert loaded_rows == rows3 and loaded_header["divergence_measure"] == DIVERGENCE_MEASURE
    lines = cache.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["kind"] == "header" and len(lines) == 4


def test_load_splices_rejects_other_caches(tmp_path):
    other = tmp_path / "x.jsonl"
    other.write_text('{"kind": "header", "schema": 2}\n')
    with pytest.raises(ValueError, match="Task 9a"):
        load_splices(other)


# ------------------------------------------------------------------ surprisal


def _strip_surprisal(path):
    """Turn a cache into one written before the surprisal field existed (schema 2 as shipped)."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        if obj.get("kind") == "header":
            obj.pop("surprisal_measure", None)
        else:
            obj.pop("surprisal", None)
            for c in obj["cuts"]:
                c.pop("del_surp", None)
                c.pop("del_surp_mean", None)
        out.append(json.dumps(obj))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return out


def test_score_paragraph_writes_surprisal_from_the_same_pass(analyzer):
    para = hand_paragraphs()[0]
    row = score_paragraph(analyzer.states, para, window=5, splitter="regex")
    surp = row["surprisal"]
    # Invariant 1: one number per token, and every cut's deleted span is seg_len long.
    assert len(surp) == row["n_tokens"] and None not in surp and row["cuts"]
    for c in row["cuts"]:
        span = surp[c["i"] + 1: c["j"] + 1]
        assert len(span) == c["seg_len"]
        assert c["del_surp"] == pytest.approx(sum(span), abs=1e-5)
        assert c["del_surp_mean"] == pytest.approx(c["del_surp"] / c["seg_len"], abs=1e-5)
        assert list(c)[-2:] == ["del_surp", "del_surp_mean"]  # new keys last
    # Invariant 2: fluency is the mean over ALL tokens (token 0 scored from BOS), not over [1:].
    assert np.mean(surp) == pytest.approx(row["fluency"], abs=2e-6)
    assert np.mean(surp[1:]) != pytest.approx(row["fluency"], abs=1e-9)
    # surprisal[t] is read from the state at position t-1: the same vector d and div use.
    ids = analyzer.states.encode(para.text)
    [res] = analyzer.states.states([ids], [[4]])
    assert surp[5] == pytest.approx(-res.states[0][ids[5]], abs=1e-5)
    [with_bos] = analyzer.states.states([ids], [[0]], return_token_logprobs=True)
    assert surp[0] == pytest.approx(-with_bos.token_logprobs[0], abs=1e-5)
    assert preregistration()["surprisal_measure"] == SURPRISAL_MEASURE
    report = check_surprisal([row])
    assert report["ok"] and report["n_within_tol_all"] == 1 and report["n_bad_cut_length"] == 0
    assert report["fluency_definition"] == FLUENCY_DEFINITION


def test_add_surprisal_is_additive_byte_identical_and_idempotent(analyzer, tmp_path):
    cache = tmp_path / "splices.jsonl"
    paras = hand_paragraphs()[:2]
    _, fresh = compute_splices(analyzer.states, paras, window=5, splitter="regex", cache_path=cache,
                               provenance={"model_id": "tiny"}, show_progress=False)
    fresh_lines = cache.read_text(encoding="utf-8").splitlines()
    old_lines = _strip_surprisal(cache)
    assert "surprisal" not in old_lines[1] and "del_surp" not in old_lines[1]

    header, rows, report = add_surprisal(analyzer.states, cache, show_progress=False)
    new_lines = cache.read_text(encoding="utf-8").splitlines()
    # Rows come out exactly as a cache scored with the fields from the start...
    assert new_lines[1:] == fresh_lines[1:] and rows == fresh
    # ...and stripping the new keys gives every old line back, byte for byte.
    for old, new in zip(old_lines, new_lines):
        obj = json.loads(new)
        if obj.get("kind") == "header":
            assert list(obj)[-1] == "surprisal_measure" and obj["surprisal_measure"] == SURPRISAL_MEASURE
            obj.pop("surprisal_measure")
        else:
            obj.pop("surprisal")
            for c in obj["cuts"]:
                c.pop("del_surp")
                c.pop("del_surp_mean")
        assert json.dumps(obj) == old
    assert report["ok"] and report["n_augmented"] == 2 and report["n_skipped"] == 0
    assert report["n_bad_paragraph_length"] == 0 and report["n_bad_cut_length"] == 0
    assert report["max_abs_mean_all_minus_fluency"] < 1e-5 and report["written"] == str(cache)

    # Idempotent: a second call scores nothing and does not touch the file.
    mtime = cache.stat().st_mtime_ns
    calls = analyzer.states.states
    analyzer.states.states = lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-scored a paragraph"))
    try:
        _, rows2, report2 = add_surprisal(analyzer.states, cache, show_progress=False)
    finally:
        analyzer.states.states = calls
    assert rows2 == rows and report2["n_augmented"] == 0 and report2["written"] is None
    assert cache.stat().st_mtime_ns == mtime

    # out_path leaves the source untouched; a mixed cache only fills the rows that lack the field.
    _strip_surprisal(cache)
    before = cache.read_bytes()
    aug = tmp_path / "aug.jsonl"
    add_surprisal(analyzer.states, cache, out_path=aug, show_progress=False)
    assert cache.read_bytes() == before and aug.read_text(encoding="utf-8").splitlines()[1:] == fresh_lines[1:]
    lines = aug.read_text(encoding="utf-8").splitlines()
    obj = json.loads(lines[1])
    obj.pop("surprisal")
    for c in obj["cuts"]:
        c.pop("del_surp"), c.pop("del_surp_mean")
    aug.write_text("\n".join([lines[0], json.dumps(obj), lines[2]]) + "\n", encoding="utf-8")
    _, rows3, report3 = add_surprisal(analyzer.states, aug, show_progress=False)
    assert report3["n_augmented"] == 1 and report3["n_skipped"] == 1 and rows3 == fresh


def test_add_surprisal_refuses_the_wrong_tokenizer_or_measure(analyzer, tmp_path):
    cache = tmp_path / "splices.jsonl"
    compute_splices(analyzer.states, hand_paragraphs()[:1], window=5, splitter="regex", cache_path=cache,
                    show_progress=False)
    lines = _strip_surprisal(cache)
    row = json.loads(lines[1])
    row["n_tokens"] += 1
    cache.write_text("\n".join([lines[0], json.dumps(row)]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=f"{row['id']}.*re-encodes"):
        add_surprisal(analyzer.states, cache, show_progress=False)
    header = json.loads(lines[0])
    header["surprisal_measure"] = "something else"
    cache.write_text("\n".join([json.dumps(header), lines[1]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="surprisal_measure"):
        add_surprisal(analyzer.states, cache, show_progress=False)
    other = tmp_path / "x.jsonl"
    other.write_text('{"kind": "header", "schema": 2}\n')
    with pytest.raises(ValueError, match="Task 9a"):
        add_surprisal(analyzer.states, other, show_progress=False)


def test_surprisal_helpers_handle_bos_none_and_bad_lengths():
    assert surprisal_from_logprobs(np.full(4, -1.0), 4) == [1.0, 1.0, 1.0, 1.0]
    assert surprisal_from_logprobs(np.full(4, -1.0), 5) == [None, 1.0, 1.0, 1.0, 1.0]  # bos_policy="none"
    with pytest.raises(ValueError, match="expected n or n-1"):
        surprisal_from_logprobs(np.full(4, -1.0), 7)
    surp = [None, 2.0, 3.0, 4.0, 5.0]
    assert deleted_surprisal(surp, 0, 2) == (5.0, 2.5)  # never touches index 0
    assert deleted_surprisal(surp, 1, 4) == (12.0, 4.0)
    with pytest.raises(ValueError, match="need 0 <= i < j < n"):
        deleted_surprisal(surp, 3, 3)
    with pytest.raises(ValueError, match="no complete surprisal"):
        deleted_surprisal([None, None, 1.0], 0, 2)
    rows = [
        {"n_tokens": 5, "fluency": 3.5, "surprisal": surp,
         "cuts": [{"i": 0, "j": 2, "seg_len": 2, "del_surp": 5.0}, {"i": 1, "j": 4, "seg_len": 4}]},
        {"n_tokens": 3, "fluency": 1.0, "surprisal": [1.0, 1.0], "cuts": []},   # too short
        {"n_tokens": 2, "fluency": 1.0, "cuts": []},                              # never augmented
    ]
    report = check_surprisal(rows)
    assert report["n_paragraphs"] == 3 and report["n_with_surprisal"] == 2 and report["n_missing"] == 1
    assert report["n_bad_paragraph_length"] == 1 and report["n_bad_cut_length"] == 1
    assert report["n_null_token0"] == 1 and report["n_cuts"] == 2
    # With a null token 0, mean over all == mean over [1:] == fluency.
    assert report["max_abs_mean_all_minus_fluency"] == pytest.approx(0.0)
    assert report["max_abs_mean_from_1_minus_fluency"] == pytest.approx(0.0)
    assert report["ok"] is False


def test_resuming_a_cache_without_surprisal_warns_and_new_rows_carry_it(analyzer, tmp_path, monkeypatch):
    from m1_analyzer.experiments import splice as splice_module

    cache = tmp_path / "splices.jsonl"
    paras = hand_paragraphs()
    compute_splices(analyzer.states, paras[:1], window=5, splitter="regex", cache_path=cache,
                    provenance={"model_id": "tiny"}, show_progress=False)
    _strip_surprisal(cache)
    warnings = []  # the package logger does not propagate to the root, so caplog would miss it
    monkeypatch.setattr(splice_module.log, "warning", lambda msg, *args: warnings.append(msg % args))
    _, rows = compute_splices(analyzer.states, paras[:2], window=5, splitter="regex", cache_path=cache,
                              provenance={"model_id": "tiny"}, show_progress=False)
    assert len(warnings) == 1 and "add_surprisal" in warnings[0] and "1 scored paragraph" in warnings[0]
    assert "surprisal" not in rows[0] and "surprisal" in rows[1]
    assert check_surprisal(rows)["n_missing"] == 1
    _, rows, report = add_surprisal(analyzer.states, cache, show_progress=False)
    assert report["n_augmented"] == 1 and report["n_skipped"] == 1 and report["ok"]
