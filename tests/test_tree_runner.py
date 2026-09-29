"""Tree recovery on the runner offline: the unmodified reference, jobs C and A on the tiny model, their
checks, the deliverables, and job B (8a with Jensen-Shannon) against the 8a pipeline."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from m1_analyzer import Analyzer, ModelConfig, RunConfig, ScoringConfig
from m1_analyzer.experiments import task_8a_js, tree_runner
from m1_analyzer.experiments import tree_runner_ref as ref
from m1_analyzer.experiments.frames_8a import FRAME_WORDS_8A, PAIRS, generate_frames_8a, save_frames_8a
from m1_analyzer.experiments.task_8a import ModelSpec, build_record_8a, score_model_8a
from m1_analyzer.testing import build_tiny_local_model

REPO_ROOT = Path(__file__).resolve().parents[1]
ANCHOR_FILE = REPO_ROOT / "data" / "anchor_tree_local.json"

WORDS = [
    ["the", "quick", "brown", "fox", "jumps"],
    ["a", "lazy", "dog", "one", "two", "three"],
    ["hello", "world", "model"],
]
TREES = [
    "(S (NP (DT the) (JJ quick) (JJ brown) (NN fox)) (VP (VBZ jumps)) (. .))",
    "(S (NP (DT a) (JJ lazy) (NN dog)) (VP (CD one) (CD two) (CD three)) (. ?) ('' ''))",
    "(S (NP (UH hello)) (NP (NN world) (NN model)))",
]
FILLER_WORDS = sorted({w for f in ref.FILLERS if f != "<del>" for w in f.split()})
EXTRA = FILLER_WORDS + [w[:1].upper() + w[1:] for w in FILLER_WORDS + sum(WORDS, [])] + ["<|endoftext|>"]


def _keys(n):
    return [f"{i},{j}" for i in range(n) for j in range(i + 1, n) if not (i == 0 and j == n - 1)]


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = build_tiny_local_model(tmp_path_factory.mktemp("tiny_tree"), extra_vocab=EXTRA, max_positions=64)
    tok, mdl, bos, device, analyzer = tree_runner.load_model_f32(path, revision=None, device="cpu")
    return tok, mdl, bos, device, analyzer


@pytest.fixture(scope="module")
def sentences():
    return [{"id": f"s{k}", "words": w, "tree": t, "keys": _keys(len(w))} for k, (w, t) in enumerate(zip(WORDS, TREES))]


@pytest.fixture(scope="module")
def t4(model, sentences, tmp_path_factory):
    tok, mdl, bos, device, _ = model
    cache = tmp_path_factory.mktemp("t4") / "t4.jsonl"
    rows = tree_runner.run_t4(tok, mdl, bos, sentences, device=device, batch=7, cache_path=cache,
                              model_meta={"model_id": "tiny", "dtype": "float32"}, show_progress=False)
    return cache, rows


def _cache_file(path, sentences, t4_rows):
    """A 1b-style cache whose totals and counts are this model's own (t4 ``total``/``n_tok``)."""
    by_id = {r["id"]: r for r in t4_rows}
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "header", "schema": 2, "proforms": list(ref.FILLERS)}) + "\n")
        for s in sentences:
            spans = {f: {k: v[:2] for k, v in by_id[s["id"]]["rows"][f].items()} for f in ref.FILLERS}
            fh.write(json.dumps({"kind": "sentence", "id": s["id"], "words": s["words"], "tree": s["tree"],
                                 "base": [0.0, 1], "spans": spans}) + "\n")
    return path


# ---------------------------------------------------------------- reference


def test_reference_is_byte_identical_to_the_one_sent():
    assert tree_runner.reference_sha256() == tree_runner.REFERENCE_SHA256
    assert tree_runner.reference_unmodified()
    assert len(ref.FILLERS) == 17 and "<del>" in ref.FILLERS


def test_end_string_skips_closing_quotes_and_defaults_to_period():
    assert ref.end_string(TREES[0]) == "."
    assert ref.end_string(TREES[1]) == "?"
    assert ref.end_string(TREES[2]) == "."


def test_jsd_distance_properties():
    rng = np.random.default_rng(0)
    a = np.log(rng.dirichlet(np.ones(50))).astype(np.float32)
    b = np.log(rng.dirichlet(np.ones(50))).astype(np.float32)
    assert ref.jsd_distance(a, a) == pytest.approx(0.0, abs=1e-6)
    assert ref.jsd_distance(a, b) == pytest.approx(ref.jsd_distance(b, a))
    disjoint = np.log(np.array([1.0, 1e-300])), np.log(np.array([1e-300, 1.0]))
    assert ref.jsd_distance(*disjoint) == pytest.approx(np.sqrt(np.log(2)), rel=1e-6)


def test_head_share_by_hand():
    a = np.log(np.array([0.7, 0.2, 0.05, 0.05]))
    b = np.log(np.array([0.1, 0.6, 0.2, 0.1]))
    d2 = (a - b) ** 2
    assert task_8a_js.head_share(a, b, k=1) == pytest.approx(d2[[0, 1]].sum() / d2.sum())
    assert task_8a_js.head_share(a, b, k=4) == pytest.approx(1.0)
    assert task_8a_js.head_share(a, a, k=1) == 0.0


# -------------------------------------------------------------------- job C


def test_t1_rows_and_resume(model, sentences, tmp_path, monkeypatch):
    tok, mdl, bos, device, _ = model
    cache = tmp_path / "t1.jsonl"
    rows = tree_runner.run_t1(tok, mdl, bos, sentences, device=device, cache_path=cache, show_progress=False,
                              model_meta={"model_id": "tiny", "dtype": "float32"})
    assert [r["id"] for r in rows] == ["s0", "s1", "s2"]
    for s, r in zip(sentences, rows):
        assert list(r["t1"]) == s["keys"]
        for l2, js in r["t1"].values():
            assert l2 > 0 and 0 <= js <= np.sqrt(np.log(2)) + 1e-9
    monkeypatch.setattr(ref, "t1_rows", lambda *a, **k: pytest.fail("resume re-scored a sentence"))
    again = tree_runner.run_t1(tok, mdl, bos, sentences, device=device, cache_path=cache, show_progress=False,
                               model_meta={"model_id": "tiny", "dtype": "float32"})
    assert again == rows
    with pytest.raises(ValueError, match="dtype"):
        tree_runner.run_t1(tok, mdl, bos, sentences, cache_path=cache, show_progress=False,
                           model_meta={"model_id": "tiny", "dtype": "bfloat16"})


def test_t1_anchor_check_passes_on_itself_and_fails_when_perturbed(model, sentences):
    tok, mdl, bos, device, _ = model
    rows = tree_runner.run_t1(tok, mdl, bos, sentences, device=device, show_progress=False)
    anchor = {"sentences": [{"id": r["id"], "t1": r["t1"]} for r in rows]}
    ok = tree_runner.check_t1_anchor(rows, anchor)
    assert ok["pass"] and ok["n_spans"] == sum(len(s["keys"]) for s in sentences)
    assert ok["l2"]["spearman"] == pytest.approx(1.0) and ok["js"]["median_rel_diff"] == 0.0
    bad = [{"id": r["id"], "t1": {k: [v[0] * 1.05, v[1]] for k, v in r["t1"].items()}} for r in rows]
    fail = tree_runner.check_t1_anchor(bad, anchor)
    assert not fail["pass"] and not fail["l2"]["pass"] and fail["js"]["pass"]
    with pytest.raises(ValueError, match="not scored"):
        tree_runner.check_t1_anchor(rows[:1], anchor)


def test_real_anchor_file_has_the_announced_contents():
    anchor = tree_runner.load_anchor(ANCHOR_FILE)
    assert anchor["model"] == tree_runner.MODEL_ID and anchor["revision"] == tree_runner.MODEL_REVISION
    assert len(anchor["sentences"]) == 20 and sum(len(s["t1"]) for s in anchor["sentences"]) == 3505
    assert [a["id"] for a in anchor["t4"]] == ["wsj_0001.mrg:1", "wsj_0003.mrg:5"]
    assert sum(len(t) for a in anchor["t4"] for t in a["rows"].values()) == 2686


# -------------------------------------------------------------------- job A


def test_t4_rows_shape_and_causality(t4, sentences):
    _, rows = t4
    for s, r in zip(sentences, rows):
        assert r["end"] == ref.end_string(s["tree"])
        assert set(r["rows"]) == set(ref.FILLERS)
        for table in r["rows"].values():
            assert list(table) == s["keys"]
            for total, n_tok, n_pre, n_suf, suf_sub, end_sub, pre_check in table.values():
                assert total <= 0 and n_tok >= n_pre + n_suf and end_sub <= 0
    caus = tree_runner.check_t4_causality(rows)
    assert caus["pass"] and caus["max_abs_pre_check"] <= 1e-3


def test_t4_batch_size_changes_no_number(model, sentences, t4):
    tok, mdl, bos, device, _ = model
    _, rows = t4
    one = tree_runner.run_t4(tok, mdl, bos, sentences[:1], device=device, batch=64, show_progress=False)
    check = tree_runner.check_t4_anchor(one, [rows[0]])
    assert check["pass"] and check["count_mismatches"] == 0 and max(check["max_abs_diff"].values()) < 1e-4


def test_t4_anchor_check_fails_on_counts_and_nats(t4):
    _, rows = t4
    bad = json.loads(json.dumps(rows[:1]))
    bad[0]["rows"]["it"]["0,1"][2] += 1
    bad[0]["rows"]["did"]["0,1"][0] -= 0.1
    check = tree_runner.check_t4_anchor(bad, [rows[0]])
    assert not check["pass"] and check["count_mismatches"] == 1 and check["max_abs_diff"]["total"] > 0.05


def test_t4_cache_check_and_load_1b_cache(t4, sentences, tmp_path):
    _, rows = t4
    header, loaded = tree_runner.load_1b_cache(_cache_file(tmp_path / "cache.jsonl", sentences, rows))
    assert [s["keys"] for s in loaded] == [s["keys"] for s in sentences]
    ok = tree_runner.check_t4_cache(rows, loaded)
    assert ok["pass"] and ok["n_tok_mismatches"] == 0 and ok["spearman_total"] == pytest.approx(1.0)
    loaded[0]["spans"]["it"]["0,1"] = [loaded[0]["spans"]["it"]["0,1"][0], 99]
    assert not tree_runner.check_t4_cache(rows, loaded)["pass"]


def test_load_1b_cache_refuses_other_fillers(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({"kind": "header", "proforms": ["it", "there"]}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fillers"):
        tree_runner.load_1b_cache(path)


def test_finalize_writes_meta_then_rows_gzipped(t4, tmp_path):
    cache, rows = t4
    meta = tree_runner.run_meta("t4", model_id="tiny", revision=None, wall_seconds=tree_runner.wall_seconds(rows),
                                checks={"causality": tree_runner.check_t4_causality(rows)})
    assert meta["reference_code_modified"] is False and meta["reference_code_sha"] == tree_runner.REFERENCE_SHA256
    out = tree_runner.finalize(rows, tmp_path / "tree_t4.jsonl.gz", meta, job="t4", expected=3)
    with gzip.open(out, "rt", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh]
    assert lines[0]["meta"]["job"] == "t4" and len(lines) == 4
    assert set(lines[1]) == {"id", "end", "orig", "rows"}
    back_meta, back = tree_runner.load_rows(out)
    assert back_meta["job"] == "t4" and [r["id"] for r in back] == ["s0", "s1", "s2"]
    with pytest.raises(ValueError, match="finish the run"):
        tree_runner.finalize(rows, tmp_path / "x.jsonl.gz", meta, job="t4", expected=1000)


# -------------------------------------------------------------------- job B


@pytest.fixture(scope="module")
def js_setup(tmp_path_factory):
    path = build_tiny_local_model(tmp_path_factory.mktemp("tiny_js"), extra_vocab=FRAME_WORDS_8A, max_positions=64)
    analyzer = Analyzer(RunConfig(model=ModelConfig(model_id=path, head="causal_lm", dtype="float32"),
                                  scoring=ScoringConfig(bos_policy="none", batch_size=64)))
    frames = save_frames_8a(generate_frames_8a({"tiny": analyzer.models.tokenizer}, n=3, seed=8),
                            tmp_path_factory.mktemp("frames") / "frames_8a_seed8.json")
    spec = ModelSpec("tiny", path)
    base = tmp_path_factory.mktemp("caches")
    score_model_8a(analyzer, spec, frames, cache_path=base / "scores_8a_tiny.jsonl", show_progress=False)
    record_8a = build_record_8a(frames, {"tiny": base / "scores_8a_tiny.jsonl"})
    header, rows = task_8a_js.score_model_8a_js(analyzer, spec, frames, cache_path=base / "scores_8a_js_tiny.jsonl",
                                                frames_per_batch=2, show_progress=False)
    return analyzer, spec, frames, record_8a, base / "scores_8a_js_tiny.jsonl", rows


def test_job_b_repeats_the_8a_l2_exactly(js_setup):
    *_, record_8a, _, rows = js_setup
    flat = task_8a_js.rows_from_cache(rows)
    assert len(flat) == 3 * len(PAIRS)
    match = task_8a_js.l2_match(flat, record_8a, "tiny")
    assert match["pass"] and match["max_rel"] == 0.0
    for r in flat:
        assert 0 <= r["js_ret"] <= np.sqrt(np.log(2)) and 0 <= r["head_ctrl"] <= 1


def test_job_b_record_schema_and_refusal(js_setup):
    _, _, frames, record_8a, cache, _ = js_setup
    record = task_8a_js.build_record_8a_js({"tiny": cache}, record_8a, frames)
    assert set(record["meta"]) >= {"date", "reference_code_sha", "l2_match_max_rel", "versions"}
    assert record["meta"]["l2_match_max_rel"] == {"tiny": 0.0}
    assert set(record["models"]["tiny"]["rows"][0]) == set(task_8a_js.ROW_KEYS)
    moved = json.loads(json.dumps(record_8a))
    moved["models"]["tiny"]["rows"][4]["ctrl"] *= 1.03
    with pytest.raises(ValueError, match="Stop and report"):
        task_8a_js.build_record_8a_js({"tiny": cache}, moved, frames)


def test_job_b_resume_scores_nothing(js_setup, monkeypatch):
    analyzer, spec, frames, _, cache, rows = js_setup
    monkeypatch.setattr(task_8a_js, "score_frames_js_batch", lambda *a, **k: pytest.fail("re-scored"))
    _, again = task_8a_js.score_model_8a_js(analyzer, spec, frames, cache_path=cache, show_progress=False)
    assert again == rows
