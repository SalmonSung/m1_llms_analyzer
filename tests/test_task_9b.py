"""Task 9b on the tiny model: measures, the generation cache, resume, the record, the invariants."""

import hashlib
import json

import numpy as np
import pytest

from m1_analyzer import Analyzer, ModelConfig, RunConfig, ScoringConfig
from m1_analyzer.experiments import decoding
from m1_analyzer.experiments.paragraphs import Paragraph, hand_paragraphs
from m1_analyzer.experiments.splice import compute_splices, splice_ids
from m1_analyzer.experiments.task_9a import analyse_9a, cut_key
from m1_analyzer.experiments.task_9b import (
    ADDITIVE_CUT_FIELDS,
    REQUESTED_CUT_FIELDS,
    GenerationProtocol,
    build_record_9b,
    check_invariants_9b,
    compute_generation_9b,
    cut_specs,
    load_9a_inputs,
    load_generation_9b,
    measure_a,
    validate_record_9b,
)
from m1_analyzer.testing import build_tiny_local_model

PROTOCOL = GenerationProtocol(W_true=5, K=4, L=6, greedy_cap=6)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def analyzer(tmp_path_factory) -> Analyzer:
    path = build_tiny_local_model(tmp_path_factory.mktemp("tiny_9b"), max_positions=256)
    return Analyzer(RunConfig(model=ModelConfig(model_id=path, head="causal_lm"), scoring=ScoringConfig(batch_size=4)))


@pytest.fixture(scope="module")
def nine_a(analyzer, tmp_path_factory):
    """A real (tiny) 9a cache and record: the inputs Task 9b starts from."""
    out = tmp_path_factory.mktemp("nine_a")
    # Two copies of the hand paragraphs under different ids, so the length-matching
    # bins hold enough cuts for a close and a far decile.
    paragraphs = list(hand_paragraphs())
    paragraphs += [Paragraph(id=p.id + "-b", text=p.text, source=p.source, info=dict(p.info)) for p in hand_paragraphs()]
    cache = out / "splices.jsonl"
    header, rows = compute_splices(analyzer.states, paragraphs, window=5, splitter="regex", cache_path=cache,
                                   strata=[[1, 10], [10, 30], [30, 80]], provenance={"model_id": "tiny"},
                                   show_progress=False)
    record = analyse_9a(rows, header, model="tiny", n_perm=20, n_boot=10, min_per_group=1)
    assert record["cuts"], "the fixture needs labelled cuts"
    record_path = out / "record_9a.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    return {"cache": cache, "record": record_path, "rows": rows, "record_dict": record,
            "sha": (_sha(cache), _sha(record_path))}


@pytest.fixture(scope="module")
def generated(analyzer, nine_a, tmp_path_factory):
    path = tmp_path_factory.mktemp("gen") / "gen_9b.jsonl"
    header, rows = compute_generation_9b(analyzer.states, analyzer.models, nine_a["record"], nine_a["cache"],
                                         cache_path=path, protocol=PROTOCOL, provenance={"model_id": "tiny"},
                                         show_progress=False)
    return {"path": path, "header": header, "rows": rows}


def test_measure_a_matches_the_state_service_and_cached_surprisal(analyzer, nine_a):
    inputs = load_9a_inputs(nine_a["record"], nine_a["cache"])
    spec = cut_specs(inputs.record, inputs.rows)[0]
    states = analyzer.states
    ids = states.encode(spec["text"])
    i, j, W = spec["i"], spec["j"], PROTOCOL.W_true
    [orig] = states.states([ids], [[]], batch_size=1, return_token_logprobs=True)
    a = measure_a(states, ids, i, j, W, orig_token_logprobs=orig.token_logprobs, cached_surprisal=spec["surprisal"])
    spliced = splice_ids(ids, i, j)
    [spl] = states.states([spliced], [[]], batch_size=1, return_token_logprobs=True)
    by_hand = [orig.token_logprobs[j + 1 + k] - spl.token_logprobs[i + 1 + k] for k in range(W)]
    assert a["true_dlogp_k"] == pytest.approx(by_hand, abs=2e-6)
    assert a["true_dlogp"] == pytest.approx(np.mean(by_hand), abs=2e-6)
    assert a["lp_orig"] == pytest.approx([-spec["surprisal"][j + 1 + k] for k in range(W)], abs=1e-5)
    assert a["cached_lp_orig_max_abs_diff"] <= 1e-5
    assert spliced[i + 1: i + 1 + W] == ids[j + 1: j + 1 + W]
    with pytest.raises(ValueError, match="follow j"):
        measure_a(states, ids, i, len(ids) - 2, W, orig_token_logprobs=orig.token_logprobs)


def test_compute_generation_9b_writes_resumes_and_guards(analyzer, nine_a, generated, monkeypatch, tmp_path):
    record = nine_a["record_dict"]
    rows = generated["rows"]
    expected = [cut_key(c["paragraph"], c["i"], c["j"]) for c in record["cuts"]]
    assert [r["key"] for r in rows] == expected
    assert [r["cut_index"] for r in rows] == list(range(len(expected)))
    assert all(r["seed"] == PROTOCOL.seed_base + r["cut_index"] for r in rows)
    lines = generated["path"].read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["task"] == "9b" and len(lines) == 1 + len(rows)
    assert (_sha(nine_a["cache"]), _sha(nine_a["record"])) == nine_a["sha"]       # the 9a inputs are untouched
    for r in rows:
        assert len(r["samples_orig"]) == PROTOCOL.K and all(len(s) == PROTOCOL.L for s in r["samples_orig"])
        assert len(r["true_dlogp_k"]) == PROTOCOL.W_true
        assert 0 <= r["first_diff_greedy"] <= PROTOCOL.greedy_cap

    # Resume: nothing is generated again, and the rows come back equal.
    def boom(*args, **kwargs):
        raise AssertionError("resume must not generate")
    monkeypatch.setattr("m1_analyzer.experiments.task_9b.sample_continuations", boom)
    header2, rows2 = compute_generation_9b(analyzer.states, analyzer.models, nine_a["record"], nine_a["cache"],
                                           cache_path=generated["path"], protocol=PROTOCOL,
                                           provenance={"model_id": "tiny"}, show_progress=False)
    assert rows2 == rows and header2["protocol"] == generated["header"]["protocol"]

    # A different protocol, or a different 9a record, refuses to resume and names the field.
    with pytest.raises(ValueError, match="protocol"):
        compute_generation_9b(analyzer.states, analyzer.models, nine_a["record"], nine_a["cache"],
                              cache_path=generated["path"], protocol=GenerationProtocol(W_true=5, K=5, L=6, greedy_cap=6),
                              provenance={"model_id": "tiny"}, show_progress=False)
    shorter = dict(record, cuts=record["cuts"][:-1])
    other = tmp_path / "record_9a_short.json"
    other.write_text(json.dumps(shorter), encoding="utf-8")
    with pytest.raises(ValueError, match="source_record"):
        compute_generation_9b(analyzer.states, analyzer.models, other, nine_a["cache"], cache_path=generated["path"],
                              protocol=PROTOCOL, provenance={"model_id": "tiny"}, show_progress=False)
    monkeypatch.undo()

    # A truncated last line is dropped and that cut regenerated identically (same seed).
    truncated = tmp_path / "gen_trunc.jsonl"
    truncated.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][: len(lines[-1]) // 2], encoding="utf-8")
    _, rows3 = compute_generation_9b(analyzer.states, analyzer.models, nine_a["record"], nine_a["cache"],
                                     cache_path=truncated, protocol=PROTOCOL, provenance={"model_id": "tiny"},
                                     show_progress=False)
    assert [r["samples_orig"] for r in rows3] == [r["samples_orig"] for r in rows]
    _, reread = load_generation_9b(truncated)
    assert len(reread) == len(rows)

    # A dry run scores only the first cuts.
    _, few = compute_generation_9b(analyzer.states, analyzer.models, nine_a["record"], nine_a["cache"],
                                   cache_path=tmp_path / "gen_limit.jsonl", protocol=PROTOCOL, limit=2, show_progress=False)
    assert [r["key"] for r in few] == expected[:2]


def test_build_record_9b_has_the_requested_shape(nine_a, generated, tmp_path):
    record_9a = nine_a["record_dict"]
    record = build_record_9b(generated["header"], generated["rows"], record_9a, model="tiny", revision="r", dtype="float32")
    validate_record_9b(record)
    assert record["protocol"]["W_true"] == 5 and record["protocol"]["K"] == 4 and record["protocol"]["L"] == 6
    assert record["protocol"]["sampling"] == "nucleus p=0.95, T=1.0, seed 20250914+idx, paired O/S"
    assert record["protocol"]["overlap"] == "token F1 on ids, cross/within"
    assert set(record["meta"]) == {"model", "revision", "dtype", "date", "notes"}
    assert record["meta"]["model"] == "tiny" and record["meta"]["revision"] == "r"
    for n, (cut, orig) in enumerate(zip(record["cuts"], record_9a["cuts"])):
        assert list(cut)[: len(REQUESTED_CUT_FIELDS)] == list(REQUESTED_CUT_FIELDS)
        assert all(name in cut for name in ADDITIVE_CUT_FIELDS)
        assert cut["state_div"] == orig["divergence"] and cut["stratum"] == orig["stratum"]
        assert cut["cut_index"] == n and cut["key"] == cut_key(orig["paragraph"], orig["i"], orig["j"])
        assert cut["true_dlogp"] == pytest.approx(np.mean(cut["true_dlogp_k"]), abs=1e-5)
    assert record["invariants"]["ok"] and record["source"]["record_9a"]["n_cuts"] == len(record["cuts"])
    text = json.dumps(record)                                        # JSON-serialisable, NaN-free
    assert "NaN" not in text
    (tmp_path / "record_9b.json").write_text(text, encoding="utf-8")


def test_check_invariants_9b(nine_a, generated):
    record_9a = nine_a["record_dict"]
    rows = [dict(r) for r in generated["rows"]]
    inv = check_invariants_9b(rows, record_9a)
    assert inv["ok"] and inv["keys_match"] and inv["order_ok"] and inv["lp_orig_ok"]
    assert inv["n_rows"] == inv["n_expected"] == len(record_9a["cuts"])
    assert inv["max_abs_cached_lp_diff"] <= 1e-5 and inv["n_cuts_within_tol"] == len(rows)
    assert inv["n_collapsed"] == 0 and inv["state_div_matches"]

    tampered = [dict(r) for r in rows]
    tampered[0]["cached_lp_orig_max_abs_diff"] = 0.5
    bad = check_invariants_9b(tampered, record_9a)
    assert bad["max_abs_cached_lp_diff"] == 0.5 and not bad["lp_orig_ok"] and not bad["ok"]

    missing = check_invariants_9b(rows[1:], record_9a)
    assert not missing["keys_match"] and missing["missing"] == [rows[0]["key"]] and not missing["ok"]
    extra = check_invariants_9b(rows + [dict(rows[0], key="nope:1-2")], record_9a)
    assert extra["extra"] == ["nope:1-2"] and not extra["ok"]
    reordered = check_invariants_9b(list(reversed(rows)), record_9a)
    assert reordered["keys_match"] and not reordered["order_ok"] and not reordered["ok"]

    collapsed = [dict(r) for r in rows]
    collapsed[0]["within"] = 0.0
    collapsed[0]["sample_overlap"] = None
    inv_c = check_invariants_9b(collapsed, record_9a)
    assert inv_c["n_collapsed"] == 1 and inv_c["collapsed"] == [rows[0]["key"]] and inv_c["ok"]


def test_validate_record_9b_catches_bad_records(nine_a, generated):
    record = build_record_9b(generated["header"], generated["rows"], nine_a["record_dict"], model="tiny")
    good = json.loads(json.dumps(record))

    def broken(mutate):
        bad = json.loads(json.dumps(good))
        mutate(bad)
        return bad

    with pytest.raises(ValueError, match="close/far"):
        validate_record_9b(broken(lambda r: r["cuts"][0].__setitem__("pair", "mid")))
    with pytest.raises(ValueError, match="true_dlogp_k"):
        validate_record_9b(broken(lambda r: r["cuts"][0]["true_dlogp_k"].pop()))
    with pytest.raises(ValueError, match="samples_orig"):
        validate_record_9b(broken(lambda r: r["cuts"][0]["samples_orig"].pop()))
    with pytest.raises(ValueError, match="samples_spliced"):
        validate_record_9b(broken(lambda r: r["cuts"][0]["samples_spliced"][0].pop()))
    with pytest.raises(ValueError, match="sample_overlap"):
        validate_record_9b(broken(lambda r: r["cuts"][0].__setitem__("sample_overlap", None)))
    with pytest.raises(ValueError, match="first_diff_greedy"):
        validate_record_9b(broken(lambda r: r["cuts"][0].__setitem__("first_diff_greedy", 99)))
    with pytest.raises(ValueError, match="protocol is missing"):
        validate_record_9b(broken(lambda r: r["protocol"].pop("K")))


def test_protocol_validates_and_names_itself():
    p = GenerationProtocol()
    assert p.as_header()["sampling"] == "nucleus p=0.95, T=1.0, seed 20250914+idx, paired O/S"
    assert p.seed_for(3) == 20250917 and p.as_header()["top_k"] is None
    with pytest.raises(ValueError):
        GenerationProtocol(K=1)
    with pytest.raises(ValueError):
        GenerationProtocol(top_p=0)


def test_load_9a_inputs_refuses_the_wrong_files(nine_a, tmp_path):
    inputs = load_9a_inputs(nine_a["record"], nine_a["cache"])
    assert inputs.source_record["n_cuts"] == len(inputs.record["cuts"]) and len(inputs.source_cache["sha256"]) == 64
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"cuts": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no cuts"):
        load_9a_inputs(bad, nine_a["cache"])
    mid = dict(inputs.record, cuts=[dict(inputs.record["cuts"][0], pair="mid")])
    bad.write_text(json.dumps(mid), encoding="utf-8")
    with pytest.raises(ValueError, match="not labelled"):
        load_9a_inputs(bad, nine_a["cache"])
    foreign = dict(inputs.record, cuts=[dict(inputs.record["cuts"][0], paragraph="nowhere")])
    bad.write_text(json.dumps(foreign), encoding="utf-8")
    with pytest.raises(ValueError, match="not in"):
        load_9a_inputs(bad, nine_a["cache"])
    other = tmp_path / "other.jsonl"
    other.write_text('{"kind": "header", "task": "1b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Task 9a header"):
        load_9a_inputs(nine_a["record"], other)


def test_eos_is_an_ordinary_token(analyzer, nine_a, monkeypatch):
    """A sample that draws EOS keeps going: every sample is exactly L ids, and the count is reported."""
    inputs = load_9a_inputs(nine_a["record"], nine_a["cache"])
    spec = cut_specs(inputs.record, inputs.rows)[0]
    eos = analyzer.models.tokenizer.eos_token_id
    forced = np.full((PROTOCOL.K, PROTOCOL.L), eos)
    monkeypatch.setattr(decoding, "sample_continuations", lambda *a, **k: forced)
    monkeypatch.setattr("m1_analyzer.experiments.task_9b.sample_continuations", lambda *a, **k: forced)
    from m1_analyzer.experiments.task_9b import score_cut
    row = score_cut(analyzer.states, analyzer.models, spec, PROTOCOL, orig_logprobs_cache={})
    assert row["n_eos_orig"] == PROTOCOL.K and all(len(s) == PROTOCOL.L for s in row["samples_orig"])
    assert row["within"] == 1.0 and row["sample_overlap"] == 1.0
