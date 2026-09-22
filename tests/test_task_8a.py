"""Task 8a offline: the frame generator on fake tokenizers, the scorer on the tiny model, the anchor, the record."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from m1_analyzer import Analyzer, ModelConfig, RunConfig, ScoringConfig
from m1_analyzer.experiments import frames_8a, task_8a
from m1_analyzer.experiments.frames_8a import (
    CODES, FRAME_WORDS_8A, FUNC, MAINS, NOUNS, PAIRS, VERB_POOL_8A, VERBS, filter_pools, format_sentence,
    frames_sha256, generate_frames_8a, load_frames_8a, passes, save_frames_8a, split_sentence, token_assertions,
    token_table, verb_pool_kept,
)
from m1_analyzer.experiments.task_8a import (
    ANCHOR_KEY, MODELS_8A, ModelSpec, anchor_check, anchor_diagnosis, anchor_summary, anchor_tokens_check,
    bos_added, build_record_8a, frame_scalars, load_anchor_file, load_scores_8a, missing_models, model_block,
    prefix_ids, rows_from_frame, score_frames_batch, score_model_8a, top_k_tokens, validate_record_8a, word_id,
)
from m1_analyzer.testing import build_tiny_local_model

REPO_ROOT = Path(__file__).resolve().parents[1]
ANCHOR_FILE = REPO_ROOT / "data" / "frames_8a_local30.json"


class FakeTokenizer:
    """Word-level: every word in `known` is one id; any other word is two ids; optional BOS."""

    def __init__(self, known, *, bos_id=None):
        self.vocab = {w: 10 + k for k, w in enumerate(sorted(known))}
        self.inverse = {i: w for w, i in self.vocab.items()}
        self.bos_token_id = bos_id
        self.eos_token_id = 2

    def _ids(self, text):
        ids = []
        for word in text.split():
            if word in self.vocab:
                ids.append(self.vocab[word])
            else:
                ids.extend([3, 4])  # a multi-token word
        return ids

    def __call__(self, text, add_special_tokens=True):
        ids = self._ids(text)
        if add_special_tokens and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids):
        return " ".join(self.inverse.get(i, {2: "<eos>", 3: "<un>", 4: "<known>"}.get(i, str(i)))
                        if i != self.bos_token_id else "<bos>" for i in ids)


ALL_WORDS = set(FRAME_WORDS_8A)


@pytest.fixture(scope="module")
def tokenizers():
    plain = FakeTokenizer(ALL_WORDS)
    with_bos = FakeTokenizer(ALL_WORDS - {"soldier", "praised", "won", "kissed"}, bos_id=1)
    return {"plain": plain, "bos": with_bos}


@pytest.fixture(scope="module")
def analyzer(tmp_path_factory) -> Analyzer:
    path = build_tiny_local_model(tmp_path_factory.mktemp("tiny_8a"), extra_vocab=FRAME_WORDS_8A, max_positions=64)
    return Analyzer(RunConfig(model=ModelConfig(model_id=path, head="causal_lm"),
                              scoring=ScoringConfig(bos_policy="none", batch_size=64)))


@pytest.fixture(scope="module")
def frames_file(tmp_path_factory, analyzer) -> Path:
    payload = generate_frames_8a({"tiny": analyzer.models.tokenizer}, n=4, seed=8)
    return save_frames_8a(payload, tmp_path_factory.mktemp("frames") / "frames_8a_seed8.json")


@pytest.fixture(scope="module")
def scored(tmp_path_factory, analyzer, frames_file):
    cache = tmp_path_factory.mktemp("cache") / "scores_8a_tiny.jsonl"
    spec = ModelSpec("tiny", analyzer.config.model.model_id)
    header, rows = score_model_8a(analyzer, spec, frames_file, cache_path=cache, frames_per_batch=3, show_progress=False)
    return spec, cache, header, rows


# ------------------------------------------------------------------ frames


def test_generator_is_deterministic_for_seed_8(tokenizers):
    a = generate_frames_8a(tokenizers, n=6, seed=8)
    b = generate_frames_8a(tokenizers, n=6, seed=8)
    assert a == b
    assert a["meta"]["n_frames"] == 6 and a["meta"]["seed"] == 8 and a["meta"]["n_frames_requested"] == 6
    assert a["meta"]["tokenizers_filtered_on"] == ["plain", "bos"]
    assert a["meta"]["generator"] == "the code block above, unmodified"
    assert a["verb_pool"] == list(VERB_POOL_8A)
    assert generate_frames_8a(tokenizers, n=6, seed=9)["frames"] != a["frames"]


def test_pool_filter_drops_words_that_are_multi_token_anywhere(tokenizers):
    nouns, verbs, mains, cvs = filter_pools(list(tokenizers.values()))
    assert "soldier" not in nouns and "praised" not in verbs and "won" not in {v for v, _ in mains}
    assert len(nouns) == len(NOUNS) - 1 and len(verbs) == len(VERBS) - 2 and len(mains) == len(MAINS) - 1
    payload = generate_frames_8a(tokenizers, n=20, seed=8)
    assert payload["meta"]["pools"] == {"pool_nouns": len(NOUNS) - 1, "pool_verbs": len(VERBS) - 2,
                                        "pool_mains": len(MAINS) - 1, "pool_cvs": 7, "draws": 20}
    for frame in payload["frames"]:
        words = {frame[s] for s in ("N1", "N2", "N3", "N4", "V2", "V3", "V4", "V")}
        assert not words & {"soldier", "praised", "won", "kissed"}


def test_every_frame_passes_both_assertions_in_every_tokenizer_and_is_unique(tokenizers):
    payload = generate_frames_8a(tokenizers, n=12, seed=8)
    seen = set()
    for frame in payload["frames"]:
        assert (frame["N1"], frame["V"]) not in seen
        seen.add((frame["N1"], frame["V"]))
        S = {code: split_sentence(frame["sentences"][code]) for code in CODES}
        assert passes(S, list(tokenizers.values()))
        for tok in tokenizers.values():
            table = token_table(tok, [frame])
            counts = {t["code"]: t["n_tok"] for t in table}
            lasts = {t["code"]: t["last_tok"] for t in table}
            assert token_assertions(counts, lasts) == []
    # The recorded-value form agrees with `passes` when a count is tampered with.
    counts["ORC_B"] += 1
    assert any("ORC/ORC_B: n_tok" in f for f in token_assertions(counts, lasts))
    lasts["WHO_B"] = "other"
    assert any("WHO/WHO_B: last_tok" in f for f in token_assertions(counts, lasts))


def test_max_draws_bounds_the_generator(tokenizers):
    payload = generate_frames_8a(tokenizers, n=10_000, seed=8, max_draws=15)
    assert payload["meta"]["pools"]["draws"] == 15 and payload["meta"]["n_frames"] <= 15


def test_function_words_must_be_single_tokens():
    with pytest.raises(AssertionError, match="function word"):
        frames_8a.generate([FakeTokenizer(ALL_WORDS - {"whom"})], n=2)


def test_sentence_round_trip_and_anchor_style_split():
    assert format_sentence("The soldier", " won the prize.") == "The soldier | won the prize."
    assert split_sentence("The soldier | won the prize.") == ("The soldier", " won the prize.")
    assert split_sentence("The soldier quickly hired the poet |.") == ("The soldier quickly hired the poet", ".")
    with pytest.raises(ValueError, match="stop"):
        split_sentence("no stop here")


def test_token_table_counts_the_bos_and_bos_added_sees_it(tokenizers):
    payload = generate_frames_8a(tokenizers, n=2, seed=8)
    plain = token_table(tokenizers["plain"], payload["frames"])
    bos = token_table(tokenizers["bos"], payload["frames"])
    assert len(plain) == len(CODES) * 2
    assert all(b["n_tok"] == p["n_tok"] + 1 for p, b in zip(plain, bos))
    assert not bos_added(tokenizers["plain"]) and bos_added(tokenizers["bos"])
    assert prefix_ids(tokenizers["bos"], "The soldier")[0] == 1


def test_verb_pool_kept_is_the_single_token_subset(tokenizers):
    assert verb_pool_kept(tokenizers["plain"]) == list(VERB_POOL_8A)
    kept = verb_pool_kept(FakeTokenizer(ALL_WORDS - {"was", "went"}))
    assert "was" not in kept and "went" not in kept and len(kept) == len(VERB_POOL_8A) - 2
    with pytest.raises(ValueError, match="not 1"):
        word_id(FakeTokenizer(ALL_WORDS - {"was"}), "was")


def test_frames_file_round_trip(tmp_path, tokenizers):
    payload = generate_frames_8a(tokenizers, n=3, seed=8)
    path = save_frames_8a(payload, tmp_path / "f.json")
    assert load_frames_8a(path) == payload
    assert len(frames_sha256(path)) == 64
    bad = dict(payload)
    bad["frames"] = [dict(payload["frames"][0], sentences={"REF": "The x | y"})]
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="lacks sentences"):
        load_frames_8a(tmp_path / "bad.json")


def test_frame_words_cover_every_pool_word():
    for w in list(NOUNS) + list(VERBS) + [v for v, _ in MAINS] + list(FUNC) + list(VERB_POOL_8A):
        assert w in ALL_WORDS


# ----------------------------------------------------------------- scoring


def test_scoring_shapes_and_ranges(scored, frames_file):
    spec, cache, header, rows = scored
    n = len(load_frames_8a(frames_file)["frames"])
    assert len(rows) == n == 4
    assert header["task"] == "8a" and header["model_key"] == "tiny" and header["bos_policy"].startswith("tokenizer default")
    assert header["frames_sha256"] == frames_sha256(frames_file)
    assert header["verb_pool_kept"] == list(VERB_POOL_8A) and header["logits_dtype"] == "float32"
    for row in rows:
        assert set(row["dist_to_ref"]) == set(CODES) - {"REF"}
        assert set(row["verbmass"]) == set(row["logp_V"]) == set(row["top10"]) == set(CODES)
        for code in CODES:
            assert 0 < row["verbmass"][code] <= 1
            assert row["logp_V"][code] <= 0
            assert len(row["top10"][code]) == 10
            probs = [p for _, p in row["top10"][code]]
            assert probs == sorted(probs, reverse=True) and all(0 < p <= 1 for p in probs)
            assert all(isinstance(t, str) for t, _ in row["top10"][code])
        assert all(d >= 0 for d in row["dist_to_ref"].values())
        record_rows = rows_from_frame(row)
        assert [(r["structure"], r["control"]) for r in record_rows] == PAIRS
        assert record_rows[0]["ret"] == row["dist_to_ref"]["ORC"] and record_rows[1]["ctrl"] == row["dist_to_ref"]["ORC_A"]


def test_distance_and_verb_mass_recomputed_by_hand(scored, analyzer, frames_file):
    spec, cache, header, rows = scored
    frame = load_frames_8a(frames_file)["frames"][0]
    tok = analyzer.models.tokenizer
    prefixes = {code: split_sentence(frame["sentences"][code])[0] for code in ("REF", "ORC")}
    seqs = [prefix_ids(tok, prefixes["REF"]), prefix_ids(tok, prefixes["ORC"])]
    ref, orc = analyzer.states.states(seqs, [[len(s) - 1] for s in seqs], batch_size=2)
    assert float(np.linalg.norm(orc.states[0] - ref.states[0])) == pytest.approx(rows[0]["dist_to_ref"]["ORC"], abs=1e-5)
    ids = [word_id(tok, w) for w in VERB_POOL_8A]
    assert float(np.exp(ref.states[0][ids]).sum()) == pytest.approx(rows[0]["verbmass"]["REF"], abs=1e-6)
    assert ref.states[0][word_id(tok, frame["V"])] == pytest.approx(rows[0]["logp_V"]["REF"], abs=1e-6)
    scal = frame_scalars({code: ref.states[0] for code in CODES}, verb_token_ids=ids, v_id=ids[0], tokenizer=tok)
    assert all(d == 0.0 for d in scal["dist_to_ref"].values())
    assert top_k_tokens(np.log(np.array([0.1, 0.6, 0.3])), tok, k=2)[0][1] == pytest.approx(0.6)


def test_scorer_refuses_a_bos_policy_that_adds_a_token(tmp_path, frames_file):
    path = build_tiny_local_model(tmp_path / "tiny_auto", extra_vocab=FRAME_WORDS_8A, max_positions=64)
    auto = Analyzer(RunConfig(model=ModelConfig(model_id=path, head="causal_lm"), scoring=ScoringConfig(bos_policy="auto")))
    with pytest.raises(ValueError, match="bos_policy='none'"):
        score_model_8a(auto, ModelSpec("tiny", path), frames_file, show_progress=False)
    with pytest.raises(ValueError, match="must not prepend"):
        score_frames_batch(auto.states, auto.models.tokenizer, [(0, load_frames_8a(frames_file)["frames"][0])],
                           verb_token_ids=[])


def test_scorer_rejects_a_frame_that_fails_the_assertions_here(analyzer, frames_file):
    frame = dict(load_frames_8a(frames_file)["frames"][0])
    sentences = dict(frame["sentences"])
    sentences["ORC_B"] = "The one two three | x."
    frame["sentences"] = sentences
    with pytest.raises(ValueError, match="token assertions"):
        score_frames_batch(analyzer.states, analyzer.models.tokenizer, [(0, frame)], verb_token_ids=[])


def test_resume_scores_nothing_and_refuses_a_changed_header(scored, analyzer, frames_file, tmp_path, monkeypatch):
    spec, cache, header, rows = scored
    calls = []
    monkeypatch.setattr(task_8a, "score_frames_batch", lambda *a, **k: calls.append(1) or [])
    header2, rows2 = score_model_8a(analyzer, spec, frames_file, cache_path=cache, show_progress=False)
    assert calls == [] and rows2 == rows and header2["date"] == header["date"]
    monkeypatch.undo()
    # A different frames file (by sha256) refuses, naming the field.
    other = save_frames_8a(generate_frames_8a({"tiny": analyzer.models.tokenizer}, n=2, seed=9), tmp_path / "other.json")
    with pytest.raises(ValueError, match="frames_(file|sha256)"):
        score_model_8a(analyzer, spec, other, cache_path=cache, show_progress=False)
    # A different model key refuses too.
    with pytest.raises(ValueError, match="model_key"):
        score_model_8a(analyzer, ModelSpec("tiny2", spec.hf_id), frames_file, cache_path=cache, show_progress=False)


def test_truncated_last_line_is_rescored_and_limit_is_a_dry_run(scored, analyzer, frames_file, tmp_path):
    spec, cache, header, rows = scored
    partial = tmp_path / "partial.jsonl"
    lines = cache.read_text(encoding="utf-8").splitlines()
    partial.write_text("\n".join(lines[:3]) + "\n" + lines[3][:40], encoding="utf-8")
    header2, rows2 = score_model_8a(analyzer, spec, frames_file, cache_path=partial, show_progress=False)
    assert [r["frame"] for r in rows2] == [0, 1, 2, 3]
    assert rows2[2]["dist_to_ref"] == rows[2]["dist_to_ref"]
    h, r = load_scores_8a(partial)
    assert len(r) == 4 and h["task"] == "8a"
    dry = tmp_path / "dry.jsonl"
    _, rows3 = score_model_8a(analyzer, spec, frames_file, cache_path=dry, limit=2, show_progress=False)
    assert [r["frame"] for r in rows3] == [0, 1]
    (tmp_path / "headless.jsonl").write_text(json.dumps({"kind": "frame", "frame": 0}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a score cache"):
        load_scores_8a(tmp_path / "headless.jsonl")


# ------------------------------------------------------------------ anchor


def _synthetic_anchor(rows, frames, path):
    reference = []
    for row in rows:
        for r in rows_from_frame(row):
            r = dict(r)
            r["stop_tok"] = row["last_tok"][r["structure"]]
            r["n_tok"] = row["n_tok"][r["structure"]]
            reference.append(r)
    path.write_text(json.dumps({"seed": 8, "n_frames": len(frames), "frames": frames, "qwen25_reference_rows": reference}))
    return path


def test_real_anchor_file_loads_with_its_schema():
    anchor = load_anchor_file(ANCHOR_FILE)
    assert len(anchor.frames) == 30 and len(anchor.rows) == 210 and anchor.meta["seed"] == 8
    assert sorted({(r["structure"], r["control"]) for r in anchor.rows}) == sorted(PAIRS)
    assert all(r["n_tok"] >= 2 and r["stop_tok"].startswith(" ") for r in anchor.rows)
    assert split_sentence(anchor.frames[0]["sentences"]["ORC"])[0] == "The soldier that the tailor followed"
    assert len(anchor.sha256) == 64
    # A word-level tokenizer reproduces the counts (no BOS in the local run) but not Qwen's ' word' last tokens.
    plain = FakeTokenizer(ALL_WORDS)
    problems = anchor_tokens_check(plain, anchor)
    assert all("last_tok" in p for p in problems) and not any("n_tok" in p for p in problems)
    assert any("n_tok" in p for p in anchor_tokens_check(FakeTokenizer(ALL_WORDS, bos_id=1), anchor))


def test_anchor_file_errors_name_the_missing_key(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"frames": [], "rows": []}))
    with pytest.raises(ValueError, match="frames"):
        load_anchor_file(tmp_path / "a.json")
    bad = json.loads(ANCHOR_FILE.read_text(encoding="utf-8"))
    del bad["qwen25_reference_rows"][0]["ctrl"]
    (tmp_path / "b.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="ctrl"):
        load_anchor_file(tmp_path / "b.json")


def test_anchor_check_passes_on_our_own_numbers_and_fails_when_perturbed(scored, analyzer, frames_file, tmp_path):
    spec, cache, header, rows = scored
    frames = load_frames_8a(frames_file)["frames"]
    path = _synthetic_anchor(rows, frames, tmp_path / "anchor.json")
    block = anchor_check(analyzer, spec, path)
    assert block["pass"] and block["n_rows"] == 28 and block["max_rel_diff_ret"] == 0.0
    assert block["token_mismatches"] == [] and block["max_abs_diff_verbmass"] == 0.0
    assert anchor_diagnosis(block).startswith("anchor passed")
    summary = anchor_summary(block)
    assert set(summary) == {"model", "frames_file", "max_rel_diff_ret", "max_rel_diff_ctrl", "max_abs_diff_verbmass",
                            "n_rows", "pass", "sha256"}
    payload = json.loads(path.read_text())
    payload["qwen25_reference_rows"][5]["ret"] *= 1.05
    payload["qwen25_reference_rows"][6]["verbmass_emb"] += 0.05
    payload["qwen25_reference_rows"][7]["stop_tok"] = " nope"
    path.write_text(json.dumps(payload))
    block = anchor_check(analyzer, spec, path)
    assert not block["pass"]
    assert block["max_rel_diff_ret"] == pytest.approx(0.05 / 1.05, rel=1e-6)
    assert block["max_abs_diff_verbmass"] == pytest.approx(0.05, abs=1e-9)
    assert len(block["token_mismatches"]) == 1
    worst = block["worst"]
    assert (worst["frame"], worst["structure"], worst["control"]) == tuple(
        payload["qwen25_reference_rows"][5][k] for k in ("frame", "structure", "control"))
    text = anchor_diagnosis(block, analyzer=analyzer, anchor_path=path)
    assert "FAILED" in text and "token mismatch" in text and "trailing space" in text and "EOS prepended" in text
    assert "does not reproduce" in text
    json.dumps(block)


def test_anchor_rows_must_name_a_frame_in_range_and_all_must_be_scored(scored, analyzer, frames_file, tmp_path):
    spec, cache, header, rows = scored
    frames = load_frames_8a(frames_file)["frames"]
    path = _synthetic_anchor(rows, frames, tmp_path / "anchor.json")
    payload = json.loads(path.read_text())
    payload["qwen25_reference_rows"][0]["frame"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="out of range"):
        anchor_check(analyzer, spec, path)
    ours = [r for row in rows for r in rows_from_frame(row)][1:]
    with pytest.raises(ValueError, match="not scored"):
        task_8a.compare_anchor_rows(ours, [r for row in rows for r in rows_from_frame(row)])


# ------------------------------------------------------------------ record


def test_build_record_merges_caches_and_validates(scored, analyzer, frames_file, tmp_path):
    spec, cache, header, rows = scored
    assert missing_models({"tiny": cache}) == list(MODELS_8A)
    second = tmp_path / "scores_8a_qwen25_0.5b.jsonl"
    lines = cache.read_text(encoding="utf-8").splitlines()
    head = json.loads(lines[0])
    head["model_key"] = ANCHOR_KEY
    second.write_text("\n".join([json.dumps(head)] + lines[1:]) + "\n", encoding="utf-8")
    anchor_path = _synthetic_anchor(rows, load_frames_8a(frames_file)["frames"], tmp_path / "anchor.json")
    block = anchor_check(analyzer, spec, anchor_path)
    record = build_record_8a(frames_file, {"tiny": cache, ANCHOR_KEY: second}, anchor=block, allow_partial=True)
    assert list(record["models"]) == [ANCHOR_KEY, "tiny"]
    assert set(record) == {"meta", "verb_pool", "frames", "models", "anchor"}
    assert record["meta"]["n_frames"] == 4 and "date" in record["meta"] and record["meta"]["seed"] == 8
    block_tiny = record["models"]["tiny"]
    assert set(block_tiny["meta"]) == {"hf_id", "revision", "weight_dtype", "logits_dtype", "bos_added", "n_params_b", "vocab_size"}
    assert block_tiny["meta"]["weight_dtype"] == "float32" and block_tiny["meta"]["bos_added"] is False
    assert len(block_tiny["rows"]) == 28 and len(block_tiny["tokens"]) == 56 and len(block_tiny["top10"]) == 56
    assert set(block_tiny["rows"][0]) == {"frame", "structure", "control", "ret", "ctrl", "verbmass_ref", "verbmass_emb",
                                          "verbmass_ctrl", "logp_V_ref", "logp_V_emb", "logp_V_ctrl"}
    assert block_tiny["top10"]["0:REF"] == rows[0]["top10"]["REF"]
    assert record["anchor"]["pass"] is True and record["anchor"]["n_rows"] == 28
    text = json.dumps(record)
    assert "NaN" not in text and "Infinity" not in text
    # Without the anchor the block is null; a cache scored on other frames is refused unless partial.
    assert build_record_8a(frames_file, {"tiny": cache})["anchor"] is None
    other = save_frames_8a(generate_frames_8a({"tiny": analyzer.models.tokenizer}, n=2, seed=9), tmp_path / "other.json")
    with pytest.raises(ValueError, match="different frames file"):
        build_record_8a(other, {"tiny": cache})
    assert build_record_8a(other, {"tiny": cache}, allow_partial=True)["models"] == {}


def test_validate_record_catches_each_defect(scored, frames_file):
    spec, cache, header, rows = scored
    good = build_record_8a(frames_file, {"tiny": cache})
    validate_record_8a(good)

    def broken(mutate):
        rec = json.loads(json.dumps(good))
        mutate(rec)
        with pytest.raises(ValueError):
            validate_record_8a(rec)

    broken(lambda r: r["models"]["tiny"]["rows"].pop())
    broken(lambda r: r["models"]["tiny"]["tokens"].pop())
    broken(lambda r: r["models"]["tiny"]["top10"]["0:REF"].pop())
    broken(lambda r: r["models"]["tiny"]["rows"][0].__setitem__("ret", -1.0))
    broken(lambda r: r["models"]["tiny"]["rows"][0].__setitem__("verbmass_emb", 1.5))
    broken(lambda r: r["models"]["tiny"]["rows"][0].__setitem__("logp_V_ref", 0.5))
    broken(lambda r: r["models"]["tiny"]["tokens"][3].__setitem__("n_tok", 99))
    broken(lambda r: r["models"]["tiny"]["verb_pool_kept"].append("flew"))
    broken(lambda r: r["models"]["tiny"]["meta"].__setitem__("logits_dtype", "bfloat16"))
    broken(lambda r: r["frames"][0]["sentences"].__setitem__("REF", "The soldier  | won."))
    broken(lambda r: r["frames"][1].update(N1=r["frames"][0]["N1"], V=r["frames"][0]["V"]))
    broken(lambda r: r["meta"].__setitem__("n_frames", 3))
    broken(lambda r: r.pop("anchor"))


def test_model_block_uses_header_meta(scored):
    spec, cache, header, rows = scored
    block = model_block(header, rows)
    assert block["meta"]["hf_id"] == header["hf_id"] and block["verb_pool_kept"] == header["verb_pool_kept"]


def test_registry_has_the_seven_models_and_overrides_copy():
    assert list(MODELS_8A) == ["qwen25_0.5b", "qwen3_0.6b", "qwen3_1.7b", "qwen3_8b", "llama31_8b", "gptoss_20b", "gemma4_31b"]
    assert MODELS_8A[ANCHOR_KEY].hf_id == "Qwen/Qwen2.5-0.5B"
    assert not MODELS_8A["gptoss_20b"].base and MODELS_8A["gemma4_31b"].gated
    spec = MODELS_8A["gemma4_31b"].with_overrides(hf_id="google/other", revision="abc")
    assert spec.hf_id == "google/other" and spec.revision == "abc" and spec.key == "gemma4_31b"
    assert MODELS_8A["gemma4_31b"].hf_id == "google/gemma-4-31b"


# --------------------------------------------------------------- device_map


def test_device_map_is_passed_through_and_the_model_is_not_moved(tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM

    from m1_analyzer.services.model_service import ModelService

    path = build_tiny_local_model(tmp_path / "tiny_map")
    seen = {}
    original = AutoModelForCausalLM.from_pretrained.__func__

    def fake(cls, model_id, **kwargs):
        placed = {k: kwargs.pop(k) for k in ("device_map", "low_cpu_mem_usage") if k in kwargs}
        seen.update(placed)
        model = original(cls, model_id, **kwargs)

        def refuse(*a, **k):
            raise AssertionError(".to() must not be called when a device_map placed the weights")

        if placed:
            model.to = refuse
        return model

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", classmethod(fake))
    service = ModelService(ModelConfig(model_id=path, head="causal_lm", device_map="cpu")).load()
    assert seen == {"device_map": "cpu", "low_cpu_mem_usage": True}
    assert service.device == "cpu" and service.metadata()["device_map"] == "cpu"
    # Default: no placement kwargs, `.to` used as before.
    seen.clear()
    ModelService(ModelConfig(model_id=path, head="causal_lm")).load()
    assert seen == {}
