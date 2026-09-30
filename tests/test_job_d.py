"""Job D offline: the v4 reference is unmodified, the fast engine reproduces ``ref.t4_rows`` on a
word-level and a byte-level BPE tokenizer, auto-sizing backs off on out-of-memory, the loop
resumes, and the deliverable has the requested format."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from m1_analyzer.experiments import job_d, tree_runner
from m1_analyzer.experiments import tree_runner_ref_v4 as ref
from m1_analyzer.testing import build_tiny_local_model

N_SENT = 5


@pytest.fixture(scope="module")
def sentences():
    main = job_d.load_sentences(set_name="main")
    held = job_d.load_sentences(set_name="heldout")
    # short and long ones, both sets
    pick = sorted(main[:40], key=lambda s: len(s["words"]))
    return [pick[0], pick[len(pick) // 2], pick[-1], held[0], held[1]][:N_SENT]


@pytest.fixture(scope="module")
def word_model(sentences, tmp_path_factory):
    words = sorted({w for s in sentences for w in s["words"]})
    fill = sorted({w for f in ref.FILLERS if f != "<del>" for w in f.split()})
    extra = fill + words + [w[:1].upper() + w[1:] for w in fill + words] + [".", "?", "!"]
    path = build_tiny_local_model(tmp_path_factory.mktemp("jd_word"), extra_vocab=extra, max_positions=128)
    tok, mdl, bos, device, _ = tree_runner.load_model_f32(path, revision=None, device="cpu", bos_token=None)
    return tok, mdl, bos


@pytest.fixture(scope="module")
def bpe_model(tmp_path_factory):
    """A byte-level BPE tokenizer (GPT-2 style: multi-token words, leading-space tokens, offsets) and a
    random 2-layer GPT-2 over it; ``<|endoftext|>`` is its BOS, as for GPT-2."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    texts = [s["text"] for s in _raw_rows()[:400]]
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=600, special_tokens=["<|endoftext|>"],
                                                           initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<|endoftext|>", eos_token="<|endoftext|>")
    torch.manual_seed(0)
    mdl = GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_positions=160, n_embd=32, n_layer=2, n_head=2)).eval()
    return tok, mdl, ref.bos_id(tok)


def _raw_rows():
    with open(job_d.SENTENCES_FILE, encoding="utf-8") as fh:
        return [r for r in map(json.loads, fh) if r.get("kind") != "header"]


def _ref_rows(tok, mdl, bos, sentences):
    out = []
    for s in sentences:
        orig, rows = ref.t4_rows(tok, mdl, bos, s["words"], s["keys"], s["end"], batch=64)
        out.append({"id": s["id"], "end": s["end"], "orig": orig, "rows": rows})
    return out


# ------------------------------------------------------------------ package


def test_reference_v4_is_the_packages():
    assert job_d.reference_sha256() == job_d.REFERENCE_SHA256
    assert job_d.reference_unmodified()


def test_v1_reference_is_untouched():
    assert tree_runner.reference_unmodified()


def test_sentences_file_sets_and_end_marks():
    main = job_d.load_sentences(set_name="main")
    held = job_d.load_sentences(set_name="heldout")
    assert (len(main), len(held)) == (1000, 2161)
    assert sum(len(s["keys"]) for s in main) == 173_966          # job A's span count
    assert all(s["keys"] == ref.span_keys(len(s["words"])) for s in main[:20])


def test_sentences_file_with_other_bytes_is_refused(tmp_path):
    bad = tmp_path / "s.jsonl"
    bad.write_bytes(job_d.SENTENCES_FILE.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="sha256"):
        job_d.load_sentences(bad)


def test_wrong_end_mark_is_refused(tmp_path):
    rows = _raw_rows()[:2]
    rows[0]["end"] = "?" if rows[0]["end"] != "?" else "."
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="end_string"):
        job_d.load_sentences(path, verify_sha=False)


def test_anchor_file_blocks():
    anchor = job_d.load_anchor()
    assert anchor["reference_code_sha"] == job_d.REFERENCE_SHA256
    for spec in job_d.MODELS_D.values():
        if spec.anchor:
            assert spec.hf_id in anchor["models"]
            assert anchor["revisions"][spec.hf_id] == spec.revision
        if spec.bos_anchor_key:
            assert anchor["bos"][spec.bos_anchor_key]["token"] == spec.bos_token


def test_queue_covers_every_requested_item_once():
    wanted = {(m.slug, s) for m in job_d.MODELS_D.values() for s in m.sets}
    assert len(job_d.QUEUE) == len(wanted) == 12
    assert [(m, s) for m, s, t in job_d.QUEUE if t == 1] == [
        ("qwen3-0.6b", "main"), ("qwen3-0.6b", "heldout"), ("qwen3-1.7b", "main"), ("llama3.1-8b", "main"),
        ("qwen3-8b", "main")]
    assert {(m, s) for m, s, _ in job_d.QUEUE} == wanted
    assert job_d.output_name("qwen3-8b", "main") == "tree_t4_qwen3-8b_main.jsonl.gz"


def test_resolve_revision_expands_a_unique_prefix():
    commits = [SimpleNamespace(commit_id="ea980cb0a6c2" + "0" * 28), SimpleNamespace(commit_id="f" * 40)]
    api = SimpleNamespace(list_repo_commits=lambda repo, token=None: commits)
    assert job_d.resolve_revision("x/y", "ea980cb0a6c2", api=api) == commits[0].commit_id
    with pytest.raises(ValueError, match="matches 0"):
        job_d.resolve_revision("x/y", "abcdef", api=api)
    assert job_d.resolve_revision("x/y", "1" * 40, api=None) == "1" * 40


# ------------------------------------------------------------------- engine


@pytest.mark.parametrize("budget", [200, 5000])
def test_engine_reproduces_reference_word_level(word_model, sentences, budget):
    tok, mdl, bos = word_model
    fast = job_d.score_sentences(tok, mdl, bos, sentences, device="cpu", budget=job_d.TokenBudget(budget))
    check = job_d.check_equivalence(fast, _ref_rows(tok, mdl, bos, sentences), dtype="float32")
    assert check["pass"], check
    assert check["abs_diff"]["total"]["max"] < 1e-4 and check["abs_diff"]["first_sub"]["max"] < 1e-4
    assert all(len(v) == 8 for r in fast for t in r["rows"].values() for v in t.values())


def test_engine_reproduces_reference_bpe(bpe_model, sentences):
    tok, mdl, bos = bpe_model
    assert tok.convert_ids_to_tokens(bos) == "<|endoftext|>"
    fast = job_d.score_sentences(tok, mdl, bos, sentences, device="cpu", budget=job_d.TokenBudget(1500))
    refr = _ref_rows(tok, mdl, bos, sentences)
    check = job_d.check_equivalence(fast, refr, dtype="float32")
    assert check["pass"], check
    # the byte-level tokenizer makes multi-token words: the frame counts are not trivial
    assert any(v[2] != v[3] for r in refr for t in r["rows"].values() for v in t.values())
    assert job_d.check_complete(fast, sentences)["pass"]


def test_first_sub_is_the_first_frame_token(bpe_model, sentences):
    """The note: first_sub equals suf_sub when one suffix token follows, end_sub when none does and the end
    mark is one token."""
    tok, mdl, bos = bpe_model
    rows = job_d.score_sentences(tok, mdl, bos, sentences, device="cpu", budget=job_d.TokenBudget(1500))
    one_suf = no_suf = 0
    for r in rows:
        single_end = len(r["orig"]["end_ids"]) == 1
        for table in r["rows"].values():
            for v in table.values():
                if v[3] == 1:
                    assert v[7] == v[4]
                    one_suf += 1
                elif v[3] == 0 and single_end:
                    assert v[7] == v[5]
                    no_suf += 1
    assert one_suf and no_suf


def test_tokenizer_checks_parallel_equals_serial(bpe_model, sentences):
    tok = bpe_model[0]
    serial = ref.tokenizer_checks(tok, sentences)
    assert job_d.tokenizer_checks_parallel(tok, sentences, workers=2, chunk=2) == serial
    gate = job_d.tokenizer_gate(serial, tok_sha=job_d.tokenizer_sha(tok))
    assert gate["pass"] and gate["is_fast"] and gate["n_substitutions"] == 17 * sum(len(s["keys"]) for s in sentences)
    assert not job_d.tokenizer_gate({**serial, "roundtrip_fail": 1})["pass"]
    assert job_d.tokenizer_gate({**serial, "is_fast": False})["failed"] == ["is_fast"]
    assert job_d.tokenizer_gate({**serial, "end_merge": 5})["pass"]         # reported, not a gate


def test_oom_halves_the_batch_and_shrinks_the_budget(word_model, sentences):
    import torch

    tok, mdl, bos = word_model
    oom = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
    calls = []

    def flaky(m, b, seqs, device):
        calls.append(len(seqs))
        if len(seqs) > 16:
            raise oom("simulated")
        return job_d._forward(m, b, seqs, device)

    budget = job_d.TokenBudget(4000)
    fast = job_d.score_sentences(tok, mdl, bos, sentences[:2], device="cpu", budget=budget, forward=flaky)
    assert budget.backoffs > 0 and budget.tokens < 4000
    assert max(calls) > 16 and job_d.check_equivalence(
        fast, _ref_rows(tok, mdl, bos, sentences[:2]), dtype="float32")["pass"]


def test_batches_respect_the_budget():
    lengths = [5, 30, 12, 30, 7, 1, 22]
    budget = job_d.TokenBudget(70, max_rows=3)
    seen = []
    for batch in job_d._batches(lengths, budget):
        assert len(batch) <= 3 and len(batch) * (max(lengths[k] for k in batch) + 1) <= 70 or len(batch) == 1
        seen += batch
    assert sorted(seen) == list(range(len(lengths)))


def test_calibrate_on_cpu_is_fixed(word_model):
    budget, report = job_d.calibrate_budget(word_model[1], word_model[2], "cpu", max_len=40, cpu_tokens=1234)
    assert budget.tokens == 1234 and report["method"].startswith("fixed")


# --------------------------------------------------------- loop and output


def test_run_resumes_and_finalizes(word_model, sentences, tmp_path):
    tok, mdl, bos = word_model
    meta = {"model_id": "tiny", "revision": None, "dtype": "float32", "bos_id": bos, "bos_token": "[EOS]"}
    cache = tmp_path / "work.jsonl"
    kw = dict(set_name="main", model_meta=meta, device="cpu", budget=job_d.TokenBudget(2000), cache_path=cache,
              show_progress=False, group_seqs=300)
    first = job_d.run_item(tok, mdl, bos, sentences, limit=2, **kw)
    assert len(first) == 2
    rows = job_d.run_item(tok, mdl, bos, sentences, **kw)
    assert [r["id"] for r in rows] == [s["id"] for s in sentences]
    assert rows[:2] == first                                   # resumed, not rescored
    with pytest.raises(ValueError, match="dtype"):
        job_d.run_item(tok, mdl, bos, sentences, **{**kw, "model_meta": {**meta, "dtype": "bfloat16"}})

    checks = {"engine_equivalence": {"pass": True}, "causality": job_d.check_causality(rows, "float32"),
              "complete": job_d.check_complete(rows, sentences)}
    assert job_d.blocking_pass(checks)
    spec = job_d.MODELS_D["gpt2"]
    m = job_d.item_meta(spec, "main", meta, n_sentences=len(sentences), n_spans=sum(len(s["keys"]) for s in sentences),
                        wall_seconds=tree_runner.wall_seconds(rows), checks=checks, budget=kw["budget"])
    with pytest.raises(ValueError, match="finish the run"):
        job_d.finalize_item(rows[:-1], sentences, tmp_path / "x.jsonl.gz", m)
    out = job_d.finalize_item(list(reversed(rows)), sentences, tmp_path / job_d.output_name("gpt2", "main"), m)
    head, body = job_d.read_deliverable(out)
    for key in ("job", "date", "model_id", "revision", "dtype", "logits_dtype", "versions", "gpu", "wall_seconds",
                "reference_code_sha", "reference_code_modified", "checks", "n_sentences", "n_spans", "fillers", "bos_id",
                "batch", "notebook", "set", "sentences_sha256", "bos_token"):
        assert key in head, key
    assert head["job"] == "d"
    assert head["reference_code_modified"] is False and head["reference_code_sha"] == job_d.REFERENCE_SHA256
    assert [r["id"] for r in body] == [s["id"] for s in sentences]
    assert set(body[0]) == {"id", "end", "orig", "rows"}
    assert len(body[0]["rows"]) == 17


def test_blocking_ignores_reported_checks():
    assert job_d.blocking_pass({"a": {"pass": True}, "extra_cache": {"pass": False},
                                "causality": {"within_tol": False, "blocking": False}})
    assert not job_d.blocking_pass({"a": {"pass": False}})


def test_cache_check_uses_only_the_cache_fillers(word_model, sentences):
    tok, mdl, bos = word_model
    rows = job_d.score_sentences(tok, mdl, bos, sentences[:2], device="cpu", budget=job_d.TokenBudget(3000))
    cache = {r["id"]: {"id": r["id"], "words": s["words"],
                       "spans": {f: {k: [v[0] + 0.001, v[1]] for k, v in r["rows"][f].items()} for f in job_d.CACHE_FILLERS}}
             for r, s in zip(rows, sentences)}
    ok = job_d.check_cache(rows, cache, sentences)
    assert ok["pass"] and ok["missing"] == 0 and ok["n_rows"] == 4 * sum(len(s["keys"]) for s in sentences[:2])
    first = next(iter(cache))
    cache[first]["spans"]["it"][sentences[0]["keys"][0]][1] += 1
    assert not job_d.check_cache(rows, cache, sentences)["pass"]


def test_span_totals_reader(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"kind": "header", "proforms": ["it"]}) + "\n"
                 + json.dumps({"id": "a", "words": ["x", "y"], "spans": {"it": {"0,1": [-1.0, 2]}}}) + "\n")
    header, rows = job_d.load_span_totals(p)
    assert header["proforms"] == ["it"] and rows["a"]["spans"]["it"]["0,1"] == [-1.0, 2]


def test_job_a_check(word_model, sentences, tmp_path):
    tok, mdl, bos = word_model
    rows = job_d.score_sentences(tok, mdl, bos, sentences[:3], device="cpu", budget=job_d.TokenBudget(3000))
    job_a = [{"id": r["id"], "end": r["end"], "orig": r["orig"],
              "rows": {f: {k: v[:7] for k, v in t.items()} for f, t in r["rows"].items()}} for r in rows]
    path = tmp_path / "a.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": {}}) + "\n")
        for r in job_a:
            fh.write(json.dumps(r) + "\n")
    ok = job_d.check_job_a(rows, path)
    assert ok["pass"] and ok["n_sentences"] == 3 and ok["max_abs_diff"]["total"] == 0.0
    assert job_d.check_job_a(rows, path, ids=[rows[0]["id"]])["n_sentences"] == 1
    bumped = json.loads(json.dumps(rows))
    f = next(iter(bumped[0]["rows"]))
    k = next(iter(bumped[0]["rows"][f]))
    bumped[0]["rows"][f][k][0] += 0.02
    assert not job_d.check_job_a(bumped, path)["pass"]
    bumped[0]["rows"][f][k][0] -= 0.02
    bumped[0]["rows"][f][k][2] += 1
    assert job_d.check_job_a(bumped, path)["int_mismatches"] == 1


def test_a_cache_from_another_reference_is_refused_and_detectable(word_model, sentences, tmp_path):
    tok, mdl, bos = word_model
    meta = {"model_id": "tiny", "revision": None, "dtype": "float32", "bos_id": bos, "bos_token": "[EOS]"}
    cache = tmp_path / "w.jsonl"
    old = {"kind": "header", "job": "t4-D", "set": "main", "model_id": "tiny", "revision": None, "dtype": "float32",
           "bos_id": bos, "reference_code_sha": "1a4fe95f" + "0" * 56, "sentences_sha256": job_d.SENTENCES_SHA256}
    cache.write_text(json.dumps(old) + "\n")
    assert job_d.cache_reference_sha(cache) != job_d.REFERENCE_SHA256
    with pytest.raises(ValueError, match="reference_code_sha"):
        job_d.run_item(tok, mdl, bos, sentences[:1], set_name="main", model_meta=meta, device="cpu",
                       budget=job_d.TokenBudget(2000), cache_path=cache, show_progress=False)
