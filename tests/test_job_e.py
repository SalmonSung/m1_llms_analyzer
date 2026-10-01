"""Job E offline: the masked reference is unmodified and imports job D's v4, the masked engine reproduces
``mref.mlm_rows`` on a RoBERTa-style (head on the masked position) and a BERT-style (full logits) tiny model,
the tokenizer checks parallelise exactly, and the deliverable has the requested fields."""

from __future__ import annotations

import json
import sys

import pytest

from m1_analyzer.experiments import job_d, job_e, tree_runner
from m1_analyzer.experiments import tree_runner_ref_v4 as ref

mref = job_e.mref


@pytest.fixture(scope="module")
def sentences():
    main = job_d.load_sentences(set_name="main")
    pick = sorted(main[:30], key=lambda s: len(s["words"]))
    return [pick[0], pick[1], pick[len(pick) // 2]]


def _tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    with open(job_d.SENTENCES_FILE, encoding="utf-8") as fh:
        texts = [r["text"] for r in map(json.loads, fh) if r.get("kind") != "header"][:400]
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=600, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    backend.post_processor = processors.ByteLevel(trim_offsets=True)
    return PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<s>", eos_token="</s>", cls_token="<s>",
                                   sep_token="</s>", pad_token="<pad>", unk_token="<unk>", mask_token="<mask>")


@pytest.fixture(scope="module")
def roberta():
    import torch
    from transformers import RobertaConfig, RobertaForMaskedLM

    tok = _tokenizer()
    torch.manual_seed(0)
    cfg = RobertaConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=200, pad_token_id=tok.pad_token_id,
                        bos_token_id=tok.cls_token_id, eos_token_id=tok.sep_token_id)
    return tok, RobertaForMaskedLM(cfg).eval()


@pytest.fixture(scope="module")
def bert():
    import torch
    from transformers import BertConfig, BertForMaskedLM

    tok = _tokenizer()
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=200, pad_token_id=tok.pad_token_id)
    return tok, BertForMaskedLM(cfg).eval()


def _ref_rows(tok, mdl, sents):
    out = []
    for s in sents:
        orig, rows = mref.mlm_rows(tok, mdl, s["words"], s["keys"], s["end"], batch=64, chunk=300)
        out.append({"id": s["id"], "end": s["end"], "orig": orig, "rows": rows})
    return out


def test_masked_reference_is_the_packages_and_imports_v4():
    assert job_e.mlm_reference_sha256() == job_e.MLM_REFERENCE_SHA256 and job_e.mlm_reference_unmodified()
    assert sys.modules["tree_runner_ref"] is ref and mref.TR is ref
    assert job_d.reference_sha256() == "3b1e5aac659e22f7afdbe0755134e04bd416e27888ab2b7f205039d8e2f5394b"


def test_anchor_file_matches_the_references():
    anchor = job_e.load_anchor()
    assert set(anchor["models"]) == {m.hf_id for m in job_e.MODELS_E.values()} | {job_e.GPT2_MEDIUM.hf_id}
    for spec in job_e.MODELS_E.values():
        block = anchor["models"][spec.hf_id]
        assert block["revision"] == spec.revision and block["kind"] == "masked"
        assert all(len(v) == 9 for s in block["sentences"] for t in s["rows"].values() for v in t.values())
    assert anchor["models"][job_e.GPT2_MEDIUM.hf_id]["revision"] == job_e.GPT2_MEDIUM.revision


def test_queue_and_names():
    assert [(m, s) for m, s, t in job_e.QUEUE_E if t == 1] == [("roberta-base", "main")]
    assert job_e.output_name("roberta-base", "main") == "tree_mlm_roberta-base_main.jsonl.gz"
    assert job_e.output_name("gpt2-medium", "heldout") == "tree_t4_gpt2-medium_heldout.jsonl.gz"


@pytest.mark.parametrize("budget", [300, 6000])
def test_engine_reproduces_mlm_rows_head_only(roberta, sentences, budget):
    tok, mdl = roberta
    fast = job_e.score_sentences_mlm(tok, mdl, sentences, device="cpu", budget=job_d.TokenBudget(budget))
    check = job_e.check_equivalence_mlm(fast, _ref_rows(tok, mdl, sentences))
    assert check["pass"], check
    assert max(v["max"] for v in check["abs_diff"].values()) < 1e-4
    assert check["orig_lp_max_abs_diff"] < 1e-4
    assert job_e.check_complete_mlm(fast, sentences)["pass"]


def test_engine_reproduces_mlm_rows_full_logits(bert, sentences):
    tok, mdl = bert
    assert not hasattr(mdl, "roberta")
    fast = job_e.score_sentences_mlm(tok, mdl, sentences[:2], device="cpu", budget=job_d.TokenBudget(2000))
    check = job_e.check_equivalence_mlm(fast, _ref_rows(tok, mdl, sentences[:2]))
    assert check["pass"], check


def test_prefix_reacts_to_the_replacement(roberta, sentences):
    """Unlike a causal model, pre_sub differs from the original prefix's score."""
    tok, mdl = roberta
    (r,) = job_e.score_sentences_mlm(tok, mdl, sentences[2:], device="cpu", budget=job_d.TokenBudget(4000))
    diffs = [abs(v[4] - sum(r["orig"]["lp"][:v[2]])) for t in r["rows"].values() for v in t.values() if v[2]]
    assert diffs and max(diffs) > 1e-3


def test_mlm_checks_parallel_equals_serial_and_gate(roberta, sentences):
    tok, _ = roberta
    serial = mref.mlm_checks(tok, sentences, causal_tok=tok)
    assert job_e.mlm_checks_parallel(tok, sentences, causal_tok=tok, workers=2, chunk=1) == serial
    assert serial["text_split_differs"] == 0 and serial["frame_differs"] == 0
    good = dict(job_e.EXPECTED_MLM_CHECKS[("roberta", "main")])
    assert job_e.mlm_tokenizer_gate(good, "roberta", "main")["pass"]
    gate = job_e.mlm_tokenizer_gate({**good, "n_forwards": good["n_forwards"] + 1}, "roberta", "main")
    assert not gate["pass"] and gate["failed"] == ["n_forwards"]
    assert job_e.mlm_tokenizer_gate({"n_sentences": 1000, "n_substitutions": 2957422, "n_forwards": 54485905},
                                    "modernbert", "main")["pass"]


def test_anchor_check_uses_all_float_fields(roberta, sentences):
    tok, mdl = roberta
    rows = job_e.score_sentences_mlm(tok, mdl, sentences[:1], device="cpu", budget=job_d.TokenBudget(3000))
    anchor = {"models": {"x": {"sentences": json.loads(json.dumps(rows))}}}
    assert job_e.check_anchor_mlm(rows, anchor, "x")["pass"]
    a = anchor["models"]["x"]["sentences"][0]
    f = next(iter(a["rows"]))
    k = next(iter(a["rows"][f]))
    a["rows"][f][k][8] += 0.06                                          # last_pre_sub
    out = job_e.check_anchor_mlm(rows, anchor, "x")
    assert not out["pass"] and out["max_abs_diff"]["last_pre_sub"] > 0.05


def test_run_resumes_and_meta(roberta, sentences, tmp_path):
    tok, mdl = roberta
    meta = {"model_id": "tiny-mlm", "revision": "r", "dtype": "float32", "bos_id": None, "tf32": False}
    kw = dict(set_name="main", model_meta=meta, device="cpu", budget=job_d.TokenBudget(3000),
              cache_path=tmp_path / "w.jsonl", show_progress=False, group_seqs=200)
    first = job_e.run_item_mlm(tok, mdl, sentences, limit=1, **kw)
    rows = job_e.run_item_mlm(tok, mdl, sentences, **kw)
    assert rows[:1] == first and len(rows) == len(sentences)
    head = job_d.cache_header(tmp_path / "w.jsonl")
    assert head["engine"] == "e1" and head["job"] == "e" and head["mlm_reference_sha"] == job_e.MLM_REFERENCE_SHA256
    with pytest.raises(ValueError, match="was written for"):
        job_d.run_item(tok, mdl, None, sentences, **kw)                 # a causal run cannot resume a masked cache
    timing = job_e.timing_block(rows, sentences[:2], sentences, masked=True)
    assert timing["projected_hours"] >= 0
    spec = job_e.MODELS_E["roberta-base"]
    m = job_e.item_meta_e(spec, "main", meta, n_sentences=len(sentences), n_spans=1, wall_seconds=1.0,
                          checks={"timing": timing}, budget=kw["budget"])
    for key in ("job", "date", "model_id", "revision", "dtype", "versions", "gpu", "wall_seconds", "reference_code_sha",
                "tree_runner_ref_sha", "reference_code_modified", "checks", "n_sentences", "n_spans", "fillers", "set",
                "sentences_sha256", "score", "batch", "notebook"):
        assert key in m, key
    assert m["job"] == "e" and m["score"] == "PLL-word-l2r" and m["reference_code_modified"] is False
    out = job_d.finalize_item(rows, sentences, tmp_path / job_e.output_name("roberta-base", "main"), m)
    head, body = job_d.read_deliverable(out)
    assert head["job"] == "e" and all(len(v) == 9 for v in body[0]["rows"]["it"].values())
