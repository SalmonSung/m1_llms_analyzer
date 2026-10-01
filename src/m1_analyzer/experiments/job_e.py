"""Job E: job D's substitutions scored by masked language models (PLL-word-l2r), plus GPT-2-medium.

The package's `mlm_runner_ref` (kept byte-identical) defines the score: for every token, its
log-probability when it and every later token of the same word are replaced by the mask token,
reading ``<s> text end-mark </s>``. Each row is
``[total, n_tok, n_pre, n_suf, pre_sub, suf_sub, end_sub, first_sub, last_pre_sub]``.

`mlm_runner_ref` does ``import tree_runner_ref``: that name is bound here to job D's v4
(`tree_runner_ref_v4`, the same file the package ships) before the import.

**The engine.** ``mref.mlm_rows`` scores one sentence at a time, 256 masked copies per pass, in job
order. `score_sentences_mlm` builds the same substitutions with the reference's helpers
(`encode_words`, `sub_text`, `word_chars`, `_common_prefix`; batched tokenisation, the tokenizer's
own word ids). It then pools every masked copy of a group of sentences, sorts the copies by length,
packs them under job D's auto-sized token budget, and reads the logits through the reference's own
`masked_logits` (RoBERTa: the output head on the masked position only). The values are float64 as in
``mref.pll``, and the rows use ``mref.mlm_rows``'s formulas. `engine_equivalence` compares the engine
with the unmodified ``mref.mlm_rows`` before every item.

GPT-2-medium is job D's procedure unchanged (`job_d.score_sentences`, ``tree_t4_gpt2-medium_<set>``).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import job_d, tree_runner
from . import tree_runner_ref_v4 as ref

_bound = sys.modules.get("tree_runner_ref")
if _bound is not None and _bound is not ref:
    raise ImportError("a module named 'tree_runner_ref' is already imported and is not job D's v4; "
                      "mlm_runner_ref must import v4.")
sys.modules["tree_runner_ref"] = ref
from . import mlm_runner_ref as mref  # noqa: E402  (needs the alias above)

MLM_REFERENCE_SHA256 = "b3dc194145769c9ab792ba5acc87ffe49f51d3ad3ab6cc57cdad7ac612a009ba"
MLM_REFERENCE_PATH = Path(mref.__file__)

_DATA = Path(__file__).resolve().parents[3] / "data" / "job_e"
ANCHOR_FILE = _DATA / "anchor_job_e.json"

ENGINE_VERSION = "e1"
SCORE = "PLL-word-l2r"
N_FIELDS = 9
#: Float fields of a masked row, by index.
MLM_FIELDS = {"total": 0, "pre_sub": 4, "suf_sub": 5, "end_sub": 6, "first_sub": 7, "last_pre_sub": 8}
ANCHOR_TOL_NATS = 0.05
EQUIV_TOL_NATS = 0.01
TIMING_FIRST = 20

#: The note's tokenizer table: mlm_checks counts must equal these exactly.
EXPECTED_MLM_CHECKS = {
    ("roberta", "main"): {"n_sentences": 1000, "n_substitutions": 2957422, "n_forwards": 53960342,
                          "text_split_differs": 0, "frame_differs": 0},
    ("roberta", "heldout"): {"n_sentences": 2161, "n_substitutions": 6664952, "n_forwards": 122230977,
                             "text_split_differs": 0, "frame_differs": 0},
    ("modernbert", "main"): {"n_sentences": 1000, "n_substitutions": 2957422, "n_forwards": 54485905},
    ("modernbert", "heldout"): {"n_sentences": 2161, "n_substitutions": 6664952, "n_forwards": 123737010},
}


@dataclass(frozen=True)
class MLMSpec:
    slug: str
    hf_id: str
    revision: str
    family: str                 # "roberta" or "modernbert": the tokenizer table row
    sets: tuple[str, ...]
    causal_pair: bool           # compare its tokenizer with GPT-2's (mlm_checks(causal_tok=...))


MODELS_E: dict[str, MLMSpec] = {m.slug: m for m in (
    MLMSpec("roberta-base", "FacebookAI/roberta-base", "e2da8e2f811d1448a5b465c236feacd80ffbac7b", "roberta",
            ("main", "heldout"), True),
    MLMSpec("roberta-large", "FacebookAI/roberta-large", "722cf37b1afa9454edce342e7895e588b6ff1d59", "roberta",
            ("main", "heldout"), True),
    MLMSpec("modernbert-large", "answerdotai/ModernBERT-large", "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
            "modernbert", ("main",), False),
)}
#: The causal partner of RoBERTa-large: job D's code, unchanged.
GPT2_MEDIUM = job_d.ModelD("gpt2-medium", "openai-community/gpt2-medium", "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
                           "float32", ("main", "heldout"), anchor=True, bos_token="<|endoftext|>")
#: GPT-2's tokenizer (job D's revision), the causal side of mlm_checks for RoBERTa.
GPT2 = job_d.MODELS_D["gpt2"]

QUEUE_E: tuple[tuple[str, str, Any], ...] = (
    ("roberta-base", "main", 1),
    ("gpt2-medium", "main", 2), ("roberta-large", "main", 2),
    ("roberta-base", "heldout", 3), ("gpt2-medium", "heldout", 3),
    ("roberta-large", "heldout", "optional"), ("modernbert-large", "main", "optional"),
)
TIER_DUE_E = {1: "2026-10-04", 2: "2026-10-06", 3: "2026-10-08", "optional": "2026-10-09"}


def is_causal(slug: str) -> bool:
    return slug == GPT2_MEDIUM.slug


def output_name(slug: str, set_name: str) -> str:
    return job_d.output_name(slug, set_name) if is_causal(slug) else f"tree_mlm_{slug}_{set_name}.jsonl.gz"


def work_name(slug: str, set_name: str) -> str:
    return job_d.work_name(slug, set_name) if is_causal(slug) else f"work_mlm_{slug}_{set_name}.jsonl"


def checks_name(slug: str, set_name: str) -> str:
    return job_d.checks_name(slug, set_name)


def mlm_reference_sha256() -> str:
    return job_d.sha256_file(MLM_REFERENCE_PATH)


def mlm_reference_unmodified() -> bool:
    return mlm_reference_sha256() == MLM_REFERENCE_SHA256


def load_anchor(path: str | Path = ANCHOR_FILE) -> dict:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("reference_code_sha") != MLM_REFERENCE_SHA256 or payload.get("tree_runner_ref_sha") != job_d.REFERENCE_SHA256:
        raise ValueError(f"{path}: written for another reference (its shas do not match the package's).")
    return payload


def cost_units_mlm(sentences: Iterable[Mapping[str, Any]]) -> int:
    """Masked work grows with copies x length: spans x words^2."""
    return sum(len(s["keys"]) * len(s["words"]) ** 2 for s in sentences)


# --------------------------------------------------------------------- model


def load_mlm_e(spec: MLMSpec, *, device: str | None = None) -> tuple[Any, Any, str, dict]:
    """``(tok, mdl, device, meta)`` through the reference's own ``load_mlm`` (float32), TF32 off."""
    import torch

    numerics = job_d.set_numerics()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok, mdl = mref.load_mlm(spec.hf_id, spec.revision, dtype=torch.float32, device=device)
    dtype = str(next(mdl.parameters()).dtype).replace("torch.", "")
    if dtype != "float32":
        raise ValueError(f"{spec.hf_id} loaded in {dtype}; job E runs in float32.")
    meta = {"model_id": spec.hf_id, "revision": spec.revision, "dtype": "float32", "bos_id": None, **numerics,
            "head_only": hasattr(mdl, "roberta") and hasattr(mdl, "lm_head")}
    return tok, mdl, device, meta


def load_causal_tok(*, cache_dir: str | None = None) -> Any:
    """GPT-2's tokenizer at job D's revision (the causal side of mlm_checks for RoBERTa)."""
    tok, _ = job_d.load_tokenizer_d(GPT2, cache_dir=cache_dir)
    return tok


# -------------------------------------------------------------------- engine


def _word_ids(w: Sequence[int | None], text: str, n_end: int) -> list[int]:
    assert None not in w and all(w[k] <= w[k + 1] for k in range(len(w) - 1)), ("word ids", text)
    nxt = (w[-1] + 1) if w else 0
    return list(w) + [nxt] * n_end


def prepare_mlm(tok: Any, words: Sequence[str], keys: Sequence[str], end: str,
                fillers: Sequence[str] = ref.FILLERS) -> dict[str, Any]:
    """The jobs of ``mref.mlm_rows`` for one sentence: ``(filler, key, ids, n_pre, n_suf, word_ids)``."""
    text = ref.text_of(words)
    end_ids = list(tok(end, add_special_tokens=False).input_ids)
    ne = len(end_ids)
    ids0, off0, w0 = mref.encode_words(tok, text, ne)
    wc = ref.word_chars(words, text)
    spans = [tuple(map(int, key.split(","))) for key in keys]
    cap_pre = {i: sum(1 for (a, b) in off0 if b <= wc[i][0]) for i in {i for i, _ in spans}}
    cap_suf = {j: sum(1 for (a, b) in off0 if a >= wc[j][1]) for j in {j for _, j in spans}}
    texts = [ref.sub_text(words, i, j, f) for f in fillers for (i, j) in spans]
    enc = tok(texts, add_special_tokens=False) if texts else None
    rev0 = ids0[::-1]
    jobs, k = [], 0
    for f in fillers:
        for key, (i, j) in zip(keys, spans):
            ids = list(enc.input_ids[k])
            wi = _word_ids(enc.word_ids(k), texts[k], ne)
            k += 1
            n_pre = min(ref._common_prefix(ids0, ids), cap_pre[i])
            n_suf = min(ref._common_prefix(rev0, ids[::-1]), cap_suf[j])
            if n_pre + n_suf > len(ids):
                n_suf = len(ids) - n_pre
            jobs.append((f, key, ids, n_pre, n_suf, wi))
    return {"text": text, "ids0": list(ids0), "w0": list(w0), "end_ids": end_ids, "jobs": jobs}


def pll_many(tok: Any, mdl: Any, seqs: Sequence[Sequence[int]], wids: Sequence[Sequence[int]], *, device: str,
             budget: job_d.TokenBudget) -> list[np.ndarray]:
    """``mref.pll`` for many sequences: every masked copy of every sequence, length-sorted and budget-packed."""
    import torch

    lens = np.array([len(x) for x in seqs], dtype=np.int64)
    width = int(lens.max()) + 2
    base = np.full((len(seqs), width), tok.pad_token_id, dtype=np.int64)
    base[:, 0] = tok.cls_token_id
    cs, ct, cu = [], [], []
    for s, (ids, wid) in enumerate(zip(seqs, wids)):
        assert len(ids) == len(wid)
        n = len(ids)
        base[s, 1:n + 1] = ids
        base[s, n + 1] = tok.sep_token_id
        last = {}
        for t in range(n):
            last[wid[t]] = t                     # word ids are monotone: the last index of each word
        for t in range(n):
            cs.append(s)
            ct.append(t)
            cu.append(last[wid[t]])
    cs, ct, cu = np.array(cs, dtype=np.int64), np.array(ct, dtype=np.int64), np.array(cu, dtype=np.int64)
    clen = lens[cs]
    out = [np.zeros(len(x), dtype=np.float64) for x in seqs]
    mask_id = tok.mask_token_id

    def forward(_mdl, _bos, idx, dev):
        idx = np.asarray(idx, dtype=np.int64)
        L = int(clen[idx].max()) + 2
        rows = base[cs[idx], :L].copy()
        r = np.arange(len(idx))
        tgt = rows[r, ct[idx] + 1].copy()
        cols = np.arange(L)[None, :]
        rows[(cols >= ct[idx, None] + 1) & (cols <= cu[idx, None] + 1)] = mask_id
        att = (cols < clen[idx, None] + 2).astype(np.int64)
        rr = torch.arange(len(idx), device=dev)
        pos = torch.from_numpy(ct[idx] + 1).to(dev)
        lg = mref.masked_logits(mdl, torch.from_numpy(rows).to(dev), torch.from_numpy(att).to(dev), rr, pos)
        lp = torch.log_softmax(lg.float(), -1)[rr, torch.from_numpy(tgt).to(dev)].cpu().numpy()
        return list(lp)

    for batch in job_d._batches(list(clen + 1), budget):
        for k, v in zip(batch, job_d.forward_safe(mdl, None, batch, device, budget, forward)):
            out[cs[k]][ct[k]] = float(v)
    return out


def score_sentences_mlm(tok: Any, mdl: Any, sentences: Sequence[Mapping[str, Any]], *, device: str,
                        budget: job_d.TokenBudget, fillers: Sequence[str] = ref.FILLERS) -> list[dict]:
    """``{"id", "end", "orig", "rows"}`` per sentence, the values of ``mref.mlm_rows``."""
    preps, seqs, wids = [], [], []
    for s in sentences:
        p = prepare_mlm(tok, s["words"], s["keys"], s["end"], fillers)
        p["orig_at"] = len(seqs)
        seqs.append(p["ids0"] + p["end_ids"])
        wids.append(p["w0"])
        p["first"] = len(seqs)
        for (_, _, ids, _, _, wi) in p["jobs"]:
            seqs.append(ids + p["end_ids"])
            wids.append(wi)
        preps.append(p)
    lps = pll_many(tok, mdl, seqs, wids, device=device, budget=budget)
    out = []
    for s, p in zip(sentences, preps):
        ids0 = p["ids0"]
        lp0e = lps[p["orig_at"]]
        orig = dict(text=p["text"], ids=ids0, lp=[float(v) for v in lp0e[:len(ids0)]], end_ids=p["end_ids"],
                    end_lp=[float(v) for v in lp0e[len(ids0):]])
        rows: dict[str, dict] = {f: {} for f in fillers}
        for k, (f, key, ids, n_pre, n_suf, _) in enumerate(p["jobs"]):
            lpe = lps[p["first"] + k]
            n = len(ids)
            lp = lpe[:n]
            rows[f][key] = [float(lp.sum()), n, n_pre, n_suf, float(lp[:n_pre].sum()),
                            float(lp[n - n_suf:].sum()) if n_suf else 0.0, float(lpe[n:].sum()), float(lpe[n - n_suf]),
                            float(lp[n_pre - 1]) if n_pre else 0.0]
        out.append({"id": s["id"], "end": s["end"], "orig": orig, "rows": rows, "n_seq": 1 + len(p["jobs"])})
    return out


def calibrate_budget_mlm(tok: Any, mdl: Any, device: str, *, max_len: int) -> tuple[job_d.TokenBudget, dict]:
    """job D's calibration with a masked probe (random ids, one masked position per row)."""
    import torch

    cfg = mdl.config
    V = int(getattr(cfg, "vocab_size", 50265))
    hidden = int(getattr(cfg, "hidden_size", 768))
    inter = int(getattr(cfg, "intermediate_size", 4 * hidden))
    head_only = hasattr(mdl, "roberta") and hasattr(mdl, "lm_head")
    per_token = 4 * (16 * hidden + 4 * inter) + (0 if head_only else 8 * V)
    rng = np.random.default_rng(0)

    def probe(n_rows: int, length: int) -> None:
        x = rng.integers(5, min(V, 30000), size=(n_rows, length + 2))
        x[:, 0], x[:, -1] = tok.cls_token_id, tok.sep_token_id
        x[:, length // 2] = tok.mask_token_id
        inp = torch.from_numpy(x).to(device)
        rr = torch.arange(n_rows, device=device)
        pos = torch.full((n_rows,), length // 2, device=device)
        lg = mref.masked_logits(mdl, inp, torch.ones_like(inp), rr, pos)
        torch.log_softmax(lg.float(), -1)[rr, inp[:, 1]].cpu()

    budget, report = job_d.calibrate_budget(mdl, None, device, max_len=max_len, probe=probe, bytes_per_token=per_token)
    budget.max_rows = max(budget.max_rows, budget.tokens // 8)
    report["max_rows"] = budget.max_rows
    report["head_only"] = head_only
    return budget, report


def max_seq_len_mlm(tok: Any, sentences: Iterable[Mapping[str, Any]], fillers: Sequence[str] = ref.FILLERS) -> int:
    return job_d.max_seq_len(tok, sentences, fillers)


def run_item_mlm(tok: Any, mdl: Any, sentences: Sequence[Mapping[str, Any]], *, set_name: str,
                 model_meta: Mapping[str, Any], device: str, budget: job_d.TokenBudget, **kw: Any) -> list[dict]:
    """job D's resumable loop with the masked engine (engine "e1", the mlm reference sha in the header)."""
    return job_d.run_item(
        tok, mdl, None, sentences, set_name=set_name, model_meta=model_meta, device=device, budget=budget,
        score_fn=lambda group: score_sentences_mlm(tok, mdl, group, device=device, budget=budget),
        engine=ENGINE_VERSION, header_extra={"job": "e", "mlm_reference_sha": mlm_reference_sha256(), "score": SCORE},
        cost_fn=cost_units_mlm, **kw)


# ------------------------------------------------------------------ checks


def mlm_checks_parallel(tok: Any, sentences: Sequence[Mapping[str, Any]], *, causal_tok: Any = None,
                        workers: int | None = None, chunk: int = 25) -> dict[str, Any]:
    """``mref.mlm_checks`` split over CPU processes; the counts are sums, so they equal one serial call."""
    return job_d.parallel_counts(lambda part: mref.mlm_checks(tok, part, causal_tok=causal_tok), sentences,
                                 workers=workers, chunk=chunk)


def mlm_tokenizer_gate(counts: Mapping[str, Any], family: str, set_name: str, *,
                       tok_sha: str | None = None, model_id: str | None = None) -> dict[str, Any]:
    """Every count of the note's table must match exactly."""
    expected = EXPECTED_MLM_CHECKS[(family, set_name)]
    failed = [k for k, v in expected.items() if counts.get(k) != v]
    return {**dict(counts), "expected": dict(expected), "failed": failed, "tokenizer_sha256": tok_sha,
            "tokenizer_of": model_id, "family": family, "pass": not failed}


def check_anchor_mlm(rows: Sequence[Mapping[str, Any]], anchor: Mapping[str, Any], hf_id: str) -> dict[str, Any]:
    """The note's anchor: integer fields identical, every float field within 0.05 nats."""
    return tree_runner.check_t4_anchor(rows, anchor["models"][hf_id]["sentences"], tol=ANCHOR_TOL_NATS,
                                       fields=tuple(MLM_FIELDS.items()))


def check_equivalence_mlm(fast_rows: Sequence[Mapping[str, Any]], ref_rows: Sequence[Mapping[str, Any]]) -> dict:
    """The engine against the unmodified ``mref.mlm_rows`` (float32: every float within 0.01 nats)."""
    return job_d.check_equivalence(fast_rows, ref_rows, dtype="float32", fields=MLM_FIELDS)


def check_complete_mlm(rows: Sequence[Mapping[str, Any]], sentences: Sequence[Mapping[str, Any]]) -> dict:
    return job_d.check_complete(rows, sentences, n_fields=N_FIELDS)


def timing_block(rows: Sequence[Mapping[str, Any]], first: Sequence[Mapping[str, Any]],
                 all_sentences: Sequence[Mapping[str, Any]], *, masked: bool) -> dict[str, Any]:
    """The note's timing: seconds for the first sentences and the projected hours for the set."""
    secs = tree_runner.wall_seconds([r for r in rows if r["id"] in {s["id"] for s in first}])
    cost = cost_units_mlm if masked else job_d.cost_units
    hours = secs / max(cost(first), 1) * cost(all_sentences) / 3600
    return {"n_sentences": len(first), "seconds": round(secs, 1), "projected_hours": round(hours, 2),
            "projection": "seconds x (cost of the set / cost of these sentences); cost = spans x words"
                          + ("^2" if masked else "")}


# -------------------------------------------------------------------- meta


def item_meta_e(spec: MLMSpec, set_name: str, model_meta: Mapping[str, Any], *, n_sentences: int, n_spans: int,
                wall_seconds: float, checks: Mapping[str, Any], budget: job_d.TokenBudget | None,
                extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The note's meta block for a masked model."""
    env = tree_runner.environment_meta()
    return {
        "job": "e", "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "model_id": spec.hf_id,
        "revision": model_meta.get("revision"), "dtype": "float32", "versions": env["versions"], "gpu": env["gpu"],
        "wall_seconds": round(float(wall_seconds), 1), "reference_code_sha": mlm_reference_sha256(),
        "tree_runner_ref_sha": job_d.reference_sha256(),
        "reference_code_modified": not (mlm_reference_unmodified() and job_d.reference_unmodified()),
        "checks": dict(checks), "n_sentences": n_sentences, "n_spans": n_spans, "fillers": list(ref.FILLERS),
        "set": set_name, "sentences_sha256": job_d.SENTENCES_SHA256, "score": SCORE,
        "batch": budget.tokens if budget else None, "notebook": "experiment_job_e.ipynb",
        "engine_version": ENGINE_VERSION,
        "engine": ("job_e.score_sentences_mlm: mref.mlm_rows's substitutions (the reference's helpers), every masked "
                   "copy of a group of sentences length-sorted and packed under a token budget ('batch' = padded "
                   "tokens per pass, sized automatically); logits via mref.masked_logits; checked against the "
                   "unmodified mref.mlm_rows (checks.engine_equivalence)"),
        "token_budget": budget.as_meta() if budget else None, "tf32": model_meta.get("tf32"),
        **(dict(extra) if extra else {}),
    }


__all__ = [
    "MLM_REFERENCE_SHA256", "ANCHOR_FILE", "ENGINE_VERSION", "SCORE", "N_FIELDS", "MLM_FIELDS", "EXPECTED_MLM_CHECKS",
    "MLMSpec", "MODELS_E", "GPT2_MEDIUM", "GPT2", "QUEUE_E", "TIER_DUE_E", "is_causal", "output_name", "work_name",
    "checks_name", "mlm_reference_sha256", "mlm_reference_unmodified", "load_anchor", "cost_units_mlm",
    "load_mlm_e", "load_causal_tok", "prepare_mlm", "pll_many", "score_sentences_mlm", "calibrate_budget_mlm",
    "max_seq_len_mlm", "run_item_mlm", "mlm_checks_parallel", "mlm_tokenizer_gate", "check_anchor_mlm",
    "check_equivalence_mlm", "check_complete_mlm", "timing_block", "item_meta_e",
]
