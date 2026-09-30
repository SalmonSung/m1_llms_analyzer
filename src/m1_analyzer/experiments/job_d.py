"""Job D: T4 frame-only substitution scores on six models and two sentence sets, on one GPU.

Job A (`tree_runner`) scored every span of the 1,000 main sentences, replaced by each
of 17 strings, on Qwen3-0.6B-Base. Job D repeats that computation for five more models and
a held-out set of 2,161 sentences (`data/job_d/sentences_3161.jsonl`). The package's
reference code v2 (`tree_runner_ref_v2`) is kept byte-identical and defines the numbers.

**The engine.** `ref.t4_rows` scores one sentence at a time. Its batches follow file
order, so short and long substitutions are padded to the same length. That is too slow for
the ~9.6 M substituted sentences per model on one A100. `score_sentences` builds exactly the
same jobs with the reference's own helpers (`text_of`, `encode`, `sub_text`,
`word_chars`, `_common_prefix`). It then pools the jobs of several sentences, sorts them by
length and packs them into batches under a **token budget**. Each forward pass is the
reference's: ``[BOS] + ids + end_ids``, right-padded, with an attention mask, and
``log_softmax(logits.float())`` gathered at the next token. The row is assembled with the
reference's formulas. Only float rounding from the different batch shapes changes, and the
`engine_equivalence` check measures that against the unmodified ``ref.t4_rows`` before
every run.

**Auto-sizing.** `calibrate_budget` finds the largest token budget that fits in the
memory left after the weights, and the fastest one. `forward_safe` halves a batch that
still runs out of memory and shrinks the budget for later batches.

**The run.** `run_item` keeps a resumable working cache per model and set (one fsynced
line per sentence, header checked on resume) and mirrors it to Drive on a timer.
`finalize_item` writes ``tree_t4_<model>_<set>.jsonl.gz`` in job A's row format, with
job A's meta fields plus ``set``, ``sentences_sha256`` and ``bos_token``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..utils.logging import get_logger
from . import tree_runner
from . import tree_runner_ref_v2 as ref
from .jsonl_cache import append_row, check_header, mirror, read_jsonl, rewrite, write_header

log = get_logger("job_d")

#: sha256 of the package files (runner note, "Inputs").
REFERENCE_SHA256 = "1a4fe95fdbfc4054a91de6c2b15bd346ae83b5df52f481017ace904907fe2d4b"
SENTENCES_SHA256 = "f2a25c1c18a967b1767eae1473d298d52d6bec3865a4e7e30007ef0370716e55"
REFERENCE_PATH = Path(ref.__file__)

_DATA = Path(__file__).resolve().parents[3] / "data" / "job_d"
SENTENCES_FILE = _DATA / "sentences_3161.jsonl"
ANCHOR_FILE = _DATA / "anchor_job_d.json"

SETS = ("main", "heldout")
CACHE_FILLERS = ("it", "there", "did", "then")

#: Thresholds.
ANCHOR_TOL_NATS = 0.05             # the note's anchor tolerance
CACHE_MIN_SPEARMAN = 0.999         # the note's cache check
CACHE_CHECK_FIRST = 50
CAUSALITY_TOL = 1e-3               # the note's reporting threshold
CAUSALITY_BF16_PER_TOKEN = 0.05    # the note's bf16 flag, per prefix token
EQUIV_TOL_F32 = 1e-2               # fast engine vs ref.t4_rows, float32: max |diff| in nats
EQUIV_TOL_BF16_MEDIAN = 0.05       # bfloat16: median |diff| ...
EQUIV_MIN_SPEARMAN_BF16 = 0.9999   # ... and rank agreement of `total`
FLOAT_FIELDS = {"total": 0, "suf_sub": 4, "end_sub": 5, "pre_check": 6}

#: log_softmax runs on row chunks whose float32 copy stays under this many bytes.
LOGSOFTMAX_CHUNK_BYTES = 1 << 30


# -------------------------------------------------------------------- models


@dataclass(frozen=True)
class ModelD:
    """One row of the note's model table."""

    slug: str                     # the file-name key: tree_t4_<slug>_<set>.jsonl.gz
    hf_id: str
    revision: str                 # the note's revision (a prefix unless it is a full sha)
    dtype: str                    # "float32" (<= 2B parameters) or "bfloat16"
    sets: tuple[str, ...]
    anchor: bool = False          # anchor_job_d.json has sentences for it
    cache_file: str | None = None      # the 1b span-cost cache for the blocking cache check
    extra_cache_file: str | None = None  # a 1b cache for a reported, non-blocking check
    bos_anchor_key: str | None = None    # its entry in the anchor's "bos" block
    bos_token: str | None = None         # the start token the note names


MODELS_D: dict[str, ModelD] = {m.slug: m for m in (
    ModelD("qwen3-0.6b", "Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd", "float32",
           ("heldout",), anchor=True, bos_anchor_key="qwen3_06b_base", bos_token="<|endoftext|>"),
    ModelD("qwen3-1.7b", "Qwen/Qwen3-1.7B-Base", "ea980cb0a6c2", "float32", ("main", "heldout"),
           cache_file="span_costs_Qwen-Qwen3-1.7B-Base_min-over-it-there-did-then.jsonl",
           bos_anchor_key="tok_qwen3_17b", bos_token="<|endoftext|>"),
    ModelD("qwen3-8b", "Qwen/Qwen3-8B-Base", "49e3418fbbbc", "bfloat16", ("main", "heldout"),
           extra_cache_file="span_costs_Qwen-Qwen3-8B-Base_min-over-it-there-did-then.jsonl",
           bos_anchor_key="tok_qwen3_8b", bos_token="<|endoftext|>"),
    ModelD("llama3.1-8b", "meta-llama/Llama-3.1-8B", "d04e592bb4f6", "bfloat16", ("main", "heldout"),
           cache_file="span_costs_meta-llama-Llama-3.1-8B_min-over-it-there-did-then.jsonl",
           bos_token="<|begin_of_text|>"),
    ModelD("olmo3-7b", "allenai/Olmo-3-1025-7B", "a81bae42db39", "bfloat16", ("main", "heldout"),
           bos_anchor_key="tok_olmo3_7b", bos_token="<|endoftext|>"),
    ModelD("gpt2", "openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e", "float32",
           ("main", "heldout"), anchor=True, bos_anchor_key="gpt2_local", bos_token="<|endoftext|>"),
)}

#: The note's delivery order (tier 1, 2, 3), smallest model first within a tier.
QUEUE: tuple[tuple[str, str, int], ...] = (
    ("qwen3-1.7b", "main", 1), ("qwen3-0.6b", "heldout", 1), ("llama3.1-8b", "main", 1), ("qwen3-8b", "main", 1),
    ("gpt2", "main", 2), ("gpt2", "heldout", 2), ("qwen3-1.7b", "heldout", 2), ("olmo3-7b", "main", 2),
    ("qwen3-8b", "heldout", 3), ("llama3.1-8b", "heldout", 3), ("olmo3-7b", "heldout", 3),
)
TIER_DUE = {1: "2026-10-04", 2: "2026-10-06", 3: "2026-10-08"}


def output_name(slug: str, set_name: str) -> str:
    return f"tree_t4_{slug}_{set_name}.jsonl.gz"


def checks_name(slug: str, set_name: str) -> str:
    return f"checks_{slug}_{set_name}.json"


def work_name(slug: str, set_name: str) -> str:
    return f"work_t4_{slug}_{set_name}.jsonl"


# ----------------------------------------------------------------- reference


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def reference_sha256() -> str:
    return sha256_file(REFERENCE_PATH)


def reference_unmodified() -> bool:
    return reference_sha256() == REFERENCE_SHA256


# -------------------------------------------------------------------- inputs


def load_sentences(path: str | os.PathLike = SENTENCES_FILE, set_name: str | None = None,
                   *, verify_sha: bool = True) -> list[dict]:
    """The sentence file's rows (optionally one set), each with ``keys = ref.span_keys(n)``.

    Refuses a file whose sha256 is not the package's, and a row whose ``end`` is not
    ``ref.end_string(tree)``.
    """
    path = Path(path)
    if verify_sha and sha256_file(path) != SENTENCES_SHA256:
        raise ValueError(f"{path}: sha256 is not the package's {SENTENCES_SHA256[:12]}...")
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "header":
                continue
            if set_name is not None and row["set"] != set_name:
                continue
            end = ref.end_string(row["tree"])
            if row["end"] != end:
                raise ValueError(f"{row['id']}: end {row['end']!r} != end_string(tree) {end!r}")
            out.append({"id": row["id"], "set": row["set"], "words": list(row["words"]), "tree": row["tree"],
                        "end": row["end"], "keys": ref.span_keys(len(row["words"]))})
    return out


def load_anchor(path: str | os.PathLike = ANCHOR_FILE) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("models", "bos", "revisions"):
        if key not in payload:
            raise ValueError(f"{path}: expected a {key!r} block (keys: {sorted(payload)}).")
    return payload


def load_span_totals(path: str | os.PathLike) -> tuple[dict, dict[str, dict]]:
    """``(header, {id: {"words", "spans"}})`` from a 1b span-cost cache (any number of fillers)."""
    header, rows, _ = read_jsonl(Path(path))
    if header is None:
        raise ValueError(f"{path} has no header line; it is not a span-cost cache.")
    return header, {r["id"]: {"id": r["id"], "words": list(r["words"]), "spans": r["spans"]} for r in rows}


def cost_units(sentences: Iterable[Mapping[str, Any]]) -> int:
    """The note's span-weighted length: spans x words, summed. Used for ETAs."""
    return sum(len(s["keys"]) * len(s["words"]) for s in sentences)


# --------------------------------------------------------------------- model


def resolve_revision(hf_id: str, revision: str, *, token: str | None = None, api: Any = None) -> str:
    """The full commit sha for the note's (possibly abbreviated) revision; exactly one match."""
    if len(revision) == 40:
        return revision
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    matches = sorted({c.commit_id for c in api.list_repo_commits(hf_id, token=token)
                      if c.commit_id.startswith(revision)})
    if len(matches) != 1:
        raise ValueError(f"{hf_id}: revision {revision!r} matches {len(matches)} commits ({matches}).")
    return matches[0]


def set_numerics() -> dict[str, Any]:
    """TF32 off, so float32 models compute in true float32 as in job A; returned for the meta block."""
    try:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        return {"tf32": False, "float32_matmul_precision": torch.get_float32_matmul_precision()}
    except Exception:  # noqa: BLE001 - reported, not fatal
        return {"tf32": None}


def load_model_d(spec: ModelD, *, revision: str | None = None, anchor: Mapping[str, Any] | None = None,
                 device: str = "auto", hf_token: str | None = None, cache_dir: str | None = None
                 ) -> tuple[Any, Any, int, str, Any, dict]:
    """``(tok, mdl, bos, device, analyzer, meta)`` for one model of the table.

    Resolves the full revision, loads through `Analyzer` in the note's dtype (LM head),
    picks BOS with ``ref.bos_id`` and checks it against the anchor's ``bos`` block and the
    token the note names.
    """
    from ..config.settings import ModelConfig, RunConfig
    from ..container import Analyzer
    from ..utils.env import resolve_hf_token

    token = resolve_hf_token(hf_token)
    full = revision or resolve_revision(spec.hf_id, spec.revision, token=token)
    if not full.startswith(spec.revision):
        raise ValueError(f"{spec.hf_id}: revision {full} is not the note's {spec.revision}.")
    if anchor is not None and spec.hf_id in anchor.get("revisions", {}) and anchor["revisions"][spec.hf_id] != full:
        raise ValueError(f"{spec.hf_id}: revision {full} is not the anchor's {anchor['revisions'][spec.hf_id]}.")
    numerics = set_numerics()
    big = spec.dtype != "float32"
    analyzer = Analyzer(RunConfig(model=ModelConfig(
        model_id=spec.hf_id, revision=full, device=device, dtype=spec.dtype, head="causal_lm",
        hf_token=hf_token, cache_dir=cache_dir, device_map="auto" if big and _cuda() else None,
    )))
    tok, mdl = analyzer.models.tokenizer, analyzer.models.model
    bos = ref.bos_id(tok)
    bos_token = tok.convert_ids_to_tokens(bos)
    if spec.bos_token is not None and bos_token != spec.bos_token:
        raise ValueError(f"{spec.hf_id}: start token {bos_token!r} (id {bos}), the note says {spec.bos_token!r}.")
    if anchor is not None and spec.bos_anchor_key:
        want = anchor["bos"][spec.bos_anchor_key]["bos_id"]
        if bos != want:
            raise ValueError(f"{spec.hf_id}: BOS id {bos} != the anchor's {want} ({spec.bos_anchor_key}).")
    md = analyzer.models.metadata()
    if str(md.get("dtype")) != spec.dtype:
        raise ValueError(f"{spec.hf_id} loaded in {md.get('dtype')}, the note says {spec.dtype}.")
    meta = {"model_id": spec.hf_id, "revision": full, "dtype": spec.dtype, "bos_id": int(bos), "bos_token": bos_token,
            **numerics}
    return tok, mdl, int(bos), str(analyzer.models.device), analyzer, meta


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


# -------------------------------------------------------------------- engine


@dataclass
class TokenBudget:
    """How many (padded) tokens one forward pass may hold; shrinks on out-of-memory."""

    tokens: int
    max_rows: int = 4096
    backoffs: int = 0
    history: list = field(default_factory=list)

    def shrink(self, factor: float = 0.85) -> None:
        self.tokens = max(64, int(self.tokens * factor))
        self.backoffs += 1
        self.history.append(self.tokens)

    def as_meta(self) -> dict[str, Any]:
        return {"tokens": self.tokens, "max_rows": self.max_rows, "oom_backoffs": self.backoffs}


def _oom_types() -> tuple:
    import torch

    types = [getattr(torch, "OutOfMemoryError", None), getattr(torch.cuda, "OutOfMemoryError", None)]
    return tuple(t for t in types if t is not None)


def _forward(mdl: Any, bos: int, seqs: Sequence[Sequence[int]], device: str) -> list[np.ndarray]:
    """The reference's `forward` for many sequences: float32 log-prob of every token after [BOS]."""
    import torch

    L = max(len(x) for x in seqs) + 1
    inp = torch.zeros(len(seqs), L, dtype=torch.long)
    att = torch.zeros_like(inp)
    for r, x in enumerate(seqs):
        inp[r, 0] = bos
        inp[r, 1:len(x) + 1] = torch.tensor(x)
        att[r, :len(x) + 1] = 1
    inp_d = inp.to(device)
    with torch.no_grad():
        logits = mdl(input_ids=inp_d, attention_mask=att.to(device)).logits
        V = logits.shape[-1]
        rows = max(1, LOGSOFTMAX_CHUNK_BYTES // max(1, L * V * 4))
        out: list[np.ndarray] = []
        for r0 in range(0, len(seqs), rows):
            ls = torch.log_softmax(logits[r0:r0 + rows, :L - 1].float(), -1)
            g = ls.gather(-1, inp_d[r0:r0 + rows, 1:].unsqueeze(-1)).squeeze(-1).cpu().numpy()
            del ls
            for r in range(r0, min(r0 + rows, len(seqs))):
                out.append(g[r - r0, :len(seqs[r])])
        del logits
    return out


def forward_safe(mdl: Any, bos: int, seqs: Sequence[Sequence[int]], device: str, budget: TokenBudget,
                 forward: Callable = None) -> list[np.ndarray]:
    """`_forward`, halving the batch (and shrinking `budget`) on CUDA out-of-memory."""
    forward = forward or _forward
    oom = False
    try:
        return forward(mdl, bos, seqs, device)
    except _oom_types():
        if len(seqs) == 1:
            raise
        oom = True
    if oom:  # outside the except block, so the failed pass's tensors are already released
        _empty_cache()
        budget.shrink()
        log.warning("Out of memory on %d sequences; halving, budget now %d tokens.", len(seqs), budget.tokens)
        h = len(seqs) // 2
        return (forward_safe(mdl, bos, seqs[:h], device, budget, forward)
                + forward_safe(mdl, bos, seqs[h:], device, budget, forward))
    raise AssertionError("unreachable")


def _empty_cache() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _batches(lengths: Sequence[int], budget: TokenBudget):
    """Indices of `lengths`, longest first, cut into batches with rows x (max length + 1) <= budget.

    Reads ``budget.tokens`` afresh for every batch, so a shrink applies at once."""
    order = sorted(range(len(lengths)), key=lambda k: -lengths[k])
    cur: list[int] = []
    width = 0
    for k in order:
        w = max(width, lengths[k] + 1)
        if cur and (w * (len(cur) + 1) > budget.tokens or len(cur) >= budget.max_rows):
            yield cur
            cur, w = [], lengths[k] + 1
        cur.append(k)
        width = w
    if cur:
        yield cur


def prepare(tok: Any, words: Sequence[str], keys: Sequence[str], end: str,
            fillers: Sequence[str] = ref.FILLERS) -> dict[str, Any]:
    """The jobs of `ref.t4_rows` for one sentence: ``(filler, key, ids, n_pre, n_suf)`` in its order.

    Same helpers and rules as the reference (text, offsets, caps, shared prefix and suffix,
    the overlap clamp); substituted texts are tokenised in one batched call.
    """
    text = ref.text_of(words)
    ids0, off0 = ref.encode(tok, text)
    end_ids = tok(end, add_special_tokens=False).input_ids
    wc = ref.word_chars(words, text)
    spans = [tuple(map(int, key.split(","))) for key in keys]
    cap_pre = {i: sum(1 for (a, b) in off0 if b <= wc[i][0]) for i in {i for i, _ in spans}}
    cap_suf = {j: sum(1 for (a, b) in off0 if a >= wc[j][1]) for j in {j for _, j in spans}}
    texts = [ref.sub_text(words, i, j, f) for f in fillers for (i, j) in spans]
    all_ids = tok(texts, add_special_tokens=False).input_ids if texts else []
    rev0 = ids0[::-1]
    jobs = []
    k = 0
    for f in fillers:
        for key, (i, j) in zip(keys, spans):
            ids = list(all_ids[k])
            k += 1
            n_pre = min(ref._common_prefix(ids0, ids), cap_pre[i])
            n_suf = min(ref._common_prefix(rev0, ids[::-1]), cap_suf[j])
            if n_pre + n_suf > len(ids):
                n_suf = len(ids) - n_pre
            jobs.append((f, key, ids, n_pre, n_suf))
    return {"text": text, "ids0": list(ids0), "end_ids": list(end_ids), "jobs": jobs}


def score_sentences(tok: Any, mdl: Any, bos: int, sentences: Sequence[Mapping[str, Any]], *, device: str,
                    budget: TokenBudget, fillers: Sequence[str] = ref.FILLERS,
                    forward: Callable | None = None) -> list[dict]:
    """``{"id", "end", "orig", "rows"}`` per sentence, the values of ``ref.t4_rows``.

    All substitutions of all `sentences` (and their originals) share length-sorted,
    budget-packed forward passes.
    """
    preps, seqs, owners = [], [], []
    for s in sentences:
        p = prepare(tok, s["words"], s["keys"], s["end"], fillers)
        p["orig_at"] = len(seqs)
        seqs.append(p["ids0"] + p["end_ids"])
        p["first"] = len(seqs)
        seqs.extend(job[2] + p["end_ids"] for job in p["jobs"])
        owners.append(len(seqs))
        preps.append(p)
    lps: list[np.ndarray | None] = [None] * len(seqs)
    for batch in _batches([len(x) for x in seqs], budget):
        for k, v in zip(batch, forward_safe(mdl, bos, [seqs[k] for k in batch], device, budget, forward)):
            lps[k] = v
    out = []
    for s, p in zip(sentences, preps):
        ids0, end_ids = p["ids0"], p["end_ids"]
        lp0e = lps[p["orig_at"]]
        lp0 = lp0e[:len(ids0)]
        orig = dict(text=p["text"], ids=ids0, lp=[float(v) for v in lp0], end_ids=end_ids,
                    end_lp=[float(v) for v in lp0e[len(ids0):]])
        rows: dict[str, dict] = {f: {} for f in fillers}
        for k, (f, key, ids, n_pre, n_suf) in enumerate(p["jobs"]):
            lpe = lps[p["first"] + k]
            lp = lpe[:len(ids)]
            rows[f][key] = [float(lp.sum()), len(ids), n_pre, n_suf, float(lp[len(ids) - n_suf:].sum()) if n_suf else 0.0,
                            float(lpe[len(ids):].sum()), float((lp[:n_pre] - lp0[:n_pre]).sum())]
        out.append({"id": s["id"], "end": s["end"], "orig": orig, "rows": rows,
                    "n_seq": 1 + len(p["jobs"])})
    return out


# ------------------------------------------------------------- auto-sizing


def calibrate_budget(mdl: Any, bos: int, device: str, *, max_len: int, typical_len: int = 24,
                     safety: float = 0.85, sweep: Sequence[float] = (1.0, 0.5, 0.25), repeats: int = 2,
                     cpu_tokens: int = 4096) -> tuple[TokenBudget, dict[str, Any]]:
    """The token budget for this model on this GPU.

    1. Start from the free memory after the weights over an estimate of the bytes one
       padded token costs (logits in the model's dtype, their float32 copy, activations).
    2. Probe at the longest possible sequence, halving on out-of-memory, until a pass fits.
    3. Time the fitting budget and fractions of it at a typical length; keep the fastest
       (preferring the larger within 3 %).
    """
    import torch

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return TokenBudget(cpu_tokens), {"device": str(device), "tokens": cpu_tokens, "method": "fixed (cpu)"}
    _empty_cache()
    free, total = torch.cuda.mem_get_info()
    cfg = getattr(mdl, "config", None)
    V = int(getattr(mdl.get_output_embeddings(), "out_features", 0) or getattr(cfg, "vocab_size", 50257))
    hidden = int(getattr(cfg, "hidden_size", 0) or getattr(cfg, "n_embd", 1024))
    inter = int(getattr(cfg, "intermediate_size", 0) or 4 * hidden)
    logit_bytes = torch.finfo(next(mdl.parameters()).dtype).bits // 8
    per_token = V * (logit_bytes + 4) + 4 * (6 * hidden + 2 * inter)
    start = int(free * safety / per_token)
    tokens = max(max_len + 1, (start // (max_len + 1)) * (max_len + 1))
    report: dict[str, Any] = {"free_gb": round(free / 2 ** 30, 2), "total_gb": round(total / 2 ** 30, 2),
                              "vocab": V, "bytes_per_token_est": per_token, "start_tokens": tokens, "probes": []}
    rng = np.random.default_rng(0)

    def run(n_tok: int, length: int) -> float:
        rows = max(1, n_tok // (length + 1))
        seqs = [list(map(int, rng.integers(100, min(V, 30000), size=length))) for _ in range(rows)]
        torch.cuda.synchronize()
        t = time.perf_counter()
        _forward(mdl, bos, seqs, device)
        torch.cuda.synchronize()
        return rows * length / (time.perf_counter() - t)

    oom = _oom_types()
    while True:
        failed = False
        try:
            torch.cuda.reset_peak_memory_stats()
            run(tokens, max_len)
            peak = torch.cuda.max_memory_allocated()
            report["probes"].append({"tokens": tokens, "ok": True, "peak_gb": round(peak / 2 ** 30, 2)})
        except oom:
            failed = True
        if not failed:
            break
        report["probes"].append({"tokens": tokens, "ok": False})
        _empty_cache()
        if tokens <= max_len + 1:
            raise RuntimeError("Even one sequence of the longest length does not fit on this GPU.")
        tokens = max(max_len + 1, tokens // 2)
    speeds = {}
    length = min(typical_len, max_len)
    for frac in sweep:
        n = max(length + 1, int(tokens * frac))
        try:
            run(n, length)  # warm-up
            speeds[n] = max(run(n, length) for _ in range(repeats))
        except oom:
            _empty_cache()
    if speeds:
        best = max(speeds.values())
        chosen = max(n for n, v in speeds.items() if v >= 0.97 * best)
    else:  # every timed pass ran out of memory: keep half the probed budget; forward_safe backs off further
        chosen = max(max_len + 1, tokens // 2)
    report.update({"sweep_tokens_per_s": {str(k): round(v) for k, v in speeds.items()}, "tokens": chosen,
                   "method": "probe + sweep"})
    _empty_cache()
    return TokenBudget(chosen), report


def max_seq_len(tok: Any, sentences: Iterable[Mapping[str, Any]], fillers: Sequence[str] = ref.FILLERS) -> int:
    """An upper bound on ``len(ids + end_ids)`` over the sentences' substitutions."""
    longest_filler = max((len(tok(" " + f, add_special_tokens=False).input_ids) for f in fillers if f != "<del>"),
                         default=0)
    worst = 0
    for s in sentences:
        n = len(tok(ref.text_of(s["words"]), add_special_tokens=False).input_ids)
        worst = max(worst, n + longest_filler + len(tok(s["end"], add_special_tokens=False).input_ids) + 2)
    return worst


# ---------------------------------------------------------------- the loop


def _header(item_meta: Mapping[str, Any], set_name: str) -> dict:
    return {"kind": "header", "job": "t4-D", "set": set_name, "model_id": item_meta.get("model_id"),
            "revision": item_meta.get("revision"), "dtype": item_meta.get("dtype"), "bos_id": item_meta.get("bos_id"),
            "reference_code_sha": reference_sha256(), "sentences_sha256": SENTENCES_SHA256,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d")}


def run_item(tok: Any, mdl: Any, bos: int, sentences: Sequence[Mapping[str, Any]], *, set_name: str,
             model_meta: Mapping[str, Any], device: str, budget: TokenBudget, cache_path: str | os.PathLike | None,
             ids: Sequence[str] | None = None, limit: int | None = None, group_seqs: int = 20000,
             mirror_path: str | os.PathLike | None = None, mirror_minutes: float = 15.0,
             show_progress: bool = True, forward: Callable | None = None) -> list[dict]:
    """Score `sentences` (resumably) into `cache_path`; returns the rows in input order.

    Sentences are scored in groups of about `group_seqs` substituted sentences; each
    sentence's line is written (fsynced) when its group finishes, so a disconnect loses at
    most one group. ``seconds`` per sentence is its share of the group's time.
    """
    header = _header(model_meta, set_name)
    done: dict[str, dict] = {}
    path = Path(cache_path) if cache_path else None
    if path is not None and path.exists():
        existing, rows, truncated = read_jsonl(path)
        if existing is not None:
            check_header(existing, header, path, unchecked={"kind", "date"})
            header = existing
        done = {r["id"]: r for r in rows}
        if truncated:
            rewrite(path, header, rows)
        log.info("Resuming %s: %d sentences already in %s.", set_name, len(done), path)
    elif path is not None:
        write_header(path, header)

    wanted = list(sentences)
    if ids is not None:
        keep = set(ids)
        wanted = [s for s in wanted if s["id"] in keep]
    if limit:
        wanted = wanted[: int(limit)]
    pending = [s for s in wanted if s["id"] not in done]
    total_units = cost_units(pending)
    groups, cur, n = [], [], 0
    for s in pending:
        cur.append(s)
        n += len(s["keys"]) * 17 + 1
        if n >= group_seqs:
            groups.append(cur)
            cur, n = [], 0
    if cur:
        groups.append(cur)

    bar = None
    if show_progress and pending:
        try:
            from tqdm.auto import tqdm

            bar = tqdm(total=len(pending), desc=f"{model_meta.get('model_id')} {set_name}", unit="sent")
        except Exception:  # noqa: BLE001
            bar = None
    fh = path.open("a", encoding="utf-8") if path is not None else None
    last_mirror, started, units_done = time.monotonic(), time.monotonic(), 0
    try:
        for group in groups:
            t0 = time.perf_counter()
            scored = score_sentences(tok, mdl, bos, group, device=device, budget=budget, forward=forward)
            dt = time.perf_counter() - t0
            n_seq = sum(r["n_seq"] for r in scored)
            for r in scored:
                r["seconds"] = round(dt * r.pop("n_seq") / n_seq, 3)
                done[r["id"]] = r
                if fh is not None:
                    append_row(fh, r)
            units_done += cost_units(group)
            if bar is not None:
                elapsed = time.monotonic() - started
                eta_h = elapsed / units_done * (total_units - units_done) / 3600 if units_done else float("nan")
                bar.set_postfix(budget=budget.tokens, eta_h=f"{eta_h:.2f}")
                bar.update(len(group))
            if mirror_path and path is not None and time.monotonic() - last_mirror >= mirror_minutes * 60:
                mirror(path, mirror_path)
                last_mirror = time.monotonic()
    finally:
        if bar is not None:
            bar.close()
        if fh is not None:
            fh.close()
        if mirror_path and path is not None and pending:
            mirror(path, mirror_path)
    return [done[s["id"]] for s in wanted if s["id"] in done]


# ------------------------------------------------------------------ checks


def _float_diffs(a: Mapping[str, Any], b: Mapping[str, Any]) -> tuple[dict[str, list[float]], list[str], list[float], list[float]]:
    diffs: dict[str, list[float]] = {k: [] for k in FLOAT_FIELDS}
    problems, ta, tb = [], [], []
    for f, table in b["rows"].items():
        for key, v in table.items():
            w = a["rows"].get(f, {}).get(key)
            if w is None:
                problems.append(f"{b['id']} {f} {key}: missing")
                continue
            if list(w[1:4]) != list(v[1:4]):
                problems.append(f"{b['id']} {f} {key}: counts {w[1:4]} != {v[1:4]}")
            for name, k in FLOAT_FIELDS.items():
                diffs[name].append(abs(float(w[k]) - float(v[k])))
            ta.append(float(w[0]))
            tb.append(float(v[0]))
    return diffs, problems, ta, tb


def check_equivalence(fast_rows: Sequence[Mapping[str, Any]], ref_rows: Sequence[Mapping[str, Any]], *,
                      dtype: str) -> dict[str, Any]:
    """The fast engine against the unmodified ``ref.t4_rows`` on the same sentences.

    Integer fields and the original ids must be identical. float32: every float within
    `EQUIV_TOL_F32` nats. bfloat16 (whose rounding depends on batch shape): median
    within `EQUIV_TOL_BF16_MEDIAN` and Spearman of ``total`` >= `EQUIV_MIN_SPEARMAN_BF16`.
    """
    mine = {r["id"]: r for r in fast_rows}
    all_diffs: dict[str, list[float]] = {k: [] for k in FLOAT_FIELDS}
    problems, ta, tb, orig = [], [], [], []
    for r in ref_rows:
        m = mine[r["id"]]
        if m["orig"]["ids"] != r["orig"]["ids"] or m["orig"]["end_ids"] != r["orig"]["end_ids"]:
            problems.append(f"{r['id']}: original ids differ")
        orig.extend(abs(x - y) for x, y in zip(m["orig"]["lp"] + m["orig"]["end_lp"], r["orig"]["lp"] + r["orig"]["end_lp"]))
        d, p, a, b = _float_diffs(m, r)
        problems += p
        ta += a
        tb += b
        for k in all_diffs:
            all_diffs[k] += d[k]
    stats = {k: {"max": float(max(v, default=0.0)), "median": float(np.median(v)) if v else 0.0}
             for k, v in all_diffs.items()}
    rho = tree_runner._spearman(ta, tb)
    if dtype == "float32":
        ok = not problems and all(s["max"] <= EQUIV_TOL_F32 for s in stats.values())
        rule = f"integers identical; every float within {EQUIV_TOL_F32} nats"
    else:
        ok = (not problems and all(s["median"] <= EQUIV_TOL_BF16_MEDIAN for s in stats.values())
              and rho >= EQUIV_MIN_SPEARMAN_BF16)
        rule = (f"integers identical; median |diff| <= {EQUIV_TOL_BF16_MEDIAN} nats; "
                f"Spearman(total) >= {EQUIV_MIN_SPEARMAN_BF16}")
    return {"n_sentences": len(ref_rows), "ids": [r["id"] for r in ref_rows], "n_rows": len(ta),
            "problems": len(problems), "problem_examples": problems[:5], "abs_diff": stats,
            "orig_lp_max_abs_diff": float(max(orig, default=0.0)), "spearman_total": rho, "rule": rule,
            "pass": bool(ok)}


def check_anchor(rows: Sequence[Mapping[str, Any]], anchor: Mapping[str, Any], hf_id: str) -> dict[str, Any]:
    """The note's anchor check (0.05 nats on total / suf_sub / end_sub; counts identical)."""
    return tree_runner.check_t4_anchor(rows, anchor["models"][hf_id]["sentences"], tol=ANCHOR_TOL_NATS)


def check_cache(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, Mapping[str, Any]],
                sentences: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The note's cache check on the four strings the 1b cache has; words must match too."""
    words = {s["id"]: s["words"] for s in sentences}
    missing = [r["id"] for r in rows if r["id"] not in cache]
    differ = [r["id"] for r in rows if r["id"] in cache and cache[r["id"]]["words"] != words[r["id"]]]
    out = tree_runner.check_t4_cache(rows, [cache[r["id"]] for r in rows if r["id"] in cache],
                                     min_spearman=CACHE_MIN_SPEARMAN, fillers=CACHE_FILLERS)
    out["sentences_missing_from_cache"] = missing
    out["sentences_with_other_words"] = differ
    out["pass"] = bool(out["pass"] and not missing and not differ)
    return out


def check_causality(rows: Sequence[Mapping[str, Any]], dtype: str) -> dict[str, Any]:
    """The note's causality report: max |pre_check|, rows over 1e-3 and, for bfloat16,
    rows over 0.05 per prefix token. Reported, not blocking (job A had 144 rows over 1e-3)."""
    out = tree_runner.check_t4_causality(rows, tol=CAUSALITY_TOL)
    per_token = [abs(float(v[6])) / v[2] for _, _, _, v in tree_runner._iter_t4(rows) if v[2]]
    flagged = sum(1 for x in per_token if x > CAUSALITY_BF16_PER_TOKEN)
    out.update({"max_abs_pre_check_per_token": float(max(per_token, default=0.0)),
                "n_flag_per_token": flagged if dtype != "float32" else None,
                "flag_per_token_tol": CAUSALITY_BF16_PER_TOKEN, "blocking": False})
    out["within_tol"] = out.pop("pass")
    return out


def check_complete(rows: Sequence[Mapping[str, Any]], sentences: Sequence[Mapping[str, Any]],
                   fillers: Sequence[str] = ref.FILLERS) -> dict[str, Any]:
    """Every sentence, every filler x span key present; every number finite; n_tok > 0."""
    mine = {r["id"]: r for r in rows}
    missing_sent = [s["id"] for s in sentences if s["id"] not in mine]
    missing_rows, bad, n = 0, [], 0
    for s in sentences:
        r = mine.get(s["id"])
        if r is None:
            continue
        vals = r["orig"]["lp"] + r["orig"]["end_lp"]
        if not all(math.isfinite(x) for x in vals):
            bad.append(f"{s['id']}: orig")
        for f in fillers:
            table = r["rows"].get(f, {})
            for key in s["keys"]:
                v = table.get(key)
                if v is None:
                    missing_rows += 1
                    continue
                n += 1
                if not all(math.isfinite(float(x)) for x in v) or v[1] <= 0:
                    bad.append(f"{s['id']} {f} {key}: {v}")
    ok = not missing_sent and not missing_rows and not bad
    return {"n_sentences": len(sentences), "n_rows": n, "missing_sentences": len(missing_sent),
            "missing_rows": missing_rows, "non_finite_or_empty": len(bad), "examples": bad[:5], "pass": bool(ok)}


def blocking_pass(checks: Mapping[str, Any]) -> bool:
    """All blocking checks pass (causality and ``extra_*`` checks are reported only)."""
    return all(block["pass"] for name, block in checks.items()
               if isinstance(block, dict) and "pass" in block and not name.startswith("extra_")
               and not block.get("blocking") is False)


# -------------------------------------------------------------------- meta


def item_meta(spec: ModelD, set_name: str, model_meta: Mapping[str, Any], *, n_sentences: int, n_spans: int,
              wall_seconds: float, checks: Mapping[str, Any], budget: TokenBudget | None,
              extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The deliverable's meta block: job A's fields, plus set, sentences_sha256 and bos_token."""
    env = tree_runner.environment_meta()
    return {
        "job": "t4", "job_letter": "D", "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "model_id": spec.hf_id, "revision": model_meta.get("revision"), "dtype": spec.dtype,
        "logits_dtype": "float32", "torch": env["versions"].get("torch"),
        "transformers": env["versions"].get("transformers"), "versions": env["versions"], "gpu": env["gpu"],
        "wall_seconds": round(float(wall_seconds), 1),
        "reference_code_sha": reference_sha256(), "reference_code_modified": not reference_unmodified(),
        "checks": dict(checks),
        "set": set_name, "sentences_sha256": SENTENCES_SHA256, "bos_token": model_meta.get("bos_token"),
        "bos_id": model_meta.get("bos_id"), "n_sentences": n_sentences, "n_spans": n_spans,
        "fillers": list(ref.FILLERS),
        "engine": ("job_d.score_sentences: ref.t4_rows's jobs (built with the reference's own helpers), "
                   "length-sorted and packed across sentences under a token budget; forward and row formulas "
                   "as in ref.forward / ref.t4_rows; checked against the unmodified ref.t4_rows "
                   "(checks.engine_equivalence)"),
        "token_budget": budget.as_meta() if budget else None, "tf32": model_meta.get("tf32"),
        "notebook": "experiment_job_d.ipynb",
        **(dict(extra) if extra else {}),
    }


def finalize_item(rows: Sequence[Mapping[str, Any]], sentences: Sequence[Mapping[str, Any]],
                  out_path: str | os.PathLike, meta: Mapping[str, Any]) -> Path:
    """Write ``tree_t4_<model>_<set>.jsonl.gz`` in sentence-file order; refuses a partial run."""
    order = {s["id"]: k for k, s in enumerate(sentences)}
    rows = sorted((r for r in rows if r["id"] in order), key=lambda r: order[r["id"]])
    return tree_runner.finalize(rows, out_path, meta, job="t4", expected=len(sentences))


def read_deliverable(path: str | os.PathLike) -> tuple[dict, list[dict]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    return lines[0]["meta"], lines[1:]


__all__ = [
    "REFERENCE_SHA256", "SENTENCES_SHA256", "SENTENCES_FILE", "ANCHOR_FILE", "SETS", "CACHE_FILLERS", "MODELS_D",
    "QUEUE", "TIER_DUE", "ModelD", "TokenBudget", "output_name", "checks_name", "work_name", "reference_sha256",
    "reference_unmodified", "load_sentences", "load_anchor", "load_span_totals", "cost_units", "resolve_revision",
    "set_numerics", "load_model_d", "prepare", "score_sentences", "forward_safe", "calibrate_budget", "max_seq_len",
    "run_item", "check_equivalence", "check_anchor", "check_cache", "check_causality", "check_complete",
    "blocking_pass", "item_meta", "finalize_item", "read_deliverable",
]
