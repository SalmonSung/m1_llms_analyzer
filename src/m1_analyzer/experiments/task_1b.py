"""Phase B of Task 1b: from cached span costs to the ``fig_1b`` record.

Task 1b asks whether substitution costs *find* constituents rather than merely
confirm ones we name. For each sentence the cached costs are reduced to one
cost per span by the replacement policy, a bracketing is induced (greedy by
default: cheapest first, never cross), and unlabelled F1 against gold is
compared with the trees that know nothing -- right-branching, left-branching,
random. Everything here is CPU-only and re-runnable in seconds.

Record (the schema `experiment_figures.fig_1b` draws; also valid for ``fig_0c``)::

    {"n_sentences": int, "treebank": str,
     "methods": [{"name": str, "f1": float, "ci": [lo, hi]}, ...],
     "by_length": [{"len": int, "n": int, "f1_sub": float, "f1_rb": float}, ...],
     "rank_curve": [{"frac_spans": float, "recall_gold": float}, ...],
     "meta": {"model", "date", "n_items", "notes"},
     "diagnostics": {...}}          # extra, ignored by the figure

Statistics: ``methods[*].f1`` is the *sentence-level mean* F1 with a 95%
percentile bootstrap over sentences (the figure's whiskers); the corpus-level
micro F1 the parsing literature also reports is in ``diagnostics``. Sentences
with no gold span after punctuation removal are excluded and counted.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np

from ..utils.logging import get_logger
from .proforms import MinOverSet, ReplacementPolicy
from .span_costs import SpanCostTable, compute_span_costs
from .spans import (
    INDUCERS,
    bracket_prf,
    left_branching,
    random_binary,
    rank_curve,
    right_branching,
)
from .stats import bootstrap_ci, paired_bootstrap_diff, trapezoid_area
from .treebank import TreebankSentence

log = get_logger("task_1b")

METHOD_SUB = "substitution-induced"
METHOD_RB = "right-branching"
METHOD_LB = "left-branching"
METHOD_RANDOM = "random bracketing"
METHODS = (METHOD_SUB, METHOD_RB, METHOD_LB, METHOD_RANDOM)

RANK_GRID = [round(x, 4) for x in np.linspace(0.0, 1.0, 26)]


# ----------------------------------------------------------- per sentence


def evaluate_sentence(
    table: SpanCostTable,
    gold: set,
    policy: ReplacementPolicy,
    *,
    inducer: str = "greedy",
    normalisation: str = "mean",
    n_random: int = 10,
    rng: np.random.Generator | None = None,
    grid: Sequence[float] = RANK_GRID,
) -> dict[str, Any]:
    """All the numbers one sentence contributes."""
    if inducer not in INDUCERS:
        raise ValueError(f"inducer must be one of {tuple(INDUCERS)}, got {inducer!r}")
    rng = rng if rng is not None else np.random.default_rng(0)
    n = table.n
    costs = table.costs_for(policy, normalisation)
    pred = set(INDUCERS[inducer](costs, n))
    p, r, f_sub, match = bracket_prf(pred, gold)
    f_rb = bracket_prf(right_branching(n), gold)[2]
    f_lb = bracket_prf(left_branching(n), gold)[2]
    f_rand = float(np.mean([bracket_prf(random_binary(n, rng), gold)[2] for _ in range(n_random)]))
    return {
        "id": table.sentence_id, "len": n,
        "n_gold": len(gold), "n_pred": len(pred), "n_match": match,
        "n_spans_scored": len(costs),
        "f1_sub": f_sub, "precision_sub": p, "recall_sub": r,
        "f1_rb": f_rb, "f1_lb": f_lb, "f1_random": f_rand,
        "rank_curve": rank_curve(costs, gold, grid),
        "pred": sorted(pred),
    }


# ---------------------------------------------------------------- corpus


def analyse_1b(
    tables: Sequence[SpanCostTable],
    sentences: Sequence[TreebankSentence] | Mapping[str, TreebankSentence],
    *,
    policy: ReplacementPolicy | None = None,
    inducer: str = "greedy",
    normalisation: str = "mean",
    n_random: int = 10,
    seed: int = 0,
    min_per_length: int = 5,
    n_boot: int = 1000,
    model: str = "unknown",
    treebank: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Build the ``fig_1b`` record from cached costs and their gold sentences."""
    policy = policy or MinOverSet()
    by_id = sentences if isinstance(sentences, Mapping) else {s.id: s for s in sentences}
    rng = np.random.default_rng(seed)

    rows, n_empty, n_missing = [], 0, 0
    for table in tables:
        sentence = by_id.get(table.sentence_id)
        if sentence is None:
            n_missing += 1
            continue
        if not sentence.gold_spans:
            n_empty += 1
            continue
        rows.append(evaluate_sentence(
            table, sentence.gold_spans, policy, inducer=inducer,
            normalisation=normalisation, n_random=n_random, rng=rng,
        ))
    if not rows:
        raise ValueError("No sentence could be evaluated (no tables matched a sentence with gold spans).")

    def column(key: str) -> np.ndarray:
        return np.array([row[key] for row in rows], dtype=float)

    methods = []
    for name, key in ((METHOD_SUB, "f1_sub"), (METHOD_RB, "f1_rb"), (METHOD_LB, "f1_lb"), (METHOD_RANDOM, "f1_random")):
        point, lo, hi = bootstrap_ci(column(key), n_boot=n_boot, seed=seed)
        methods.append({"name": name, "f1": round(point, 4), "ci": [round(lo, 4), round(hi, 4)]})

    by_length = []
    for length in sorted({row["len"] for row in rows}):
        group = [row for row in rows if row["len"] == length]
        if len(group) < min_per_length:
            continue
        by_length.append({
            "len": int(length), "n": len(group),
            "f1_sub": round(float(np.mean([g["f1_sub"] for g in group])), 4),
            "f1_rb": round(float(np.mean([g["f1_rb"] for g in group])), 4),
        })

    curve = np.mean([row["rank_curve"] for row in rows], axis=0)
    rank = [{"frac_spans": float(x), "recall_gold": round(float(y), 4)} for x, y in zip(RANK_GRID, curve)]

    tm, tp, tg = (int(sum(row[k] for row in rows)) for k in ("n_match", "n_pred", "n_gold"))
    micro_f1 = 2 * tm / (tp + tg) if (tp + tg) else 0.0
    gap, gap_lo, gap_hi = paired_bootstrap_diff(column("f1_sub"), column("f1_rb"), n_boot=n_boot, seed=seed)
    gap_lb = paired_bootstrap_diff(column("f1_sub"), column("f1_lb"), n_boot=n_boot, seed=seed)
    gap_rand = paired_bootstrap_diff(column("f1_sub"), column("f1_random"), n_boot=n_boot, seed=seed)
    area = trapezoid_area(RANK_GRID, curve)
    treebank = treebank or next(iter(by_id.values())).source

    conventions = (
        "traces and punctuation removed, unaries collapsed, unlabelled spans, "
        "single words and whole sentence excluded"
    )
    note_parts = [
        f"inducer={inducer}", f"policy={policy.name}", f"cost={normalisation} log-prob per token",
        f"random={n_random} trees/sentence", f"micro F1={micro_f1:.3f}",
        f"gap vs RB={gap:+.3f} [{gap_lo:+.3f}, {gap_hi:+.3f}]", f"rank area={area:.3f}",
    ]
    if n_empty:
        note_parts.append(f"{n_empty} flat sentence(s) without gold excluded")
    if notes:
        note_parts.append(notes)

    record = {
        "n_sentences": len(rows),
        "treebank": treebank,
        "methods": methods,
        "by_length": by_length,
        "rank_curve": rank,
        "meta": {
            "model": model,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "n_items": len(rows),
            "notes": "; ".join(note_parts),
        },
        "diagnostics": {
            "conventions": conventions,
            "inducer": inducer, "policy": policy.name, "normalisation": normalisation,
            "n_random": n_random, "seed": seed, "n_boot": n_boot,
            "micro": {"n_match": tm, "n_pred": tp, "n_gold": tg, "f1": round(micro_f1, 4)},
            "gap_vs_rb": {"diff": round(gap, 4), "ci": [round(gap_lo, 4), round(gap_hi, 4)]},
            "gap_vs_lb": {"diff": round(gap_lb[0], 4), "ci": [round(gap_lb[1], 4), round(gap_lb[2], 4)]},
            "gap_vs_random": {"diff": round(gap_rand[0], 4), "ci": [round(gap_rand[1], 4), round(gap_rand[2], 4)]},
            "rank_area": round(area, 4),
            "n_empty_gold": n_empty, "n_unmatched_tables": n_missing,
            "n_failed_variants": int(sum(t.failed for t in tables)),
            "per_sentence": [
                {k: row[k] for k in ("id", "len", "n_gold", "n_pred", "n_match", "f1_sub", "f1_rb", "f1_lb", "f1_random")}
                for row in rows
            ],
        },
    }
    validate_record(record)
    return record


# ------------------------------------------------------------ validation


def validate_record(record: Mapping[str, Any]) -> None:
    """Fail loudly if the record would not draw as ``fig_1b`` / ``fig_0c``."""
    for key in ("n_sentences", "treebank", "methods", "by_length", "rank_curve", "meta"):
        if key not in record:
            raise ValueError(f"record is missing {key!r}")
    names = [m["name"] for m in record["methods"]]
    for needed in (METHOD_SUB, METHOD_RB, METHOD_RANDOM):
        if needed not in names:
            raise ValueError(f"methods must include {needed!r}; got {names}")
    for m in record["methods"]:
        if not (0.0 <= m["f1"] <= 1.0):
            raise ValueError(f"F1 out of range for {m['name']}: {m['f1']}")
        if m.get("ci") is not None and len(m["ci"]) != 2:
            raise ValueError(f"ci for {m['name']} must be [lo, hi]")
    for row in record["by_length"]:
        for key in ("len", "n", "f1_sub", "f1_rb"):
            if key not in row:
                raise ValueError(f"by_length row missing {key!r}: {row}")
    for point in record["rank_curve"]:
        if not (0.0 <= point["frac_spans"] <= 1.0 and 0.0 <= point["recall_gold"] <= 1.0):
            raise ValueError(f"rank_curve point out of range: {point}")
    if not isinstance(record["meta"], Mapping) or "model" not in record["meta"]:
        raise ValueError("meta must be a mapping naming at least the model")


def verdict(record: Mapping[str, Any]) -> str:
    """The runbook's pass / fail reading, as a sentence."""
    f1 = {m["name"]: m for m in record["methods"]}
    sub, rb = f1[METHOD_SUB], f1[METHOD_RB]
    diag = record.get("diagnostics", {})
    gap = diag.get("gap_vs_rb", {}).get("ci")
    gap_lb = diag.get("gap_vs_lb", {}).get("ci")
    clears_rb = gap is not None and gap[0] > 0
    clears_lb = gap_lb is not None and gap_lb[0] > 0
    lengths = record["by_length"]
    below = [row["len"] for row in lengths if row["f1_sub"] < row["f1_rb"]]
    area = diag.get("rank_area")
    lines = [
        f"substitution-induced F1 {sub['f1']:.3f} {sub['ci']} vs right-branching {rb['f1']:.3f} {rb['ci']}.",
    ]
    if clears_rb and clears_lb:
        lines.append("PASS: clears both baselines by their paired-bootstrap intervals.")
    elif clears_rb:
        lines.append("PARTIAL: clears right-branching but not left-branching -- may be reproducing a direction.")
    else:
        lines.append("FAIL: ties or loses to right-branching -- the test confirms boundaries, it does not find them.")
    if below:
        lines.append(f"Below right-branching at lengths {below} (a method that only wins on short sentences has not won).")
    if area is not None:
        lines.append(f"Rank-curve area {area:.3f} (0.5 = chance): " + (
            "costs rank gold spans early." if area >= 0.6 else "costs barely rank gold spans above chance."
        ))
    return "\n".join(lines)


# ------------------------------------------------------------ end to end


def run_task_1b(
    scorer,
    sentences: Sequence[TreebankSentence],
    *,
    policy: ReplacementPolicy | None = None,
    cache_path=None,
    provenance: dict[str, Any] | None = None,
    inducer: str = "greedy",
    normalisation: str = "mean",
    n_random: int = 10,
    seed: int = 0,
    min_per_length: int = 5,
    model: str = "unknown",
    treebank: str | None = None,
    notes: str = "",
    show_progress: bool = True,
    **score_kwargs: Any,
) -> tuple[dict[str, Any], list[SpanCostTable]]:
    """Phase A then phase B in one call; returns ``(record, tables)``."""
    policy = policy or MinOverSet()
    tables = compute_span_costs(
        scorer, sentences, policy, cache_path=cache_path, provenance=provenance,
        show_progress=show_progress, **score_kwargs,
    )
    record = analyse_1b(
        tables, sentences, policy=policy, inducer=inducer, normalisation=normalisation,
        n_random=n_random, seed=seed, min_per_length=min_per_length, model=model,
        treebank=treebank, notes=notes,
    )
    return record, tables
