"""Phase B of Task 9a: from the splice cache to the ``fig_9a`` record.

Does deleting whole sentences between two *close* endpoint states change what
follows less than deleting between two *far* ones? Divergence is dominated by
how much text was deleted (Spearman with deleted length ~ +0.9 on the demo
paragraph), and endpoint distance itself grows with deleted length, so the
comparison is made **within strata of deleted-segment length**, and the
close / far labels are assigned **within length-matching bins** of
``match_width`` tokens inside each stratum (bottom / top decile of endpoint
distance, pooled across paragraphs), so close and far cuts delete the same
amount of text to within a bin. Pooled deciles would leave the short stratum
with no far cuts and the long one with no close cuts; deciles over a whole
stratum of doubling width still leave the far decile deleting more than the
close one, which a synthetic null exposed as a spurious 1.2x.

The pass criterion is one stratified permutation test: the statistic is the
stratum-size-weighted mean of ``log(median far / median close)`` over strata
with enough cuts in both groups; close / far labels are shuffled *within* each
matching bin, so the null keeps every bin's label counts. Cuts from one paragraph share its states, so the confidence interval
on the stratified ratio comes from a paragraph-level cluster bootstrap. The
pooled comparison, the per-stratum Mann-Whitney values and the continuous
(residual) analysis are reported as diagnostics, never as the verdict.

Grammaticality: admissibility makes the splice grammatical by construction, so
``grammatical`` records the *hand audit's* outcome. The audit sample is drawn
blind (the sheet shows text only, never a divergence), the outcome is read back
from the filled sheet, unaudited cuts carry ``audited: false``, and a failed
audit is reported as a violation of the rule, then excluded from the
comparison as the figure requires.

Record (the schema `experiment_figures.fig_9a` draws, plus extras)::

    {"cuts": [{"pair": "close"|"far", "grammatical": bool, "seg_len": int,
               "divergence": float, "audited": bool, "paragraph": str, "i": int,
               "j": int, "boundary": str, "endpoint_distance": float,
               "stratum": str, "fluency": float,
               "del_surp": float, "del_surp_mean": float   # when the cache carries them
              }, ...],
     "divergence_measure": str, "window": int,
     "strata": [{"label", "lo", "hi", "n_close", "n_far", "median_close",
                 "median_far", "ratio", "p_mannwhitney", "n_total"}, ...],
     "stratified": {"ratio", "ci", "p", "test", "n_perm", "n_boot", "n_paragraphs"},
     "pooled": {"ratio", "p", "n_close", "n_far", "note"},
     "audit": {"n_audited", "n_failed", "failed", "by_pair"},
     "meta": {"model", "date", "n_items", "notes"},
     "diagnostics": {...}}
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from ..utils.logging import get_logger
from .boundaries import SENTENCE, starts_sentence
from .splice import DEFAULT_DECILE, DEFAULT_MATCH_WIDTH, DEFAULT_STRATA, DIVERGENCE_MEASURE, check_surprisal

log = get_logger("task_9a")

CLOSE, FAR, MID = "close", "far", "mid"
TEST_NAME = (
    "stratified permutation test: stratum-size-weighted mean of log(median far / median close) "
    "over length strata, close/far labels shuffled within length-matching bins, one-sided (far > close)"
)
AUDIT_COLUMNS = ("key", "paragraph", "i", "j", "pair", "boundary", "seg_len", "join", "spliced_text",
                 "grammatical", "note")
_TRUE = {"y", "yes", "true", "t", "1", "ok", "grammatical"}
_FALSE = {"n", "no", "false", "f", "0", "ungrammatical", "bad"}


# ---------------------------------------------------------------- cuts


def cut_key(paragraph_id: str, i: int, j: int) -> str:
    return f"{paragraph_id}:{i}-{j}"


def flatten_cuts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One dict per cut with its paragraph's id and fluency attached."""
    cuts = []
    for row in rows:
        for c in row["cuts"]:
            cut = dict(c)
            cut["paragraph"] = row["id"]
            cut["key"] = cut_key(row["id"], c["i"], c["j"])
            cut["paragraph_fluency"] = row.get("fluency")
            cuts.append(cut)
    return cuts


def stratum_label(lo: int, hi: int) -> str:
    return f"{lo}–{hi - 1} tokens"


def stratum_of(seg_len: int, strata: Sequence[Sequence[int]]) -> int | None:
    for k, (lo, hi) in enumerate(strata):
        if lo <= seg_len < hi:
            return k
    return None


def bin_of(seg_len: int, strata: Sequence[Sequence[int]], match_width: int) -> tuple[int, int] | None:
    """``(stratum, bin)``: the matching bin of `match_width` tokens, aligned to the stratum's lower edge."""
    k = stratum_of(seg_len, strata)
    if k is None:
        return None
    return k, (seg_len - strata[k][0]) // match_width


def assign_pairs(
    cuts: Sequence[dict[str, Any]],
    *,
    strata: Sequence[Sequence[int]] = DEFAULT_STRATA,
    decile: float = DEFAULT_DECILE,
    boundary: str = SENTENCE,
    match_width: int = DEFAULT_MATCH_WIDTH,
    min_bin: int = 4,
) -> None:
    """Label each cut in place: ``stratum`` (index or None), ``bin`` and ``pair``.

    Within each length-matching bin (`match_width` tokens, nested in the
    strata), over cuts of the given boundary type pooled across paragraphs, the
    bottom `decile` of endpoint distance is ``close``, the top `decile` is
    ``far``, the rest ``mid``. Bins with fewer than `min_bin` cuts are left
    unlabelled (``mid``). Cuts of other boundary types or outside every
    stratum get ``pair = None``. The pooled label (deciles over everything at
    once, the length-confounded version) goes to ``pair_pooled``.
    """
    for cut in cuts:
        cut["stratum"] = stratum_of(int(cut["seg_len"]), strata)
        cut["bin"] = bin_of(int(cut["seg_len"]), strata, match_width)
        cut["pair"] = None
        cut["pair_pooled"] = None
    eligible = [c for c in cuts if c["boundary"] == boundary]
    bins = sorted({c["bin"] for c in eligible if c["bin"] is not None})
    for b in bins:
        group = [c for c in eligible if c["bin"] == b]
        _label(group, decile, "pair", min_bin)
    _label(eligible, decile, "pair_pooled", 2)


def _label(group: list[dict[str, Any]], decile: float, field: str, minimum: int) -> None:
    if len(group) < minimum:
        for c in group:
            c[field] = MID
        return
    d = np.array([c["d"] for c in group], dtype=float)
    lo, hi = np.quantile(d, [decile, 1.0 - decile])
    for c, value in zip(group, d):
        c[field] = CLOSE if value <= lo else FAR if value >= hi else MID


# ---------------------------------------------------------------- audit


def audit_sample(
    cuts: Sequence[Mapping[str, Any]],
    *,
    n_close: int = 10,
    n_far: int = 10,
    n_other: int = 10,
    seed: int = 0,
    boundary: str = SENTENCE,
) -> list[dict[str, Any]]:
    """A seeded random sample of labelled cuts for the hand audit.

    ``n_other`` is drawn from the remaining admissible cuts of the boundary type
    (``mid`` and outside-stratum), so the audit also checks the rule where the
    comparison does not look.
    """
    rng = np.random.default_rng(seed)
    pool = [c for c in cuts if c["boundary"] == boundary]
    groups = {
        CLOSE: [c for c in pool if c.get("pair") == CLOSE],
        FAR: [c for c in pool if c.get("pair") == FAR],
        "other": [c for c in pool if c.get("pair") not in (CLOSE, FAR)],
    }
    wanted = {CLOSE: n_close, FAR: n_far, "other": n_other}
    sample = []
    for name, group in groups.items():
        k = min(wanted[name], len(group))
        if k < wanted[name]:
            log.warning("Audit: only %d %s cut(s) available, %d asked for.", len(group), name, wanted[name])
        if k:
            picked = rng.choice(len(group), size=k, replace=False)
            sample.extend(dict(group[int(p)], audit_group=name) for p in sorted(picked))
    return sample


def write_audit_csv(sample: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]],
                    path: str | os.PathLike, marker: str = " ⟦cut⟧ ") -> Path:
    """The sheet a person fills in: text only, never a divergence."""
    from .splice import splice_text

    texts = {r["id"]: r["text"] for r in rows}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        for c in sample:
            text = texts[c["paragraph"]]
            spliced = text[: c["char_i"]] + marker + text[c["char_j"]:].lstrip()
            writer.writerow({
                "key": c["key"], "paragraph": c["paragraph"], "i": c["i"], "j": c["j"],
                "pair": c.get("audit_group", c.get("pair")), "boundary": c["boundary"], "seg_len": c["seg_len"],
                "join": c.get("join", ""), "spliced_text": spliced, "grammatical": "", "note": "",
            })
    return path


def load_audit_csv(path: str | os.PathLike, *, strict: bool = True) -> dict[str, tuple[bool, str]]:
    """``{key: (grammatical, note)}`` from a filled sheet; unfilled rows raise (or are skipped)."""
    out: dict[str, tuple[bool, str]] = {}
    missing = []
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            value = (row.get("grammatical") or "").strip().lower()
            if value in _TRUE:
                out[row["key"]] = (True, row.get("note") or "")
            elif value in _FALSE:
                out[row["key"]] = (False, row.get("note") or "")
            else:
                missing.append(row["key"])
    if missing and strict:
        raise ValueError(
            f"{path}: {len(missing)} row(s) have no grammatical judgement (first: {missing[0]!r}). "
            "Fill the `grammatical` column with y/n for every row, or pass strict=False to skip them."
        )
    return out


def apply_audit(cuts: Sequence[dict[str, Any]], audit: Mapping[str, tuple[bool, str]] | None) -> None:
    """Set ``grammatical`` / ``audited`` / ``audit_note`` on every cut in place."""
    audit = audit or {}
    unknown = set(audit) - {c["key"] for c in cuts}
    if unknown:
        raise ValueError(f"audit names {len(unknown)} cut(s) not in this cache (first: {sorted(unknown)[0]!r})")
    for c in cuts:
        if c["key"] in audit:
            c["grammatical"], c["audit_note"] = audit[c["key"]]
            c["audited"] = True
        else:
            c["grammatical"], c["audit_note"], c["audited"] = True, "", False


def boundary_check(rows: Sequence[Mapping[str, Any]], cuts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Do the cache's sentence endpoints actually sit at sentence boundaries?

    Zero for any cache written under schema 2, whose boundary rule rejects a
    position that is not followed by a sentence start. A non-zero count means the
    cache predates that rule, and the count is *not* evenly spread: such a
    position is mid-sentence, so its state is atypical for a sentence end and
    lands in the FAR decile far more often than the close one (13.5% vs 0.5% on
    the 200-paragraph Qwen run that motivated the rule). Read ``close`` and
    ``far`` here before reading the ratio: contamination concentrated in one arm
    inflates it in the direction of the hypothesis.

    ``n_cuts_affected`` counts cuts touching such a position at either endpoint --
    a bad head leaves the splice ending mid-sentence just as a bad tail leaves it
    resuming mid-sentence.
    """
    text_of = {r["id"]: r["text"] for r in rows}
    bad: set[tuple[str, int]] = set()
    checked = 0
    for r in rows:
        char_of: dict[int, int] = {}
        for c in r["cuts"]:
            if c["boundary"] != SENTENCE:
                continue
            char_of[c["i"]] = c["char_i"]
            char_of[c["j"]] = c["char_j"]
        for token, char in char_of.items():
            checked += 1
            if not starts_sentence(r["text"], char):
                bad.add((r["id"], token))

    def touches(c: Mapping[str, Any]) -> bool:
        return (c["paragraph"], c["i"]) in bad or (c["paragraph"], c["j"]) in bad

    sentence_cuts = [c for c in cuts if c["boundary"] == SENTENCE]
    by_pair = {}
    for name in (CLOSE, FAR):
        group = [c for c in sentence_cuts if c["pair"] == name]
        hit = sum(touches(c) for c in group)
        by_pair[name] = {"n": len(group), "affected": hit,
                         "rate": _r(hit / len(group)) if group else float("nan")}
    examples = []
    for pid, token in sorted(bad)[:5]:
        char = next(c["char_i"] if c["i"] == token else c["char_j"]
                    for r in rows if r["id"] == pid
                    for c in r["cuts"] if c["boundary"] == SENTENCE and token in (c["i"], c["j"]))
        examples.append({"paragraph": pid, "token": token,
                         "before": text_of[pid][max(0, char - 40):char],
                         "after": text_of[pid][char:char + 40]})
    return {
        "n_endpoints_checked": checked,
        "n_endpoints_not_followed_by_a_sentence": len(bad),
        "n_cuts_affected": sum(touches(c) for c in sentence_cuts),
        "n_cuts": len(sentence_cuts),
        "by_pair": by_pair,
        "examples": examples,
        "clean": not bad,
    }


# ----------------------------------------------------------- statistics


def _by_stratum(cuts: Sequence[Mapping[str, Any]], n_strata: int) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for k in range(n_strata):
        close = np.array([c["div"] for c in cuts if c["stratum"] == k and c["pair"] == CLOSE], dtype=float)
        far = np.array([c["div"] for c in cuts if c["stratum"] == k and c["pair"] == FAR], dtype=float)
        out.append((close, far))
    return out


def _by_bin(cuts: Sequence[Mapping[str, Any]]) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """``(stratum, close, far)`` per matching bin, the unit the permutation shuffles within."""
    bins = sorted({c["bin"] for c in cuts if c["bin"] is not None and c["pair"] in (CLOSE, FAR)})
    out = []
    for b in bins:
        close = np.array([c["div"] for c in cuts if c["bin"] == b and c["pair"] == CLOSE], dtype=float)
        far = np.array([c["div"] for c in cuts if c["bin"] == b and c["pair"] == FAR], dtype=float)
        out.append((b[0], close, far))
    return out


def _strata_from_bins(bins: Sequence[tuple[int, np.ndarray, np.ndarray]], n_strata: int) -> list[tuple[np.ndarray, np.ndarray]]:
    close = [[] for _ in range(n_strata)]
    far = [[] for _ in range(n_strata)]
    for k, c, f in bins:
        close[k].append(c)
        far[k].append(f)
    return [(np.concatenate(c) if c else np.array([]), np.concatenate(f) if f else np.array([])) for c, f in zip(close, far)]


def stratified_log_ratio(groups: Sequence[tuple[np.ndarray, np.ndarray]], min_per_group: int) -> float:
    """Stratum-size-weighted mean of log(median far / median close); nan without a usable stratum."""
    num, den = 0.0, 0.0
    for close, far in groups:
        if len(close) < min_per_group or len(far) < min_per_group:
            continue
        mc, mf = float(np.median(close)), float(np.median(far))
        if mc <= 0 or mf <= 0:
            continue
        w = len(close) + len(far)
        num += w * np.log(mf / mc)
        den += w
    return num / den if den else float("nan")


def stratified_permutation_test(
    bins: Sequence[tuple[int, np.ndarray, np.ndarray]], n_strata: int, *,
    min_per_group: int = 10, n_perm: int = 5000, seed: int = 0,
) -> tuple[float, float]:
    """``(observed log ratio, one-sided p)``.

    `bins` are ``(stratum, close, far)`` per length-matching bin. The statistic
    is computed over strata; the labels are shuffled within each bin, so the
    null keeps every bin's close / far counts and hence the length balance.
    """
    observed = stratified_log_ratio(_strata_from_bins(bins, n_strata), min_per_group)
    if not np.isfinite(observed):
        return observed, float("nan")
    rng = np.random.default_rng(seed)
    pooled = [(k, np.concatenate([c, f]), len(c)) for k, c, f in bins]
    hits = 0
    for _ in range(n_perm):
        shuffled = []
        for k, values, nc in pooled:
            perm = rng.permutation(values)
            shuffled.append((k, perm[:nc], perm[nc:]))
        if stratified_log_ratio(_strata_from_bins(shuffled, n_strata), min_per_group) >= observed:
            hits += 1
    return observed, (hits + 1) / (n_perm + 1)


def cluster_bootstrap_ratio(
    cuts: Sequence[Mapping[str, Any]], n_strata: int, *, min_per_group: int = 10, n_boot: int = 1000, seed: int = 0,
) -> tuple[float, float]:
    """95% percentile CI of the stratified ratio, resampling PARAGRAPHS with replacement."""
    by_par: dict[str, list[tuple[int, str, float]]] = {}
    for c in cuts:
        if c["pair"] in (CLOSE, FAR) and c["stratum"] is not None:
            by_par.setdefault(c["paragraph"], []).append((c["stratum"], c["pair"], float(c["div"])))
    ids = sorted(by_par)
    if len(ids) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        sample = rng.choice(len(ids), size=len(ids), replace=True)
        close = [[] for _ in range(n_strata)]
        far = [[] for _ in range(n_strata)]
        for s in sample:
            for k, pair, div in by_par[ids[int(s)]]:
                (close if pair == CLOSE else far)[k].append(div)
        stat = stratified_log_ratio([(np.array(c), np.array(f)) for c, f in zip(close, far)], min_per_group)
        if np.isfinite(stat):
            values.append(stat)
    if len(values) < 10:
        return float("nan"), float("nan")
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(np.exp(lo)), float(np.exp(hi))


def _mannwhitney_greater(far: np.ndarray, close: np.ndarray) -> float:
    if len(far) < 1 or len(close) < 1:
        return float("nan")
    return float(stats.mannwhitneyu(far, close, alternative="greater").pvalue)


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(stats.spearmanr(a, b).correlation)


def _r(x: float, nd: int = 4) -> float:
    return float("nan") if x is None or not np.isfinite(x) else round(float(x), nd)


# -------------------------------------------------------------- record


def analyse_9a(
    rows: Sequence[Mapping[str, Any]],
    header: Mapping[str, Any],
    *,
    audit: Mapping[str, tuple[bool, str]] | None = None,
    model: str = "unknown",
    notes: str = "",
    seed: int = 0,
    n_perm: int = 5000,
    n_boot: int = 1000,
    min_per_group: int = 10,
    strata: Sequence[Sequence[int]] | None = None,
    decile: float | None = None,
) -> dict[str, Any]:
    """Build the ``fig_9a`` record from a splice cache (and the filled audit).

    `strata` and `decile` default to the cache header, i.e. the pre-registered
    values; passing different ones is recorded in the notes as a deviation.
    """
    window = int(header["window"])
    boundary = header.get("primary_boundary", SENTENCE)
    pre_strata = [tuple(s) for s in header.get("strata", DEFAULT_STRATA)]
    pre_decile = float(header.get("decile", DEFAULT_DECILE))
    match_width = int(header.get("match_width", DEFAULT_MATCH_WIDTH))
    deviations = []
    if strata is not None and [tuple(s) for s in strata] != pre_strata:
        deviations.append(f"strata {list(map(list, strata))} differ from the pre-registered {list(map(list, pre_strata))}")
    if decile is not None and decile != pre_decile:
        deviations.append(f"decile {decile} differs from the pre-registered {pre_decile}")
    strata = [tuple(s) for s in (strata or pre_strata)]
    decile = decile if decile is not None else pre_decile

    cuts = flatten_cuts(rows)
    if not cuts:
        raise ValueError("The cache holds no cuts; nothing to analyse.")
    assign_pairs(cuts, strata=strata, decile=decile, boundary=boundary, match_width=match_width)
    apply_audit(cuts, audit)

    primary = [c for c in cuts if c["boundary"] == boundary]
    labelled = [c for c in primary if c["pair"] in (CLOSE, FAR)]
    entering = [c for c in labelled if c["grammatical"]]
    groups = _by_stratum(entering, len(strata))

    strata_rows = []
    for k, ((lo, hi), (close, far)) in enumerate(zip(strata, groups)):
        n_total = sum(1 for c in primary if c["stratum"] == k)
        ratio = float(np.median(far) / np.median(close)) if len(close) and len(far) and np.median(close) > 0 else float("nan")
        seg_close = [c["seg_len"] for c in entering if c["stratum"] == k and c["pair"] == CLOSE]
        seg_far = [c["seg_len"] for c in entering if c["stratum"] == k and c["pair"] == FAR]
        strata_rows.append({
            "label": stratum_label(lo, hi), "lo": lo, "hi": hi, "n_total": n_total,
            "n_close": int(len(close)), "n_far": int(len(far)),
            "median_seg_len_close": _r(np.median(seg_close), 1) if seg_close else float("nan"),
            "median_seg_len_far": _r(np.median(seg_far), 1) if seg_far else float("nan"),
            "median_close": _r(np.median(close)) if len(close) else float("nan"),
            "median_far": _r(np.median(far)) if len(far) else float("nan"),
            "ratio": _r(ratio), "p_mannwhitney": _r(_mannwhitney_greater(far, close)),
            "usable": bool(len(close) >= min_per_group and len(far) >= min_per_group),
        })

    log_ratio, p_strat = stratified_permutation_test(_by_bin(entering), len(strata), min_per_group=min_per_group,
                                                     n_perm=n_perm, seed=seed)
    ci_lo, ci_hi = cluster_bootstrap_ratio(entering, len(strata), min_per_group=min_per_group, n_boot=n_boot, seed=seed)
    n_paragraphs = len({c["paragraph"] for c in entering})

    # Pooled deciles: the length-confounded comparison the spec warns about.
    pooled_close = np.array([c["div"] for c in primary if c["grammatical"] and c["pair_pooled"] == CLOSE])
    pooled_far = np.array([c["div"] for c in primary if c["grammatical"] and c["pair_pooled"] == FAR])
    pooled = {
        "n_close": int(len(pooled_close)), "n_far": int(len(pooled_far)),
        "median_close": _r(np.median(pooled_close)) if len(pooled_close) else float("nan"),
        "median_far": _r(np.median(pooled_far)) if len(pooled_far) else float("nan"),
        "ratio": _r(np.median(pooled_far) / np.median(pooled_close)) if len(pooled_close) and len(pooled_far) else float("nan"),
        "p_mannwhitney": _r(_mannwhitney_greater(pooled_far, pooled_close)),
        "median_seg_len_close": _r(np.median([c["seg_len"] for c in primary if c["grammatical"] and c["pair_pooled"] == CLOSE])) if len(pooled_close) else float("nan"),
        "median_seg_len_far": _r(np.median([c["seg_len"] for c in primary if c["grammatical"] and c["pair_pooled"] == FAR])) if len(pooled_far) else float("nan"),
        "note": "pooled deciles with unmatched segment lengths; length-confounded, NOT the pass criterion",
    }

    # Continuous secondary: the demo's residual analysis, over every grammatical primary cut.
    gram = [c for c in primary if c["grammatical"]]
    seg = np.array([c["seg_len"] for c in gram], float)
    dist = np.array([c["d"] for c in gram], float)
    div = np.array([c["div"] for c in gram], float)
    continuous: dict[str, Any] = {
        "n": len(gram),
        "spearman_seg_div": _r(_spearman(seg, div)),
        "spearman_dist_div": _r(_spearman(dist, div)),
        "spearman_seg_dist": _r(_spearman(seg, dist)),
    }
    if len(gram) >= 3 and np.ptp(seg) > 0:
        X = np.column_stack([np.ones_like(seg), np.log(seg)])
        res_div = div - X @ np.linalg.lstsq(X, div, rcond=None)[0]
        res_dist = dist - X @ np.linalg.lstsq(X, dist, rcond=None)[0]
        continuous["spearman_resid_dist_resid_div"] = _r(_spearman(res_dist, res_div))
        close_r = [r for c, r in zip(gram, res_div) if c["pair"] == CLOSE]
        far_r = [r for c, r in zip(gram, res_div) if c["pair"] == FAR]
        continuous["resid_median_close"] = _r(np.median(close_r)) if close_r else float("nan")
        continuous["resid_median_far"] = _r(np.median(far_r)) if far_r else float("nan")
        fl = np.array([c["fl"] - (c["paragraph_fluency"] or np.nan) for c in gram], float)
        if np.isfinite(fl).all():
            continuous["spearman_fluency_delta_div"] = _r(_spearman(fl, div))
    # Deleted-segment surprisal as a second matching variable (additive: only when the
    # cache carries it). The partial correlation controls for BOTH log length and the
    # information the deleted span held; if it survives, endpoint distance carries
    # something beyond "more information deleted, more changed".
    if len(gram) >= 4 and all("del_surp" in c for c in gram):
        del_surp = np.array([c["del_surp"] for c in gram], float)
        continuous["spearman_del_surp_div"] = _r(_spearman(del_surp, div))
        continuous["spearman_del_surp_seg"] = _r(_spearman(del_surp, seg))
        continuous["spearman_del_surp_dist"] = _r(_spearman(del_surp, dist))
        if np.ptp(seg) > 0 and np.ptp(del_surp) > 0:
            X2 = np.column_stack([np.ones_like(seg), np.log(seg), del_surp])
            r_div = div - X2 @ np.linalg.lstsq(X2, div, rcond=None)[0]
            r_dist = dist - X2 @ np.linalg.lstsq(X2, dist, rcond=None)[0]
            continuous["partial_spearman_dist_div_given_logseg_del_surp"] = _r(_spearman(r_dist, r_div))

    audited = [c for c in cuts if c["audited"]]
    failed = [c for c in audited if not c["grammatical"]]
    audit_summary = {
        "n_audited": len(audited), "n_failed": len(failed),
        "failed": [{"key": c["key"], "pair": c["pair"], "seg_len": c["seg_len"], "note": c["audit_note"]} for c in failed],
        "by_pair": {name: {"n": sum(1 for c in audited if (c["pair"] if c["pair"] in (CLOSE, FAR) else "other") == name),
                           "failed": sum(1 for c in failed if (c["pair"] if c["pair"] in (CLOSE, FAR) else "other") == name)}
                    for name in (CLOSE, FAR, "other")},
        "loaded": audit is not None,
    }

    secondary = {}
    for kind in sorted({c["boundary"] for c in cuts} - {boundary}):
        others = [dict(c) for c in cuts if c["boundary"] == kind]
        assign_pairs(others, strata=strata, decile=decile, boundary=kind, match_width=match_width)
        ok = [c for c in others if c["pair"] in (CLOSE, FAR) and c["grammatical"]]
        g = _by_stratum(ok, len(strata))
        lr, p = stratified_permutation_test(_by_bin(ok), len(strata), min_per_group=min_per_group,
                                            n_perm=max(200, n_perm // 10), seed=seed)
        secondary[kind] = {"n_cuts": len(others), "stratified_ratio": _r(np.exp(lr)) if np.isfinite(lr) else float("nan"),
                           "p": _r(p), "n_per_stratum": [[len(c), len(f)] for c, f in g]}

    record_cuts = []
    for c in labelled:
        lo, hi = strata[c["stratum"]]
        entry = {
            "pair": c["pair"], "grammatical": bool(c["grammatical"]), "seg_len": int(c["seg_len"]),
            "divergence": float(c["div"]), "audited": bool(c["audited"]), "paragraph": c["paragraph"],
            "i": int(c["i"]), "j": int(c["j"]), "boundary": c["boundary"], "endpoint_distance": float(c["d"]),
            "stratum": stratum_label(lo, hi), "fluency": float(c["fl"]),
        }
        if "del_surp" in c:  # same names as the cache, so the cut joins back without a rename table
            entry["del_surp"] = float(c["del_surp"])
            entry["del_surp_mean"] = float(c["del_surp_mean"])
        record_cuts.append(entry)

    usable = [s for s in strata_rows if s["usable"]]
    note_parts = [
        f"boundary={boundary}", f"splitter={header.get('splitter')}", f"decile={decile}", f"match_width={match_width}",
        f"strata={[list(s) for s in strata]}",
        f"stratified ratio={np.exp(log_ratio):.3f} p={p_strat:.4f}" if np.isfinite(log_ratio) else "stratified ratio undefined",
        f"{len(usable)}/{len(strata)} strata usable (>= {min_per_group} per group)",
        f"pooled ratio={pooled['ratio']:.3f} (length-confounded)" if np.isfinite(pooled["ratio"]) else "pooled ratio undefined",
        f"audit {audit_summary['n_failed']}/{audit_summary['n_audited']} failed" if audit_summary["n_audited"] else "audit not loaded",
    ]
    note_parts += [f"DEVIATION: {d}" for d in deviations]
    if notes:
        note_parts.append(notes)

    record = {
        "cuts": record_cuts,
        "divergence_measure": header.get("divergence_measure", DIVERGENCE_MEASURE),
        "window": window,
        "strata": strata_rows,
        "stratified": {
            "ratio": _r(np.exp(log_ratio)) if np.isfinite(log_ratio) else float("nan"),
            "ci": [_r(ci_lo), _r(ci_hi)], "p": _r(p_strat, 5), "test": TEST_NAME,
            "n_perm": n_perm, "n_boot": n_boot, "min_per_group": min_per_group, "match_width": match_width,
            "n_paragraphs": n_paragraphs, "n_close": int(sum(len(c) for c, _ in groups)),
            "n_far": int(sum(len(f) for _, f in groups)),
        },
        "pooled": pooled,
        "audit": audit_summary,
        "meta": {
            "model": model,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "n_items": len(entering),
            "notes": "; ".join(note_parts),
        },
        "diagnostics": {
            "n_paragraphs": len(rows), "n_cuts": len(cuts), "n_primary_cuts": len(primary),
            "n_outside_strata": sum(1 for c in primary if c["stratum"] is None),
            "n_unlabelled_in_small_bins": sum(1 for c in primary if c["stratum"] is not None and c["pair"] == MID
                                              and sum(1 for o in primary if o["bin"] == c["bin"]) < 4),
            "n_not_retokenising": sum(1 for c in cuts if not c.get("retokenises", True)),
            "n_rejected_boundaries": int(sum(r.get("n_rejected_boundaries", 0) for r in rows)),
            "seg_len_range": [int(min(c["seg_len"] for c in primary)), int(max(c["seg_len"] for c in primary))] if primary else None,
            "boundary_check": boundary_check(rows, primary),
            "continuous": continuous,
            "secondary_boundaries": secondary,
            "preregistration": {k: v for k, v in header.items() if k != "kind"},
            "seed": seed,
        },
    }
    if any("surprisal" in r for r in rows):
        record["diagnostics"]["surprisal"] = check_surprisal(rows)
    validate_record(record)
    return record


# ------------------------------------------------------------ validation


def validate_record(record: Mapping[str, Any]) -> None:
    """Fail loudly if the record would not draw as ``fig_9a``."""
    for key in ("cuts", "divergence_measure", "window", "meta"):
        if key not in record:
            raise ValueError(f"record is missing {key!r}")
    for c in record["cuts"]:
        for key in ("pair", "grammatical", "seg_len", "divergence"):
            if key not in c:
                raise ValueError(f"cut is missing {key!r}: {c}")
        if c["pair"] not in (CLOSE, FAR):
            raise ValueError(f"pair must be close or far, got {c['pair']!r}")
        if c["divergence"] < 0:
            raise ValueError(f"divergence must be >= 0: {c}")
    if "strata" in record:
        for s in record["strata"]:
            for key in ("lo", "hi", "n_close", "n_far"):
                if key not in s:
                    raise ValueError(f"stratum row missing {key!r}: {s}")
    if "stratified" in record and "p" not in record["stratified"]:
        raise ValueError("stratified must carry p")
    if not isinstance(record["meta"], Mapping) or "model" not in record["meta"]:
        raise ValueError("meta must be a mapping naming at least the model")


def verdict(record: Mapping[str, Any]) -> str:
    """The runbook's pass / fail reading, as a sentence."""
    st = record.get("stratified", {})
    ratio, p, ci = st.get("ratio"), st.get("p"), st.get("ci", [None, None])
    usable = [s for s in record.get("strata", []) if s.get("usable")]
    lines = []
    if ratio is None or not np.isfinite(ratio) or p is None or not np.isfinite(p):
        lines.append("UNDETERMINED: no length stratum has enough close and far cuts (n per group below the minimum); "
                     "score more paragraphs.")
    else:
        per = ", ".join(f"{s['label']}: {s['n_close']}/{s['n_far']}" for s in record["strata"])
        lines.append(f"stratified far/close median divergence {ratio:.3f}x [{ci[0]:.3f}, {ci[1]:.3f}], "
                     f"permutation p = {p:.4f} over {len(usable)} usable stratum/strata (n close/far: {per}).")
        if p < 0.05 and ratio > 1.0:
            lines.append("PASS: far cuts diverge more within length strata -- the skip reading's core claim holds.")
        elif 0.9 <= ratio <= 1.1:
            lines.append("FAIL: the stratified ratio sits near 1.0 -- the skip reading loses its core claim; "
                         "Act 7's 'not tested at this depth' row becomes a negative result.")
        else:
            lines.append("FAIL: not significant within length strata at p < 0.05.")
    pooled = record.get("pooled", {})
    if pooled.get("ratio") is not None and np.isfinite(pooled["ratio"]):
        lines.append(f"Pooled (unmatched lengths) ratio {pooled['ratio']:.2f}x with median deleted "
                     f"{pooled.get('median_seg_len_close')} vs {pooled.get('median_seg_len_far')} tokens: "
                     "NOT a pass criterion (length-confounded).")
    cont = record.get("diagnostics", {}).get("continuous", {})
    if cont.get("spearman_seg_div") is not None and np.isfinite(cont["spearman_seg_div"]):
        lines.append(f"Spearman(deleted tokens, divergence) = {cont['spearman_seg_div']:+.2f}, "
                     f"(endpoint distance, divergence) = {cont.get('spearman_dist_div', float('nan')):+.2f}, "
                     f"(deleted tokens, endpoint distance) = {cont.get('spearman_seg_dist', float('nan')):+.2f}.")
    bc = record.get("diagnostics", {}).get("boundary_check", {})
    if bc and not bc.get("clean", True):
        close_rate = bc["by_pair"][CLOSE]["rate"]
        far_rate = bc["by_pair"][FAR]["rate"]
        lines.append(
            f"WARNING: {bc['n_endpoints_not_followed_by_a_sentence']}/{bc['n_endpoints_checked']} endpoints are "
            f"not followed by a sentence start, affecting {bc['n_cuts_affected']}/{bc['n_cuts']} cuts "
            f"({close_rate:.1%} of close, {far_rate:.1%} of far). This cache predates the schema-2 boundary "
            "rule; contamination concentrated in the far arm inflates the ratio. Re-run phase A, or exclude "
            "them via the audit and report the sensitivity."
        )

    audit = record.get("audit", {})
    if audit.get("n_audited"):
        if audit["n_failed"]:
            lines.append(f"Audit: {audit['n_failed']}/{audit['n_audited']} audited cuts were NOT grammatical -- the "
                         "admissibility rule failed at that rate; the failing cuts are excluded and drawn hollow.")
        else:
            lines.append(f"Audit: {audit['n_audited']}/{audit['n_audited']} audited cuts grammatical; the rule held.")
    else:
        lines.append("Audit sheet not loaded: every cut is grammatical by construction only (audited=false).")
    return "\n".join(lines)
