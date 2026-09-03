"""Figures for the open experiments in sentence_structure_experiments.md.

HOW THIS FILE IS USED
    On the experiment machine, each experiment writes ONE JSON record in the
    schema given in its figure function's docstring, then calls

        import experiment_figures as EF
        EF.fig_4b(json.load(open("run_4b.json")), path="fig_4b.png")

    Every fig_* function takes (record, path=None) and returns a matplotlib
    Figure; if path is given the figure is also saved. Nothing else is needed.

    Running this file as a script renders every experiment's figure twice, on
    mock data shaped like the runbook's "if true" and "if false" predictions,
    side by side, into fig_mock_<task>.png. Mock figures are watermarked.

CONVENTIONS SHARED BY ALL RECORDS
    * Every distance is L2 between next-token LOG-probability vectors, exactly
      as in the teaching doc; "normalised" (r) means divided by the median
      non-adjacent pair distance of the same sentence.
    * Costs are nats per predicted token (surprisal rise), as in Act 3.
    * Every record may carry "meta": {"model", "date", "n_items", "notes"};
      it is printed in the caption strip if present.
    * Colour is threaded: real box / embedded / close / trained = BLUE;
      straddling / control / far / untrained = ORANGE; a third comparator
      (arbitrary pairs, random bracketing, nearest-noun rule) = PURPLE;
      reference lines and "no effect" = GREY. No red/green pairs anywhere.

DEPENDENCIES  numpy, scipy, matplotlib, pillow (for the mock composites).
"""
import json
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

BLUE, ORANGE, PURPLE, GREY, TXT = "#2b6ca3", "#d9822b", "#6a51a3", "#8a8f94", "#222222"
S_T, S_L, S_S = 11.0, 9.5, 8.5          # title / label / tick font ladder
mpl.rcParams.update({"font.size": S_L, "axes.titlesize": S_T, "axes.labelsize": S_L,
                     "xtick.labelsize": S_S, "ytick.labelsize": S_S,
                     "legend.fontsize": S_S, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
RNG = np.random.default_rng


# --------------------------------------------------------------------- helpers
def _wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, cen - half, cen + half


def _boot_ci(vals, fn=np.mean, n_boot=2000, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) < 2:
        return fn(vals), fn(vals), fn(vals)
    r = RNG(seed)
    bs = [fn(r.choice(vals, len(vals), replace=True)) for _ in range(n_boot)]
    return fn(vals), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def _ratio_ci(a, b, n_boot=2000, seed=0):
    """mean(b)/mean(a) with a paired bootstrap CI (a, b same length)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = RNG(seed)
    idx = np.arange(len(a))
    bs = []
    for _ in range(n_boot):
        s = r.choice(idx, len(idx), replace=True)
        bs.append(b[s].mean() / a[s].mean())
    return b.mean() / a.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def _strip(ax, x, vals, col, w=0.18, seed=0, s=14, alpha=0.75):
    """Jittered strip with a median tick - for small n (figure-style 6.1)."""
    r = RNG(seed)
    vals = np.asarray(vals, float)
    ax.scatter(x + r.uniform(-w, w, len(vals)), vals, s=s, color=col, alpha=alpha,
               edgecolor="none", zorder=3)
    ax.plot([x - w * 1.5, x + w * 1.5], [np.median(vals)] * 2, color=col, lw=2.2,
            zorder=4, solid_capstyle="butt")


def _ecdf(ax, vals, col, label):
    v = np.sort(np.asarray(vals, float))
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=col, lw=1.8,
            label=label)


def _watermark(fig, rec):
    if rec.get("mock"):
        fig.text(0.5, 0.5, "MOCK DATA \u00b7 %s" % rec.get("outcome", "").upper(),
                 ha="center", va="center", fontsize=40, color=GREY, alpha=0.13,
                 rotation=25, zorder=0)


def _caption(fig, rec, text):
    """Caption strip. Wrapped to the figure width (~13 chars per inch at this
    size) so a long corpus name or note from the experiment machine cannot run
    off the canvas - it did, on a real record, before this wrap existed."""
    import textwrap
    meta = rec.get("meta", {})
    tail = ("  \u00b7  " + ", ".join("%s: %s" % (k, v) for k, v in meta.items())) if meta else ""
    width = max(60, int(fig.get_figwidth() * 16))
    fig.text(0.01, 0.005, "\n".join(textwrap.wrap(text + tail, width)), fontsize=S_S - 0.5, color=GREY, va="bottom", linespacing=1.3)


def _finish(fig, rec, path):
    _watermark(fig, rec)
    if path:
        fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    return fig


def _goodness(ax, text):
    """Direction-of-goodness cue, upright, right of the panel top; the panel's
    title is lifted above it so the two never share a line."""
    ax.text(1.0, 1.005, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=S_S, color=GREY, style="italic")
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=15)


# ===================================================================== 0a
def fig_0a(rec, path=None):
    """Task 0a - does an n-gram model reproduce the substitution separation?

    RECORD
      {"spans": [{"span": str, "sentence_id": int, "label": "box"|"straddle",
                  "cost_gpt2": float, "cost_ngram": float}, ...],
       "ngram": {"order": int, "corpus": str, "tokens": int},   # for the caption
       "meta": {...}}
      cost_* = surprisal rise in nats/token after replacing the span, exactly
      as in Act 3 (same sentences, spans and replacement words for both models).

    FIGURE  Two abutting strips sharing y: per-span cost by label, GPT-2 and
      the n-gram model. Median ticks; the gap between medians printed once.
    PASS  GPT-2 separates the groups; the n-gram model does not (or far less).
    FAIL  Both separate - the test measures bigram plausibility.
    """
    sp = rec["spans"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    fig.subplots_adjust(left=0.16)
    for ax, key, name in zip(axes, ("cost_gpt2", "cost_ngram"),
                             ("GPT-2 124M", "%d-gram (Kneser-Ney)" % rec["ngram"]["order"])):
        b = [s[key] for s in sp if s["label"] == "box"]
        t = [s[key] for s in sp if s["label"] == "straddle"]
        _strip(ax, 0, b, BLUE); _strip(ax, 1, t, ORANGE)
        gap = np.median(t) - np.median(b)
        ax.set_title("%s: gap %.2f nats" % (name, gap), loc="left")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["real box\n(n=%d)" % len(b),
                                                   "straddles\n(n=%d)" % len(t)])
        ax.set_xlim(-0.6, 1.6); ax.margins(y=0.08)
    axes[0].set_ylabel("cost of replacing the span (nats/token)", labelpad=8)
    _goodness(axes[1], "larger gap = the model sees the box")
    _caption(fig, rec, "n-gram trained on %s (%s tokens); same spans and replacements as Act 3."
             % (rec["ngram"]["corpus"], "{:,}".format(rec["ngram"]["tokens"])))
    fig.subplots_adjust(bottom=0.22, top=0.86)
    return _finish(fig, rec, path)


def mock_0a(outcome="true", n=40, seed=1):
    r = RNG(seed)
    sp = []
    for i in range(n):
        box = i % 2 == 0
        g = r.normal(0.75 if box else 1.80, 0.25)
        ng = (r.normal(0.9 if box else 1.05, 0.35) if outcome == "true"
              else r.normal(0.8 if box else 1.7, 0.3))
        sp.append(dict(span="span %d" % i, sentence_id=i // 8,
                       label="box" if box else "straddle", cost_gpt2=float(g),
                       cost_ngram=float(max(ng, 0.05))))
    return dict(spans=sp, ngram=dict(order=5, corpus="WikiText-103", tokens=103_000_000),
                mock=True, outcome=outcome, meta={"model": "mock"})


# ===================================================================== 0b
def fig_0b(rec, path=None):
    """Task 0b - do the state-space effects survive on an untrained model?

    RECORD
      {"effects": [{"task": "6"|"8"|"9", "label": str,
                    "trained":   {"value": float, "ci": [lo, hi]},
                    "untrained": {"value": float, "ci": [lo, hi]}}, ...],
       "seeds": int,                # how many random inits were averaged
       "meta": {...}}
      Effect sizes are all ratios with 1.0 = no effect and >1 = the effect:
        task 6: median r(crossing) / median r(box)
        task 8: mean d(control) / mean d(embedded)
        task 9: median divergence(far cut) / median divergence(close cut)

    FIGURE  One dumbbell per task, untrained (orange) -> trained (blue), CIs as
      whiskers, a grey line at 1.0.
    PASS  Untrained CIs include 1.0 for every task.
    FAIL  Untrained shows part of the effect - subtract it before interpreting.
    """
    E = rec["effects"]
    fig, ax = plt.subplots(figsize=(6.8, 0.9 + 0.75 * len(E)))
    for i, e in enumerate(E):
        y = len(E) - 1 - i
        for cond, col, dy in (("untrained", ORANGE, 0.12), ("trained", BLUE, -0.12)):
            v, (lo, hi) = e[cond]["value"], e[cond]["ci"]
            ax.plot([lo, hi], [y + dy] * 2, color=col, lw=1.6, zorder=3)
            ax.scatter([v], [y + dy], s=48, color=col, zorder=4, edgecolor="white", lw=0.8)
        ax.text(ax.get_xlim()[0], y, "", va="center")
    ax.axvline(1.0, color=GREY, lw=1.0, ls="--", zorder=1)
    ax.set_yticks(range(len(E))); ax.set_yticklabels([e["label"] for e in E][::-1])
    ax.set_xlabel("effect size (ratio; 1.0 = no effect)")
    ax.set_title("Does a randomly initialised GPT-2 already show the effect?", loc="left")
    ax.scatter([], [], color=ORANGE, label="untrained (%d random inits)" % rec["seeds"])
    ax.scatter([], [], color=BLUE, label="trained")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2); ax.margins(x=0.08, y=0.25)
    _goodness(ax, "untrained at 1.0 = effect comes from training")
    _caption(fig, rec, "Whiskers: 95% CI. Same protocols as Acts 7-8 and the splice test, weights re-initialised.")
    fig.subplots_adjust(left=0.30, bottom=0.34, top=0.85)
    return _finish(fig, rec, path)


def mock_0b(outcome="true", seed=2):
    r = RNG(seed)
    labs = {"6": "6  box vs crossing endpoints", "8": "8  clause return vs control",
            "9": "9  far vs close splice"}
    tr = {"6": 1.59, "8": 1.11, "9": 1.20}
    E = []
    for t, lab in labs.items():
        u = 1.0 + r.normal(0, 0.02) if outcome == "true" else 1.0 + 0.55 * (tr[t] - 1) + r.normal(0, 0.02)
        E.append(dict(task=t, label=lab, trained=dict(value=tr[t], ci=[tr[t] - 0.12, tr[t] + 0.12]),
                      untrained=dict(value=float(u), ci=[float(u - 0.07), float(u + 0.07)])))
    return dict(effects=E, seeds=5, mock=True, outcome=outcome, meta={"model": "mock"})


# ===================================================================== 0c / 1b
def _f1_bars(ax, methods, focal="substitution-induced"):
    names = [m["name"] for m in methods]
    ys = np.arange(len(methods))[::-1]
    for y, m in zip(ys, methods):
        col = BLUE if m["name"] == focal else (PURPLE if "random" in m["name"] else GREY)
        ax.barh(y, m["f1"], color=col, alpha=0.9 if col == BLUE else 0.55, height=0.62)
        if m.get("ci"):
            ax.plot(m["ci"], [y, y], color=TXT, lw=1.2)
        ax.text(m["f1"] + 0.012, y, "%.2f" % m["f1"], va="center", fontsize=S_S)
    rb = [m for m in methods if "right-branching" in m["name"]]
    if rb:
        ax.axvline(rb[0]["f1"], color=GREY, lw=1.0, ls="--", zorder=1)
    ax.set_yticks(ys); ax.set_yticklabels(names); ax.set_xlim(0, 1.0)
    ax.set_xlabel("unlabelled bracket F1 against gold")


def fig_0c(rec, path=None):
    """Task 0c - does substitution beat the trivial right-branching tree?

    RECORD
      {"n_sentences": int, "treebank": str,
       "methods": [{"name": str, "f1": float, "ci": [lo, hi]}, ...],
       "meta": {...}}
      names should include exactly: "substitution-induced", "right-branching",
      "random bracketing"; optionally "published LM induction (Kim 2020)".
      F1 is unlabelled bracket F1 over the same sentences for every method.

    FIGURE  Horizontal bars; right-branching drawn as the dashed line to beat.
    PASS  substitution-induced clears the right-branching line by its CI.
    FAIL  It ties or loses.
    """
    fig, ax = plt.subplots(figsize=(6.6, 0.8 + 0.6 * len(rec["methods"])))
    _f1_bars(ax, rec["methods"])
    ax.set_title("Bracketing quality on %d %s sentences" % (rec["n_sentences"], rec["treebank"]), loc="left")
    _goodness(ax, "higher = closer to the linguist's tree")
    _caption(fig, rec, "Dashed line: right-branching baseline. Whiskers: 95% bootstrap CI over sentences.")
    fig.subplots_adjust(left=0.36, bottom=0.25, top=0.85)
    return _finish(fig, rec, path)


def mock_0c(outcome="true", seed=3):
    sub = 0.58 if outcome == "true" else 0.40
    return dict(n_sentences=500, treebank="UD English-EWT",
                methods=[dict(name="substitution-induced", f1=sub, ci=[sub - 0.03, sub + 0.03]),
                         dict(name="right-branching", f1=0.41, ci=[0.39, 0.43]),
                         dict(name="random bracketing", f1=0.24, ci=[0.22, 0.26]),
                         dict(name="published LM induction (Kim 2020)", f1=0.62, ci=None)],
                mock=True, outcome=outcome, meta={"model": "mock"})


def fig_1b(rec, path=None):
    """Task 1b - does substitution FIND constituents it was never told about?

    RECORD  (superset of the 0c record)
      {"n_sentences": int, "treebank": str,
       "methods": [...as in fig_0c...],
       "by_length": [{"len": int, "n": int, "f1_sub": float, "f1_rb": float}, ...],
       "rank_curve": [{"frac_spans": float, "recall_gold": float}, ...],
       "meta": {...}}
      by_length: F1 per sentence length (in words) for substitution and
        right-branching.  rank_curve: rank ALL spans by cost, accept the
        cheapest fraction x, record the fraction of gold constituents
        recovered - the diagonal is chance.

    FIGURE  A: F1 bars (0c). B: F1 vs sentence length, both methods.
      C: recall-of-gold vs fraction of spans accepted, with the chance diagonal.
    PASS  A clears right-branching; B stays above it at every length;
      C bows well above the diagonal.
    FAIL  A ties right-branching; C hugs the diagonal - the ranking confirms
      boundaries it is given but cannot find them.
    """
    fig = plt.figure(figsize=(11.5, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.42)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))
    _f1_bars(axA, rec["methods"])
    axA.set_title("A  Against gold, all sentences", loc="left")
    L = rec["by_length"]
    x = [l["len"] for l in L]
    axB.plot(x, [l["f1_sub"] for l in L], "-o", color=BLUE, ms=4, label="substitution-induced")
    axB.plot(x, [l["f1_rb"] for l in L], "-o", color=GREY, ms=4, label="right-branching")
    axB.set_xlabel("sentence length (words)"); axB.set_ylabel("unlabelled F1")
    axB.set_title("B  By sentence length", loc="left"); axB.legend(frameon=False)
    axB.set_ylim(0, 1); axB.margins(x=0.05)
    C = rec["rank_curve"]
    axC.plot([c["frac_spans"] for c in C], [c["recall_gold"] for c in C], color=BLUE, lw=2)
    axC.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="chance")
    axC.set_xlabel("fraction of spans accepted (cheapest first)")
    axC.set_ylabel("fraction of gold constituents recovered")
    axC.set_title("C  Does cost rank gold boxes first?", loc="left")
    axC.set_xlim(0, 1); axC.set_ylim(0, 1); axC.legend(frameon=False, loc="lower right")
    _caption(fig, rec, "%d %s sentences; every contiguous span scored with a fixed proform per length class."
             % (rec["n_sentences"], rec["treebank"]))
    fig.subplots_adjust(left=0.13, bottom=0.2, top=0.87, right=0.98)
    return _finish(fig, rec, path)


def mock_1b(outcome="true", seed=4):
    r = RNG(seed)
    base = mock_0c(outcome, seed)
    lens = list(range(5, 31, 3))
    by_len = [dict(len=l, n=int(40 - l), f1_sub=float(np.clip((0.66 if outcome == "true" else 0.44)
              - 0.006 * (l - 5) + r.normal(0, 0.015), 0, 1)),
                   f1_rb=float(np.clip(0.50 - 0.006 * (l - 5) + r.normal(0, 0.015), 0, 1))) for l in lens]
    xs = np.linspace(0, 1, 26)
    pow_ = 0.45 if outcome == "true" else 0.92
    curve = [dict(frac_spans=float(x), recall_gold=float(x ** pow_)) for x in xs]
    base.update(by_length=by_len, rank_curve=curve)
    return base


# ===================================================================== 1a
def fig_1a(rec, path=None):
    """Task 1a - does the substitution separation survive at volume, with a
    replacement policy fixed BLIND to the label?

    RECORD
      {"spans": [{"sentence_id": int, "span": str, "label": "box"|"straddle",
                  "length_class": str, "cost": float}, ...],
       "policy": str,        # e.g. "every 3-4 word span -> 'it'"
       "meta": {...}}

    FIGURE  A: separation (median straddle cost - median box cost) as n grows,
      by subsampling the spans; bootstrap band. B: the two final cost
      distributions, stacked with a shared x (figure-style 6.4).
    PASS  The band excludes zero and flattens as n grows.
    FAIL  The separation shrinks toward zero.
    """
    sp = rec["spans"]
    b = np.array([s["cost"] for s in sp if s["label"] == "box"])
    t = np.array([s["cost"] for s in sp if s["label"] == "straddle"])
    r = RNG(0)
    ns = np.unique(np.geomspace(8, min(len(b), len(t)), 10).astype(int))
    med, lo, hi = [], [], []
    for n in ns:
        bs = [np.median(r.choice(t, n)) - np.median(r.choice(b, n)) for _ in range(400)]
        med.append(np.median(bs)); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
    fig = plt.figure(figsize=(9.6, 3.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.08, wspace=0.35)
    axA = fig.add_subplot(gs[:, 0]); axB1 = fig.add_subplot(gs[0, 1]); axB2 = fig.add_subplot(gs[1, 1], sharex=axB1)
    axA.fill_between(ns, lo, hi, color=BLUE, alpha=0.18, lw=0)
    axA.plot(ns, med, "-o", color=BLUE, ms=4)
    axA.axhline(0, color=GREY, ls="--", lw=1)
    axA.set_xscale("log"); axA.set_xticks(ns[::3]); axA.set_xticklabels([str(n) for n in ns[::3]])
    axA.set_xlabel("spans per group (subsampled)"); axA.set_ylabel("straddle \u2212 box, medians (nats/token)", labelpad=8)
    axA.set_title("A  Separation as the sample grows", loc="left"); axA.margins(x=0.06, y=0.15)
    bins = np.linspace(0, max(b.max(), t.max()) * 1.05, 28)
    axB1.hist(b, bins, color=BLUE, alpha=0.85); axB2.hist(t, bins, color=ORANGE, alpha=0.85)
    axB1.set_title("B  Final distributions", loc="left")
    axB1.text(0.98, 0.85, "real boxes (n=%d)" % len(b), transform=axB1.transAxes, ha="right", color=BLUE)
    axB2.text(0.98, 0.85, "straddles (n=%d)" % len(t), transform=axB2.transAxes, ha="right", color=ORANGE)
    axB2.set_xlabel("cost (nats/token)"); plt.setp(axB1.get_xticklabels(), visible=False)
    for ax in (axB1, axB2):
        ax.set_yticks([])
    _goodness(axA, "band above zero = boxes cheaper")
    _caption(fig, rec, "Replacement policy: %s (blind to the label). %d sentences." % (
        rec["policy"], len({s["sentence_id"] for s in sp})))
    fig.subplots_adjust(left=0.12, bottom=0.2, top=0.87, right=0.98)
    return _finish(fig, rec, path)


def mock_1a(outcome="true", n=300, seed=5):
    r = RNG(seed)
    sp = []
    for i in range(n):
        box = i % 2 == 0
        mu = (0.9 if box else 1.55) if outcome == "true" else (1.15 if box else 1.25)
        sp.append(dict(sentence_id=i // 12, span="span %d" % i, label="box" if box else "straddle",
                       length_class="3-4", cost=float(max(r.gamma(4, mu / 4), 0.02))))
    return dict(spans=sp, policy="every 2-3 word span -> 'it', 4-6 word span -> 'that'",
                mock=True, outcome=outcome, meta={"model": "mock"})


# ===================================================================== 3a
def fig_3a(rec, path=None):
    """Task 3a - is the recovered tree the CORRECT tree?

    RECORD
      {"sentences": [{"id": int, "n_gold": int, "n_pred": int, "n_match": int}, ...],
       "disagreements": [{"type": str, "count": int}, ...],
       "treebank": str, "meta": {...}}
      Per sentence: number of gold brackets, predicted brackets, and brackets
      in both (unlabelled). disagreements: hand-categorised mismatches, e.g.
      "PP attachment", "coordination", "VP boundary", "other".

    FIGURE  A: per-sentence precision vs recall, point size = n_gold, the
      corpus F1 printed. B: what the disagreements are, as bars.
    PASS  Cloud in the upper right; disagreements concentrated in known-
      ambiguous attachment types.
    FAIL  Systematic but different - a cloud off the diagonal with one
      dominant non-attachment disagreement type.
    """
    S = rec["sentences"]
    P = np.array([s["n_match"] / max(s["n_pred"], 1) for s in S])
    R = np.array([s["n_match"] / max(s["n_gold"], 1) for s in S])
    tm, tp, tg = (sum(s[k] for s in S) for k in ("n_match", "n_pred", "n_gold"))
    f1 = 2 * tm / (tp + tg)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 3.9), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.45})
    axA.scatter(R, P, s=[10 + 3 * s["n_gold"] for s in S], color=BLUE, alpha=0.45, edgecolor="none")
    axA.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
    axA.set_xlim(-0.02, 1.02); axA.set_ylim(-0.02, 1.02)
    axA.set_xlabel("recall of gold brackets"); axA.set_ylabel("precision of recovered brackets")
    axA.set_title("A  Per sentence (n=%d): corpus F1 %.2f" % (len(S), f1), loc="left")
    D = sorted(rec["disagreements"], key=lambda d: -d["count"])
    axB.barh(range(len(D))[::-1], [d["count"] for d in D],
             color=[PURPLE if "attach" in d["type"].lower() else GREY for d in D], height=0.6)
    axB.set_yticks(range(len(D))[::-1]); axB.set_yticklabels([d["type"] for d in D])
    axB.set_xlabel("mismatched brackets"); axB.set_title("B  Where the two trees disagree", loc="left")
    axB.margins(x=0.1)
    _caption(fig, rec, "%s gold parses; unlabelled brackets only (substitution yields no category labels). "
             "Purple: attachment ambiguities." % rec["treebank"])
    fig.subplots_adjust(left=0.08, bottom=0.2, top=0.86, right=0.98)
    return _finish(fig, rec, path)


def mock_3a(outcome="true", n=120, seed=6):
    r = RNG(seed)
    S = []
    for i in range(n):
        g = int(r.integers(4, 14)); p = int(max(2, g + r.integers(-2, 3)))
        frac = r.beta(7, 3) if outcome == "true" else r.beta(3, 4)
        S.append(dict(id=i, n_gold=g, n_pred=p, n_match=int(round(min(g, p) * frac))))
    dis = ([("PP attachment", 61), ("coordination", 28), ("VP boundary", 14), ("other", 9)] if outcome == "true"
           else [("subject/verb split", 74), ("PP attachment", 22), ("coordination", 15), ("other", 11)])
    return dict(sentences=S, disagreements=[dict(type=t, count=c) for t, c in dis],
                treebank="UD English-EWT", mock=True, outcome=outcome, meta={"model": "mock"})


# ===================================================================== 4a
def fig_4a(rec, path=None):
    """Task 4a - does agreement survive distance and attractor type?

    RECORD
      {"conditions": [{"name": str, "n": int, "correct": int,
                       "published_acc": float|null}, ...],
       "by_distance": [{"distance": int, "n": int, "correct": int}, ...],
       "meta": {...}}
      conditions: BLiMP / SyntaxGym agreement subsets by name; correct =
      items where the model prefers the grammatical verb. published_acc: the
      benchmark's reported GPT-2 124M accuracy, if any. by_distance: your
      template sweep of words between head and verb.

    FIGURE  A: accuracy per condition with Wilson 95% CI; hollow marker =
      published GPT-2 figure; chance line at 0.5. B: accuracy vs distance.
    PASS  All conditions well above chance; B degrades gracefully.
    FAIL  B falls to chance within a few words.
    """
    C = rec["conditions"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.5})
    for i, c in enumerate(C):
        y = len(C) - 1 - i
        p, lo, hi = _wilson(c["correct"], c["n"])
        axA.plot([lo, hi], [y, y], color=BLUE, lw=1.6); axA.scatter([p], [y], s=44, color=BLUE, zorder=4)
        if c.get("published_acc") is not None:
            axA.scatter([c["published_acc"]], [y], s=60, facecolor="white", edgecolor=TXT, lw=1.2, zorder=5)
    axA.axvline(0.5, color=GREY, ls="--", lw=1)
    axA.set_yticks(range(len(C))); axA.set_yticklabels(["%s (n=%d)" % (c["name"], c["n"]) for c in C][::-1])
    axA.set_xlim(0.3, 1.02); axA.set_xlabel("accuracy (grammatical verb preferred)")
    axA.set_title("A  Per benchmark condition", loc="left")
    axA.scatter([], [], facecolor="white", edgecolor=TXT, label="published GPT-2 124M")
    axA.scatter([], [], color=BLUE, label="this run, Wilson 95% CI")
    axA.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    D = rec["by_distance"]
    xs = [d["distance"] for d in D]
    ps = [_wilson(d["correct"], d["n"]) for d in D]
    axB.fill_between(xs, [p[1] for p in ps], [p[2] for p in ps], color=BLUE, alpha=0.18, lw=0)
    axB.plot(xs, [p[0] for p in ps], "-o", color=BLUE, ms=4)
    axB.axhline(0.5, color=GREY, ls="--", lw=1)
    axB.set_ylim(0.3, 1.02); axB.set_xlabel("words between head noun and verb"); axB.set_ylabel("accuracy")
    axB.set_title("B  Template sweep (n=%d per distance)" % D[0]["n"], loc="left"); axB.margins(x=0.08)
    _goodness(axB, "higher = follows the head")
    _caption(fig, rec, "Dashed line: chance. Benchmarks run as published; the sweep covers distances they lack.")
    fig.subplots_adjust(left=0.3, bottom=0.3, top=0.86, right=0.98)
    return _finish(fig, rec, path)


def mock_4a(outcome="true", seed=7):
    r = RNG(seed)
    names = [("BLiMP: distractor in PP", 1000, 0.90), ("BLiMP: distractor in RC", 1000, 0.84),
             ("SyntaxGym: agreement, PP", 400, 0.88), ("SyntaxGym: agreement, RC", 400, 0.80),
             ("SyntaxGym: reflexive", 400, 0.72)]
    C = [dict(name=n, n=k, correct=int(k * (a + r.normal(0, 0.02))), published_acc=a) for n, k, a in names]
    D = []
    for dist in range(1, 7):
        acc = (0.92 - 0.035 * dist) if outcome == "true" else max(0.5, 0.92 - 0.14 * dist) + 0.01
        D.append(dict(distance=dist, n=200, correct=int(200 * min(acc + r.normal(0, 0.015), 0.99))))
    return dict(conditions=C, by_distance=D, mock=True, outcome=outcome, meta={"model": "mock"})


# ===================================================================== 4b
RULES = ("structural", "first_noun", "nearest_noun")
RULE_COL = {"structural": BLUE, "first_noun": ORANGE, "nearest_noun": PURPLE}
RULE_LAB = {"structural": "structural head", "first_noun": "first noun", "nearest_noun": "nearest noun"}


def fig_4b(rec, path=None):
    """Task 4b - THE Act 5 discriminator: does the verb follow the structural
    head when the head is neither the first nor the nearest noun?

    RECORD
      {"items": [{"construction": str, "sentence": str,
                  "logp": {"structural": float, "first_noun": float,
                           "nearest_noun": float}}, ...],
       "meta": {...}}
      Each item is built so the three candidate rules pick three DIFFERENT
      verb forms (or, for auxiliary fronting, three different auxiliaries);
      logp[rule] = log-probability of the form that rule predicts. Items where
      two rules coincide must be excluded upstream, not zero-filled.
      construction in {"fronted PP", "object relative", "coordinated subject",
      "auxiliary fronting", ...}.

    FIGURE  A: per construction, the share of items won by each rule (argmax
      of logp), stacked; n per construction. B: per item, margin of the
      structural form over the BEST linear rival, as a strip; zero line.
    PASS  Structural wins a clear majority in every construction; margins
      mostly positive.
    FAIL  First-noun wins - Act 5 demonstrated a linear heuristic.
    """
    it = rec["items"]
    cons = sorted({i["construction"] for i in it})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.55})
    for k, cname in enumerate(cons):
        rows = [i for i in it if i["construction"] == cname]
        win = [max(RULES, key=lambda rr: i["logp"][rr]) for i in rows]
        left = 0
        for rule in RULES:
            share = win.count(rule) / len(rows)
            axA.barh(len(cons) - 1 - k, share, left=left, color=RULE_COL[rule], height=0.6,
                     label=RULE_LAB[rule] if k == 0 else None)
            if share >= 0.12:
                axA.text(left + share / 2, len(cons) - 1 - k, "%.0f%%" % (100 * share), ha="center",
                         va="center", color="white", fontsize=S_S, weight="bold")
            left += share
        marg = [i["logp"]["structural"] - max(i["logp"]["first_noun"], i["logp"]["nearest_noun"]) for i in rows]
        _strip(axB, k, marg, BLUE, seed=k)
    axA.set_yticks(range(len(cons)))
    axA.set_yticklabels(["%s (n=%d)" % (c, sum(1 for i in it if i["construction"] == c)) for c in cons][::-1])
    axA.set_xlim(0, 1); axA.set_xlabel("share of items won by each rule")
    axA.set_title("A  Which rule does the model follow?", loc="left")
    axA.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    axB.axhline(0, color=GREY, ls="--", lw=1)
    axB.set_xticks(range(len(cons))); axB.set_xticklabels([c.replace(" ", "\n", 1) for c in cons])
    axB.set_ylabel("log P(structural form) \u2212 log P(best linear form)")
    axB.set_title("B  Margin of the structural form", loc="left"); axB.margins(x=0.15, y=0.1)
    _goodness(axB, "above zero = structural")
    _caption(fig, rec, "Every item separates the three rules' predictions; %d items total." % len(it))
    fig.subplots_adjust(left=0.24, bottom=0.3, top=0.86, right=0.98)
    return _finish(fig, rec, path)


def mock_4b(outcome="true", seed=8):
    r = RNG(seed)
    it = []
    for cname in ("fronted PP", "object relative", "coordinated subject", "auxiliary fronting"):
        for j in range(24):
            if outcome == "true":
                s, f, n = r.normal(-1.0, 0.6), r.normal(-2.6, 0.8), r.normal(-2.9, 0.8)
            else:
                s, f, n = r.normal(-2.4, 0.7), r.normal(-1.1, 0.6), r.normal(-2.8, 0.8)
            it.append(dict(construction=cname, sentence="item %d" % j,
                           logp=dict(structural=float(s), first_noun=float(f), nearest_noun=float(n))))
    return dict(items=it, mock=True, outcome=outcome, meta={"model": "mock"})


# ===================================================================== 5a
def fig_5a(rec, path=None):
    """Task 5a - is the agreement decision readable from the GEOMETRY once the
    comparison is posed correctly?

    RECORD
      {"items": [{"id": str, "head_number": "sg"|"pl",
                  "proj_logit": float,        # (s_item - s_match) . u_number
                  "probe_correct": bool}, ...],   # linear probe on 768-d residual
       "pca_sweep": [{"k": int, "acc": float}, ...],   # secondary analysis
       "n_probe_train": int, "meta": {...}}
      proj_logit: state difference (item minus its length-matched opposite-
      number twin) projected on u_number = mean(are,were,have,do) -
      mean(is,was,has,does) coordinates; positive should mean plural head.
      pca_sweep: agreement-test accuracy after projecting out the top-k
      principal directions of state movement, k = 0 .. K.

    FIGURE  A: projection per item, coloured by head number; sign accuracy
      printed. B: two accuracies with Wilson CI - logit projection sign and
      residual probe - against chance. C: PCA sweep.
    PASS  A separates by sign; B both well above chance.
    FAIL  B at chance with correct targets - a genuine representational null.
    """
    it = rec["items"]
    sg = [i["proj_logit"] for i in it if i["head_number"] == "sg"]
    pl = [i["proj_logit"] for i in it if i["head_number"] == "pl"]
    sign_ok = sum(1 for i in it if (i["proj_logit"] > 0) == (i["head_number"] == "pl"))
    probe_ok = sum(1 for i in it if i["probe_correct"])
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(11.4, 3.8), gridspec_kw={"width_ratios": [1, 0.8, 1], "wspace": 0.5})
    _strip(axA, 0, sg, ORANGE); _strip(axA, 1, pl, BLUE)
    axA.axhline(0, color=GREY, ls="--", lw=1)
    axA.set_xticks([0, 1]); axA.set_xticklabels(["singular head\n(n=%d)" % len(sg), "plural head\n(n=%d)" % len(pl)])
    axA.set_ylabel("projection on the number direction (nats)")
    axA.set_title("A  Sign right on %d of %d" % (sign_ok, len(it)), loc="left"); axA.set_xlim(-0.6, 1.6); axA.margins(y=0.1)
    for x, (k, lab) in enumerate(((sign_ok, "logit-space\nprojection"), (probe_ok, "residual\nlinear probe"))):
        p, lo, hi = _wilson(k, len(it))
        axB.bar(x, p, color=BLUE, width=0.55, alpha=0.9); axB.plot([x, x], [lo, hi], color=TXT, lw=1.3)
    axB.axhline(0.5, color=GREY, ls="--", lw=1); axB.set_ylim(0, 1.02)
    axB.set_xticks([0, 1]); axB.set_xticklabels(["logit-space\nprojection", "residual\nlinear probe"])
    axB.set_ylabel("accuracy"); axB.set_title("B  Two readouts", loc="left")
    P = rec["pca_sweep"]
    axC.plot([p["k"] for p in P], [p["acc"] for p in P], "-o", color=BLUE, ms=4)
    axC.axhline(0.5, color=GREY, ls="--", lw=1); axC.set_ylim(0, 1.02)
    axC.set_xlabel("top-k principal directions removed"); axC.set_ylabel("agreement-test accuracy")
    axC.set_title("C  Secondary: PCA sweep", loc="left"); axC.margins(x=0.06)
    _goodness(axB, "higher = decision is in the geometry")
    _caption(fig, rec, "Targets are length-matched full sentences, not bare prefixes. Probe trained on %d held-out items. "
             "Panel C cannot separate 'signal absent' from 'signal removed with a top component'." % rec["n_probe_train"])
    fig.subplots_adjust(left=0.07, bottom=0.24, top=0.86, right=0.98)
    return _finish(fig, rec, path)


def mock_5a(outcome="true", n=60, seed=9):
    r = RNG(seed)
    it = []
    for i in range(n):
        pl = i % 2 == 1
        mu = (1.4 if pl else -1.4) if outcome == "true" else 0.0
        it.append(dict(id="i%d" % i, head_number="pl" if pl else "sg", proj_logit=float(r.normal(mu, 0.9)),
                       probe_correct=bool(r.random() < (0.9 if outcome == "true" else 0.52))))
    sweep = [dict(k=k, acc=float(np.clip((0.55 + 0.05 * min(k, 6)) if outcome == "true" else 0.5 + r.normal(0, 0.03), 0, 1)))
             for k in range(0, 21, 2)]
    return dict(items=it, pca_sweep=sweep, n_probe_train=200, mock=True, outcome=outcome, meta={"model": "mock"})


# ===================================================================== 6a
def _dumbbell_strata(ax, strata, key, xlabel, title):
    """Per-stratum box-vs-cross medians as dumbbells, pooled row at the bottom."""
    rows = list(strata) + [dict(**{key: "pooled"}, box=sum((s["box"] for s in strata), []),
                                cross=sum((s["cross"] for s in strata), []))]
    for i, s in enumerate(rows):
        y = len(rows) - 1 - i
        mb, mc = np.median(s["box"]), np.median(s["cross"])
        ax.plot([mb, mc], [y, y], color=GREY, lw=1.4, zorder=2)
        ax.scatter([mb], [y], s=46, color=BLUE, zorder=4); ax.scatter([mc], [y], s=46, color=ORANGE, marker="D", zorder=4)
        ax.text(1.005, y, "n = %d / %d" % (len(s["box"]), len(s["cross"])), transform=ax.get_yaxis_transform(),
                va="center", fontsize=S_S, color=GREY)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([str(s[key]) for s in rows][::-1])
    ax.axvline(1.0, color=GREY, ls=":", lw=1)
    ax.set_xlabel(xlabel); ax.set_title(title, loc="left"); ax.margins(x=0.12, y=0.2)
    ax.scatter([], [], color=BLUE, label="real box (median)"); ax.scatter([], [], color=ORANGE, marker="D", label="crossing run (median)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2)


def fig_6a(rec, path=None):
    """Task 6a - does the endpoint-closeness effect hold within EXACT span
    length?

    RECORD
      {"strata": [{"length": int, "box": [float, ...], "cross": [float, ...]}, ...],
       "meta": {...}}
      box / cross: normalised endpoint distances r (Act 7 definition) for every
      real constituent / crossing run of exactly that many words.

    FIGURE  One dumbbell per length: median r for boxes (blue) and crossing
      runs (orange), n per group; pooled row at the bottom; dotted line at
      r = 1 (a typical pair).
    PASS  Box left of crossing in every stratum with adequate n.
    FAIL  The gap appears only where lengths differ - a length artefact.
    """
    fig, ax = plt.subplots(figsize=(7.4, 1.0 + 0.55 * (len(rec["strata"]) + 1)))
    _dumbbell_strata(ax, rec["strata"], "length", "normalised endpoint distance r (1 = typical pair)",
                     "Box endpoints closer than crossing endpoints, at every span length?")
    ax.set_ylabel("span length (words)")
    _goodness(ax, "blue left of orange = effect")
    _caption(fig, rec, "Same-sentence normalisation as Act 7; one-sided Mann-Whitney per stratum recommended.")
    fig.subplots_adjust(left=0.14, bottom=0.3, top=0.85, right=0.84)
    return _finish(fig, rec, path)


def mock_6a(outcome="true", seed=10):
    r = RNG(seed)
    S = []
    for L in (2, 3, 4, 5, 6):
        nb, nc = int(r.integers(8, 20)), int(r.integers(10, 30))
        if outcome == "true":
            b = r.lognormal(np.log(0.72), 0.25, nb); c = r.lognormal(np.log(1.10), 0.25, nc)
        else:
            b = r.lognormal(np.log(0.95 + 0.04 * L), 0.25, nb); c = r.lognormal(np.log(0.95 + 0.04 * L), 0.25, nc)
        S.append(dict(length=L, box=[float(x) for x in b], cross=[float(x) for x in c]))
    return dict(strata=S, mock=True, outcome=outcome, meta={"model": "mock"})


# ===================================================================== 8a
def fig_8a(rec, path=None):
    """Task 8a - is Act 8's 1.11x a real effect size, or an artefact of six
    hand-built items and their last-token confound?

    RECORD
      {"items": [{"id": str, "structure": str, "last_tok_cat": str,
                  "d_emb": float, "d_ctrl": float}, ...],
       "meta": {...}}
      d_emb = L2(embedded state, reference state); d_ctrl = L2(control state,
      reference). last_tok_cat = category of the token immediately before the
      read-off point in the EMBEDDED pass (e.g. "verb", "noun", "relativizer");
      the control is built to end on the same category, so a stratum compares
      like with like. structure = clause type (object relative, subject
      relative, complement clause, ...).

    FIGURE  A: ECDF of per-item ratio d_ctrl/d_emb; mean-of-means ratio with
      paired-bootstrap CI printed; 1.0 line. B: ratio per last-token category
      with CI. C: leave-one-out spread of the mean ratio.
    PASS  CI excludes 1.0 overall AND within every category; C is narrow.
    FAIL  CI includes 1.0, or the effect lives in one category.
    """
    it = rec["items"]
    de = np.array([i["d_emb"] for i in it]); dc = np.array([i["d_ctrl"] for i in it])
    ratio, lo, hi = _ratio_ci(de, dc)
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(11.6, 3.8), gridspec_kw={"width_ratios": [1, 1.1, 0.8], "wspace": 0.5})
    _ecdf(axA, dc / de, BLUE, "per item")
    axA.axvline(1.0, color=GREY, ls="--", lw=1); axA.axvspan(lo, hi, color=BLUE, alpha=0.15, lw=0)
    axA.axvline(ratio, color=BLUE, lw=1.4)
    axA.set_xlabel("d(control) / d(embedded), per item"); axA.set_ylabel("fraction of items")
    axA.set_title("A  Mean ratio %.3f\u00d7 [%.3f, %.3f], n=%d" % (ratio, lo, hi, len(it)), loc="left"); axA.margins(x=0.05)
    cats = sorted({i["last_tok_cat"] for i in it})
    for k, cname in enumerate(cats):
        sub = [i for i in it if i["last_tok_cat"] == cname]
        rr, l2, h2 = _ratio_ci([i["d_emb"] for i in sub], [i["d_ctrl"] for i in sub], seed=k)
        y = len(cats) - 1 - k
        axB.plot([l2, h2], [y, y], color=BLUE, lw=1.6); axB.scatter([rr], [y], s=46, color=BLUE, zorder=4)
        axB.text(1.005, y, "n = %d" % len(sub), transform=axB.get_yaxis_transform(), va="center", fontsize=S_S, color=GREY)
    axB.axvline(1.0, color=GREY, ls="--", lw=1)
    axB.set_yticks(range(len(cats))); axB.set_yticklabels(cats[::-1]); axB.set_xlabel("mean ratio within category")
    axB.set_title("B  By last token before read-off", loc="left"); axB.margins(x=0.15, y=0.3)
    loo = [dc[np.arange(len(it)) != j].mean() / de[np.arange(len(it)) != j].mean() for j in range(len(it))]
    axC.hist(loo, bins=18, color=BLUE, alpha=0.85); axC.axvline(1.0, color=GREY, ls="--", lw=1)
    axC.set_xlabel("mean ratio, one item left out"); axC.set_yticks([])
    axC.set_title("C  Leave-one-out spread %.3f\u2013%.3f" % (min(loo), max(loo)), loc="left"); axC.margins(x=0.1)
    _goodness(axA, "right of 1.0 = clause return")
    _caption(fig, rec, "Reference = bare sentence; items vary clause structure, not nouns. Shaded: paired-bootstrap 95% CI of the mean ratio.")
    fig.subplots_adjust(left=0.06, bottom=0.2, top=0.86, right=0.97)
    return _finish(fig, rec, path)


def mock_8a(outcome="true", n=120, seed=11):
    r = RNG(seed)
    cats = ["noun", "verb", "relativizer", "adjective"]
    structs = ["object relative", "subject relative", "complement clause", "adverbial clause"]
    it = []
    for i in range(n):
        de = r.lognormal(np.log(450), 0.25)
        mult = (1.12 + r.normal(0, 0.08)) if outcome == "true" else (1.0 + r.normal(0, 0.08) + (0.25 if i % 4 == 2 else 0))
        it.append(dict(id="i%d" % i, structure=structs[i % 4], last_tok_cat=cats[i % 4],
                       d_emb=float(de), d_ctrl=float(de * mult)))
    return dict(items=it, mock=True, outcome=outcome, meta={"model": "mock"})


# ===================================================================== 8b
def fig_8b(rec, path=None):
    """Task 8b - does the return effect compose with embedding depth?

    RECORD
      {"depths": [{"depth": int, "items": [{"d_emb": float, "d_ctrl": float}, ...]}, ...],
       "meta": {...}}
      One entry per depth (1, 2, 3); controls are token-count matched at each
      depth. Depth 3 is reported but not interpreted (unprocessable for humans
      too).

    FIGURE  Mean ratio with paired-bootstrap CI vs depth; 1.0 line; depth 3
      drawn on a hatched background labelled "reported, not interpreted".
    PASS  Depth 2 CI excludes 1.0 - the operation composes at least once.
    FAIL  Effect vanishes past depth 1.
    """
    D = sorted(rec["depths"], key=lambda d: d["depth"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for d in D:
        de = [i["d_emb"] for i in d["items"]]; dc = [i["d_ctrl"] for i in d["items"]]
        rr, lo, hi = _ratio_ci(de, dc, seed=d["depth"])
        ax.plot([d["depth"]] * 2, [lo, hi], color=BLUE, lw=1.8); ax.scatter([d["depth"]], [rr], s=60, color=BLUE, zorder=4)
        ax.text(d["depth"], hi, "n=%d" % len(de), ha="center", va="bottom", fontsize=S_S, color=GREY)
    ax.axhline(1.0, color=GREY, ls="--", lw=1)
    if any(d["depth"] >= 3 for d in D):
        ax.axvspan(2.5, max(d["depth"] for d in D) + 0.5, color=GREY, alpha=0.12, hatch="//", lw=0)
        ax.text(3, ax.get_ylim()[0], "reported, not interpreted", ha="center", va="bottom", fontsize=S_S, color=GREY)
    ax.set_xticks([d["depth"] for d in D]); ax.set_xlabel("embedding depth")
    ax.set_ylabel("d(control) / d(embedded), mean ratio"); ax.set_title("Does one clause boundary compose into two?", loc="left")
    ax.margins(x=0.2, y=0.25)
    _goodness(ax, "above 1.0 = return")
    _caption(fig, rec, "Paired-bootstrap 95% CI. The test of composition is depth 1 -> 2; depth-3 centre embedding exceeds human processing.")
    fig.subplots_adjust(left=0.14, bottom=0.2, top=0.86)
    return _finish(fig, rec, path)


def mock_8b(outcome="true", seed=12):
    r = RNG(seed)
    D = []
    for depth, mult in ((1, 1.12), (2, 1.10 if outcome == "true" else 1.0), (3, 1.02)):
        items = [dict(d_emb=float(x), d_ctrl=float(x * (mult + r.normal(0, 0.07)))) for x in r.lognormal(np.log(450), 0.25, 40)]
        D.append(dict(depth=depth, items=items))
    return dict(depths=D, mock=True, outcome=outcome, meta={"model": "mock"})


# ===================================================================== 9a (covers task 9's control)
def fig_9a(rec, path=None):
    """Task 9 with its control, and 9a at power - omittability by splicing.

    RECORD
      {"cuts": [{"pair": "close"|"far", "grammatical": bool, "seg_len": int,
                 "divergence": float}, ...],
       "divergence_measure": str, "window": int, "meta": {...}}
      One row per deletion: pair = whether the two endpoint states were close
      or far; grammatical = whether the spliced text is grammatical (judged
      BEFORE looking at divergence); seg_len = tokens deleted; divergence =
      the pre-registered downstream measure over `window` aligned positions.
      Only grammatical cuts enter the comparison (Task 9's control); the
      ungrammatical ones are drawn hollow and never summarised.

    FIGURE  A: ECDF of divergence for close vs far GRAMMATICAL cuts; medians,
      ratio and one-sided Mann-Whitney p printed. B: far/close median ratio by
      segment-length stratum with n.
    PASS  Far cuts diverge more at p < 0.05 once grammaticality is matched.
    FAIL  Ratio near 1.0 with matched grammaticality.
    """
    cuts = rec["cuts"]
    g = [c for c in cuts if c["grammatical"]]
    cl = np.array([c["divergence"] for c in g if c["pair"] == "close"])
    fa = np.array([c["divergence"] for c in g if c["pair"] == "far"])
    p = stats.mannwhitneyu(fa, cl, alternative="greater").pvalue
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.55})
    _ecdf(axA, cl, BLUE, "close pair, grammatical cut (n=%d)" % len(cl))
    _ecdf(axA, fa, ORANGE, "far pair, grammatical cut (n=%d)" % len(fa))
    ung = [c["divergence"] for c in cuts if not c["grammatical"]]
    if ung:
        axA.scatter(ung, np.full(len(ung), 0.02), s=14, facecolor="white", edgecolor=GREY, lw=0.8,
                    label="ungrammatical splice, excluded (n=%d)" % len(ung))
    axA.set_xlabel("downstream divergence over %d positions" % rec["window"])
    axA.set_ylabel("fraction of cuts"); axA.legend(frameon=False, loc="lower right"); axA.set_ylim(-0.03, 1.03)
    axA.set_title("A  Medians %.1f vs %.1f = %.2f\u00d7, one-sided p = %.3f" % (
        np.median(cl), np.median(fa), np.median(fa) / np.median(cl), p), loc="left"); axA.margins(x=0.05)
    lens = sorted({c["seg_len"] for c in g})
    edges = np.quantile(lens, [0, 0.33, 0.66, 1.0]).astype(int)
    for k in range(3):
        lo_, hi_ = edges[k], edges[k + 1]
        sub = [c for c in g if lo_ <= c["seg_len"] <= hi_]
        c_ = [c["divergence"] for c in sub if c["pair"] == "close"]; f_ = [c["divergence"] for c in sub if c["pair"] == "far"]
        if len(c_) > 2 and len(f_) > 2:
            rr = np.median(f_) / np.median(c_)
            bs = [np.median(RNG(k).choice(f_, len(f_))) / np.median(RNG(k + 7).choice(c_, len(c_))) for _ in range(300)]
            axB.plot([np.percentile(bs, 2.5), np.percentile(bs, 97.5)], [k, k], color=ORANGE, lw=1.6)
            axB.scatter([rr], [k], s=46, color=ORANGE, zorder=4)
            axB.text(1.005, k, "n = %d / %d" % (len(c_), len(f_)), transform=axB.get_yaxis_transform(),
                     va="center", fontsize=S_S, color=GREY)
    axB.axvline(1.0, color=GREY, ls="--", lw=1)
    axB.set_yticks(range(3)); axB.set_yticklabels(["%d\u2013%d tokens" % (edges[k], edges[k + 1]) for k in range(3)])
    axB.set_xlabel("far / close median divergence"); axB.set_title("B  By deleted-segment length", loc="left")
    axB.margins(x=0.15, y=0.3)
    _goodness(axA, "orange right of blue = close pairs are skippable")
    _caption(fig, rec, "Divergence: %s. Grammaticality judged before divergence was computed; ungrammatical splices never enter a summary." % rec["divergence_measure"])
    fig.subplots_adjust(left=0.08, bottom=0.2, top=0.86, right=0.9)
    return _finish(fig, rec, path)


def mock_9a(outcome="true", n=240, seed=13):
    r = RNG(seed)
    cuts = []
    for i in range(n):
        far = i % 2 == 1
        gram = r.random() < (0.9 if not far else 0.55)
        base = 150 if not far else (185 if outcome == "true" else 152)
        div = r.lognormal(np.log(base * (1.35 if not gram else 1.0)), 0.3)
        cuts.append(dict(pair="far" if far else "close", grammatical=bool(gram),
                         seg_len=int(r.integers(4, 40)), divergence=float(div)))
    return dict(cuts=cuts, divergence_measure="mean L2 of log-prob states", window=12,
                mock=True, outcome=outcome, meta={"model": "mock"})


# ===================================================================== 9b
def fig_9b(rec, path=None):
    """Task 9b - is the GENERATED TEXT preserved, not merely the state?

    RECORD
      {"splices": [{"pair": "close"|"far", "state_div": float,
                    "text_overlap": float}, ...],
       "overlap_measure": str, "decoding": str, "meta": {...}}
      For each grammatical splice: state_div as in 9a; text_overlap between
      continuations generated from the original and the spliced context under
      a decoding protocol fixed in advance (e.g. token F1 over 30 tokens, or
      shared content-word fraction), in [0, 1].

    FIGURE  A: text overlap for close vs far splices, strips with medians.
      B: state divergence vs text overlap, Spearman rho printed - does the
      proxy track the thing the paper's claim is about?
    PASS  A: close > far; B: clear negative relation.
    FAIL  B flat - state divergence measures the wrong thing.
    """
    S = rec["splices"]
    cl = [s["text_overlap"] for s in S if s["pair"] == "close"]; fa = [s["text_overlap"] for s in S if s["pair"] == "far"]
    rho = stats.spearmanr([s["state_div"] for s in S], [s["text_overlap"] for s in S]).correlation
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 3.8), gridspec_kw={"width_ratios": [0.8, 1.2], "wspace": 0.4})
    _strip(axA, 0, cl, BLUE); _strip(axA, 1, fa, ORANGE)
    axA.set_xticks([0, 1]); axA.set_xticklabels(["close pair\n(n=%d)" % len(cl), "far pair\n(n=%d)" % len(fa)])
    axA.set_ylabel("continuation overlap (%s)" % rec["overlap_measure"]); axA.set_xlim(-0.6, 1.6); axA.set_ylim(-0.02, 1.02)
    axA.set_title("A  Text after the splice", loc="left")
    for pair, col in (("close", BLUE), ("far", ORANGE)):
        sub = [s for s in S if s["pair"] == pair]
        axB.scatter([s["state_div"] for s in sub], [s["text_overlap"] for s in sub], s=16, color=col, alpha=0.6, edgecolor="none", label=pair + " pair")
    axB.set_xlabel("state divergence (the proxy)"); axB.set_ylabel("continuation overlap (the claim)")
    axB.set_title("B  Does the proxy track the claim?  Spearman \u03c1 = %.2f" % rho, loc="left")
    axB.legend(frameon=False, loc="upper right"); axB.set_ylim(-0.02, 1.02); axB.margins(x=0.05)
    _goodness(axA, "higher = text preserved")
    _caption(fig, rec, "Decoding: %s, fixed before the run. Grammatical splices only." % rec["decoding"])
    fig.subplots_adjust(left=0.1, bottom=0.2, top=0.86, right=0.98)
    return _finish(fig, rec, path)


def mock_9b(outcome="true", n=160, seed=14):
    r = RNG(seed)
    S = []
    for i in range(n):
        far = i % 2 == 1
        sd = r.lognormal(np.log(185 if far else 150), 0.3)
        if outcome == "true":
            ov = np.clip(0.95 - 0.0028 * sd + r.normal(0, 0.08), 0, 1)
        else:
            ov = np.clip(0.45 + r.normal(0, 0.12), 0, 1)
        S.append(dict(pair="far" if far else "close", state_div=float(sd), text_overlap=float(ov)))
    return dict(splices=S, overlap_measure="token F1, 30 tokens", decoding="greedy, 30 tokens",
                mock=True, outcome=outcome, meta={"model": "mock"})


# ===================================================================== 10a
LEVELS = ["NP", "PP", "VP", "SBAR", "S"]


def fig_10a(rec, path=None):
    """Task 10a - is there a graded HIERARCHY of scales, or two unrelated facts?

    RECORD
      {"levels": [{"level": "NP"|"PP"|"VP"|"SBAR"|"S", "box": [float, ...],
                   "cross": [float, ...]}, ...],
       "meta": {...}}
      box / cross: normalised endpoint distances r for gold constituents of
      that category and for length-matched crossing runs at the same scale.
      Order is syntactic level, small to large: NP, PP, VP, SBAR, S.

    FIGURE  A: dumbbells per level (box vs cross medians, n). B: the effect
      size (cross / box median ratio) per level with bootstrap CI, joined by a
      line - is it monotone?
    PASS  Effect at every level, ordered by level.
    FAIL  Effect at one level only, or no ordering.
    """
    Lv = sorted(rec["levels"], key=lambda l: LEVELS.index(l["level"]))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.5})
    _dumbbell_strata(axA, Lv, "level", "normalised endpoint distance r", "A  Box vs crossing endpoints, by syntactic level")
    axA.set_ylabel("constituent type (small \u2192 large)")
    ratios = []
    for k, l in enumerate(Lv):
        bs = [np.median(RNG(k).choice(l["cross"], len(l["cross"]))) / np.median(RNG(k + 3).choice(l["box"], len(l["box"]))) for _ in range(400)]
        rr = np.median(l["cross"]) / np.median(l["box"]); ratios.append(rr)
        axB.plot([k, k], [np.percentile(bs, 2.5), np.percentile(bs, 97.5)], color=BLUE, lw=1.6)
    axB.plot(range(len(Lv)), ratios, "-o", color=BLUE, ms=5)
    axB.axhline(1.0, color=GREY, ls="--", lw=1)
    axB.set_xticks(range(len(Lv))); axB.set_xticklabels([l["level"] for l in Lv])
    axB.set_ylabel("crossing / box median ratio"); mono = all(np.diff(ratios) >= 0) or all(np.diff(ratios) <= 0)
    axB.set_title("B  Effect size by level: %s" % ("monotone" if mono else "not monotone"), loc="left"); axB.margins(x=0.1, y=0.2)
    _goodness(axB, "above 1.0 = effect at that scale")
    _caption(fig, rec, "Gold categories from a treebank; crossing runs length-matched within each level.")
    fig.subplots_adjust(left=0.1, bottom=0.3, top=0.86, right=0.98)
    return _finish(fig, rec, path)


def mock_10a(outcome="true", seed=15):
    r = RNG(seed)
    Lv = []
    for k, lev in enumerate(LEVELS):
        n = int(r.integers(15, 40))
        if outcome == "true":
            bm = 0.80 - 0.06 * k; cm = 1.10
        else:
            bm = 0.72 if lev in ("NP", "S") else 1.05; cm = 1.10
        Lv.append(dict(level=lev, box=[float(x) for x in r.lognormal(np.log(bm), 0.22, n)],
                       cross=[float(x) for x in r.lognormal(np.log(cm), 0.22, n + 5)]))
    return dict(levels=Lv, mock=True, outcome=outcome, meta={"model": "mock"})


# ===================================================================== 10b
def fig_10b(rec, path=None):
    """Task 10b - does the alignment sharpen with model scale?

    RECORD
      {"checkpoints": [{"name": str, "params": int,
                        "effects": {"6": {"value": float, "ci": [lo, hi]},
                                    "8": {...}, "9": {...}}}, ...],
       "reference": {"name": "GPT-2 124M", "params": 124000000, "effects": {...}},
       "meta": {...}}
      Effect ratios exactly as in fig_0b (1.0 = none). checkpoints = the
      Pythia suite, 70M -> 12B; reference = this document's model, drawn as a
      hollow marker.

    FIGURE  Three abutting panels sharing a log-x of parameter count, one per
      task, effect ratio with CI; 1.0 line; GPT-2 124M as a hollow marker.
    PASS  Effect sizes rise with scale.
    FAIL  Flat or falling - Act 8's weakness is not a small-model artefact.
    """
    C = sorted(rec["checkpoints"], key=lambda c: c["params"])
    labs = {"6": "6  box vs crossing", "8": "8  clause return", "9": "9  splice divergence"}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True, gridspec_kw={"wspace": 0.3})
    for ax, t in zip(axes, ("6", "8", "9")):
        x = [c["params"] for c in C]; v = [c["effects"][t]["value"] for c in C]
        lo = [c["effects"][t]["ci"][0] for c in C]; hi = [c["effects"][t]["ci"][1] for c in C]
        ax.fill_between(x, lo, hi, color=BLUE, alpha=0.15, lw=0); ax.plot(x, v, "-o", color=BLUE, ms=4, label="Pythia suite")
        rf = rec["reference"]
        ax.scatter([rf["params"]], [rf["effects"][t]["value"]], s=64, facecolor="white", edgecolor=TXT, lw=1.2, zorder=5, label=rf["name"])
        ax.axhline(1.0, color=GREY, ls="--", lw=1); ax.set_xscale("log")
        ax.set_title(labs[t], loc="left"); ax.margins(x=0.1, y=0.2)
        ax.set_xticks([1e8, 1e9, 1e10]); ax.set_xticklabels(["100M", "1B", "10B"])
    axes[0].set_ylabel("effect ratio (1.0 = none)"); axes[1].set_xlabel("parameters")
    axes[0].legend(frameon=False, loc="upper left")
    _goodness(axes[2], "rising = sharpens with scale")
    _caption(fig, rec, "Same protocols as Acts 7-8 and the splice test; Pythia is the paper's own model family (their Figure 1 is 12B).")
    fig.subplots_adjust(left=0.07, bottom=0.22, top=0.86, right=0.98)
    return _finish(fig, rec, path)


def mock_10b(outcome="true", seed=16):
    r = RNG(seed)
    sizes = [("70M", 70e6), ("160M", 160e6), ("410M", 410e6), ("1B", 1e9), ("2.8B", 2.8e9), ("6.9B", 6.9e9), ("12B", 12e9)]
    base = {"6": 1.55, "8": 1.10, "9": 1.18}
    C = []
    for k, (n, p) in enumerate(sizes):
        eff = {}
        for t in base:
            v = base[t] + (0.06 * k if outcome == "true" else r.normal(0, 0.02)) * (1 if t != "8" else 0.7)
            eff[t] = dict(value=float(v), ci=[float(v - 0.08), float(v + 0.08)])
        C.append(dict(name="Pythia-" + n, params=int(p), effects=eff))
    ref = dict(name="GPT-2 124M", params=124_000_000,
               effects={t: dict(value=v, ci=[v - 0.1, v + 0.1]) for t, v in (("6", 1.59), ("8", 1.11), ("9", 1.20))})
    return dict(checkpoints=C, reference=ref, mock=True, outcome=outcome, meta={"model": "mock"})


# ===================================================================== 10c
def fig_10c(rec, path=None):
    """Task 10c - are sentence-final periods close because of the SENTENCE
    between them, or because every period predicts a sentence start?

    RECORD
      {"pairs": [{"kind": "within"|"cross"|"arbitrary", "r": float}, ...],
       "meta": {...}}
      within = two sentence-final periods in the SAME paragraph; cross = two
      sentence-final periods from UNRELATED paragraphs; arbitrary = any two
      non-adjacent positions. r = normalised distance on one shared scale.

    FIGURE  Three strips with medians and n; the arbitrary-pair median is the
      grey reference.
    PASS  within markedly closer than cross.
    FAIL  cross as close as within - a punctuation effect; drop the
      sentence-scale point from Act 7 and Task 10.
    """
    P = rec["pairs"]
    groups = [("within", "same paragraph\nperiod pairs", BLUE), ("cross", "unrelated paragraphs\nperiod pairs", ORANGE),
              ("arbitrary", "arbitrary\nnon-adjacent pairs", PURPLE)]
    fig, ax = plt.subplots(figsize=(7, 3.9))
    for x, (kind, lab, col) in enumerate(groups):
        v = [p["r"] for p in P if p["kind"] == kind]
        _strip(ax, x, v, col, seed=x, s=12, alpha=0.55)
        ax.text(x, -0.02, "n = %d" % len(v), ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=S_S, color=GREY)
    ax.set_xticks(range(3)); ax.set_xticklabels([g[1] for g in groups]); ax.tick_params(axis="x", pad=16)
    ax.set_ylabel("normalised distance r"); ax.set_xlim(-0.6, 2.6); ax.margins(y=0.08)
    w = np.median([p["r"] for p in P if p["kind"] == "within"]); c = np.median([p["r"] for p in P if p["kind"] == "cross"])
    ax.set_title("Period pairs: same paragraph %.2f vs unrelated %.2f (medians)" % (w, c), loc="left")
    _goodness(ax, "lower = closer")
    _caption(fig, rec, "One normalisation scale for all three groups (median arbitrary-pair distance).")
    fig.subplots_adjust(left=0.12, bottom=0.28, top=0.86)
    return _finish(fig, rec, path)


def mock_10c(outcome="true", seed=17):
    r = RNG(seed)
    P = [dict(kind="within", r=float(x)) for x in r.lognormal(np.log(0.22), 0.3, 40)]
    P += [dict(kind="cross", r=float(x)) for x in r.lognormal(np.log(0.75 if outcome == "true" else 0.24), 0.3, 60)]
    P += [dict(kind="arbitrary", r=float(x)) for x in r.lognormal(np.log(1.0), 0.35, 200)]
    return dict(pairs=P, mock=True, outcome=outcome, meta={"model": "mock"})


# ================================================================ registry
TASKS = ["0a", "0b", "0c", "1a", "1b", "3a", "4a", "4b", "5a", "6a", "8a", "8b", "9a", "9b", "10a", "10b", "10c"]
FIG = {t: globals()["fig_" + t] for t in TASKS}
MOCK = {t: globals()["mock_" + t] for t in TASKS}


def render_mock_pair(task, outdir="."):
    """Render the 'if true' and 'if false' mock figures for one task and stack
    them vertically, each under a readable header, into fig_mock_<task>.png."""
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import font_manager
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 34)
    paths = []
    for outcome in ("true", "false"):
        p = os.path.join(outdir, "_mock_%s_%s.png" % (task, outcome))
        fig = FIG[task](MOCK[task](outcome), path=p); plt.close(fig); paths.append(p)
    ims = [Image.open(p).convert("RGB") for p in paths]
    w = max(im.width for im in ims)
    ims = [im.resize((w, int(im.height * w / im.width))) for im in ims]
    head, gap = 70, 30
    out = Image.new("RGB", (w, sum(im.height for im in ims) + 2 * head + gap), "white")
    dr = ImageDraw.Draw(out)
    y = 0
    for im, lab, col in zip(ims, ("IF TRUE  \u2014  mock data shaped like the pass prediction",
                                  "IF FALSE  \u2014  mock data shaped like the fail prediction"),
                            ((43, 108, 163), (217, 130, 43))):
        dr.rectangle([0, y, w, y + head], fill=(245, 246, 247))
        dr.text((24, y + 16), "Task %s   %s" % (task, lab), fill=col, font=font)
        out.paste(im, (0, y + head)); y += head + im.height + gap
    final = os.path.join(outdir, "fig_mock_%s.png" % task)
    out.save(final, optimize=True)
    return final


if __name__ == "__main__":
    for t in TASKS:
        print(render_mock_pair(t))
