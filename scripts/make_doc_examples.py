#!/usr/bin/env python
"""Generate the illustrative examples the docs site shows, from the repo's own fixtures.

Everything under ``docs/assets/examples/`` is produced here and committed, so the
GitHub Pages build only needs ``mkdocs-material`` (no torch). Nothing is a real
result: the numbers come from ``FakeSpanScorer`` (a scorer that knows the gold
tree), from synthetic Task 9a rows, from ``mock_<task>`` data, or from the tiny
random GPT-2 that backs the test suite. The *shapes* -- cache rows, records, the
audit sheet, the figures -- are the real ones.

    python scripts/make_doc_examples.py            # torch parts run when torch imports
    python scripts/make_doc_examples.py --no-torch # skip the tiny-model artifacts
    python scripts/make_doc_examples.py --check    # torch-free text fragments unchanged? (CI)

Writes only .json / .jsonl / .csv / .txt / .png: a ``.md`` here would become a site
page, and a ``.py`` would trip ``tests/test_architecture_doc.py``.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT_DIR = REPO / "docs" / "assets" / "examples"
ROUND = 4
MAX_LIST = 4
#: Header / meta fields that change on every run and carry no information for a reader.
VOLATILE = {"date", "library_versions", "device", "device_info", "seconds", "source_record",
            "source_cache", "elapsed_s", "run_id", "created_at"}
#: The torch-free text fragments `--check` compares byte for byte.
TEXT_FRAGMENTS = (
    "theory_example.json", "spans_basics.txt", "substitute.txt", "detokenize.txt", "induce_trace.txt",
    "span_costs_header.json", "span_costs_row.json", "record_1b.json", "verdict_1b.txt",
    "cuts_by_hand.txt", "record_9a_synthetic.json", "verdict_9a.txt",
)


# ------------------------------------------------------------------ helpers


def _round(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, ROUND)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v) for v in x]
    return x


def _trim(x: Any, max_list: int = MAX_LIST) -> Any:
    """Shorten long lists for display: the first `max_list` items, then a literal "..."."""
    if isinstance(x, dict):
        return {k: _trim(v, max_list) for k, v in x.items()}
    if isinstance(x, list):
        items = [_trim(v, max_list) for v in x[:max_list]]
        return items + ["..."] if len(x) > max_list else items
    return x


def _scrub(x: Any) -> Any:
    """Replace run-dependent fields with placeholders so the output is reproducible."""
    if isinstance(x, dict):
        return {k: ("<varies per run>" if k in VOLATILE else _scrub(v)) for k, v in x.items()}
    if isinstance(x, list):
        return [_scrub(v) for v in x]
    return x


def display(obj: Any, *, trim: bool = True) -> Any:
    obj = _scrub(_round(obj))
    return _trim(obj) if trim else obj


def write_json(out: Path, name: str, obj: Any, *, trim: bool = True) -> None:
    (out / name).write_text(json.dumps(display(obj, trim=trim), indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")


def write_text(out: Path, name: str, lines: list[str]) -> None:
    (out / name).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def fig(out: Path, name: str, draw, record: dict) -> None:
    """Render one figure with the mock watermark on."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    record = dict(record)
    record["mock"] = True
    plt.close(draw(record, path=str(out / name)))


# ------------------------------------------------------------ Task 1b (no torch)


def task_1b_examples(out: Path) -> None:
    from m1_analyzer.experiments import (
        MinOverSet, compute_span_costs, detokenize_ptb, hand_examples, run_task_1b, substitute,
        theory_example, verdict, experiment_figures as EF,
    )
    from m1_analyzer.experiments.jsonl_cache import read_jsonl
    from m1_analyzer.experiments.spans import (
        bracket_prf, crosses, enumerate_spans, greedy_induce, left_branching, right_branching, spans_to_brackets,
    )
    from m1_analyzer.experiments.treebank import CONVENTIONS, TreebankSentence
    from m1_analyzer.testing import FakeSpanScorer

    s = theory_example()
    gold = sorted(s.gold_spans)
    write_json(out, "theory_example.json", {
        "id": s.id, "source": s.source, "words": s.words, "text": detokenize_ptb(s.words),
        "gold_spans": [list(g) for g in gold], "gold_brackets": spans_to_brackets(s.words, s.gold_spans),
        "conventions": CONVENTIONS,
    }, trim=False)

    n = s.n
    spans = enumerate_spans(n)
    write_text(out, "spans_basics.txt", [
        f"sentence: {' '.join(s.words)}   (n = {n} words)",
        f"enumerate_spans({n}) -> {len(spans)} spans, first five: {spans[:5]}",
        f"crosses((0, 2), (1, 4)) -> {crosses((0, 2), (1, 4))}    crosses((0, 2), (3, 6)) -> {crosses((0, 2), (3, 6))}",
        f"right_branching(6) -> {sorted(right_branching(6))}",
        f"left_branching(6)  -> {sorted(left_branching(6))}",
        f"gold spans of the theory sentence: {[tuple(g) for g in gold]}",
        f"as brackets: {spans_to_brackets(s.words, s.gold_spans)}",
    ])

    def sub(i: int, j: int, proform: str) -> list[str]:
        words = substitute(s.words, i, j, proform)
        return [f"substitute(words, {i}, {j}, {proform!r})",
                f"  words -> {words}",
                f"  text  -> {detokenize_ptb(words)}"]

    lines = [f"original: {detokenize_ptb(s.words)}", ""]
    for i, j, p in [(3, 6, "it"), (0, 2, "it"), (8, 12, "did"), (9, 12, "do so"), (7, 12, "<del>"), (4, 6, "blorp")]:
        lines += sub(i, j, p) + [""]
    lines.append("cost(i, j) = mean log-prob per token of the original - that of the variant; the")
    lines.append("policy (min over the real proforms) picks the cheapest proform; controls are scored but never chosen.")
    write_text(out, "substitute.txt", lines)

    write_text(out, "detokenize.txt", [
        "PTB tokens -> text the model reads (detokenize_ptb):",
        "  ['The', 'company', 'does', \"n't\", 'own', '5', '%', '-LRB-', 'yet', '-RRB-', '.']",
        "  -> " + detokenize_ptb(["The", "company", "does", "n't", "own", "5", "%", "-LRB-", "yet", "-RRB-", "."]),
    ])

    policy = MinOverSet()
    scorer = FakeSpanScorer([s])
    [table] = compute_span_costs(scorer, [s], policy, show_progress=False)
    costs = table.costs_for(policy)
    kept, visited = greedy_induce(costs, n, trace=True)
    lines = ["Greedy induction on the theory sentence, costs from FakeSpanScorer (gold cheap, crossing dear):", "",
             f"{'span':>9}  {'cost':>7}  decision"]
    for span, cost, blocker in visited[:16]:
        decision = "kept" if blocker is None else f"skipped: crosses {blocker}"
        lines.append(f"{str(span):>9}  {cost:7.3f}  {decision}")
    if len(visited) > 16:
        lines.append(f"... {len(visited) - 16} more spans visited")
    p, r, f1, _ = bracket_prf(kept, s.gold_spans)
    _, _, f1_rb, _ = bracket_prf(right_branching(n), s.gold_spans)
    _, _, f1_lb, _ = bracket_prf(left_branching(n), s.gold_spans)
    lines += ["", f"kept ({len(kept)} spans): {sorted(kept)}",
              f"as brackets: {spans_to_brackets(s.words, kept)}",
              f"gold:        {spans_to_brackets(s.words, s.gold_spans)}",
              f"bracket P/R/F1 vs gold = {p:.3f} / {r:.3f} / {f1:.3f}   right-branching F1 = {f1_rb:.3f}   left-branching F1 = {f1_lb:.3f}"]
    write_text(out, "induce_trace.txt", lines)

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "span_costs.jsonl"
        sentences = hand_examples()
        with_controls = MinOverSet(controls=["blorp", "<del>"])
        compute_span_costs(FakeSpanScorer(sentences, proforms=with_controls.proforms), sentences, with_controls,
                           cache_path=cache, provenance={"model_id": "FakeSpanScorer (illustrative)"},
                           show_progress=False)
        header, rows, _ = read_jsonl(cache)
    write_json(out, "span_costs_header.json", header, trim=False)
    row = dict(rows[0])
    row["spans"] = {p: dict(list(t.items())[:3] + [("...", f"{len(t) - 3} more spans")]) if len(t) > 3 else t
                    for p, t in list(row["spans"].items())[:3]}
    if len(rows[0]["spans"]) > 3:
        row["spans"]["..."] = f"{len(rows[0]['spans']) - 3} more proforms / controls"
    write_json(out, "span_costs_row.json", row, trim=False)

    # Enough sentences of one length for `by_length` to have a row (the test suite's recipe).
    base = hand_examples()
    sentences = base + [TreebankSentence(id=f"hand-1-copy{k}", words=base[1].words, gold_spans=base[1].gold_spans,
                                         source=base[1].source) for k in range(6)]
    record, _ = run_task_1b(FakeSpanScorer(sentences), sentences, policy=MinOverSet(),
                            model="FakeSpanScorer (illustrative)", show_progress=False, seed=0,
                            notes="docs example: a scorer that knows the gold tree; not a model")
    write_json(out, "record_1b.json", record)
    write_text(out, "verdict_1b.txt", [verdict(record)])
    fig(out, "fig_1b_fake.png", EF.fig_1b, record)
    fig(out, "fig_0c_fake.png", EF.fig_0c, record)


# ------------------------------------------------------------ Task 9a (no torch)


def synthetic_rows(effect: float, *, n_paragraphs: int = 100, cuts_per: int = 15, seed: int = 0) -> list[dict]:
    """Cache rows shaped like the demo: divergence tracks deleted length; `effect` adds a
    far-minus-close gap independent of length (the same generator the tests use)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_paragraphs):
        text = "X. " * 300
        cuts = []
        for k in range(cuts_per):
            seg = int(rng.integers(10, 200))
            d = 150 + 0.6 * seg + rng.normal(0, 40)
            base = 60 + 0.9 * seg
            div = base * (1 + effect * (d - 150 - 0.6 * seg) / 40) + rng.normal(0, 8)
            i = 5 + k
            cuts.append({"i": i, "j": i + seg, "boundary": "sentence" if k % 5 else "clause", "seg_len": seg,
                         "char_i": 3 * i, "char_j": 3 * (i + seg), "d": float(d), "div": float(max(div, 1.0)),
                         "div_k": [float(max(div, 1.0))] * 3, "fl": 3.8 + float(rng.normal(0, 0.05)),
                         "retokenises": True, "join": "X. ⟦cut⟧ X."})
        rows.append({"kind": "paragraph", "id": f"p{p}", "text": text, "n_tokens": 600, "fluency": 3.8,
                     "boundaries": {"sentence": [c["i"] for c in cuts]}, "n_rejected_boundaries": 0, "cuts": cuts})
    return rows


def task_9a_examples(out: Path) -> None:
    from m1_analyzer.experiments import (
        admissible_cuts, analyse_9a, boundary_positions, hand_paragraphs, preregistration, splice_ids,
        splice_text, verdict_9a, experiment_figures as EF,
    )

    p = hand_paragraphs()[0]
    offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", p.text)]
    n_tokens = len(offsets)
    boundaries, rejected = boundary_positions(p.text, offsets, splitter="regex")
    cuts = admissible_cuts(boundaries, n_tokens, window=5)
    lines = [f"paragraph {p.id!r} ({n_tokens} whitespace tokens; a real run uses the model's tokenizer offsets):",
             f"  {p.text}", "",
             f"boundary_positions(text, offsets, splitter='regex') -> {boundaries}, rejected sentence ends: {rejected}",
             f"admissible_cuts(boundaries, n_tokens={n_tokens}, window=5) -> {len(cuts)} cuts; the first five:"]
    for c in cuts[:5]:
        lines.append(f"  Cut(i={c.i}, j={c.j}, boundary={c.boundary!r})  deletes {c.seg_len} tokens")
    c = cuts[0]
    lines += ["", f"the first cut, (i, j) = ({c.i}, {c.j}):",
              f"  splice_text -> {splice_text(p.text, offsets, c.i, c.j, marker=' ⟦cut⟧ ')}",
              f"  splice_ids(list(range({n_tokens})), {c.i}, {c.j}) -> {splice_ids(list(range(n_tokens)), c.i, c.j)}",
              "", "Phase A then asks the model for the next-token state at i and at j of the ORIGINAL (their L2",
              "distance is d) and at i+1..i+window of the SPLICED text versus j+1..j+window of the original",
              "(the per-position div_k; the median is div)."]
    write_text(out, "cuts_by_hand.txt", lines)

    header = preregistration(window=20, splitter="regex", provenance={"model_id": "synthetic (illustrative)"})
    rows = synthetic_rows(0.5)
    record = analyse_9a(rows, header, model="synthetic (illustrative)", n_perm=300, n_boot=200,
                        notes="docs example: synthetic rows with a planted far > close effect")
    shown = dict(record)
    shown["cuts"] = record["cuts"][:3] + ["..."] if len(record["cuts"]) > 3 else record["cuts"]
    write_json(out, "record_9a_synthetic.json", shown)
    write_text(out, "verdict_9a.txt", [verdict_9a(record)])
    fig(out, "fig_9a_synthetic.png", EF.fig_9a, record)


def mock_figures(out: Path) -> None:
    from m1_analyzer.experiments import experiment_figures as EF

    for task in ("1b", "0c", "9a", "9b"):
        EF.render_mock_pair(task, str(out))
        for outcome in ("true", "false"):
            (out / f"_mock_{task}_{outcome}.png").unlink(missing_ok=True)


# ------------------------------------------------------ tiny model (needs torch)


def tiny_model_examples(out: Path) -> None:
    from m1_analyzer import Analyzer, ExtractionConfig, ModelConfig, RunConfig, ScoringConfig, StorageConfig
    from m1_analyzer.experiments import (
        GenerationProtocol, Paragraph, add_surprisal, analyse_9a, assign_pairs, audit_sample, build_record_9b,
        compute_generation_9b, compute_splices, flatten_cuts, hand_paragraphs, load_generation_9b, write_audit_csv,
    )
    from m1_analyzer.experiments.jsonl_cache import read_jsonl
    from m1_analyzer.testing import build_tiny_local_model

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model_path = build_tiny_local_model(tmp / "tiny", max_positions=256)

        # Hidden-state extraction: the JSON the base pipeline writes.
        analyzer = Analyzer(RunConfig(
            model=ModelConfig(model_id=model_path),
            extraction=ExtractionConfig(layers=[-1, "middle"], batch_size=4),
            storage=StorageConfig(output_dir=str(tmp / "outputs")),
        ))
        result = analyzer.batch(["hello world", "the quick brown fox jumps over the lazy dog"])
        paths = analyzer.save(result, name="docs_example")
        payload = json.loads(Path(paths.json_path).read_text(encoding="utf-8"))
        payload["run"]["model_id"] = "<local tiny random GPT-2: 4 layers, hidden size 16>"
        for r in payload["records"]:
            for layer in r["layers"].values():
                layer["values"] = layer["values"][:4] + ["..."] if layer["values"] else layer["values"]
        write_json(out, "extraction_tiny.json", payload, trim=False)
        analyzer.unload()

        # Task 9a on the tiny model: a real cache row, then the record.
        analyzer = Analyzer(RunConfig(model=ModelConfig(model_id=model_path, head="causal_lm"),
                                      scoring=ScoringConfig(batch_size=4)))
        paragraphs = list(hand_paragraphs())
        paragraphs += [Paragraph(id=p.id + "-b", text=p.text, source=p.source, info=dict(p.info)) for p in hand_paragraphs()]
        cache = tmp / "splices.jsonl"
        header, rows = compute_splices(analyzer.states, paragraphs, window=5, splitter="regex", cache_path=cache,
                                       strata=[[1, 10], [10, 30], [30, 80]],
                                       provenance={"model_id": "tiny random GPT-2 (illustrative)"}, show_progress=False)
        header, rows, report = add_surprisal(analyzer.states, cache, show_progress=False)
        write_json(out, "splices_tiny_header.json", header, trim=False)
        row = dict(rows[0])
        row["cuts"] = row["cuts"][:2] + ["..."] if len(row["cuts"]) > 2 else row["cuts"]
        write_json(out, "splices_tiny_row.json", row)
        write_json(out, "surprisal_report.json", report, trim=False)
        cuts = flatten_cuts(rows)
        assign_pairs(cuts, strata=header["strata"], decile=header["decile"], match_width=header["match_width"])
        sheet = write_audit_csv(audit_sample(cuts, n_close=2, n_far=2, n_other=2, seed=0), rows, tmp / "audit.csv")
        shutil.copy(sheet, out / "audit_tiny.csv")
        record_9a = analyse_9a(rows, header, model="tiny random GPT-2 (illustrative)", n_perm=50, n_boot=20,
                               min_per_group=1)
        shown = dict(record_9a)
        shown["cuts"] = record_9a["cuts"][:3] + ["..."] if len(record_9a["cuts"]) > 3 else record_9a["cuts"]
        write_json(out, "record_9a_tiny.json", shown)
        record_path = tmp / "record_9a.json"
        record_path.write_text(json.dumps(record_9a), encoding="utf-8")

        # Task 9b on the same tiny model, from the 9a files just written.
        gen = tmp / "gen_9b.jsonl"
        protocol = GenerationProtocol(W_true=5, K=4, L=6, greedy_cap=6)
        compute_generation_9b(analyzer.states, analyzer.models, record_path, cache, cache_path=gen,
                              protocol=protocol, provenance={"model_id": "tiny random GPT-2 (illustrative)"},
                              show_progress=False)
        header_9b, rows_9b = load_generation_9b(gen)
        write_json(out, "gen_9b_tiny_header.json", header_9b, trim=False)
        row = dict(rows_9b[0])
        row["samples_orig_decoded"] = [analyzer.states.decode(s) for s in row["samples_orig"][:2]]
        row["samples_spliced_decoded"] = [analyzer.states.decode(s) for s in row["samples_spliced"][:2]]
        write_json(out, "gen_9b_tiny_row.json", row)
        record_9b = build_record_9b(header_9b, rows_9b, record_9a, model="tiny random GPT-2 (illustrative)")
        shown = dict(record_9b)
        shown["cuts"] = record_9b["cuts"][:2] + ["..."] if len(record_9b["cuts"]) > 2 else record_9b["cuts"]
        write_json(out, "record_9b_tiny.json", shown)
        analyzer.unload()


# ------------------------------------------------------------------- driver


def generate(out: Path, *, torch_ok: bool, figures: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    task_1b_examples(out)
    task_9a_examples(out)
    if figures:
        mock_figures(out)
    if torch_ok:
        tiny_model_examples(out)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:  # pragma: no cover - depends on the environment
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-torch", action="store_true", help="skip the tiny-model artifacts")
    parser.add_argument("--check", action="store_true",
                        help="regenerate the torch-free text fragments into a temp dir and diff them against "
                             "the committed ones; exit 1 on drift")
    parser.add_argument("--out", default=str(OUT_DIR), help=f"output directory (default {OUT_DIR})")
    args = parser.parse_args(argv)

    if args.check:
        committed = Path(args.out)
        with tempfile.TemporaryDirectory() as tmp:
            fresh = Path(tmp) / "examples"
            generate(fresh, torch_ok=False, figures=False)
            drift = [name for name in TEXT_FRAGMENTS
                     if not (committed / name).exists() or not filecmp.cmp(fresh / name, committed / name, shallow=False)]
        if drift:
            print("docs/assets/examples is out of date for: " + ", ".join(drift))
            print("Re-run `python scripts/make_doc_examples.py` and commit the result.")
            return 1
        print(f"{len(TEXT_FRAGMENTS)} example fragments match the generator.")
        return 0

    use_torch = not args.no_torch and torch_available()
    if not args.no_torch and not use_torch:
        print("torch is not importable: skipping the tiny-model artifacts (pass --no-torch to silence this).")
    generate(Path(args.out), torch_ok=use_torch)
    written = sorted(p.name for p in Path(args.out).iterdir())
    print(f"wrote {len(written)} files to {args.out}:")
    for name in written:
        print("  " + name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
