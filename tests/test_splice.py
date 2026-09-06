"""Phase A of Task 9a on the tiny model: cuts, splices, states, cache, resume."""

import json

import numpy as np
import pytest

from m1_analyzer import Analyzer, ModelConfig, RunConfig, ScoringConfig
from m1_analyzer.experiments.paragraphs import Paragraph, hand_paragraphs
from m1_analyzer.experiments.splice import (
    DIVERGENCE_MEASURE,
    Cut,
    admissible_cuts,
    compute_splices,
    cut_count,
    load_splices,
    paragraph_boundaries,
    preregistration,
    score_paragraph,
    splice_ids,
    splice_text,
)
from m1_analyzer.testing import build_tiny_local_model


@pytest.fixture(scope="module")
def analyzer(tmp_path_factory) -> Analyzer:
    # Paragraphs are longer than the session-wide tiny model's 32 positions.
    path = build_tiny_local_model(tmp_path_factory.mktemp("tiny_long"), max_positions=256)
    return Analyzer(RunConfig(model=ModelConfig(model_id=path, head="causal_lm"), scoring=ScoringConfig(batch_size=4)))


def test_admissible_cuts_pair_same_type_endpoints_with_room_after_j():
    cuts = admissible_cuts({"sentence": [3, 9, 15, 21], "clause": [6, 12]}, n_tokens=27, window=5)
    sentence = [(c.i, c.j) for c in cuts if c.boundary == "sentence"]
    assert sentence == [(3, 9), (3, 15), (3, 21), (9, 15), (9, 21), (15, 21)]  # j=21 leaves exactly 5
    assert [(c.i, c.j) for c in cuts if c.boundary == "clause"] == [(6, 12)]  # never a sentence-clause pair
    # window=5 needs n - (j + 1) >= 5: with 26 tokens j=21 leaves 4 -> excluded.
    cuts = admissible_cuts({"sentence": [3, 9, 15, 21]}, n_tokens=26, window=5)
    assert [(c.i, c.j) for c in cuts] == [(3, 9), (3, 15), (9, 15)]
    assert all(c.seg_len == c.j - c.i for c in cuts)
    assert [(c.i, c.j) for c in admissible_cuts({"clause": [6, 12]}, 26, 5)] == [(6, 12)]
    with pytest.raises(ValueError):
        admissible_cuts({"sentence": [1, 2]}, 10, 0)


def test_splice_ids_and_text_delete_strictly_between_endpoints():
    ids = list(range(10))
    assert splice_ids(ids, 2, 6) == [0, 1, 2, 7, 8, 9]
    assert Cut(2, 6).seg_len == 4
    with pytest.raises(ValueError):
        splice_ids(ids, 6, 2)
    text = "aa bb. cc dd. ee ff."
    offsets = [(0, 2), (3, 6), (7, 9), (10, 13), (14, 16), (17, 20)]
    assert splice_text(text, offsets, 1, 3) == "aa bb. ee ff."
    assert splice_text(text, offsets, 1, 3, marker=" |") == "aa bb. | ee ff."


def test_preregistration_validates_and_names_everything():
    header = preregistration(window=20, splitter="regex", provenance={"model_id": "m"})
    assert header["divergence_measure"] == DIVERGENCE_MEASURE
    assert header["strata"] == [[20, 40], [40, 80], [80, 160]] and header["decile"] == 0.1
    assert header["primary_boundary"] == "sentence" and header["model_id"] == "m"
    with pytest.raises(ValueError, match="primary_boundary"):
        preregistration(boundary_kinds=("clause",))
    with pytest.raises(ValueError, match="decile"):
        preregistration(decile=0.7)
    with pytest.raises(ValueError, match="stratum"):
        preregistration(strata=[[40, 20]])


def test_score_paragraph_measures_what_the_docstring_says(analyzer):
    para = hand_paragraphs()[0]
    ids, offsets, boundaries, rejected = paragraph_boundaries(analyzer.states, para, splitter="regex")
    assert rejected == 0 and len(boundaries["sentence"]) == 8
    row = score_paragraph(analyzer.states, para, window=5, splitter="regex")
    assert row["n_tokens"] == len(ids) and len(row["cuts"]) == len(admissible_cuts(boundaries, len(ids), 5))
    cut = row["cuts"][0]
    assert cut["seg_len"] == cut["j"] - cut["i"] and len(cut["div_k"]) == 5
    assert cut["div"] == pytest.approx(float(np.median(cut["div_k"])), abs=1e-3)
    assert cut["retokenises"] is True and "⟦cut⟧" in cut["join"]
    # Recompute one cut by hand from the state service.
    spliced = splice_ids(ids, cut["i"], cut["j"])
    [orig] = analyzer.states.states([ids], [[cut["i"], cut["j"]] + [cut["j"] + 1 + k for k in range(5)]])
    [spl] = analyzer.states.states([spliced], [[cut["i"] + 1 + k for k in range(5)]])
    d = np.linalg.norm(orig.states[0] - orig.states[1])
    div_k = np.linalg.norm(spl.states - orig.states[2:], axis=1)
    assert cut["d"] == pytest.approx(d, abs=1e-3)
    assert np.allclose(cut["div_k"], div_k, atol=1e-3)
    # The spliced text's own fluency, in nats per token, from the same pass.
    assert cut["fl"] == pytest.approx(-analyzer.score_one(analyzer.states.decode(spliced)).mean_logprob, abs=1e-3)
    # The rejoined TEXT is what the audit sees; the tiny word-level tokenizer decodes
    # with a space before punctuation, so compare modulo whitespace.
    rejoined = splice_text(para.text, offsets, cut["i"], cut["j"])
    assert rejoined.replace(" ", "") == analyzer.states.decode(spliced).replace(" ", "")


def test_a_paragraph_without_cuts_still_gets_a_row(analyzer):
    para = Paragraph(id="one", text="the quick brown fox jumps over the lazy dog.")
    row = score_paragraph(analyzer.states, para, window=5, splitter="regex")
    assert row["cuts"] == [] and row["fluency"] > 0


def test_cache_resumes_and_guards_the_preregistration(analyzer, tmp_path):
    cache = tmp_path / "splices.jsonl"
    paras = hand_paragraphs()
    header, rows = compute_splices(analyzer.states, paras[:2], window=5, splitter="regex", cache_path=cache,
                                   provenance={"model_id": "tiny"}, show_progress=False)
    assert cut_count(rows) > 0 and [r["id"] for r in rows] == [p.id for p in paras[:2]]
    calls = analyzer.states.states
    analyzer.states.states = lambda *a, **k: (_ for _ in ()).throw(AssertionError("scored a finished paragraph"))
    try:
        header2, rows2 = compute_splices(analyzer.states, paras[:2], window=5, splitter="regex", cache_path=cache,
                                         provenance={"model_id": "tiny"}, show_progress=False)
    finally:
        analyzer.states.states = calls
    assert rows2 == rows and header2["window"] == 5
    with pytest.raises(ValueError, match="window=5"):
        compute_splices(analyzer.states, paras, window=6, splitter="regex", cache_path=cache, show_progress=False)
    with pytest.raises(ValueError, match="model_id"):
        compute_splices(analyzer.states, paras, window=5, splitter="regex", cache_path=cache,
                        provenance={"model_id": "other"}, show_progress=False)
    # The third paragraph is added on resume; a truncated last line is repaired first.
    with cache.open("a", encoding="utf-8") as fh:
        fh.write('{"kind": "paragraph", "id": "broken", "cu')
    header3, rows3 = compute_splices(analyzer.states, paras, window=5, splitter="regex", cache_path=cache,
                                     provenance={"model_id": "tiny"}, show_progress=False)
    assert [r["id"] for r in rows3] == [p.id for p in paras]
    loaded_header, loaded_rows = load_splices(cache)
    assert loaded_rows == rows3 and loaded_header["divergence_measure"] == DIVERGENCE_MEASURE
    lines = cache.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["kind"] == "header" and len(lines) == 4


def test_load_splices_rejects_other_caches(tmp_path):
    other = tmp_path / "x.jsonl"
    other.write_text('{"kind": "header", "schema": 2}\n')
    with pytest.raises(ValueError, match="Task 9a"):
        load_splices(other)
