"""Paragraph selection: cleaning, filters, seeded sampling, file sources."""

import json

import pytest

from m1_analyzer.experiments.paragraphs import (
    Paragraph,
    clean_paragraph,
    hand_paragraphs,
    iter_text_file,
    load_paragraph_file,
    select_paragraphs,
)


def _words(text):
    return len(text.split())


def test_clean_paragraph_requires_sentence_punctuation():
    assert clean_paragraph("  one two.\n three four!  ") == "one two. three four!"
    assert clean_paragraph('He said "yes."') == 'He said "yes."'
    assert clean_paragraph("== Heading ==") is None
    assert clean_paragraph("   ") is None


def test_select_filters_and_samples_deterministically():
    cands = []
    for k in range(40):
        n = 8 + k  # 8..47 words
        cands.append((f"p{k}", " ".join(["word"] * (n - 4)) + ". a b. c d.", {"k": k}))
    cands.append(("heading", "Section title", {}))
    cands.append(("short", "a b. c d. e f.", {}))
    chosen = select_paragraphs(cands, _words, n=5, min_tokens=20, max_tokens=30, min_sentences=3,
                               splitter="regex", seed=1, pool_factor=2)
    again = select_paragraphs(cands, _words, n=5, min_tokens=20, max_tokens=30, min_sentences=3,
                              splitter="regex", seed=1, pool_factor=2)
    assert [p.id for p in chosen] == [p.id for p in again]
    assert len(chosen) == 5
    assert all(20 <= p.info["n_tokens"] <= 30 and p.info["n_sentences"] >= 3 for p in chosen)
    assert [p.info["pool_index"] for p in chosen] == sorted(p.info["pool_index"] for p in chosen)
    other = select_paragraphs(cands, _words, n=5, min_tokens=20, max_tokens=30, min_sentences=3,
                              splitter="regex", seed=2, pool_factor=2)
    assert [p.id for p in other] != [p.id for p in chosen]


def test_select_warns_or_raises_when_too_few_qualify():
    cands = [("a", "one two three. four five.", {})]
    assert len(select_paragraphs(cands, _words, n=3, min_tokens=1, max_tokens=50, min_sentences=2, splitter="regex")) == 1
    with pytest.raises(ValueError, match="No paragraph qualified"):
        select_paragraphs(cands, _words, n=1, min_tokens=100, max_tokens=200, splitter="regex")


def test_text_and_jsonl_files(tmp_path):
    txt = tmp_path / "corpus.txt"
    txt.write_text("first one. second one. third one.\n\n\nsecond block. still here. and more.\n", encoding="utf-8")
    assert [pid for pid, _, _ in iter_text_file(txt)] == ["corpus:0", "corpus:1"]
    jl = tmp_path / "corpus.jsonl"
    jl.write_text("\n".join(json.dumps({"id": f"x{k}", "text": "a b. c d. e f."}) for k in range(3)) + "\n")
    paras = load_paragraph_file(jl, _words, n=2, min_tokens=1, max_tokens=50, min_sentences=2, splitter="regex")
    assert len(paras) == 2 and paras[0].source == str(jl)


def test_hand_paragraphs_round_trip():
    for p in hand_paragraphs():
        assert Paragraph.from_json(p.to_json()) == p
        assert clean_paragraph(p.text) == p.text
