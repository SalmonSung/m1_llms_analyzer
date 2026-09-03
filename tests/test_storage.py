"""Serialisation: schema, precision, sidecar policy, round-trip, atomicity."""

import json
from pathlib import Path

import numpy as np
import pytest

from m1_analyzer import load_run
from m1_analyzer.testing import TINY_HIDDEN


def _write(analyzer, texts, name="run"):
    result = analyzer.batch(texts)
    return result, analyzer.save(result, name=name)


def test_json_is_written_with_manifest(analyzer, sample_texts):
    _, paths = _write(analyzer, sample_texts)
    payload = json.loads(Path(paths.json_path).read_text())

    assert payload["schema_version"] == "1.0"
    run = payload["run"]
    for key in (
        "run_id", "created_at", "model_id", "device", "dtype", "pooling", "seed",
        "layers_requested", "layers_resolved", "layer_index_convention",
        "num_hidden_layers", "hidden_size", "library_versions", "config_fingerprint",
    ):
        assert key in run, f"manifest is missing {key}"
    assert payload["counts"]["records"] == len(sample_texts)
    assert payload["failures"] == []


def test_values_are_inline_and_rounded_by_default(analyzer, sample_texts):
    _, paths = _write(analyzer, sample_texts)
    payload = json.loads(Path(paths.json_path).read_text())
    values = payload["records"][0]["layers"]["-1"]["values"]

    assert paths.npy_path is None
    assert len(values) == TINY_HIDDEN
    # float_precision=6 must actually shrink the text, not just round the float.
    assert all(len(str(v).split(".")[-1]) <= 6 for v in values if isinstance(v, float))


def test_sidecar_written_when_forced(make_analyzer, sample_texts):
    analyzer = make_analyzer(npy_mode="always")
    _, paths = _write(analyzer, sample_texts, name="sidecar")
    assert paths.npy_path is not None
    payload = json.loads(Path(paths.json_path).read_text())
    entry = payload["records"][0]["layers"]["-1"]
    assert entry["values"] is None
    assert entry["npy_ref"]
    assert payload["storage"]["values_in_sidecar"] is True


def test_auto_mode_switches_to_sidecar_above_threshold(make_analyzer, sample_texts):
    analyzer = make_analyzer(pooling="none", npy_threshold_floats=1)
    _, paths = _write(analyzer, sample_texts, name="auto")
    assert paths.npy_path is not None


def test_round_trip_inline(analyzer, sample_texts):
    result, paths = _write(analyzer, sample_texts, name="inline")
    loaded = load_run(paths.json_path)
    for record, entry in zip(result.records, loaded["records"]):
        np.testing.assert_allclose(
            entry["layers"]["-1"]["values"], record.layer(-1), rtol=1e-4, atol=1e-6
        )


def test_round_trip_sidecar_is_lossless(make_analyzer, sample_texts):
    analyzer = make_analyzer(npy_mode="always")
    result, paths = _write(analyzer, sample_texts, name="lossless")
    loaded = load_run(paths.json_path)
    for record, entry in zip(result.records, loaded["records"]):
        # float32 in, float32 out -- exact equality, unlike the rounded JSON path.
        np.testing.assert_array_equal(
            entry["layers"]["-1"]["values"], record.layer(-1).astype(np.float32)
        )


def test_missing_sidecar_raises_a_helpful_error(make_analyzer, sample_texts):
    analyzer = make_analyzer(npy_mode="always")
    _, paths = _write(analyzer, sample_texts, name="orphan")
    Path(paths.npy_path).unlink()
    with pytest.raises(FileNotFoundError, match="Keep the .json and .npz together"):
        load_run(paths.json_path)


def test_per_token_shapes_survive_round_trip(make_analyzer):
    analyzer = make_analyzer(pooling="none", npy_mode="always")
    result, paths = _write(analyzer, ["hello world", "a b c"], name="tokens")
    loaded = load_run(paths.json_path)
    for record, entry in zip(result.records, loaded["records"]):
        assert entry["layers"]["-1"]["values"].shape == record.layer(-1).shape


def test_input_text_can_be_withheld(make_analyzer, sample_texts):
    analyzer = make_analyzer(include_input_text=False)
    _, paths = _write(analyzer, sample_texts, name="notext")
    payload = json.loads(Path(paths.json_path).read_text())
    assert "text" not in payload["records"][0]
    assert payload["records"][0]["id"]  # ids still allow joining back to inputs


def test_failures_are_persisted(analyzer, monkeypatch, sample_texts):
    original = analyzer.inference._forward

    def flaky(texts, *args, **kwargs):
        if any(t == "a b c" for t in texts):
            raise RuntimeError("synthetic")
        return original(texts, *args, **kwargs)

    monkeypatch.setattr(analyzer.inference, "_forward", flaky)
    _, paths = _write(analyzer, sample_texts, name="withfail")
    payload = json.loads(Path(paths.json_path).read_text())
    assert len(payload["failures"]) == 1
    assert payload["failures"][0]["error_type"] == "RuntimeError"


def test_no_temp_files_left_behind(analyzer, sample_texts):
    _, paths = _write(analyzer, sample_texts, name="clean")
    leftovers = [p.name for p in Path(paths.json_path).parent.iterdir() if p.name.startswith(".")]
    assert leftovers == []


def test_filenames_are_sanitised(analyzer, sample_texts):
    _, paths = _write(analyzer, sample_texts, name="my run/2024:01")
    assert "/" not in Path(paths.json_path).name
    assert ":" not in Path(paths.json_path).name


def test_run_helper_extracts_and_saves(analyzer, sample_texts):
    result, paths = analyzer.run(sample_texts, name="oneshot")
    assert len(result.records) == len(sample_texts)
    assert Path(paths.json_path).exists()


def test_save_accepts_a_single_record(analyzer):
    record = analyzer.invoke("hello world")
    paths = analyzer.save(record, name="single")
    payload = json.loads(Path(paths.json_path).read_text())
    assert payload["counts"]["records"] == 1


def test_no_token_appears_in_output(make_analyzer, sample_texts, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_supersecret_value")
    analyzer = make_analyzer()
    _, paths = _write(analyzer, sample_texts, name="secret")
    assert "hf_supersecret_value" not in Path(paths.json_path).read_text()


def test_manifest_records_the_pooling_actually_used(analyzer):
    """A per-call override must be reflected in the saved manifest, not the default."""
    result = analyzer.batch(["hello world"], pooling="mean")
    paths = analyzer.save(result, name="override")
    payload = json.loads(Path(paths.json_path).read_text())
    assert analyzer.config.extraction.pooling == "last_token"
    assert payload["run"]["pooling"] == "mean"


def test_manifest_records_layers_actually_used(make_analyzer):
    analyzer = make_analyzer(layers=-1)
    result = analyzer.batch(["hello world"], layers=[0, "middle"])
    paths = analyzer.save(result, name="override_layers")
    payload = json.loads(Path(paths.json_path).read_text())
    assert payload["run"]["layers_resolved"] == [0, 2]
