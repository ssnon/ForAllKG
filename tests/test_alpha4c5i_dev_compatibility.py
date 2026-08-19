from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import campaigns.sers_alpha4_epoch.alpha4.alpha4c5i_dev_compatibility as compat


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_parse_h1_counts_requires_exact_printed_invariants():
    stdout = """
Trend evidence: 15
Precision local results: 15
CrossContext assessments: 10
Grounding relations: 10
"""
    assert compat.parse_h1_counts(stdout) == {
        "trend_evidence_count": 15,
        "precision_count": 15,
        "cross_context_count": 10,
        "grounding_count": 10,
    }


def test_closed_reserve_paths_are_rejected_before_read(tmp_path: Path):
    path = (
        tmp_path
        / "evaluation/sers_alpha4c5h1/reserve_b_v1/"
        "trend_aware_hypothesis_input.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not even parsed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="closed Reserve"):
        compat.load_and_verify_dev_input(
            root=tmp_path,
            path=path,
            dev_paper_ids=[f"SYNTH_{i:02d}" for i in range(53)],
        )


def test_exact_dev_set_is_required(tmp_path: Path, monkeypatch):
    path = tmp_path / "evaluation/dev/trend_aware_hypothesis_input.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")

    fake = SimpleNamespace(
        domain_profile_id="sers_au_ag",
        trend_corpus_binding=SimpleNamespace(
            paper_ids=[f"SYNTH_{i:02d}" for i in range(52)]
        ),
    )
    monkeypatch.setattr(
        compat.TrendAwareHypothesisInput,
        "model_validate_json",
        lambda text: fake,
    )
    monkeypatch.setattr(
        compat,
        "verify_trend_aware_input_sources",
        lambda source: None,
    )

    with pytest.raises(ValueError, match="exact 53-paper DEV set"):
        compat.load_and_verify_dev_input(
            root=tmp_path,
            path=path,
            dev_paper_ids=[f"SYNTH_{i:02d}" for i in range(53)],
        )


def test_exact_dev_set_passes(tmp_path: Path, monkeypatch):
    path = tmp_path / "evaluation/dev/trend_aware_hypothesis_input.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    dev = [f"SYNTH_{i:02d}" for i in range(53)]

    fake = SimpleNamespace(
        domain_profile_id="sers_au_ag",
        trend_corpus_binding=SimpleNamespace(paper_ids=list(reversed(dev))),
    )
    monkeypatch.setattr(
        compat.TrendAwareHypothesisInput,
        "model_validate_json",
        lambda text: fake,
    )
    monkeypatch.setattr(
        compat,
        "verify_trend_aware_input_sources",
        lambda source: None,
    )

    observed = compat.load_and_verify_dev_input(
        root=tmp_path,
        path=path,
        dev_paper_ids=dev,
    )
    assert observed is fake


def test_component_hash_verifier_detects_drift(tmp_path: Path, monkeypatch):
    rel = "synthetic/component.py"
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        compat,
        "EXPECTED_5I_COMPONENT_SHA256",
        {rel: compat.sha256_file(path)},
    )
    assert compat.verify_5i_component_hashes(tmp_path) == []

    path.write_text("x = 2\n", encoding="utf-8")
    issues = compat.verify_5i_component_hashes(tmp_path)
    assert any("SHA drift" in issue for issue in issues)


def test_summary_is_zero_llm_and_no_closed_reserve(tmp_path: Path):
    input_path = tmp_path / "evaluation/dev/input.json"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text("{}\n", encoding="utf-8")
    source = SimpleNamespace(
        input_id="synthetic_input",
        input_sha256="a" * 64,
    )
    summary = compat.make_structural_summary(
        root=tmp_path,
        input_path=input_path,
        source=source,
        h1_counts={
            "trend_evidence_count": 15,
            "precision_count": 15,
                "cross_context_count": 10,
            "grounding_count": 10,
        },
        probe={
            "empty_abstention_validation_passed": True,
            "scientific_hypotheses_generated": 0,
            "llm_calls": 0,
        },
        preview={},
    )
    assert summary["llm_calls"] == 0
    assert summary["closed_reserve_a_used"] is False
    assert summary["closed_reserve_b_used"] is False
    assert summary["reserve_b_rerun"] is False
    assert summary[
        "passes_deterministic_downstream_compatibility"
    ] is True


def test_test_source_contains_no_real_reserve_identity():
    source_text = Path(__file__).read_text(encoding="utf-8")
    forbidden = "SERS" + "_API_"
    assert forbidden not in source_text


def test_dev_split_loader_reads_canonical_partition_shape(
    tmp_path: Path,
    monkeypatch,
):
    split_path = tmp_path / "blind_split.json"
    dev = [f"SYNTH_{i:02d}" for i in range(53)]
    payload = {
        "split_id": compat.EXPECTED_BLIND_SPLIT_ID,
        "split_sha256":
            compat.EXPECTED_BLIND_SPLIT_SEMANTIC_SHA256,
        "partitions": {
            "development": {
                "paper_ids": dev,
            },
            "reserve_a": {
                "paper_ids": ["A"],
            },
            "reserve_b": {
                "paper_ids": ["B"],
            },
        },
    }
    _write(split_path, payload)
    monkeypatch.setattr(
        compat,
        "DEFAULT_BLIND_SPLIT",
        Path("blind_split.json"),
    )
    monkeypatch.setattr(
        compat,
        "EXPECTED_BLIND_SPLIT_RAW_SHA256",
        compat.sha256_file(split_path),
    )

    assert compat.load_exact_dev_paper_ids(tmp_path) == sorted(dev)


def test_dev_split_loader_supports_assignment_shape(
    tmp_path: Path,
    monkeypatch,
):
    split_path = tmp_path / "blind_split.json"
    dev = [f"SYNTH_{i:02d}" for i in range(53)]
    payload = {
        "split_id": compat.EXPECTED_BLIND_SPLIT_ID,
        "assignments": [
            *[
                {"paper_id": paper_id, "partition": "development"}
                for paper_id in dev
            ],
            {"paper_id": "A", "partition": "reserve_a"},
            {"paper_id": "B", "partition": "reserve_b"},
        ],
    }
    _write(split_path, payload)
    monkeypatch.setattr(
        compat,
        "DEFAULT_BLIND_SPLIT",
        Path("blind_split.json"),
    )
    monkeypatch.setattr(
        compat,
        "EXPECTED_BLIND_SPLIT_RAW_SHA256",
        compat.sha256_file(split_path),
    )

    assert compat.load_exact_dev_paper_ids(tmp_path) == sorted(dev)


def test_dev_split_loader_rejects_raw_sha_drift(
    tmp_path: Path,
    monkeypatch,
):
    split_path = tmp_path / "blind_split.json"
    _write(
        split_path,
        {
            "split_id": compat.EXPECTED_BLIND_SPLIT_ID,
            "development_paper_ids": [
                f"SYNTH_{i:02d}" for i in range(53)
            ],
        },
    )
    monkeypatch.setattr(
        compat,
        "DEFAULT_BLIND_SPLIT",
        Path("blind_split.json"),
    )
    monkeypatch.setattr(
        compat,
        "EXPECTED_BLIND_SPLIT_RAW_SHA256",
        "0" * 64,
    )

    with pytest.raises(ValueError, match="raw SHA drift"):
        compat.load_exact_dev_paper_ids(tmp_path)


def test_h1_compatibility_reuses_existing_summary_without_rerun(
    tmp_path: Path,
    monkeypatch,
):
    summary = tmp_path / "summary.json"
    _write(
        summary,
        {
            "passes_downstream_compatibility": True,
            "scientific_semantics_modified": False,
            "precision_algorithm_modified": False,
            "trend_semantics_id":
                compat.EXPECTED_TREND_SEMANTICS_ID,
            "precision_semantics_id":
                compat.EXPECTED_PRECISION_SEMANTICS_ID,
            "reserve_a_used": False,
            "reserve_b_used": False,
            "count_thresholds_used_for_acceptance": False,
            "llm_calls": 0,
            "trend_evidence_count": 15,
            "precision_local_result_count": 15,
            "cross_context_assessment_count": 10,
            "grounding_relation_count": 10,
        },
    )
    monkeypatch.setattr(
        compat,
        "DEFAULT_H1_DEV_SUMMARY",
        Path("summary.json"),
    )

    def _forbid_subprocess(*args, **kwargs):
        raise AssertionError(
            "5i must not rerun alpha4c.5h.1 DEV compatibility"
        )

    monkeypatch.setattr(
        compat.subprocess,
        "run",
        _forbid_subprocess,
    )

    counts, provenance = compat.run_h1_dev_compatibility(
        tmp_path
    )
    assert counts == {
        "trend_evidence_count": 15,
        "precision_count": 15,
        "cross_context_count": 10,
        "grounding_count": 10,
    }
    assert "rerun_performed=false" in provenance


def test_h1_existing_summary_count_drift_fails_closed(
    tmp_path: Path,
    monkeypatch,
):
    summary = tmp_path / "summary.json"
    _write(
        summary,
        {
            "passes_downstream_compatibility": True,
            "scientific_semantics_modified": False,
            "precision_algorithm_modified": False,
            "trend_semantics_id":
                compat.EXPECTED_TREND_SEMANTICS_ID,
            "precision_semantics_id":
                compat.EXPECTED_PRECISION_SEMANTICS_ID,
            "reserve_a_used": False,
            "reserve_b_used": False,
            "count_thresholds_used_for_acceptance": False,
            "llm_calls": 0,
            "trend_evidence_count": 14,
            "precision_local_result_count": 15,
            "cross_context_assessment_count": 10,
            "grounding_relation_count": 10,
        },
    )
    monkeypatch.setattr(
        compat,
        "DEFAULT_H1_DEV_SUMMARY",
        Path("summary.json"),
    )

    with pytest.raises(
        RuntimeError,
        match="DEV upstream invariant drift",
    ):
        compat.run_h1_dev_compatibility(tmp_path)
