"""Offline fail-closed tests; no private source text or real corpus required."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest

from scripts.discovery.audit_empirical_evidence_readiness_m4b1 import (
    build_audit, collect_files, digest, load_source, required_tags, run,
)


def source():
    def observable(name, role="INDEPENDENT_DISCRIMINATOR"):
        return {"name": name, "role": role, "independent_of_sers_outcome": "UNKNOWN"}
    return {
        "schema_version": "m4a2-confrontation-readiness-audit-v1",
        "status": "REVIEW_REQUIRED_NO_SCIENCE_CERTIFICATION",
        "case_count": 1,
        "fresh_llm_or_network_calls": 0,
        "scientific_truth_or_novelty_certified": False,
        "physical_feasibility_certified": False,
        "reviewer_template_completed": False,
        "deletion_or_production_authority": False,
        "source_files_mutated": False,
        "cases": [{
            "case_id": "QA_TEST", "readiness": "NOT_READY_EMPIRICAL_ADJUDICATION",
            "independent_measurement_certified": False,
            "scientific_truth_or_falsification_certified": False,
            "hypothesis_refs": ["B00"],
            "proposed_experiment_UNREVIEWED": {
                "observables": [observable("SERS band ratios", "TARGET_OUTCOME"),
                                observable("molecular orientation")]
            }
        }]
    }


def setup(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    input_path = tmp_path / "source.json"
    input_path.write_text(json.dumps(source()), encoding="utf-8")
    return data, input_path


def args(tmp_path: Path, data: Path, input_path: Path):
    return argparse.Namespace(m4a2_report=input_path, expected_m4a2_sha256=None,
                              data_root=[data], output_dir=tmp_path / "out",
                              max_depth=7, max_files=100, max_header_bytes=2000000)


def test_empty_inventory_is_not_evidence_absence(tmp_path):
    d, f = setup(tmp_path)
    result = run(args(tmp_path, d, f))
    assert result["file_count"] == 0
    assert all(x["availability_status"] == "NOT_LOCATED_IN_SCANNED_SCOPE"
               for x in result["cases"][0]["observables"])
    assert result["claims_about_unscanned_data"] is False
    assert result["science_truth_or_novelty_certified"] is False


def test_csv_columns_discovered_without_proof(tmp_path):
    d, f = setup(tmp_path)
    (d / "series.csv").write_text("sample_id,orientation_angle,raman_band_ratio\nx,40,1.7\n")
    result = run(args(tmp_path, d, f))
    assert result["file_count"] == 1
    assert "ORIENTATION" in result["files"][0]["field_tags_HEURISTIC"]
    assert "SERS" in result["files"][0]["field_tags_HEURISTIC"]
    assert result["cases"][0]["observables"][0]["candidate_file_count"] == 1
    assert result["cases"][0]["observables"][1]["candidate_file_count"] == 1
    assert result["cases"][0]["observables"][1]["independent_measurement_witness_verified"] is False
    assert result["files"][0]["actual_measurements_read"] is False


def test_template_blank_even_with_candidate(tmp_path):
    d, f = setup(tmp_path)
    (d / "angle.csv").write_text("orientation_angle\n5\n")
    run(args(tmp_path, d, f))
    with (tmp_path / "out" / "M4B1_EVIDENCE_LINKAGE_TEMPLATE_PRIVATE.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert all(r["source_validation_status"] == "UNREVIEWED" and not r["linked_source_file_PRIVATE"] for r in rows)


def test_bad_schema_rejected(tmp_path):
    d, f = setup(tmp_path)
    v = source(); v["schema_version"] = "unknown"
    f.write_text(json.dumps(v))
    with pytest.raises(ValueError, match="schema mismatch"):
        run(args(tmp_path, d, f))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("field", ["scientific_truth_or_novelty_certified", "physical_feasibility_certified", "reviewer_template_completed", "deletion_or_production_authority", "source_files_mutated"])
def test_forged_authority_rejected(tmp_path, field):
    d, f = setup(tmp_path)
    v = source(); v[field] = True
    f.write_text(json.dumps(v))
    with pytest.raises(ValueError, match=field):
        run(args(tmp_path, d, f))


def test_wrong_sha_rejected(tmp_path):
    d, f = setup(tmp_path)
    a = args(tmp_path, d, f)
    a.expected_m4a2_sha256 = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        run(a)


def test_existing_output_not_overwritten(tmp_path):
    d, f = setup(tmp_path)
    a = args(tmp_path, d, f)
    a.output_dir.mkdir()
    (a.output_dir / "keep").write_text("keep")
    with pytest.raises(ValueError, match="must not already exist"):
        run(a)
    assert (a.output_dir / "keep").read_text() == "keep"


def test_bounded_scan_does_not_claim_complete(tmp_path):
    d, f = setup(tmp_path)
    for i in range(4): (d / f"series{i}.csv").write_text("raman_band_ratio\n1\n")
    a = args(tmp_path, d, f); a.max_files = 2
    result = run(a)
    assert result["file_count"] == 2
    assert result["scan_truncated"] is True
    assert result["status"] == "PARTIAL_INVENTORY_REVIEW_REQUIRED"


def test_depth_bounded(tmp_path):
    d, f = setup(tmp_path)
    (d / "nested").mkdir()
    (d / "nested" / "data.csv").write_text("orientation_angle\n2\n")
    a = args(tmp_path, d, f); a.max_depth = 0
    result = run(a)
    assert result["file_count"] == 0


def test_symlink_never_traversed(tmp_path):
    d, f = setup(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "secret.csv").write_text("orientation_angle\n2\n")
    (d / "link").symlink_to(elsewhere, target_is_directory=True)
    result = run(args(tmp_path, d, f))
    assert result["file_count"] == 0


def test_large_file_only_metadata(tmp_path):
    d, f = setup(tmp_path)
    (d / "orientation.csv").write_text("angle\n" + "5\n" * 200)
    a = args(tmp_path, d, f); a.max_header_bytes = 50
    result = run(a)
    assert result["files"][0]["inspection_status"] == "TOO_LARGE_FOR_HEADER_INSPECTION"
    assert result["files"][0]["small_file_sha256"] is None


def test_pdf_metadata_only(tmp_path):
    d, f = setup(tmp_path)
    (d / "sers_paper.pdf").write_bytes(b"not parsed")
    result = run(args(tmp_path, d, f))
    assert result["files"][0]["inspection_status"] == "METADATA_ONLY_UNSUPPORTED_FORMAT"
    assert result["files"][0]["columns_or_keys"] == []


def test_bad_case_missing_observable_rejected(tmp_path):
    d, f = setup(tmp_path)
    v = source(); v["cases"][0]["proposed_experiment_UNREVIEWED"]["observables"] = []
    f.write_text(json.dumps(v))
    with pytest.raises(ValueError, match="missing observables"):
        run(args(tmp_path, d, f))


def test_input_source_unchanged(tmp_path):
    d, f = setup(tmp_path)
    before = digest(f)
    run(args(tmp_path, d, f))
    assert digest(f) == before


def test_metadata_role_has_no_certification(tmp_path):
    v = source(); report = build_audit(v, source_sha="f"*64, rows=[], roots=[], truncated=False, warnings=[])
    assert report["automatic_research_idea_or_hypothesis_promotion"] is False
    assert report["measurement_independence_certified"] is False


def test_unknown_data_folder_rejected(tmp_path):
    d, f = setup(tmp_path)
    a = args(tmp_path, d, f); a.data_root = [tmp_path / "missing"]
    with pytest.raises((ValueError, FileNotFoundError)):
        run(a)


def test_json_schema_discovered(tmp_path):
    d, f = setup(tmp_path)
    (d / "dataset.json").write_text(json.dumps([{"orientation_angle": 1, "near_field": 2}]))
    result = run(args(tmp_path, d, f))
    assert result["files"][0]["inspection_status"] == "JSON_KEYS_SAMPLED"
    assert "FIELD" in result["files"][0]["field_tags_HEURISTIC"]


def test_duplicate_root_rejected(tmp_path):
    d, f = setup(tmp_path)
    a = args(tmp_path, d, f); a.data_root = [d, d]
    with pytest.raises(ValueError, match="duplicate scan roots"):
        run(a)


def test_independent_orientation_search_not_sers_result_leak(tmp_path):
    d, f = setup(tmp_path)
    (d / "only_sers.csv").write_text("raman_band_ratio\n1.3\n")
    result = run(args(tmp_path, d, f))
    obs = result["cases"][0]["observables"]
    assert obs[0]["candidate_file_count"] == 1
    assert obs[1]["candidate_file_count"] == 0
    assert "SERS" not in obs[1]["search_tags_HEURISTIC"]
