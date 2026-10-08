"""Pure local tests for the M2-B audit; no LLM or remote provider."""
from __future__ import annotations

from types import SimpleNamespace as NS
import json
from pathlib import Path

import pytest

from scripts.discovery import audit_scientific_portfolio_counterfactual_m2b as m


def _candidate(candidate_id, *, origin="FRONTIER", operator=None, sources=(), external=False, cross_source=False):
    return NS(candidate_id=candidate_id, source_object_id="source:" + candidate_id,
              origin=origin, operator_id=operator, source_kind=None,
              parent_source_kinds=list(sources), external_literature_lineage=external,
              cross_source_composition=cross_source)


def test_scenarios_select_expected_origin_and_lineage():
    rows = [
        _candidate("f", sources=["OPEN_WORLD_AXIS"], external=True),
        _candidate("e", origin="EVOLUTION", operator="BACKBONE_MUTATION", sources=["HIGHER_ORDER"]),
        _candidate("x", origin="EVOLUTION", operator="CANDIDATE_INTERPRETATION", sources=["DIRECT_HIGHER_ORDER", "HIGHER_ORDER"], cross_source=True),
    ]
    assert m._filter_candidate_ids(NS(candidates=rows), m.scenarios()["FRONTIER_ONLY"]) == {"f"}
    assert m._filter_candidate_ids(NS(candidates=rows), m.scenarios()["NO_EXPLICIT_EXTERNAL_LINEAGE"]) == {"e", "x"}
    assert m._filter_candidate_ids(NS(candidates=rows), m.scenarios()["NO_CROSS_SOURCE_EVOLUTION"]) == {"f", "e"}
    assert m._filter_candidate_ids(NS(candidates=rows), m.scenarios()["NO_HIGHER_ORDER_ANCESTRY"]) == {"f"}


def test_replay_creates_synthetic_source_identity(monkeypatch):
    rows = [_candidate("a"), _candidate("b")]
    seen = {}
    def fake_selector(*, pool, evaluation, **kwargs):
        assert pool.pool_id != "real-pool"
        assert evaluation.source_pool_id == pool.pool_id
        assert evaluation.source_pool_sha256 == pool.pool_sha256
        seen["ids"] = [x.candidate_id for x in pool.candidates]
        return NS(entries=[])
    monkeypatch.setattr(m, "build_scientific_portfolio_selection", fake_selector)
    m.replay_frozen_selection(
        pool=NS(pool_id="real-pool", candidates=rows),
        evaluation=NS(report_id="real-eval", evaluations=[NS(candidate_id="a"), NS(candidate_id="b")]),
        eligible_ids={"b"}, max_retained=8, max_per_profile=2, label="TEST",
    )
    assert seen["ids"] == ["b"]


def test_m1_pin_mismatch_is_rejected(tmp_path):
    f = tmp_path / "pool.json"
    f.write_text("{}")
    with pytest.raises(ValueError, match="SHA mismatch"):
        m._validate_m1_pin({"artifacts": [{"key": "stage7_candidate_pool", "present": True, "source_sha256": "0" * 64}]}, "stage7_candidate_pool", f)


def test_output_guard_protects_existing_dirs(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        m.run_audit(case_dir=root, output_dir=tmp_path / "out")


def test_readonly_end_to_end_with_pinned_frozen_inputs(tmp_path, monkeypatch):
    case = tmp_path / "case"
    source = case / "scientific_portfolio_shadow"
    source.mkdir(parents=True)
    pool_data = {
        "pool_id": "real-pool", "pool_sha256": "pool-sha",
        "candidates": [
            {"candidate_id": "f", "source_object_id": "frontier:f", "origin": "FRONTIER", "source_kind": "OPEN_WORLD_AXIS", "parent_source_kinds": [], "external_literature_lineage": True, "cross_source_composition": False, "operator_id": None},
            {"candidate_id": "e", "source_object_id": "evolution:e", "origin": "EVOLUTION", "source_kind": None, "parent_source_kinds": ["HIGHER_ORDER"], "external_literature_lineage": False, "cross_source_composition": False, "operator_id": "BACKBONE_MUTATION"},
        ],
    }
    eval_data = {"source_pool_id": "real-pool", "source_pool_sha256": "pool-sha", "report_id": "real-eval", "report_sha256": "eval-sha", "evaluations": [{"candidate_id": "f"}, {"candidate_id": "e"}]}
    selection_data = {
        "source_pool_id": "real-pool", "source_pool_sha256": "pool-sha",
        "source_evaluation_report_id": "real-eval", "max_retained_candidates": 2,
        "max_retained_per_profile": 2,
        "entries": [
            {"candidate_id": "f", "assigned_profile": "MECHANISM_FOCUSED", "pareto_layer": 1},
            {"candidate_id": "e", "assigned_profile": "HIGH_INFORMATION", "pareto_layer": 1},
        ],
    }
    for filename, content in [("candidate_pool.json", pool_data), ("evaluation.json", eval_data), ("selection.json", selection_data)]:
        (source / filename).write_text(json.dumps(content), encoding="utf-8")

    def to_ns(value):
        if isinstance(value, dict):
            return NS(**{k: to_ns(v) for k, v in value.items()})
        if isinstance(value, list):
            return [to_ns(v) for v in value]
        return value

    class FakeModel:
        @classmethod
        def model_validate_json(cls, text):
            return to_ns(json.loads(text))

    monkeypatch.setattr(m, "ScientificPortfolioCandidatePool", FakeModel)
    monkeypatch.setattr(m, "ScientificPortfolioEvaluationReport", FakeModel)
    monkeypatch.setattr(m, "ScientificPortfolioSelectionReport", FakeModel)

    def fake_selector(*, pool, evaluation, max_retained_candidates, **kwargs):
        ids = {x.candidate_id for x in pool.candidates}
        rows = [to_ns(x) for x in selection_data["entries"] if x["candidate_id"] in ids]
        return NS(entries=rows[:max_retained_candidates])

    monkeypatch.setattr(m, "build_scientific_portfolio_selection", fake_selector)
    manifest = {
        "source_case": str(case),
        "artifacts": [
            {"key": "stage7_candidate_pool", "present": True, "source_sha256": m._sha_path(source / "candidate_pool.json")},
            {"key": "stage7_selection", "present": True, "source_sha256": m._sha_path(source / "selection.json")},
        ],
    }
    manifest_path = tmp_path / "m1.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in source.glob("*.json")}
    dest = tmp_path / "output"
    result = m.run_audit(case_dir=case, output_dir=dest, freeze_manifest=manifest_path)
    assert result["status"] == "BASELINE_PARITY_PASSED"
    report = json.loads((dest / "selection_sensitivity.json").read_text())
    by_name = {x["scenario"]: x for x in report["scenarios"]}
    assert by_name["BASELINE"]["selected_candidate_ids"] == ["f", "e"]
    assert by_name["FRONTIER_ONLY"]["selected_candidate_ids"] == ["f"]
    assert by_name["EVOLUTION_ONLY"]["selected_candidate_ids"] == ["e"]
    assert before == {p.name: p.read_bytes() for p in source.glob("*.json")}
    assert (dest / "frozen_input_evaluation.json").is_file()
    with pytest.raises(FileExistsError):
        m.run_audit(case_dir=case, output_dir=dest, freeze_manifest=manifest_path)
