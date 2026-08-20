from __future__ import annotations

import json
from pathlib import Path

from scripts.demo_viewer_runtime import build_demo_viewer, discover_feasibility_dir, load_demo_payload


def _dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "run"
    feas = run / "feasibility_v02"
    hid = "hypothesis:test"
    _dump(
        feas / "feasibility" / "intake.json",
        {
            "schema_version": "feasibility-intake-v0",
            "intake_id": "intake:test",
            "intake_sha256": "abc",
            "question": "How does coordination affect HER?",
            "corpus_id": "test_corpus",
            "task_id": "task:test",
            "abstention_reason": None,
            "hypotheses": [
                {
                    "hypothesis_id": hid,
                    "title": "Coordination controls HER",
                    "statement": "Coordination may change HER activity.",
                    "hypothesis_type": "descriptor_mediation",
                    "inferential_bridge": "Evidence is extended into a testable relation.",
                    "assumptions": ["Comparable structures exist."],
                    "source_paper_ids": ["Paper_1"],
                    "premises": [
                        {
                            "statement_id": "stmt:1",
                            "text": "Paper 1 reports a coordination/activity relation.",
                            "epistemic_role": "reported",
                            "claim_kind": "mechanism",
                            "paper_ids": ["Paper_1"],
                            "requires_verification": False,
                        }
                    ],
                    "predictions": [
                        {
                            "observation_id": "pred:1",
                            "observable": "HER activity",
                            "expected_direction": "qualitative_change",
                            "rationale": "Coordination changes the local state.",
                        }
                    ],
                    "falsifiers": [
                        {
                            "criterion_id": "fal:1",
                            "observable": "HER activity",
                            "falsifying_outcome": "No change under matched comparison.",
                        }
                    ],
                    "semantic_gate_status": "eligible_with_warnings",
                    "semantic_warning_dimensions": ["directional_specificity"],
                    "semantic_fail_dimensions": [],
                }
            ],
        },
    )
    _dump(
        feas / "scope" / "hypothesis_test.json",
        {
            "hypothesis_id": hid,
            "scope_id": "scope:test",
            "catalyst_class": "dual_atom",
            "hypothesis_level": "comparative_study",
            "reaction": "HER",
            "environments": [],
            "metals": [],
            "coordination_variables": ["nitrogen_coordination_number"],
            "independent_variables": ["nitrogen_coordination_number"],
            "dependent_observables": ["HER_activity"],
            "requires_candidate_concretization": True,
            "scope_confidence": "high",
            "scope_warnings": ["metal_identity_not_concrete"],
        },
    )
    _dump(
        feas / "validation" / "hypothesis_test.json",
        {
            "hypothesis_id": hid,
            "specification_id": "spec:test",
            "validation_strategy": "comparative_computational_study",
            "requires_candidate_concretization": True,
            "controlled_variables": ["support_model"],
            "varied_variables": ["nitrogen_coordination_number"],
            "primary_observables": ["HER_activity"],
            "secondary_observables": ["structural_stability"],
            "required_comparisons": ["compare matched coordination structures"],
            "candidate_concretization_requirements": ["select explicit metal pair(s)"],
            "required_physics_checks": ["pair_stability"],
            "not_applicable_physics_checks": ["isolated_site_stability"],
            "not_applicable_experimental_capabilities": ["isolated_single_atom_synthesis"],
            "success_patterns": ["HER_activity: expected_direction=qualitative_change"],
            "falsification_patterns": ["HER_activity: no change"],
            "next_actions": ["concretize validation systems"],
        },
    )
    _dump(
        feas / "physics" / "hypothesis_test.json",
        {
            "hypothesis_id": hid,
            "report_id": "physics:test",
            "disposition": "requires_computation",
            "confidence": "medium",
            "checks": [
                {
                    "check_type": "pair_stability",
                    "status": "requires_computation",
                    "basis": "unavailable",
                    "rationale": "No computation backend configured.",
                }
            ],
            "unresolved_checks": ["pair_stability"],
            "not_applicable_checks": ["isolated_site_stability"],
            "next_required_computations": ["Resolve pair_stability."],
        },
    )
    _dump(
        feas / "experimental" / "hypothesis_test.json",
        {
            "hypothesis_id": hid,
            "report_id": "experimental:test",
            "disposition": "conditionally_plausible",
            "precedent_status": "not_assessed",
            "checks": [
                {
                    "check_type": "performance_testability",
                    "status": "pass",
                    "rationale": "HER activity is directly testable.",
                }
            ],
            "required_characterization": ["atomic_resolution_microscopy"],
            "required_electrochemical_tests": ["her_polarization_and_kinetic_testing"],
        },
    )
    _dump(
        feas / "decision" / "portfolio.json",
        {
            "cards": [
                {
                    "hypothesis_id": hid,
                    "decision_id": "decision:test",
                    "final_disposition": "requires_validation_design",
                    "required_computations": ["Resolve pair_stability."],
                    "required_characterization": ["atomic_resolution_microscopy"],
                    "required_electrochemical_tests": ["her_polarization_and_kinetic_testing"],
                    "key_uncertainties": ["physics:pair_stability"],
                }
            ]
        },
    )
    _dump(feas / "manifest.json", {"schema_version": "feasibility-e2e-manifest-v02"})
    return run, feas


def test_discover_and_load_payload(tmp_path: Path) -> None:
    run, feas = _fixture(tmp_path)
    assert discover_feasibility_dir(run) == feas.resolve()
    payload = load_demo_payload(feas, run_dir=run)
    assert payload["question"] == "How does coordination affect HER?"
    assert payload["paper_ids"] == ["Paper_1"]
    assert payload["hypotheses"][0]["decision"]["final_disposition"] == "requires_validation_design"


def test_build_self_contained_html(tmp_path: Path) -> None:
    run, _ = _fixture(tmp_path)
    output = run / "demo" / "index.html"
    result = build_demo_viewer(run_dir=run, output=output)
    html = result.read_text(encoding="utf-8")
    assert "GraphAgentsDAC Hypothesis Lineage &amp; Validation Viewer" in html or "GraphAgentsDAC Hypothesis Lineage & Validation Viewer" in html
    assert "Coordination controls HER" in html
    assert "Verification matrix" in html
    assert "requires_validation_design" in html
    assert '<script src=' not in html
    assert '<link rel=' not in html
