from pathlib import Path

from pipeline_core.discovery.prospective_identification_materialization_shadow import artifact_from_contract, compact_shadow_record
from pipeline_core.discovery.relation_validity_aware_generation import IdentificationContract


class _Candidate:
    hypothesis_id = "hypothesis:test"
    premise_statement_ids = ["s1", "s2"]
    gap_statement_ids = ["g1"]
    def model_dump(self, mode="json"):
        return {"hypothesis_id": self.hypothesis_id, "premise_statement_ids": list(self.premise_statement_ids), "gap_statement_ids": list(self.gap_statement_ids)}


class _Context:
    context_id = "hypothesis_context:test"


def _prospective_contract():
    return IdentificationContract(
        proposed_relation="X may alter Y under a matched test.", independent_variable="X", dependent_observable="Y",
        comparison_context="future matched comparison", required_controls=["same acquisition"],
        measurement_compatibility="current records are not paired", measurement_compatibility_mode="PROSPECTIVE_MATCH_REQUIRED",
        measurement_support_statement_ids=["s1"], contextual_measurement_statement_ids=["s2"],
        directional_evidence_basis="grounded prediction", directionality_mode="GROUNDED_PREDICTION", directional_support_statement_ids=["s1"],
        potential_confounders=["batch"], task_alignment_rationale="same task", premise_statement_ids=["s1", "s2"],
        assessment="PARTIALLY_IDENTIFIED", limitation_reason="relation not currently established",
        current_evidence_status="PARTIAL_GROUNDING", prospective_identifiability="PROSPECTIVELY_IDENTIFIABLE",
        current_relation_support_statement_ids=[], grounded_bridge_statement_ids=["s1", "s2"],
        prospective_test_design="Vary X while holding acquisition fixed and measure Y on matched samples.",
        prospective_falsifier="Y shows no reproducible relation to X under the matched test.", ungrounded_required_concepts=[],
    )


def _not_operationalizable_contract():
    return IdentificationContract(
        proposed_relation="Ungrounded concept Z controls Y.", independent_variable="Z", dependent_observable="Y",
        comparison_context="not grounded", required_controls=[], measurement_compatibility="cannot be grounded",
        measurement_compatibility_mode="NOT_COMPARABLE", measurement_support_statement_ids=[], contextual_measurement_statement_ids=["s1", "s2"],
        directional_evidence_basis="not identifiable", directionality_mode="NOT_IDENTIFIABLE", directional_support_statement_ids=[],
        potential_confounders=[], task_alignment_rationale="same topic only", premise_statement_ids=["s1", "s2"],
        assessment="NOT_IDENTIFIABLE", limitation_reason="requires ungrounded Z", current_evidence_status="CONTEXT_ONLY",
        prospective_identifiability="NOT_OPERATIONALIZABLE", current_relation_support_statement_ids=[], grounded_bridge_statement_ids=[],
        prospective_test_design=None, prospective_falsifier=None, ungrounded_required_concepts=["Z"],
    )


def test_prospective_shadow_has_no_authority():
    artifact = artifact_from_contract(source_stage="test", context=_Context(), candidate=_Candidate(), contract=_prospective_contract())
    row = compact_shadow_record(artifact)
    assert artifact.contract_integrity_passed is True
    assert row["prospective_identifiability"] == "PROSPECTIVELY_IDENTIFIABLE"
    assert row["shadow_has_generation_authority"] is False
    assert row["shadow_has_selection_authority"] is False
    assert row["production_selection_changed"] is False


def test_not_operationalizable_is_observed_not_enforced():
    artifact = artifact_from_contract(source_stage="test", context=_Context(), candidate=_Candidate(), contract=_not_operationalizable_contract())
    row = compact_shadow_record(artifact)
    assert row["would_abstain_if_authoritative"] is True
    assert row["shadow_has_generation_authority"] is False
    assert row["shadow_has_selection_authority"] is False
    assert row["production_selection_changed"] is False


def test_materialization_hooks_cover_all_generation_routes():
    root = Path(__file__).resolve().parents[2]
    feedback = (root / "pipeline_core/discovery/sers_novelty_feedback_closed_loop.py").read_text(encoding="utf-8")
    adaptive = (root / "pipeline_core/discovery/adaptive_discovery_controller.py").read_text(encoding="utf-8")
    retrav = (root / "pipeline_core/discovery/adaptive_graph_retraversal_generation.py").read_text(encoding="utf-8")
    runner = (root / "scripts/discovery/run_adaptive_discovery_controller_shadow.py").read_text(encoding="utf-8")
    assert "run_prospective_identification_shadow(" in feedback
    assert "closed_loop_7_75_feedback_generation" in feedback
    assert "run_prospective_identification_shadow(" in adaptive
    assert "adaptive_7_76_axis_mutation" in adaptive
    assert "run_prospective_identification_shadow(" in retrav
    assert "adaptive_7_77_graph_context_reset" in retrav
    assert "prospective_identification_stage=" in runner
    assert "adaptive_7_76_" in runner
