from __future__ import annotations

import pipeline_core.discovery.scientific_relation_ir as relation_ir_module
from domains.registry import get_domain_profile
from pipeline_core.discovery.atomic_scientific_specification import (
    CompiledAtomicSpecification,
)
from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimScientificStructure,
)
from pipeline_core.discovery.reframing.atomic_cross_lane_relation_ir_adapter import (
    compile_atomic_report_relation_ir as compile_cross_lane_relation_ir,
)
from pipeline_core.discovery.reframing.atomic_cross_lane_synthesis import (
    AtomicCrossLaneSynthesisReport,
    CompiledAtomicCrossLaneHypothesis,
)
from pipeline_core.discovery.scientific_relation_ir import (
    NullRelationTypingAdapter,
    compile_atomic_report_relation_ir as compile_compatibility_relation_ir,
    compile_atomic_specifications_relation_ir_report,
)


def _spec() -> CompiledAtomicSpecification:
    return CompiledAtomicSpecification(
        local_id="atomic:1",
        claim_id="claim:1",
        kind="moderator_interaction",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text="Factor M moderates descriptor D and outcome O.",
        rationale="test",
        source_candidate_ids=["candidate:1"],
        premise_statement_ids=["statement:1"],
        gap_statement_ids=[],
        prior_art_identity_terms=["Factor M"],
        relation_endpoint_anchors=["descriptor D", "outcome O"],
        scope_qualifier_spans=[],
        directional_qualifier_spans=[],
        relation_nucleus_terms=["descriptor D", "outcome O"],
        distinguishing_terms=["Factor M"],
        required_bridge=(
            "Factor M moderates the relation between descriptor D and outcome O."
        ),
        observable="outcome O",
        predicted_observation="Outcome O changes with descriptor D.",
        falsification_condition="Outcome O does not change with descriptor D.",
        prediction_observation_id="prediction:1",
        falsification_criterion_id="falsifier:1",
        search_concepts=["descriptor D", "outcome O"],
        search_queries=["descriptor D outcome O"],
        scientific_structure=NoveltyClaimScientificStructure(),
        scientific_structure_reason_codes=[],
    )


def _report(spec: CompiledAtomicSpecification) -> AtomicCrossLaneSynthesisReport:
    hypothesis = CompiledAtomicCrossLaneHypothesis(
        hypothesis_id="hypothesis:1",
        source_candidate_ids=["candidate:1", "candidate:2"],
        source_lanes=[
            "RELATIONAL_DISCOVERY",
            "SCIENTIFIC_REFRAMING",
        ],
        synthesis_kind="cross_lane_explanatory_synthesis",
        title="Test synthesis",
        hypothesis_statement=(
            "Factor M moderates the relation between descriptor D and outcome O."
        ),
        premise_statement_ids=["statement:1"],
        gap_statement_ids=[],
        assumptions=[],
        atomic_specifications=[spec],
    )
    return AtomicCrossLaneSynthesisReport(
        report_id="atomic_report:test",
        source_candidate_portfolio_id="candidate_portfolio:test",
        source_task_id="task:test",
        source_context_id="context:test",
        question="How does Factor M affect descriptor D and outcome O?",
        backend_name="test",
        model_name="test",
        hypotheses=[hypothesis],
        hypothesis_count=1,
        atomic_specification_count=1,
        llm_calls_performed=0,
    )


def test_core_relation_ir_has_no_cross_lane_report_type_dependency():
    assert not hasattr(
        relation_ir_module,
        "AtomicCrossLaneSynthesisReport",
    )


def test_cross_lane_adapter_matches_neutral_report_compiler():
    spec = _spec()
    report = _report(spec)
    profile = get_domain_profile("sers_au_ag")
    adapter = NullRelationTypingAdapter()

    neutral = compile_atomic_specifications_relation_ir_report(
        source_atomic_report_id=report.report_id,
        source_contract="atomic-cross-lane-scientific-synthesis-report-v1",
        specifications=[("hypothesis:1", spec)],
        domain_profile=profile,
        typing_adapter=adapter,
    )
    adapted = compile_cross_lane_relation_ir(
        report=report,
        domain_profile=profile,
        typing_adapter=adapter,
    )

    assert adapted.model_dump(mode="json") == neutral.model_dump(mode="json")


def test_legacy_report_compiler_name_is_compatibility_equivalent():
    spec = _spec()
    report = _report(spec)
    profile = get_domain_profile("sers_au_ag")
    adapter = NullRelationTypingAdapter()

    adapted = compile_cross_lane_relation_ir(
        report=report,
        domain_profile=profile,
        typing_adapter=adapter,
    )
    compatibility = compile_compatibility_relation_ir(
        report=report,
        domain_profile=profile,
        typing_adapter=adapter,
    )

    assert compatibility.model_dump(mode="json") == adapted.model_dump(
        mode="json"
    )
    assert compatibility.report_id == adapted.report_id
