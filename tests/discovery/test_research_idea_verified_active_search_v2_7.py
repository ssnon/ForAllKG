from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace as NS

from pipeline_core.discovery.research_idea_active_realization import (
    ExactFeedbackAttachmentReport,
    FeedbackAttachmentRecord,
)
from pipeline_core.discovery.research_idea_closed_loop import RealizationLifecycleReport
from pipeline_core.discovery.research_idea_closed_loop_v2_6 import CreditContinuityReportV2
from pipeline_core.discovery.research_idea_contracts import ResearchIdeaKernel, ResearchIdeaNode
from pipeline_core.discovery.research_idea_offspring_execution import OffspringExecutionReport
from pipeline_core.discovery.research_idea_population_semantics import (
    IdeaRealizationLink,
    ScientificFeedbackFacetObservation,
)
from pipeline_core.discovery.research_idea_program_families import (
    build_scientific_program_families,
)
from pipeline_core.discovery.research_idea_verified_active_search import (
    FreshResidualVerificationReport,
    attach_feedback_with_fresh_residual,
    build_cost_audit,
    build_idea_evolution_requests,
    normalize_lifecycle_strict,
)


CTX = "ctx:test"
SHA = "c" * 64


def _node(idea_id: str, core: list[str], *, generation: int = 0, parents=None, scope=None):
    kernel = ResearchIdeaKernel(
        canonical_intent="test program",
        core_scientific_commitments=core,
        scope_commitments=scope or [],
    )
    raw = json.dumps(kernel.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return ResearchIdeaNode(
        idea_id=idea_id,
        generation_index=generation,
        parent_idea_ids=list(parents or []),
        source_context_id=CTX,
        source_context_sha256=SHA,
        origin_kind=("FRONTIER" if generation == 0 else "GENERATIONAL_OFFSPRING"),
        source_object_id=f"src:{idea_id}",
        kernel=kernel,
        kernel_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        task_relation_mode="DIRECT",
    )


def _link(idea_id="i1", rid="r1", hid="h1", *, status="MATERIALIZED", attempt=1):
    return IdeaRealizationLink(
        realization_id=rid,
        idea_id=idea_id,
        generation_index=2,
        hypothesis_id=hid,
        source_context_id=CTX,
        realization_kind="INITIAL",
        attempt_index=attempt,
        materialization_status=status,
    )


def _obs(idea_id="i1", rid="r1", hid="h1", *, ident=None, complete=False):
    return ScientificFeedbackFacetObservation(
        observation_id=f"obs:{rid}",
        idea_id=idea_id,
        realization_id=rid,
        hypothesis_id=hid,
        materialization_status="MATERIALIZED",
        grounding_integrity="PASSED",
        prospective_status=("COMPLETE" if complete else None),
        prospective_identifiability=ident,
        prospective_contract_integrity_passed=(True if complete else None),
    )


def _lifecycle(obs):
    link = _link()
    return RealizationLifecycleReport(
        report_id="life:1",
        report_sha256="l" * 64,
        generation_index=2,
        source_offspring_execution_report_id="exec:1",
        source_context_id=CTX,
        source_context_sha256=SHA,
        target_idea_ids=["i1"],
        links=[link],
        observations=[obs],
        local_states=[],
        output_portfolio_id="portfolio:1",
        materialized_hypothesis_count=1,
        usable_grounded_realization_count=0,
        rescued_within_same_idea_count=0,
        local_search_exhausted_count=0,
        prospective_audit_count=int(obs.prospective_status == "COMPLETE"),
        prospective_not_operationalizable_count=int(obs.prospective_identifiability == "NOT_OPERATIONALIZABLE"),
        realization_llm_call_count=0,
        prospective_audit_llm_call_count=0,
        max_realizations_per_idea=2,
        max_repair_attempts=1,
    )


def _feedback(residual: str | None):
    return ExactFeedbackAttachmentReport(
        report_id="fb:1",
        report_sha256="f" * 64,
        source_lifecycle_report_id="life:1",
        records=[
            FeedbackAttachmentRecord(
                hypothesis_id="h1",
                idea_id="i1",
                prospective_evaluation_state="EVALUATED",
                residual_evaluation_state=("EVALUATED" if residual else "NOT_EVALUATED"),
                prospective_identifiability="NOT_OPERATIONALIZABLE",
                residual_epistemic_state=residual,
            )
        ],
        prospective_evaluated_count=1,
        prospective_not_evaluated_count=0,
        prospective_ambiguous_count=0,
        residual_evaluated_count=int(residual is not None),
        residual_not_evaluated_count=int(residual is None),
        residual_ambiguous_count=0,
    )


def test_scientific_program_family_groups_mechanism_variants_but_keeps_distinct_endpoints():
    nodes = [
        _node("a0", ["gap confinement --MODULATES--> mediator M", "mediator M --CONTROLS--> hotspot distribution"]),
        _node("a1", ["gap confinement --MODULATES--> mediator N", "mediator N --CONTROLS--> hotspot redistribution"], generation=2, parents=["a0"]),
        _node("b0", ["reporter affinity --MODULATES--> binding state", "binding state --CONTROLS--> spectral ratio"]),
        _node("b1", ["reporter affinity --MODULATES--> adsorption state", "adsorption state --CONTROLS--> spectral ratio"], generation=2, parents=["b0"]),
    ]
    report = build_scientific_program_families(nodes, similarity_floor=0.40, endpoint_floor=0.35)
    assert report.program_family_count == 2
    family_by_idea = {row.idea_id: row.program_family_key for row in report.assignments}
    assert family_by_idea["a0"] == family_by_idea["a1"]
    assert family_by_idea["b0"] == family_by_idea["b1"]
    assert family_by_idea["a0"] != family_by_idea["b0"]


def test_complete_link_guard_prevents_bridge_chaining_into_giant_family():
    nodes = [
        _node("a", ["alpha field --MODULATES--> shared bridge"]),
        _node("b", ["shared bridge --MODULATES--> beta field"]),
        _node("c", ["beta field --MODULATES--> gamma output"]),
    ]
    report = build_scientific_program_families(
        nodes,
        similarity_floor=0.22,
        endpoint_floor=0.20,
    )
    assert report.endpoint_anchor_guard_prevents_bridge_chaining is True
    assert report.program_family_count >= 2


def test_strict_lifecycle_does_not_treat_not_evaluated_materialization_as_usable():
    lifecycle = normalize_lifecycle_strict(_lifecycle(_obs(complete=False)))
    state = lifecycle.local_states[0]
    assert state.materialized_count == 1
    assert state.usable_grounded_realization_count == 0
    assert state.local_search_exhausted is False


def test_strict_lifecycle_accepts_completed_prospective_identifiable_realization():
    lifecycle = normalize_lifecycle_strict(_lifecycle(_obs(ident="PROSPECTIVELY_IDENTIFIABLE", complete=True)))
    assert lifecycle.local_states[0].usable_grounded_realization_count == 1


def test_fresh_residual_attachment_requires_exact_portfolio_lineage(tmp_path):
    lifecycle = normalize_lifecycle_strict(_lifecycle(_obs(ident="NOT_OPERATIONALIZABLE", complete=True)))
    bad = {"source_portfolio_id": "wrong", "hypotheses": []}
    try:
        attach_feedback_with_fresh_residual(
            lifecycle=lifecycle,
            fresh_residual_state=bad,
            prospective_search_roots=[tmp_path],
        )
    except ValueError as exc:
        assert "portfolio lineage mismatch" in str(exc)
    else:
        raise AssertionError("expected exact-lineage failure")


def test_known_region_after_local_attempts_emits_nonexecuting_evolution_request():
    lifecycle = normalize_lifecycle_strict(_lifecycle(_obs(ident="NOT_OPERATIONALIZABLE", complete=True)), extra_local_budget=2)
    families = build_scientific_program_families([_node("i1", ["A --CAUSES--> B"], generation=2, parents=["p"])])
    feedback = _feedback("PRIOR_ART_BACKED_OR_NO_RESIDUAL")
    active = NS(
        action_records=[NS(idea_id="i1", action="EVIDENCE_REAXIS")],
    )
    report = build_idea_evolution_requests(
        generation_index=2,
        lifecycle=lifecycle,
        final_feedback=feedback,
        action_reports=[active],
        families=families,
    )
    assert report.request_count == 1
    req = report.requests[0]
    assert req.requested_mutation_scope in {"OPERATIONALIZATION_ESCAPE", "KNOWN_REGION_ESCAPE"}
    assert req.request_executes_g4_generation is False
    assert req.program_family_key is not None


def test_cost_audit_keeps_cost_and_scientific_outcomes_separate():
    family = build_scientific_program_families([_node("i1", ["A --CAUSES--> B"])])
    active = NS(local_llm_call_count=10, prospective_audit_llm_call_count=4, rescued_within_same_idea_ids=["i1"])
    verification = FreshResidualVerificationReport(
        report_id="v:1",
        report_sha256="v" * 64,
        generation_index=2,
        round_index=0,
        source_portfolio_id="p",
        hypothesis_count=1,
        status="COMPLETE",
        verification_subprocess_call_count=5,
        fresh_exact_lineage_residual_created=True,
    )
    initial = CreditContinuityReportV2(
        report_id="c0", report_sha256="0" * 64, traces=[], outcome_counts={},
        generation2_same_idea_rescue_count=0, generation3_same_idea_rescue_count=0,
        total_same_idea_rescue_count=0, child_rescue_count=2, productive_lineage_count=0,
        unresolved_count=7,
    )
    final = initial.model_copy(update={"report_id": "c1", "child_rescue_count": 2, "unresolved_count": 5})
    execution = OffspringExecutionReport.model_construct(llm_call_count=8)
    audit = build_cost_audit(
        family_report=family,
        active_reports=[active],
        fresh_verification_reports=[verification],
        initial_credit=initial,
        final_credit=final,
        generation3_execution=execution,
    )
    assert audit.same_idea_rescue_count == 1
    assert audit.same_idea_rescue_per_local_llm_call == 0.1
    assert audit.unresolved_reduction == 2
    assert audit.scalar_fitness_used is False


def test_program_family_contract_does_not_use_operator_as_authority():
    parent = _node("p", ["A --MODULATES--> M", "M --CONTROLS--> B"])
    child = _node("c", ["A --MODULATES--> N", "N --CONTROLS--> B"], generation=2, parents=["p"])
    child = child.model_copy(update={"operator_id": "AXIS_MUTATION"})
    report = build_scientific_program_families([parent, child], similarity_floor=0.35, endpoint_floor=0.30)
    assert report.operator_id_is_not_family_authority is True
    assert report.family_is_not_hard_selection_gate is True


def test_four_backbone_population_remains_four_program_families():
    nodes = []
    for family in range(4):
        for index in range(16):
            nodes.append(
                _node(
                    f"f{family}:{index}",
                    [
                        f"endpoint {family} source --MODULATES--> mediator {index}",
                        f"mediator {index} --CONTROLS--> endpoint {family} target",
                    ],
                    scope=[f"modifier regime {index % 4}"],
                )
            )
    report = build_scientific_program_families(
        nodes,
        similarity_floor=0.38,
        endpoint_floor=0.55,
    )
    assert report.program_family_count == 4
