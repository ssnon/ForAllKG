from pipeline_core.discovery.research_idea_epistemic_archive import (
    EpistemicDecompositionReport,
    EpistemicRealizationRecord,
    IdeaRealizationArchive,
    MissingEvidenceRequirement,
    MultiRealizationArchiveReport,
    RealizationArchiveEntry,
)
from pipeline_core.discovery.research_idea_parallel_partial_search import (
    build_parallel_partial_search_report,
)


def req(*, rid: str, idea: str, kind: str = "RELATION_OR_MECHANISM_EVIDENCE"):
    return MissingEvidenceRequirement(
        requirement_id=rid,
        idea_id=idea,
        generation_index=3,
        kind=kind,
        description="Direct evidence for the proposed relation is missing.",
        recovery_routes=["LITERATURE_ACQUISITION", "CONTEXT_REBUILD"],
        reason_codes=["TEST_EVIDENCE_GAP"],
    )


def rec(*, eid: str, idea: str, maturity: str, status: str, requirements=None, hypothesis_id=None):
    return EpistemicRealizationRecord(
        epistemic_realization_id=eid,
        idea_id=idea,
        generation_index=3,
        realization_id=f"r:{eid}",
        hypothesis_id=hypothesis_id,
        source_context_id="ctx",
        attempt_index=1,
        realization_kind="TEST",
        materialization_status=status,
        canonical_intent="Test a mechanism",
        scientific_commitments=["A --CONTROLS--> B"],
        grounded_premise_statement_ids=["s1"] if hypothesis_id else [],
        grounded_source_paper_ids=["p1"] if hypothesis_id else [],
        prediction_texts=["B changes under intervention"],
        falsifier_texts=["B does not change"],
        grounding_coverage="STRICT" if hypothesis_id else "NONE",
        epistemic_maturity=maturity,
        missing_evidence_requirements=list(requirements or []),
    )


def entry(row, retained=True):
    return RealizationArchiveEntry(
        idea_id=row.idea_id,
        generation_index=row.generation_index,
        epistemic_realization_id=row.epistemic_realization_id,
        realization_id=row.realization_id,
        hypothesis_id=row.hypothesis_id,
        materialization_status=row.materialization_status,
        grounding_coverage=row.grounding_coverage,
        epistemic_maturity=row.epistemic_maturity,
        missing_evidence_requirement_count=len(row.missing_evidence_requirements),
        prediction_present=bool(row.prediction_texts),
        falsifier_present=bool(row.falsifier_texts),
        materialization_rank=2 if row.hypothesis_id else 0,
        operationalization_rank=1,
        evaluation_completeness_rank=1,
        falsifiability_rank=1,
        epistemic_slice_key=row.epistemic_maturity,
        archive_disposition="RETAIN_ONLY_REALIZATION",
        archive_retained=retained,
    )


def artifacts(rows):
    decomposition = EpistemicDecompositionReport(
        report_id="d",
        report_sha256="d",
        records=rows,
        record_count=len(rows),
        idea_count=len({row.idea_id for row in rows}),
        maturity_counts={},
        requirement_kind_counts={},
        literature_recovery_requirement_count=sum(
            "LITERATURE_ACQUISITION" in requirement.recovery_routes
            for row in rows
            for requirement in row.missing_evidence_requirements
        ),
        kg_retraversal_requirement_count=0,
    )
    grouped = {}
    for row in rows:
        grouped.setdefault(row.idea_id, []).append(row)
    archives = [
        IdeaRealizationArchive(
            idea_id=idea,
            generation_index=3,
            entries=[entry(row) for row in idea_rows],
            retained_epistemic_realization_ids=[row.epistemic_realization_id for row in idea_rows],
            retained_realization_ids=[row.realization_id for row in idea_rows],
            retained_hypothesis_ids=[row.hypothesis_id for row in idea_rows if row.hypothesis_id],
            residual_state_diversity=[],
            maturity_diversity=sorted({row.epistemic_maturity for row in idea_rows}),
        )
        for idea, idea_rows in grouped.items()
    ]
    archive = MultiRealizationArchiveReport(
        report_id="a",
        report_sha256="a",
        archives=archives,
        idea_count=len(archives),
        realization_entry_count=sum(len(x.entries) for x in archives),
        retained_entry_count=sum(len(x.entries) for x in archives),
        multi_realization_idea_count=sum(len(x.entries) > 1 for x in archives),
        epistemic_diversity_retention_count=0,
    )
    return decomposition, archive


def test_speculative_falsifiable_member_remains_reproductive_without_grounding():
    row = rec(eid="e1", idea="i1", maturity="SPECULATIVE_BUT_FALSIFIABLE", status="ABSTAINED")
    decomposition, archive = artifacts([row])
    report = build_parallel_partial_search_report(decomposition=decomposition, archive=archive)
    member = report.members[0]
    assert member.eligible_for_same_idea_search
    assert member.eligible_for_child_idea_parenting
    assert member.eligible_for_program_selection
    assert "PRIOR_ART_PROBE" in member.search_lanes
    assert report.speculative_falsifiable_realizations_remain_search_reproductive
    assert not report.evidence_recovery_is_blocking_inner_loop


def test_evidence_seeking_member_is_not_blocked_by_acquisition():
    row = rec(
        eid="e1",
        idea="i1",
        maturity="EVIDENCE_SEEKING",
        status="COMPILE_REJECTED",
        requirements=[req(rid="q1", idea="i1")],
    )
    decomposition, archive = artifacts([row])
    report = build_parallel_partial_search_report(decomposition=decomposition, archive=archive)
    member = report.members[0]
    assert member.eligible_for_same_idea_search
    assert member.eligible_for_child_idea_parenting
    assert member.eligible_for_prior_art_probe
    assert not member.eligible_for_evidence_acquisition_escalation
    assert report.epistemic_debts[0].disposition == "PROBE_ELIGIBLE"


def test_persistent_debt_only_enables_optional_acquisition_escalation():
    rows = [
        rec(eid="e1", idea="i1", maturity="EVIDENCE_SEEKING", status="ABSTAINED", requirements=[req(rid="q1", idea="i1")]),
        rec(eid="e2", idea="i1", maturity="EVIDENCE_SEEKING", status="ABSTAINED", requirements=[req(rid="q2", idea="i1")]),
    ]
    decomposition, archive = artifacts(rows)
    report = build_parallel_partial_search_report(
        decomposition=decomposition,
        archive=archive,
        acquisition_persistence_threshold=2,
    )
    assert report.epistemic_debts[0].disposition == "ACQUISITION_ESCALATION_ELIGIBLE"
    assert report.acquisition_escalation_eligible_debt_count == 1
    assert all(row.eligible_for_evidence_acquisition_escalation for row in report.members)
    assert all(row.eligible_for_child_idea_parenting for row in report.members)
    assert not report.epistemic_debts[0].automatic_acquisition_executed


def test_grounded_and_speculative_realizations_coexist_in_parallel_archive_search():
    grounded = rec(
        eid="g",
        idea="i1",
        maturity="OPERATIONAL_GROUNDED",
        status="MATERIALIZED",
        hypothesis_id="h1",
    )
    speculative = rec(
        eid="s",
        idea="i1",
        maturity="SPECULATIVE_BUT_FALSIFIABLE",
        status="ABSTAINED",
    )
    decomposition, archive = artifacts([grounded, speculative])
    report = build_parallel_partial_search_report(decomposition=decomposition, archive=archive)
    state = report.idea_states[0]
    assert state.remains_in_search_population
    assert state.has_parallel_realization_paths
    assert len(state.child_parent_member_ids) == 2
    assert report.child_parent_eligible_member_count == 2


def test_parallel_search_emits_nonexecuting_child_evolution_handoff():
    row = rec(eid="e1", idea="i1", maturity="SPECULATIVE_BUT_FALSIFIABLE", status="ABSTAINED")
    decomposition, archive = artifacts([row])
    report = build_parallel_partial_search_report(decomposition=decomposition, archive=archive)
    assert report.evolution_handoff_count == 1
    handoff = report.evolution_handoffs[0]
    assert handoff.idea_id == "i1"
    assert "TRANSFORM" in handoff.recommended_channels
    assert not handoff.executes_offspring_generation
    assert handoff.grounding_is_not_precondition_for_child_idea_generation


def test_no_retrieval_or_llm_calls_are_implied_by_shadow_report():
    row = rec(eid="e1", idea="i1", maturity="PARTIALLY_GROUNDED", status="ABSTAINED")
    decomposition, archive = artifacts([row])
    report = build_parallel_partial_search_report(decomposition=decomposition, archive=archive)
    assert report.new_retrieval_calls is False
    assert report.new_llm_calls is False
    assert report.offspring_generation_executed is False
    assert report.canonical_graph_mutated is False
