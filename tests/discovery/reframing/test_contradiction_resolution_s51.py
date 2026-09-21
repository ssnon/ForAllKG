from __future__ import annotations

import pytest

from pipeline_core.discovery.reframing.contradiction_resolution import (
    ContradictionResolutionBatchDraft,
    ContradictionResolutionDraft,
    ContradictionResolutionGeneration,
    ContradictionResolutionShadowRuntime,
    build_contradiction_resolution_input,
    compile_contradiction_resolution_candidate,
)
from pipeline_core.discovery.reframing.evidence_tension import (
    EvidenceLevelTensionWitness,
    ScientificEvidenceTensionReport,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    DifferentialPredictionDraft,
    DiscriminatingTestDraft,
    ReframeFalsifierDraft,
    ScientificModelDraft,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    ReframeEvidenceStatement,
    ScientificReframeEvidencePacket,
)


def _evidence() -> ScientificReframeEvidencePacket:
    return ScientificReframeEvidencePacket(
        task_id="task:1",
        question=(
            "When does far-field resonance matching predict the SERS optimum, "
            "and when does it fail?"
        ),
        source_context_id="context:1",
        source_context_sha256="a" * 64,
        premise_statements=[
            ReframeEvidenceStatement(
                statement_id="a1",
                text=(
                    "Closer LSPR-excitation matching was associated with stronger "
                    "SERS response in one reported system."
                ),
                epistemic_role="reported",
                claim_kind="association",
                paper_ids=["P1"],
            ),
            ReframeEvidenceStatement(
                statement_id="b1",
                text=(
                    "In another interacting structure, matching the LSPR peak to "
                    "the laser was not a significant contributor to SERS."
                ),
                epistemic_role="reported",
                claim_kind="mechanism",
                paper_ids=["P2"],
            ),
            ReframeEvidenceStatement(
                statement_id="c1",
                text="A third observation reports a dielectric-dependent spectral shift.",
                epistemic_role="reported",
                claim_kind="observation",
                paper_ids=["P3"],
            ),
        ],
        gap_statements=[
            ReframeEvidenceStatement(
                statement_id="g1",
                text="The packet does not establish which context controls the difference.",
                epistemic_role="unresolved",
                claim_kind="scope_limit",
                paper_ids=["P1", "P2"],
            )
        ],
        grounded_source_chunk_count=3,
        unresolved_grounded_node_count=0,
    )


def _witness(
    *,
    witness_id: str = "evidence_tension:abc",
    types: list[str] | None = None,
    side_a: list[str] | None = None,
    side_b: list[str] | None = None,
    independent: bool = True,
) -> EvidenceLevelTensionWitness:
    return EvidenceLevelTensionWitness(
        witness_id=witness_id,
        source_tension_id="tension:1",
        source_tension_type="context_dependency",
        focal_statement_id=None,
        side_a_statement_ids=side_a or ["a1"],
        side_b_statement_ids=side_b or ["b1"],
        grounded_statement_ids=sorted(set((side_a or ["a1"]) + (side_b or ["b1"]))),
        paper_ids=["P1", "P2"],
        claim_kinds=["association", "mechanism"],
        tension_types=types or ["CONTEXT", "MECHANISTIC"],
        classification_bases=[],
        relevant_condition_signature_count=2,
        relevant_condition_names=["structure"],
        paired_grounded_sides=True,
        paired_response_signal=True,
        independence_basis="cross_paper" if independent else "single_family",
        independent_family_signal=independent,
    )


def _report(witnesses: list[EvidenceLevelTensionWitness]) -> ScientificEvidenceTensionReport:
    counts: dict[str, int] = {}
    for witness in witnesses:
        for kind in witness.tension_types:
            counts[kind] = counts.get(kind, 0) + 1
    return ScientificEvidenceTensionReport(
        report_id="tension_report:1",
        source_task_id="task:1",
        source_context_id="context:1",
        source_context_sha256="a" * 64,
        source_explorer_report_id="explorer:1",
        source_explorer_report_sha256="b" * 64,
        source_evidence_sha256="c" * 64,
        source_packet_sha256="d" * 64,
        extraction_mode="packet_assisted",
        witnesses=witnesses,
        tension_type_counts=counts,
    )


def _draft(witness_id: str = "evidence_tension:abc") -> ContradictionResolutionDraft:
    return ContradictionResolutionDraft(
        local_id="r1",
        title="Context-conditioned reconciliation of resonance matching",
        tension_witness_ids=[witness_id],
        premise_statement_ids=["a1", "b1", "c1"],
        gap_statement_ids=["g1"],
        side_a_statement_ids=["a1"],
        side_b_statement_ids=["b1"],
        baseline_model=ScientificModelDraft(
            summary="One global resonance-matching rule controls SERS response.",
            assumptions=["The same matching rule applies across structures."],
            explained_statement_ids=["a1"],
            expected_observations=["Closer matching should always improve response."],
        ),
        resolution_model=ScientificModelDraft(
            summary=(
                "Resonance matching matters when the measured far-field mode remains "
                "coupled to the locally sampled enhancement, but its contribution can "
                "become weak when structural interaction redistributes the local response."
            ),
            assumptions=["Structural context can change the contribution of matching."],
            explained_statement_ids=["a1", "b1", "c1"],
            expected_observations=[
                "The sign or strength of the matching-response relation should vary with structure."
            ],
        ),
        apparent_contradiction=(
            "LSPR-excitation proximity is associated with stronger SERS in one context "
            "but is reported as not significantly contributing in another."
        ),
        resolution_principle=(
            "Condition the contribution of resonance matching on structural coupling and local response redistribution."
        ),
        resolution_kind="context_partition",
        distinguishing_context_variables=["degree of structural interaction"],
        proposed_resolution_constructs=[],
        differential_predictions=[
            DifferentialPredictionDraft(
                local_id="dp1",
                observable="association between LSPR detuning and SERS response",
                baseline_expectation="the association remains similar across structures",
                alternative_expectation="the association weakens in strongly interacting structures",
                discriminating_outcome="a context-dependent loss of the association favors the resolution model",
            )
        ],
        falsifiers=[
            ReframeFalsifierDraft(
                local_id="f1",
                falsifying_outcome=(
                    "Matched structural comparisons show the same LSPR-detuning dependence in both contexts."
                ),
            )
        ],
        discriminating_test=DiscriminatingTestDraft(
            test_design=(
                "Compare matched structures spanning interaction strength while measuring far-field LSPR and SERS excitation profiles."
            ),
            primary_observables=["LSPR detuning", "SERS response", "structural interaction"],
            baseline_favoring_outcome="one common detuning-response relation fits all structures",
            alternative_favoring_outcome="the detuning-response relation changes with structural interaction",
        ),
        unresolved_questions=["Which structural descriptor best predicts the change?"],
    )


def test_input_marks_positive_vs_null_context_witness_eligible():
    resolution_input = build_contradiction_resolution_input(
        evidence=_evidence(), tensions=_report([_witness()])
    )
    assert resolution_input.eligible_witness_ids == ["evidence_tension:abc"]
    seed = resolution_input.seeds[0]
    assert "positive_vs_null_effect" in seed.semantic_incompatibility_signals
    assert seed.scientific_conflict_authority is False


def test_strong_directional_type_is_eligible_without_lexical_signal():
    evidence = _evidence().model_copy(
        update={
            "premise_statements": [
                _evidence().premise_statements[0].model_copy(
                    update={"text": "Observation A has response alpha."}
                ),
                _evidence().premise_statements[1].model_copy(
                    update={"text": "Observation B has response beta."}
                ),
                _evidence().premise_statements[2],
            ]
        }
    )
    resolution_input = build_contradiction_resolution_input(
        evidence=evidence,
        tensions=_report([_witness(types=["DIRECTIONAL"])]),
    )
    assert resolution_input.eligible_witness_ids == ["evidence_tension:abc"]


def test_unpaired_or_nonindependent_witness_is_not_eligible():
    witness = _witness(independent=False)
    resolution_input = build_contradiction_resolution_input(
        evidence=_evidence(), tensions=_report([witness])
    )
    assert resolution_input.eligible_witness_ids == []


def test_input_rejects_task_mismatch():
    report = _report([_witness()]).model_copy(update={"source_task_id": "task:other"})
    with pytest.raises(ValueError, match="task_id"):
        build_contradiction_resolution_input(evidence=_evidence(), tensions=report)


def test_compile_requires_both_witness_sides():
    resolution_input = build_contradiction_resolution_input(
        evidence=_evidence(), tensions=_report([_witness()])
    )
    draft = _draft().model_copy(update={"side_b_statement_ids": ["a1"]})
    with pytest.raises(ValueError, match="side B"):
        compile_contradiction_resolution_candidate(
            evidence=_evidence(), resolution_input=resolution_input, draft=draft
        )


def test_compile_resolution_model_must_explain_both_sides():
    resolution_input = build_contradiction_resolution_input(
        evidence=_evidence(), tensions=_report([_witness()])
    )
    draft = _draft().model_copy(
        update={
            "resolution_model": _draft().resolution_model.model_copy(
                update={"explained_statement_ids": ["a1", "c1"]}
            )
        }
    )
    with pytest.raises(ValueError, match="side B"):
        compile_contradiction_resolution_candidate(
            evidence=_evidence(), resolution_input=resolution_input, draft=draft
        )


def test_compile_valid_candidate_preserves_shadow_authority_boundaries():
    resolution_input = build_contradiction_resolution_input(
        evidence=_evidence(), tensions=_report([_witness()])
    )
    candidate = compile_contradiction_resolution_candidate(
        evidence=_evidence(), resolution_input=resolution_input, draft=_draft()
    )
    assert candidate.operator_id == "CONTRADICTION_RESOLUTION"
    assert candidate.epistemic_status == "hypothesis_only"
    assert candidate.tension_witness_is_conflict_authority is False
    assert candidate.contradiction_claim_established is False
    assert candidate.production_selection_authority is False


def test_compile_repairs_unique_one_edit_witness_id_copy_error():
    canonical = "evidence_tension:0123456789abcdef"
    typo = "evidence_tension:0123456789abcdeg"
    resolution_input = build_contradiction_resolution_input(
        evidence=_evidence(), tensions=_report([_witness(witness_id=canonical)])
    )
    candidate = compile_contradiction_resolution_candidate(
        evidence=_evidence(), resolution_input=resolution_input, draft=_draft(typo)
    )
    assert candidate.tension_witness_ids == [canonical]
    assert any(
        "one-edit tension witness ID copy error" in row
        for row in candidate.provenance_normalizations
    )


class _Backend:
    backend_name = "fake"
    model_name = "fake-model"

    def __init__(self, draft: ContradictionResolutionBatchDraft):
        self.draft = draft
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return ContradictionResolutionGeneration(draft=self.draft)


def test_runtime_skips_without_eligible_tension_and_calls_zero_llm():
    resolution_input = build_contradiction_resolution_input(
        evidence=_evidence(), tensions=_report([_witness(independent=False)])
    )
    backend = _Backend(ContradictionResolutionBatchDraft(candidates=[]))
    report, prompt = ContradictionResolutionShadowRuntime(backend).run(
        evidence=_evidence(), resolution_input=resolution_input
    )
    assert backend.calls == 0
    assert prompt is None
    assert report.decision == "skipped_no_eligible_tension"
    assert report.llm_calls_performed == 0


def test_runtime_generates_one_shadow_candidate():
    resolution_input = build_contradiction_resolution_input(
        evidence=_evidence(), tensions=_report([_witness()])
    )
    backend = _Backend(
        ContradictionResolutionBatchDraft(candidates=[_draft()])
    )
    report, prompt = ContradictionResolutionShadowRuntime(backend).run(
        evidence=_evidence(), resolution_input=resolution_input
    )
    assert backend.calls == 1
    assert prompt is not None
    assert report.decision == "generated"
    assert len(report.candidates) == 1
    assert report.source_tensions_are_conflict_authority is False
    assert report.canonical_graph_mutated is False
