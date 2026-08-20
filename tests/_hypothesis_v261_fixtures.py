from __future__ import annotations

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterionDraft,
    HypothesisContext,
    HypothesisEvidenceStatement,
    HypothesisGapContext,
    HypothesisMotifContext,
    HypothesisPolicy,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
    HypothesisRouteContext,
    HypothesisDesignLeverContext,
    PredictedObservationDraft,
)


def make_context() -> HypothesisContext:
    return HypothesisContext(
        context_id="hypothesis_context:test-v261",
        context_sha256="context-sha-v261",
        source_packet_id="packet:test-v261",
        source_packet_sha256="packet-sha-v261",
        source_report_id="report:test-v261",
        source_report_sha256="report-sha-v261",
        task_id="task:test-v261",
        question="How could coordination-dependent adsorption be electronically mediated?",
        corpus_id="dac_her_test",
        evidence_statements=[
            HypothesisEvidenceStatement(
                statement_id="s:reported",
                text="Coordination modulates hydrogen adsorption energetics in the reported system.",
                epistemic_role="reported",
                claim_kind="mechanism",
                paper_ids=["Kiwook_1"],
                scientific_support_node_ids=["n:reported"],
                eligible_as_premise=True,
            ),
            HypothesisEvidenceStatement(
                statement_id="s:candidate",
                text="An exploratory charge-redistribution association may accompany adsorption changes and requires verification.",
                epistemic_role="evidence_synthesis",
                claim_kind="association",
                paper_ids=["Kiwook_2"],
                scientific_support_node_ids=["n:candidate"],
                requires_verification=True,
                eligible_as_premise=True,
                premise_restrictions=["candidate_requires_verification"],
            ),
            HypothesisEvidenceStatement(
                statement_id="s:gap",
                text="The supplied context does not establish charge-transfer mediation of the coordination effect.",
                epistemic_role="unresolved",
                claim_kind="scope_limit",
                paper_ids=["Kiwook_1"],
                scientific_support_node_ids=["n:reported"],
                eligible_as_premise=False,
                eligible_as_gap=True,
                premise_restrictions=[
                    "scope_limit_not_positive_premise",
                    "unresolved_not_positive_premise",
                ],
            ),
            HypothesisEvidenceStatement(
                statement_id="s:restricted",
                text="The packet supports an adsorption connection but not an established electronic mediator.",
                epistemic_role="evidence_synthesis",
                claim_kind="scope_limit",
                paper_ids=["Kiwook_1"],
                scientific_support_node_ids=["n:reported"],
                eligible_as_premise=False,
                eligible_as_gap=False,
                premise_restrictions=["scope_limit_not_positive_premise"],
            ),
        ],
        mechanism_routes=[
            HypothesisRouteContext(
                route_id="route:test",
                statement_ids=["s:reported"],
                paper_ids=["Kiwook_1"],
                structural_type="DIRECT_MECHANISTIC",
            )
        ],
        mechanistic_motifs=[
            HypothesisMotifContext(
                motif_id="motif:test",
                label="coordination-dependent adsorption",
                statement_ids=["s:reported"],
                paper_ids=["Kiwook_1"],
            )
        ],
        reported_design_levers=[
            HypothesisDesignLeverContext(
                lever_id="lever:test",
                label="coordination environment",
                statement_ids=["s:reported"],
                paper_ids=["Kiwook_1"],
            )
        ],
        research_gaps=[
            HypothesisGapContext(
                gap_id="gap:test",
                statement_id="s:gap",
                reason="missing_direct_relation_in_packet",
            )
        ],
        partial_absence_blocked_paper_ids=["Kiwook_10"],
        policy=HypothesisPolicy(),
    )


def make_valid_draft(*, premise_id: str = "s:reported") -> HypothesisPortfolioDraft:
    return HypothesisPortfolioDraft(
        hypotheses=[
            HypothesisProposalDraft(
                local_id="h1",
                title="Electronic mediation of coordination-dependent adsorption",
                hypothesis_statement=(
                    "Coordination-dependent adsorption changes may be mediated by a corresponding redistribution of electronic charge."
                ),
                hypothesis_type="descriptor_mediation",
                premise_statement_ids=[premise_id],
                gap_statement_ids=["s:gap"],
                inferential_bridge=(
                    "The reported coordination-to-adsorption relation and the unresolved electronic mediator motivate testing charge redistribution as the connecting mechanism."
                ),
                predicted_observations=[
                    PredictedObservationDraft(
                        local_id="p1",
                        observable="electronic charge distribution and hydrogen adsorption energetics",
                        expected_direction="shift",
                        rationale=(
                            "If the proposed mediator is operative, coordination-dependent adsorption changes should co-vary with electronic redistribution."
                        ),
                    )
                ],
                falsification_criteria=[
                    FalsificationCriterionDraft(
                        local_id="f1",
                        observable="electronic charge distribution and hydrogen adsorption energetics",
                        falsifying_outcome=(
                            "Hydrogen adsorption energetics change with coordination while electronic charge distribution remains unchanged."
                        ),
                    )
                ],
                assumptions=[
                    "The compared coordination environments are otherwise chemically comparable."
                ],
            )
        ]
    )


def make_bad_novelty_draft() -> HypothesisPortfolioDraft:
    draft = make_valid_draft()
    row = draft.hypotheses[0].model_copy(
        update={
            "hypothesis_statement": (
                "We propose a novel mechanism in which coordination-dependent adsorption changes are mediated by charge redistribution."
            )
        }
    )
    return draft.model_copy(update={"hypotheses": [row]})
