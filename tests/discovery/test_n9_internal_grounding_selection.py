from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisContext,
    HypothesisEvidenceProfile,
    HypothesisEvidenceStatement,
    HypothesisPolicy,
    PredictedObservation,
)
from pipeline_core.discovery.novelty_internal_closure_review import (
    authoritative_premise_statements,
)


def _statement(
    statement_id: str,
    *,
    role: str = "reported",
    eligible: bool = True,
    verify: bool = False,
):
    return HypothesisEvidenceStatement(
        statement_id=statement_id,
        text="ICOHP correlates with hydrogen adsorption free energy.",
        epistemic_role=role,
        claim_kind="mechanism",
        paper_ids=["paper:1"],
        scientific_support_node_ids=[
            "node:1"
        ],
        scientific_support_edge_ids=[
            "edge:1"
        ],
        eligible_as_premise=eligible,
        requires_verification=verify,
    )


def _context():
    return HypothesisContext(
        context_id="context:1",
        context_sha256="sha-context",
        source_packet_id="packet:1",
        source_packet_sha256="sha-packet",
        source_report_id="report:1",
        source_report_sha256="sha-report",
        task_id="task:1",
        question="test",
        corpus_id="corpus:1",
        domain_profile_id="dac_her",
        evidence_statements=[
            _statement("stmt:good"),
            _statement(
                "stmt:not-selected"
            ),
            _statement(
                "stmt:verify",
                verify=True,
            ),
        ],
        policy=HypothesisPolicy(),
    )


def _hypothesis():
    return HypothesisCard(
        hypothesis_id="hypothesis:1",
        domain_profile_id="dac_her",
        source_context_id="context:1",
        source_context_sha256="sha-context",
        source_report_id="report:1",
        source_report_sha256="sha-report",
        title="test",
        hypothesis_statement="test hypothesis",
        hypothesis_type="context_dependency",
        premise_statement_ids=[
            "stmt:good",
            "stmt:verify",
        ],
        inferential_bridge="test bridge",
        predicted_observations=[
            PredictedObservation(
                observation_id="obs:1",
                observable="x",
                expected_direction="shift",
                rationale="test",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="false:1",
                observable="x",
                falsifying_outcome="no shift",
            )
        ],
        evidence_profile=(
            HypothesisEvidenceProfile(
                premise_count=2,
                gap_count=0,
                source_paper_count=1,
                candidate_premise_count=0,
                reported_premise_count=2,
                synthesis_premise_count=0,
            )
        ),
    )


def test_only_selected_authoritative_premises_are_supplied():
    rows = authoritative_premise_statements(
        hypothesis=_hypothesis(),
        context=_context(),
    )

    assert [
        row.statement_id
        for row in rows
    ] == [
        "stmt:good"
    ]

def test_context_id_mismatch_fails_closed():
    hypothesis = _hypothesis().model_copy(
        update={
            "source_context_id":
                "context:wrong",
        }
    )

    rows = authoritative_premise_statements(
        hypothesis=hypothesis,
        context=_context(),
    )

    assert rows == []


def test_context_sha_mismatch_fails_closed():
    hypothesis = _hypothesis().model_copy(
        update={
            "source_context_sha256":
                "sha-wrong",
        }
    )

    rows = authoritative_premise_statements(
        hypothesis=hypothesis,
        context=_context(),
    )

    assert rows == []
