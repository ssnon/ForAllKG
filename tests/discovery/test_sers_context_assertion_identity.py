from __future__ import annotations

from domains.sers.hypothesis_context_contracts import (
    HypothesisContextAssertionDraft,
    HypothesisContextInterpretationDraft,
)
from domains.sers.hypothesis_context_interpreter import (
    _canonicalize_interpretation_draft,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisEvidenceProfile,
)


def card() -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id="hypothesis:authoritative",
        domain_profile_id="sers_au_ag",
        source_context_id="context:test",
        source_context_sha256="context-sha",
        source_report_id="report:test",
        source_report_sha256="report-sha",
        title="test",
        hypothesis_statement="Central scientific statement.",
        hypothesis_type="context_dependency",
        premise_statement_ids=["stmt:test"],
        inferential_bridge="Bridge scientific statement.",
        predicted_observations=[],
        falsification_criteria=[],
        assumptions=[
            "Assumption one.",
            "Assumption two.",
            "Assumption three.",
        ],
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=1,
            gap_count=0,
            source_paper_count=1,
            candidate_premise_count=0,
            reported_premise_count=1,
            synthesis_premise_count=0,
        ),
    )


def draft_with_stale_identity(
) -> HypothesisContextInterpretationDraft:
    stale = "hypothesis:authoritativeX"

    return HypothesisContextInterpretationDraft(
        hypothesis_id=stale,
        source_signature_ids=["signature:model-noise"],
        assertions=[
            HypothesisContextAssertionDraft(
                assertion_id=f"central:{stale}",
                assertion_kind="central",
                assertion_text="Central scientific statement.",
                mentions=[],
            ),
            HypothesisContextAssertionDraft(
                assertion_id=f"bridge:{stale}",
                assertion_kind="bridge",
                assertion_text="Bridge scientific statement.",
                mentions=[],
            ),
            HypothesisContextAssertionDraft(
                assertion_id=f"assumption:{stale}:0",
                assertion_kind="assumption",
                assertion_text="Assumption one.",
                mentions=[],
            ),
            HypothesisContextAssertionDraft(
                assertion_id=f"assumption:{stale}:1",
                assertion_kind="assumption",
                assertion_text="Assumption two.",
                mentions=[],
            ),
            HypothesisContextAssertionDraft(
                assertion_id=f"assumption:{stale}:2",
                assertion_kind="assumption",
                assertion_text="Assumption three.",
                mentions=[],
            ),
        ],
    )


def test_canonicalizes_assertion_identity_from_authoritative_card():
    result = _canonicalize_interpretation_draft(
        draft=draft_with_stale_identity(),
        card=card(),
        source_signatures=[],
    )

    assert result.hypothesis_id == (
        "hypothesis:authoritative"
    )

    assert [
        row.assertion_id
        for row in result.assertions
    ] == [
        "central:hypothesis:authoritative",
        "bridge:hypothesis:authoritative",
        "assumption:hypothesis:authoritative:0",
        "assumption:hypothesis:authoritative:1",
        "assumption:hypothesis:authoritative:2",
    ]


def test_does_not_hide_semantic_assertion_mismatch():
    draft = draft_with_stale_identity()

    broken = draft.assertions[2].model_copy(
        update={
            "assertion_text":
                "Semantically different assumption.",
        }
    )

    draft = draft.model_copy(
        update={
            "assertions": [
                draft.assertions[0],
                draft.assertions[1],
                broken,
                draft.assertions[3],
                draft.assertions[4],
            ]
        }
    )

    result = _canonicalize_interpretation_draft(
        draft=draft,
        card=card(),
        source_signatures=[],
    )

    # No exact authoritative semantic counterpart:
    # stale identity remains visible for compiler rejection.
    assert result.assertions[2].assertion_id == (
        "assumption:hypothesis:authoritativeX:0"
    )
