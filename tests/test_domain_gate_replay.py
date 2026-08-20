from __future__ import annotations

from pipeline_core.chunking import ChunkSpec
from pipeline_core.corpus.domain_gate_replay import (
    build_domain_gate_replay_fixture,
    build_zero_loss_summary,
    verify_fixture_contract,
)
from domains.extraction_registry import get_extraction_adapter
from pipeline_core.draft_schema import KnowledgeGraphDraft
from domains.dac_her.micro_reextract_prompts import (
    build_domain_gate_recovery_prompt,
)


def _empty_draft() -> KnowledgeGraphDraft:
    return KnowledgeGraphDraft(
        paper_id="P",
        chunk_id="P:main:c",
        section="abstract",
        document_id="main",
        document_role="main",
        page_ids=[],
        asset_ids=[],
        entities=[],
        experiments=[],
        calculations=[],
        measurements=[],
        measurement_groups=[],
        observation_claims=[],
        mechanism_claims=[],
        edges=[],
    )


def _fixture():
    adapter = get_extraction_adapter("catalysis_mechanism")
    chunk = ChunkSpec(
        paper_id="P",
        section="abstract",
        index=0,
        core_text="Frozen source.",
        left_context="left",
        right_context="right",
        chunk_id="P:main:c",
        document_id="main",
        document_role="main",
    )
    rejected = _empty_draft()
    prompt = build_domain_gate_recovery_prompt(
        paper_id=chunk.paper_id,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        document_role=chunk.document_role,
        section=chunk.section,
        page_ids=chunk.page_ids,
        asset_ids=chunk.asset_ids,
        core_text=chunk.core_text,
        left_context=chunk.left_context,
        right_context=chunk.right_context,
        asset_context=chunk.asset_context,
        rejected_graph_payload=rejected.model_dump(),
        domain_error="reserved structured node type(s) Experiment",
    )
    return build_domain_gate_replay_fixture(
        extraction_adapter=adapter,
        chunk=chunk,
        rejected_draft=rejected,
        domain_error=ValueError(
            "reserved structured node type(s) Experiment"
        ),
        system_prompt=adapter.micro_reextract_system_prompt,
        user_prompt=prompt,
        captured_model="test-model",
        captured_provider=None,
        max_completion_tokens=4000,
    )


def test_fixture_freezes_prompt_and_schema_contract():
    fixture = _fixture()
    adapter = get_extraction_adapter("catalysis_mechanism")

    assert fixture.user_prompt.startswith("PAPER_ID:")
    assert fixture.core_text == "Frozen source."
    assert fixture.full_response_model == "KnowledgeGraphDraft"
    assert fixture.compact_response_model == "BroadMechanismGraphDraft"
    assert (
        fixture.compact_schema_estimated_tokens
        < fixture.full_schema_estimated_tokens
    )
    assert verify_fixture_contract(fixture, adapter) == []


def _row(
    *,
    condition: str,
    final: bool = True,
    mechanism: bool = True,
    mechanism_claims: int = 1,
    mechanism_edges: int = 2,
    issues=None,
    measurement_issues: int = 0,
    input_tokens: int = 100,
):
    return {
        "fixture_id": "f",
        "condition": condition,
        "llm_success": True,
        "domain_gate_pass": True,
        "strict_valid": final,
        "finalization_success": final,
        "mechanism_connected": mechanism,
        "provider_input_tokens": input_tokens,
        "provider_output_tokens": 10,
        "provider_total_tokens": input_tokens + 10,
        "measurement_issue_count": measurement_issues,
        "node_count": 4,
        "edge_count": 3,
        "mechanism_claim_count": mechanism_claims,
        "mechanism_incident_edge_count": mechanism_edges,
        "finalized_node_count": 4 if final else 0,
        "finalized_edge_count": 3 if final else 0,
        "issue_counts": issues or {},
        "finalization_issue_counts": {},
    }


def test_zero_loss_gate_never_auto_adopts():
    summary = build_zero_loss_summary(
        [
            _row(condition="full", input_tokens=100),
            _row(condition="compact", input_tokens=80),
        ]
    )
    assert summary["verdict"] == (
        "PASS_AUTOMATED_ZERO_LOSS_GATE_"
        "MANUAL_SEMANTIC_REVIEW_STILL_REQUIRED"
    )


def test_zero_loss_gate_rejects_any_observed_mechanism_loss():
    summary = build_zero_loss_summary(
        [
            _row(condition="full", mechanism_claims=2),
            _row(condition="compact", mechanism_claims=1),
        ]
    )
    assert summary["verdict"] == (
        "DO_NOT_ADOPT_OBSERVED_QUALITY_LOSS_SIGNAL"
    )


def test_zero_loss_gate_rejects_new_compact_issue_family():
    summary = build_zero_loss_summary(
        [
            _row(condition="full"),
            _row(
                condition="compact",
                issues={"MECHANISM_MISSING_SUPPORT": 1},
            ),
        ]
    )
    assert summary["verdict"] == (
        "DO_NOT_ADOPT_OBSERVED_QUALITY_LOSS_SIGNAL"
    )
