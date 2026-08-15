from __future__ import annotations

import hashlib
import json

import pytest

from dac_her.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPolicy,
)
from dac_her.hypothesis_trend_grounding import (
    GroundingSourceArtifact,
    HypothesisTrendGroundingBundle,
    HypothesisTrendGroundingPolicy,
    HypothesisTrendRelationGrounding,
    capabilities_for_status,
)
from dac_her.hypothesis_trend_input import (
    HYPOTHESIS_TREND_INPUT_CONTRACT_SEMANTICS_ID,
    HypothesisTrendInputPolicy,
    _canonical_json,
    _lanes_for_grounding,
    _sha256_json,
    build_trend_aware_hypothesis_input,
    project_trend_input_views,
    validate_trend_grounding_bundle_sha,
)


INPUT_SEM = "sers_au_ag_hypothesis_trend_input_v1_alpha4c5b"


def _sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_context(corpus_id: str) -> HypothesisContext:
    payload = {
        "schema_version": "hypothesis-context-v1",
        "context_id": "context:test",
        "source_packet_id": "packet:test",
        "source_packet_sha256": "packet-sha",
        "source_report_id": "report:test",
        "source_report_sha256": "report-sha",
        "task_id": "task:test",
        "question": "test question",
        "corpus_id": corpus_id,
        "domain_profile_id": "sers_au_ag",
        "evidence_statements": [],
        "mechanism_routes": [],
        "mechanistic_motifs": [],
        "reported_design_levers": [],
        "research_gaps": [],
        "partial_absence_blocked_paper_ids": [],
        "policy": HypothesisPolicy().model_dump(mode="json"),
    }
    payload["context_sha256"] = _sha256_json(payload)
    return HypothesisContext(**payload)


def _make_grounding(status: str):
    caps = capabilities_for_status(
        status,
        directions=["positive"],
    )
    pairs = []
    repeated = []
    reversal = []
    context_specific = []
    unresolved = []

    if status == "repeated":
        pairs = ["pair:1"]
        repeated = ["pair:1"]
    elif status == "reversed":
        pairs = ["pair:1"]
        reversal = ["pair:1"]
    elif status == "context_specific":
        pairs = ["pair:1"]
        context_specific = ["pair:1"]
    elif status == "insufficient":
        pairs = []
    else:
        raise AssertionError(status)

    return HypothesisTrendRelationGrounding(
        grounding_id=f"grounding:{status}",
        contract_semantics_id=(
            "hypothesis_trend_grounding_contract_v1_alpha4c5a"
        ),
        grounding_semantics_id=(
            "sers_au_ag_hypothesis_trend_grounding_v1_alpha4c5a"
        ),
        domain_profile_id="sers_au_ag",
        relation_id="relation:1",
        independent_variable_key="particle_size",
        dependent_observable_key="sers_performance",
        control_family="structural",
        observable_semantics="qualitative_sers_performance",
        local_result_ids=["local:1"],
        paper_ids=["paper:1"],
        member_trend_ids=["trend:1"],
        directions=["positive"],
        shapes=["monotonic"],
        evidence_kinds=["reported_claim"],
        evidence_bases=["reported_directional_claim"],
        source_claim_ids=["claim:1"],
        source_node_ids=["claim:1"],
        association_only_result_ids=[],
        source_asserted_causal_trend_ids=[],
        source_requires_verification_trend_ids=[],
        cross_context_assessment_id="assessment:1",
        cross_context_status=status,
        pairwise_contrast_ids=pairs,
        repeated_pair_ids=repeated,
        reversal_pair_ids=reversal,
        context_specific_pair_ids=context_specific,
        unresolved_pair_ids=unresolved,
        differentiating_dimensions=(
            ["particle_morphology"]
            if status in {"reversed", "context_specific"}
            else []
        ),
        unresolved_dimensions=(
            ["excitation_wavelength"]
            if status == "insufficient"
            else []
        ),
        cross_context_reason_codes=["test"],
        **caps,
    )


def _make_bundle(tmp_path, status="insufficient", zero=False):
    summary_path = tmp_path / "summary.json"
    summary = {
        "trend_id": "trend:test",
        "domain_profile_id": "sers_au_ag",
        "trend_semantics_id": "sers_au_ag_trend_v5_alpha4c2121",
        "corpus_id": "corpus:test",
        "corpus_mode": "evidence",
        "paper_ids": ["paper:1", "paper:2"],
        "structural_gate": True,
    }
    summary_path.write_text(
        json.dumps(summary, sort_keys=True),
        encoding="utf-8",
    )
    artifact = GroundingSourceArtifact(
        role="trend_summary",
        path=str(summary_path),
        sha256=_sha_file(summary_path),
    )
    groundings = [] if zero else [_make_grounding(status)]
    payload = {
        "schema_version": "hypothesis-trend-grounding-bundle-v1",
        "bundle_id": "bundle:test",
        "contract_semantics_id": (
            "hypothesis_trend_grounding_contract_v1_alpha4c5a"
        ),
        "grounding_semantics_id": (
            "sers_au_ag_hypothesis_trend_grounding_v1_alpha4c5a"
        ),
        "domain_profile_id": "sers_au_ag",
        "source_trend_semantics_id": (
            "sers_au_ag_trend_v5_alpha4c2121"
        ),
        "source_precision_semantics_id": (
            "sers_au_ag_trend_precision_v5_alpha4c21211"
        ),
        "source_cross_context_contract_semantics_id": (
            None if zero else "cross_context_trend_contract_v1_alpha4c3a"
        ),
        "source_cross_context_assessment_semantics_id": (
            None
            if zero
            else "cross_context_trend_assessment_v1_alpha4c3c"
        ),
        "source_artifacts": [artifact.model_dump(mode="json")],
        "groundings": [
            row.model_dump(mode="json") for row in groundings
        ],
        "relation_count": len(groundings),
        "local_result_count": len(groundings),
        "cross_context_status_counts": (
            {} if zero else {status: 1}
        ),
        "support_role_counts": (
            {}
            if zero
            else {groundings[0].support_role: 1}
        ),
        "local_empirical_premise_count": (
            0
            if zero
            else int(groundings[0].local_empirical_premise_allowed)
        ),
        "cross_context_replicated_premise_count": (
            0
            if zero
            else int(
                groundings[0].
                cross_context_replicated_premise_allowed
            )
        ),
        "context_dependency_signal_count": (
            0
            if zero
            else int(
                groundings[0].context_dependency_premise_allowed
            )
        ),
        "reversal_counterevidence_count": (
            0
            if zero
            else int(
                groundings[0].reversal_counterevidence_required
            )
        ),
        "replication_gap_signal_count": (
            0
            if zero
            else int(
                groundings[0].replication_gap_signal_allowed
            )
        ),
        "zero_yield": zero,
        "policy": (
            HypothesisTrendGroundingPolicy().model_dump(mode="json")
        ),
    }
    payload["bundle_sha256"] = _sha256_json(payload)
    return HypothesisTrendGroundingBundle(**payload)


def test_lane_mapping_keeps_namespaces_separate():
    assert _lanes_for_grounding(
        _make_grounding("insufficient")
    ) == (
        "local_empirical_support",
        "replication_gap",
    )
    assert _lanes_for_grounding(
        _make_grounding("repeated")
    ) == (
        "local_empirical_support",
        "cross_paper_replicated_support",
    )
    assert _lanes_for_grounding(
        _make_grounding("context_specific")
    ) == (
        "local_empirical_support",
        "context_dependency_signal",
    )
    assert _lanes_for_grounding(
        _make_grounding("reversed")
    ) == (
        "local_empirical_support",
        "context_dependency_signal",
        "reversal_boundary",
    )


def test_v2_like_insufficient_builds_local_plus_gap_only(tmp_path):
    bundle = _make_bundle(tmp_path, status="insufficient")
    context = _make_context("corpus:test")
    value = build_trend_aware_hypothesis_input(
        grounded_context=context,
        trend_grounding=bundle,
        input_semantics_id=INPUT_SEM,
    )
    assert value.lane_counts == {
        "local_empirical_support": 1,
        "replication_gap": 1,
    }
    assert all(row.maker_selectable is False for row in value.trend_views)
    assert all(row.causal_use_allowed is False for row in value.trend_views)
    assert all(row.universal_use_allowed is False for row in value.trend_views)
    assert "premise_statement_ids" not in (
        value.trend_views[0].model_fields
    )
    assert value.policy == HypothesisTrendInputPolicy()


def test_corpus_binding_is_from_locked_trend_summary(tmp_path):
    bundle = _make_bundle(tmp_path, status="insufficient")
    wrong_context = _make_context("different:corpus")
    with pytest.raises(
        ValueError,
        match="SHA-locked Trend summary corpus_id",
    ):
        build_trend_aware_hypothesis_input(
            grounded_context=wrong_context,
            trend_grounding=bundle,
            input_semantics_id=INPUT_SEM,
        )


def test_tampered_grounding_sha_fails_closed(tmp_path):
    bundle = _make_bundle(tmp_path, status="insufficient")
    tampered = bundle.model_copy(
        update={"bundle_sha256": "0" * 64}
    )
    with pytest.raises(ValueError, match="bundle_sha256 mismatch"):
        validate_trend_grounding_bundle_sha(tampered)


def test_zero_yield_builds_no_views_and_no_gap(tmp_path):
    bundle = _make_bundle(tmp_path, zero=True)
    context = _make_context("corpus:test")
    value = build_trend_aware_hypothesis_input(
        grounded_context=context,
        trend_grounding=bundle,
        input_semantics_id=INPUT_SEM,
    )
    assert value.trend_views == []
    assert value.lane_counts == {}
    assert value.trend_grounding.zero_yield is True
