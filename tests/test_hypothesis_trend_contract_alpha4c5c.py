from __future__ import annotations

import json

import pytest

from dac_her.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPolicy,
)
from dac_her.hypothesis_trend_compiler import (
    TrendAwareHypothesisCompiler,
    TrendHypothesisCompileError,
)
from dac_her.hypothesis_trend_contracts import (
    TrendAwareFalsificationCriterionDraft,
    TrendAwareHypothesisPortfolioDraft,
    TrendAwareHypothesisProposalDraft,
    TrendAwarePredictedObservationDraft,
    TrendReferenceDraft,
)
from dac_her.hypothesis_trend_grounding import (
    GroundingSourceArtifact,
    HypothesisTrendGroundingBundle,
    HypothesisTrendGroundingPolicy,
    HypothesisTrendRelationGrounding,
    capabilities_for_status,
)
from dac_her.hypothesis_trend_input import (
    _sha256_json,
    build_trend_aware_hypothesis_input,
)
from dac_her.hypothesis_trend_validator import (
    TrendAwareHypothesisValidator,
)


def _context(corpus_id: str) -> HypothesisContext:
    payload = {
        "schema_version": "hypothesis-context-v1",
        "context_id": "context:test",
        "source_packet_id": "packet:test",
        "source_packet_sha256": "packet-sha",
        "source_report_id": "report:test",
        "source_report_sha256": "report-sha",
        "task_id": "task:test",
        "question": "test",
        "corpus_id": corpus_id,
        "domain_profile_id": "sers_au_ag",
        "evidence_statements": [],
        "mechanism_routes": [],
        "mechanistic_motifs": [],
        "reported_design_levers": [],
        "research_gaps": [],
        "partial_absence_blocked_paper_ids": [],
        "policy":
            HypothesisPolicy().model_dump(
                mode="json"
            ),
    }
    payload["context_sha256"] = _sha256_json(payload)
    return HypothesisContext(**payload)


def _grounding(status: str, papers):
    caps = capabilities_for_status(
        status,
        directions=["positive"],
    )
    pairs = []
    repeated = []
    reversal = []
    context = []
    if status == "repeated":
        pairs = ["pair:1"]
        repeated = ["pair:1"]
    elif status == "reversed":
        pairs = ["pair:1"]
        reversal = ["pair:1"]
    elif status == "context_specific":
        pairs = ["pair:1"]
        context = ["pair:1"]

    return HypothesisTrendRelationGrounding(
        grounding_id=f"grounding:{status}",
        contract_semantics_id=(
            "hypothesis_trend_grounding_contract_v1_alpha4c5a"
        ),
        grounding_semantics_id=(
            "sers_au_ag_hypothesis_trend_grounding_v1_alpha4c5a"
        ),
        domain_profile_id="sers_au_ag",
        relation_id=f"relation:{status}",
        independent_variable_key="particle_size",
        dependent_observable_key="sers_performance",
        control_family="structural",
        observable_semantics="qualitative_sers_performance",
        local_result_ids=["local:1"],
        paper_ids=list(papers),
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
        context_specific_pair_ids=context,
        unresolved_pair_ids=[],
        differentiating_dimensions=(
            ["morphology"]
            if status
            in {"reversed", "context_specific"}
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


def _bundle(tmp_path, status="insufficient", papers=("p1",)):
    summary = {
        "trend_id": "trend:test",
        "domain_profile_id": "sers_au_ag",
        "trend_semantics_id":
            "sers_au_ag_trend_v5_alpha4c2121",
        "corpus_id": "corpus:test",
        "corpus_mode": "evidence",
        "paper_ids": sorted(set(papers) | {"p-extra"}),
        "structural_gate": True,
    }
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(summary, sort_keys=True),
        encoding="utf-8",
    )
    import hashlib
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    artifact = GroundingSourceArtifact(
        role="trend_summary",
        path=str(path),
        sha256=sha,
    )
    g = _grounding(status, papers)
    payload = {
        "schema_version":
            "hypothesis-trend-grounding-bundle-v1",
        "bundle_id": "bundle:test",
        "contract_semantics_id": (
            "hypothesis_trend_grounding_contract_v1_alpha4c5a"
        ),
        "grounding_semantics_id": (
            "sers_au_ag_hypothesis_trend_grounding_v1_alpha4c5a"
        ),
        "domain_profile_id": "sers_au_ag",
        "source_trend_semantics_id":
            "sers_au_ag_trend_v5_alpha4c2121",
        "source_precision_semantics_id": (
            "sers_au_ag_trend_precision_v5_alpha4c21211"
        ),
        "source_cross_context_contract_semantics_id":
            "cross_context_trend_contract_v1_alpha4c3a",
        "source_cross_context_assessment_semantics_id":
            "cross_context_trend_assessment_v1_alpha4c3c",
        "source_artifacts": [
            artifact.model_dump(mode="json")
        ],
        "groundings": [g.model_dump(mode="json")],
        "relation_count": 1,
        "local_result_count": 1,
        "cross_context_status_counts": {status: 1},
        "support_role_counts": {g.support_role: 1},
        "local_empirical_premise_count":
            int(g.local_empirical_premise_allowed),
        "cross_context_replicated_premise_count":
            int(
                g.cross_context_replicated_premise_allowed
            ),
        "context_dependency_signal_count":
            int(g.context_dependency_premise_allowed),
        "reversal_counterevidence_count":
            int(g.reversal_counterevidence_required),
        "replication_gap_signal_count":
            int(g.replication_gap_signal_allowed),
        "zero_yield": False,
        "policy":
            HypothesisTrendGroundingPolicy().
            model_dump(mode="json"),
    }
    payload["bundle_sha256"] = _sha256_json(payload)
    return HypothesisTrendGroundingBundle(**payload)


def _source(tmp_path, status="insufficient", papers=("p1",)):
    return build_trend_aware_hypothesis_input(
        grounded_context=_context("corpus:test"),
        trend_grounding=_bundle(
            tmp_path,
            status=status,
            papers=papers,
        ),
        input_semantics_id=(
            "sers_au_ag_hypothesis_trend_input_v1_alpha4c5b"
        ),
    )


def _draft(refs):
    return TrendAwareHypothesisPortfolioDraft(
        hypotheses=[
            TrendAwareHypothesisProposalDraft(
                local_id="h1",
                title="test",
                hypothesis_statement=(
                    "The scoped relation should remain "
                    "qualitatively observable."
                ),
                hypothesis_type="context_dependency",
                premise_statement_ids=[],
                gap_statement_ids=[],
                trend_references=refs,
                inferential_bridge=(
                    "The empirical relation is used only "
                    "within its allowed provenance scope."
                ),
                predicted_observations=[
                    TrendAwarePredictedObservationDraft(
                        local_id="p1",
                        observable="sers_performance",
                        expected_direction="increase",
                        rationale="scoped qualitative prediction",
                    )
                ],
                falsification_criteria=[
                    TrendAwareFalsificationCriterionDraft(
                        local_id="f1",
                        observable="sers_performance",
                        falsifying_outcome=(
                            "the scoped increase is not observed"
                        ),
                    )
                ],
            )
        ],
        abstention_reason=None,
    )


def _view(source, lane):
    return next(
        row for row in source.trend_views
        if row.lane == lane
    )


def test_insufficient_requires_replication_gap(tmp_path):
    source = _source(tmp_path)
    local = _view(source, "local_empirical_support")
    draft = _draft([
        TrendReferenceDraft(
            view_id=local.view_id,
            use_role="positive_empirical_support",
        )
    ])
    with pytest.raises(TrendHypothesisCompileError) as exc:
        TrendAwareHypothesisCompiler().compile(
            source,
            draft,
        )
    assert any(
        row.code == "MISSING_REPLICATION_GAP_COMPANION"
        for row in exc.value.issues
    )


def test_insufficient_local_plus_gap_trend_only_passes(tmp_path):
    source = _source(tmp_path)
    draft = _draft([
        TrendReferenceDraft(
            view_id=_view(
                source,
                "local_empirical_support",
            ).view_id,
            use_role="positive_empirical_support",
        ),
        TrendReferenceDraft(
            view_id=_view(
                source,
                "replication_gap",
            ).view_id,
            use_role="replication_gap",
        ),
    ])
    portfolio = TrendAwareHypothesisCompiler().compile(
        source,
        draft,
    )
    result = TrendAwareHypothesisValidator().validate(
        source,
        portfolio,
    )
    assert result.passes
    card = portfolio.hypotheses[0]
    assert card.premise_statement_ids == []
    assert card.cross_paper_synthesis is False
    assert card.evidence_profile.trend_positive_support_count == 1
    assert card.evidence_profile.trend_gap_count == 1


def test_replication_gap_cannot_be_positive_support(tmp_path):
    source = _source(tmp_path)
    gap = _view(source, "replication_gap")
    draft = _draft([
        TrendReferenceDraft(
            view_id=gap.view_id,
            use_role="positive_empirical_support",
        )
    ])
    with pytest.raises(TrendHypothesisCompileError) as exc:
        TrendAwareHypothesisCompiler().compile(
            source,
            draft,
        )
    assert any(
        row.code == "TREND_USE_LANE_MISMATCH"
        for row in exc.value.issues
    )


def test_context_specific_requires_context_qualification(tmp_path):
    source = _source(
        tmp_path,
        status="context_specific",
        papers=("p1", "p2"),
    )
    local = _view(source, "local_empirical_support")
    with pytest.raises(TrendHypothesisCompileError) as exc:
        TrendAwareHypothesisCompiler().compile(
            source,
            _draft([
                TrendReferenceDraft(
                    view_id=local.view_id,
                    use_role="positive_empirical_support",
                )
            ]),
        )
    assert any(
        row.code
        == "MISSING_CONTEXT_QUALIFICATION_COMPANION"
        for row in exc.value.issues
    )


def test_reversed_requires_context_and_counterevidence(tmp_path):
    source = _source(
        tmp_path,
        status="reversed",
        papers=("p1", "p2"),
    )
    local = _view(source, "local_empirical_support")
    with pytest.raises(TrendHypothesisCompileError) as exc:
        TrendAwareHypothesisCompiler().compile(
            source,
            _draft([
                TrendReferenceDraft(
                    view_id=local.view_id,
                    use_role="positive_empirical_support",
                )
            ]),
        )
    codes = {row.code for row in exc.value.issues}
    assert "MISSING_CONTEXT_QUALIFICATION_COMPANION" in codes
    assert "MISSING_REVERSAL_BOUNDARY_COMPANION" in codes


def test_repeated_cross_paper_support_is_explicit(tmp_path):
    source = _source(
        tmp_path,
        status="repeated",
        papers=("p1", "p2"),
    )
    cross = _view(
        source,
        "cross_paper_replicated_support",
    )
    portfolio = TrendAwareHypothesisCompiler().compile(
        source,
        _draft([
            TrendReferenceDraft(
                view_id=cross.view_id,
                use_role="cross_paper_empirical_support",
            )
        ]),
    )
    result = TrendAwareHypothesisValidator().validate(
        source,
        portfolio,
    )
    assert result.passes
    card = portfolio.hypotheses[0]
    assert card.cross_paper_synthesis is True
    assert (
        card.evidence_profile.
        trend_cross_paper_support_count
        == 1
    )


def test_unknown_trend_view_fails_closed(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(TrendHypothesisCompileError) as exc:
        TrendAwareHypothesisCompiler().compile(
            source,
            _draft([
                TrendReferenceDraft(
                    view_id="trend_view:not-real",
                    use_role="positive_empirical_support",
                )
            ]),
        )
    assert any(
        row.code == "UNKNOWN_TREND_VIEW"
        for row in exc.value.issues
    )


def test_validator_detects_tampered_support_scope(tmp_path):
    source = _source(tmp_path)
    draft = _draft([
        TrendReferenceDraft(
            view_id=_view(
                source,
                "local_empirical_support",
            ).view_id,
            use_role="positive_empirical_support",
        ),
        TrendReferenceDraft(
            view_id=_view(
                source,
                "replication_gap",
            ).view_id,
            use_role="replication_gap",
        ),
    ])
    portfolio = TrendAwareHypothesisCompiler().compile(
        source,
        draft,
    )
    card = portfolio.hypotheses[0].model_copy(
        update={"support_paper_ids": ["fabricated"]}
    )
    tampered = portfolio.model_copy(
        update={"hypotheses": [card]}
    )
    result = TrendAwareHypothesisValidator().validate(
        source,
        tampered,
    )
    assert not result.passes
    assert any(
        row.code == "PAPER_SCOPE_MISMATCH"
        for row in result.issues
    )
