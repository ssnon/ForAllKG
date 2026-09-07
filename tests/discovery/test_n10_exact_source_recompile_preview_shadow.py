from pipeline_core.discovery.external_novelty_contracts import NoveltyClaimDraft
from pipeline_core.discovery.hypothesis_contracts import HypothesisCard
from pipeline_core.discovery.novelty_claim_decomposition import (
    _plan_exact_source_atomic_recompile_shadow,
    _preview_exact_source_atomic_recompile_shadow,
)


EXACT_SOURCE = (
    "Under fixed temperature, Catalyst A increases product yield "
    "relative to Catalyst B."
)


def _card(
    *,
    bridge: str = EXACT_SOURCE,
    assumptions: list[str] | None = None,
) -> HypothesisCard:
    return HypothesisCard.model_validate(
        {
            "hypothesis_id": "hypothesis:s14n",
            "domain_profile_id": "domain:s14n",
            "source_context_id": "context:s14n",
            "source_context_sha256": "context-sha",
            "source_report_id": "report:s14n",
            "source_report_sha256": "report-sha",
            "title": "Synthetic exact-source preview fixture",
            "hypothesis_statement": EXACT_SOURCE,
            "hypothesis_type": "context_dependency",
            "premise_statement_ids": ["premise:1"],
            "gap_statement_ids": ["gap:1"],
            "inferential_bridge": bridge,
            "predicted_observations": [
                {
                    "observation_id": "obs:1",
                    "observable": "Catalyst A product yield",
                    "expected_direction": "increase",
                    "rationale": (
                        "Under fixed temperature, Catalyst A product yield "
                        "increases relative to Catalyst B."
                    ),
                }
            ],
            "falsification_criteria": [
                {
                    "criterion_id": "fals:1",
                    "observable": "Catalyst A product yield",
                    "falsifying_outcome": (
                        "Catalyst A does not increase product yield relative "
                        "to Catalyst B under fixed temperature."
                    ),
                }
            ],
            "assumptions": list(assumptions or []),
            "source_paper_ids": ["paper:synthetic"],
            "gap_paper_ids": [],
            "cross_paper_synthesis": False,
            "candidate_dependency": "none",
            "evidence_profile": {
                "premise_count": 1,
                "gap_count": 1,
                "source_paper_count": 1,
                "candidate_premise_count": 0,
                "reported_premise_count": 1,
                "synthesis_premise_count": 0,
            },
            "status": "hypothesized",
            "novelty_status": "not_assessed",
        }
    )


def _claim(
    *,
    prediction_id: str = "obs:1",
) -> NoveltyClaimDraft:
    return NoveltyClaimDraft.model_validate(
        {
            "local_id": "claim_1",
            "kind": "context_condition",
            "importance": "core",
            "novelty_selection_role": "NOVELTY_BEARING",
            "text": "Catalyst A affects product yield relative to Catalyst B.",
            "rationale": "Synthetic preview fixture only.",
            "search_concepts": ["Catalyst A"],
            "search_queries": ["Catalyst A product yield"],
            "distinguishing_terms": ["fixed temperature"],
            "prior_art_identity_terms": ["Catalyst A"],
            "relation_nucleus_terms": ["product yield"],
            "semantic_fidelity_binding": {
                "proposition_basis": EXACT_SOURCE,
                "relation_endpoint_anchors": [
                    "Catalyst A",
                    "product yield relative to Catalyst B",
                ],
                "scope_qualifier_spans": ["Under fixed temperature"],
                "directional_qualifier_spans": ["increases"],
                "prediction_observation_id": prediction_id,
                "falsification_criterion_id": "fals:1",
            },
            "required_bridge": "",
            "predicted_observation": (
                "Under fixed temperature, Catalyst A product yield increases "
                "relative to Catalyst B."
            ),
            "falsification_condition": (
                "Catalyst A does not increase product yield relative to "
                "Catalyst B under fixed temperature."
            ),
        }
    )


def _plan(card: HypothesisCard, claim: NoveltyClaimDraft) -> dict[str, object]:
    return _plan_exact_source_atomic_recompile_shadow(
        card,
        claim,
        sanitized_required_bridge="",
    )


def test_unique_novelty_bearing_candidate_compiles_to_pass_shadow_preview() -> None:
    card = _card()
    claim = _claim()
    plan = _plan(card, claim)

    preview = _preview_exact_source_atomic_recompile_shadow(
        card,
        claim,
        recompile_plan=plan,
    )

    assert plan["classification"] == "EXACT_SOURCE_RECOMPILE_CANDIDATE"
    assert preview["preview_status"] == "PASS_SHADOW_PREVIEW"
    assert preview["ready_for_bounded_recompile"] is True
    assert preview["diagnostic_only"] is True
    assert preview["production_authority"] is False
    assert preview["production_recompile_enabled"] is False
    assert preview["recompile_performed"] is False
    assert preview["exact_source_text"] == EXACT_SOURCE
    assert preview["sanitized_required_bridge"] == EXACT_SOURCE
    assert preview["semantic_fidelity_shadow"]["reason_codes"] == []
    taxonomy = preview["semantic_fidelity_taxonomy_shadow"]
    assert taxonomy["binding_contract"]["status"] == "VALID"
    assert taxonomy["claim_fidelity"]["status"] == "NO_FLAG"
    assert taxonomy["specification_completeness"]["status"] == "NO_FLAG"
    assert taxonomy["specification_self_containment"]["status"] == "NO_FLAG"
    assert taxonomy["overall_review_status"] == "PASS_SHADOW"
    assert preview["preview_reason_codes"] == []


def test_preview_mutation_is_limited_to_three_draft_fields() -> None:
    card = _card()
    claim = _claim()
    before = claim.model_dump(mode="json")

    preview = _preview_exact_source_atomic_recompile_shadow(
        card,
        claim,
        recompile_plan=_plan(card, claim),
    )
    after = preview["preview_claim"]

    assert after is not None
    assert claim.model_dump(mode="json") == before
    assert preview["mutation_audit"]["changed_fields"] == [
        "required_bridge",
        "text",
    ]
    assert preview["mutation_audit"]["unexpected_changed_fields"] == []
    assert preview["mutation_audit"]["all_other_fields_preserved"] is True
    assert after["text"] == EXACT_SOURCE
    assert after["required_bridge"] == EXACT_SOURCE
    assert after["semantic_fidelity_binding"]["proposition_basis"] == EXACT_SOURCE


def test_post_compile_binding_failure_blocks_ready_status() -> None:
    card = _card()
    claim = _claim(prediction_id="obs:missing")
    plan = _plan(card, claim)

    preview = _preview_exact_source_atomic_recompile_shadow(
        card,
        claim,
        recompile_plan=plan,
    )

    assert plan["classification"] == "EXACT_SOURCE_RECOMPILE_CANDIDATE"
    assert preview["preview_status"] == "REVIEW_REQUIRED_PREVIEW"
    assert preview["ready_for_bounded_recompile"] is False
    assert "atomic_prediction_source_id_unknown" in (
        preview["semantic_fidelity_shadow"]["reason_codes"]
    )
    assert "recompile_preview_binding_contract_not_valid" in (
        preview["preview_reason_codes"]
    )
    assert "recompile_preview_overall_not_pass_shadow" in (
        preview["preview_reason_codes"]
    )


def test_non_candidate_never_builds_preview_claim() -> None:
    card = _card(
        bridge=(
            "Catalyst A increases product yield relative to Catalyst B."
        )
    )
    claim = _claim()
    plan = _plan(card, claim)

    preview = _preview_exact_source_atomic_recompile_shadow(
        card,
        claim,
        recompile_plan=plan,
    )

    assert plan["classification"] == "NO_EXACT_SOURCE_RECOMPILE_CANDIDATE"
    assert preview["preview_status"] == "NOT_ELIGIBLE"
    assert preview["ready_for_bounded_recompile"] is False
    assert preview["preview_claim"] is None
    assert preview["recompile_performed"] is False


def test_ambiguous_plan_never_builds_preview_claim() -> None:
    card = _card(
        assumptions=[
            (
                "Under fixed temperature, Catalyst A strongly increases "
                "product yield relative to Catalyst B."
            )
        ]
    )
    claim = _claim()
    plan = _plan(card, claim)

    preview = _preview_exact_source_atomic_recompile_shadow(
        card,
        claim,
        recompile_plan=plan,
    )

    assert plan["classification"] == "AMBIGUOUS_EXACT_SOURCE_CANDIDATES"
    assert plan["candidate_count"] == 2
    assert preview["preview_status"] == "NOT_ELIGIBLE"
    assert preview["ready_for_bounded_recompile"] is False
    assert preview["preview_claim"] is None
