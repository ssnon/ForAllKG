from copy import deepcopy

from pipeline_core.discovery.external_novelty_contracts import NoveltyClaimDraft
from pipeline_core.discovery.hypothesis_contracts import HypothesisCard
from pipeline_core.discovery.novelty_claim_decomposition import (
    _authorize_exact_source_atomic_recompile_shadow,
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
            "hypothesis_id": "hypothesis:s14p",
            "domain_profile_id": "domain:s14p",
            "source_context_id": "context:s14p",
            "source_context_sha256": "context-sha",
            "source_report_id": "report:s14p",
            "source_report_sha256": "report-sha",
            "title": "Synthetic mutation-boundary fixture",
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


def _claim(*, prediction_id: str = "obs:1") -> NoveltyClaimDraft:
    return NoveltyClaimDraft.model_validate(
        {
            "local_id": "claim_1",
            "kind": "context_condition",
            "importance": "core",
            "novelty_selection_role": "NOVELTY_BEARING",
            "text": "Catalyst A affects product yield relative to Catalyst B.",
            "rationale": "Synthetic authority fixture only.",
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


def _preview(
    card: HypothesisCard,
    claim: NoveltyClaimDraft,
    plan: dict[str, object],
) -> dict[str, object]:
    return _preview_exact_source_atomic_recompile_shadow(
        card,
        claim,
        recompile_plan=plan,
    )


def _authority(
    claim: NoveltyClaimDraft,
    plan: dict[str, object],
    preview: dict[str, object],
) -> dict[str, object]:
    return _authorize_exact_source_atomic_recompile_shadow(
        claim,
        recompile_plan=plan,
        recompile_preview=preview,
    )


def test_clean_preview_satisfies_authority_shadow_contract() -> None:
    card = _card()
    claim = _claim()
    plan = _plan(card, claim)
    preview = _preview(card, claim, plan)

    authority = _authority(claim, plan, preview)

    assert authority["authority_status"] == "AUTHORIZED_SHADOW"
    assert authority["bounded_recompile_contract_satisfied"] is True
    assert authority["authority_reason_codes"] == []
    assert authority["diagnostic_only"] is True
    assert authority["production_authority"] is False
    assert authority["production_recompile_enabled"] is False
    assert authority["recompile_performed"] is False
    assert authority["candidate_source_path"] == "inferential_bridge.unit[0]"
    assert authority["exact_source_text"] == EXACT_SOURCE
    assert authority["unexpected_changed_fields"] == []
    assert authority["nested_binding_changed_fields"] == []


def test_nonready_preview_is_denied_at_mutation_boundary() -> None:
    card = _card()
    claim = _claim(prediction_id="obs:missing")
    plan = _plan(card, claim)
    preview = _preview(card, claim, plan)

    authority = _authority(claim, plan, preview)

    assert preview["ready_for_bounded_recompile"] is False
    assert authority["authority_status"] == "DENIED_SHADOW"
    assert authority["bounded_recompile_contract_satisfied"] is False
    assert "recompile_authority_preview_not_ready" in (
        authority["authority_reason_codes"]
    )


def test_candidate_source_path_tamper_is_denied() -> None:
    card = _card()
    claim = _claim()
    plan = _plan(card, claim)
    preview = deepcopy(_preview(card, claim, plan))
    preview["candidate_source_path"] = "assumptions[9].unit[0]"

    authority = _authority(claim, plan, preview)

    assert authority["authority_status"] == "DENIED_SHADOW"
    assert "recompile_authority_source_identity_mismatch" in (
        authority["authority_reason_codes"]
    )


def test_exact_source_tamper_is_denied() -> None:
    card = _card()
    claim = _claim()
    plan = _plan(card, claim)
    preview = deepcopy(_preview(card, claim, plan))
    preview["exact_source_text"] = (
        "Under fixed temperature, Catalyst A decreases product yield "
        "relative to Catalyst B."
    )

    authority = _authority(claim, plan, preview)

    assert authority["authority_status"] == "DENIED_SHADOW"
    assert "recompile_authority_source_identity_mismatch" in (
        authority["authority_reason_codes"]
    )


def test_unexpected_top_level_preview_mutation_is_denied() -> None:
    card = _card()
    claim = _claim()
    plan = _plan(card, claim)
    preview = deepcopy(_preview(card, claim, plan))
    preview["preview_claim"]["rationale"] = "tampered rationale"

    authority = _authority(claim, plan, preview)

    assert authority["authority_status"] == "DENIED_SHADOW"
    assert "rationale" in authority["unexpected_changed_fields"]
    assert "recompile_authority_unexpected_field_mutation" in (
        authority["authority_reason_codes"]
    )
    assert "recompile_authority_mutation_audit_mismatch" in (
        authority["authority_reason_codes"]
    )


def test_nested_binding_tamper_is_denied() -> None:
    card = _card()
    claim = _claim()
    plan = _plan(card, claim)
    preview = deepcopy(_preview(card, claim, plan))
    preview["preview_claim"]["semantic_fidelity_binding"][
        "relation_endpoint_anchors"
    ] = ["Catalyst A", "tampered endpoint"]

    authority = _authority(claim, plan, preview)

    assert authority["authority_status"] == "DENIED_SHADOW"
    assert "relation_endpoint_anchors" in (
        authority["nested_binding_changed_fields"]
    )
    assert "recompile_authority_binding_mutation_out_of_bounds" in (
        authority["authority_reason_codes"]
    )


def test_preview_metadata_tamper_is_denied() -> None:
    card = _card()
    claim = _claim()
    plan = _plan(card, claim)
    preview = deepcopy(_preview(card, claim, plan))
    preview["production_recompile_enabled"] = True

    authority = _authority(claim, plan, preview)

    assert authority["authority_status"] == "DENIED_SHADOW"
    assert "recompile_authority_preview_metadata_guard_failed" in (
        authority["authority_reason_codes"]
    )


def test_ambiguous_plan_is_denied_without_preview_authority() -> None:
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
    preview = _preview(card, claim, plan)

    authority = _authority(claim, plan, preview)

    assert plan["classification"] == "AMBIGUOUS_EXACT_SOURCE_CANDIDATES"
    assert preview["preview_status"] == "NOT_ELIGIBLE"
    assert authority["authority_status"] == "DENIED_SHADOW"
    assert "recompile_authority_plan_not_unique_candidate" in (
        authority["authority_reason_codes"]
    )
    assert "recompile_authority_preview_not_ready" in (
        authority["authority_reason_codes"]
    )
