from copy import deepcopy

from pipeline_core.discovery.external_novelty_contracts import NoveltyClaimDraft
from pipeline_core.discovery.hypothesis_contracts import HypothesisCard
from pipeline_core.discovery.novelty_claim_decomposition import (
    _authorize_scope_wrapper_exact_source_atomic_recompile_shadow,
    _plan_exact_source_atomic_recompile_shadow,
    _plan_scope_wrapper_exact_source_atomic_recompile_shadow,
    _preview_scope_wrapper_exact_source_atomic_recompile_shadow,
)


SOURCE = (
    "Under the condition of fixed temperature, Catalyst A increases "
    "product yield relative to Catalyst B."
)
BINDING_SCOPE = "At fixed temperature"
SOURCE_SCOPE = "Under the condition of fixed temperature"


def _card(
    *,
    bridge: str = SOURCE,
    assumptions: list[str] | None = None,
) -> HypothesisCard:
    return HypothesisCard.model_validate(
        {
            "hypothesis_id": "hypothesis:s14x",
            "domain_profile_id": "domain:s14x",
            "source_context_id": "context:s14x",
            "source_context_sha256": "context-sha",
            "source_report_id": "report:s14x",
            "source_report_sha256": "report-sha",
            "title": "Synthetic scope-wrapper preview fixture",
            "hypothesis_statement": SOURCE,
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
                        "At fixed temperature, Catalyst A product yield "
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
                        "to Catalyst B at fixed temperature."
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
            "text": (
                "At fixed temperature, Catalyst A increases product yield "
                "relative to Catalyst B."
            ),
            "rationale": "Synthetic wrapper preview fixture only.",
            "search_concepts": ["Catalyst A"],
            "search_queries": ["Catalyst A product yield"],
            "distinguishing_terms": ["fixed temperature"],
            "prior_art_identity_terms": ["Catalyst A"],
            "relation_nucleus_terms": ["product yield"],
            "semantic_fidelity_binding": {
                "proposition_basis": (
                    "At fixed temperature, Catalyst A increases product yield "
                    "relative to Catalyst B."
                ),
                "relation_endpoint_anchors": [
                    "Catalyst A",
                    "product yield relative to Catalyst B",
                ],
                "scope_qualifier_spans": [BINDING_SCOPE],
                "directional_qualifier_spans": ["increases"],
                "prediction_observation_id": prediction_id,
                "falsification_criterion_id": "fals:1",
            },
            "required_bridge": "",
            "predicted_observation": (
                "At fixed temperature, Catalyst A product yield increases "
                "relative to Catalyst B."
            ),
            "falsification_condition": (
                "Catalyst A does not increase product yield relative to "
                "Catalyst B at fixed temperature."
            ),
        }
    )


def _plans(card: HypothesisCard, claim: NoveltyClaimDraft):
    exact = _plan_exact_source_atomic_recompile_shadow(
        card,
        claim,
        sanitized_required_bridge="",
    )
    wrapper = _plan_scope_wrapper_exact_source_atomic_recompile_shadow(
        card,
        claim,
        sanitized_required_bridge="",
        exact_source_plan=exact,
    )
    return exact, wrapper


def _preview(card, claim, wrapper):
    return _preview_scope_wrapper_exact_source_atomic_recompile_shadow(
        card,
        claim,
        recompile_plan=wrapper,
    )


def _authority(claim, wrapper, preview):
    return _authorize_scope_wrapper_exact_source_atomic_recompile_shadow(
        claim,
        recompile_plan=wrapper,
        recompile_preview=preview,
    )


def test_clean_wrapper_preview_rebinds_only_exact_source_and_scope() -> None:
    card = _card()
    claim = _claim()
    before = claim.model_dump(mode="json")
    exact, wrapper = _plans(card, claim)
    preview = _preview(card, claim, wrapper)

    assert exact["classification"] == "NO_EXACT_SOURCE_RECOMPILE_CANDIDATE"
    assert wrapper["classification"] == (
        "SCOPE_WRAPPER_EXACT_SOURCE_RECOMPILE_CANDIDATE"
    )
    assert preview["preview_status"] == "PASS_SHADOW_PREVIEW"
    assert preview["ready_for_bounded_recompile"] is True
    assert preview["preview_reason_codes"] == []
    assert preview["candidate_source_path"] == "inferential_bridge.unit[0]"
    assert preview["exact_source_text"] == SOURCE
    assert preview["binding_scope"] == BINDING_SCOPE
    assert preview["source_scope"] == SOURCE_SCOPE
    assert preview["sanitized_required_bridge"] == SOURCE
    assert preview["semantic_fidelity_shadow"]["reason_codes"] == []
    assert preview["semantic_fidelity_taxonomy_shadow"]["overall_review_status"] == (
        "PASS_SHADOW"
    )

    payload = preview["preview_claim"]
    assert payload["text"] == SOURCE
    assert payload["required_bridge"] == SOURCE
    binding = payload["semantic_fidelity_binding"]
    assert binding["proposition_basis"] == SOURCE
    assert binding["scope_qualifier_spans"] == [SOURCE_SCOPE]
    assert binding["relation_endpoint_anchors"] == (
        before["semantic_fidelity_binding"]["relation_endpoint_anchors"]
    )
    assert binding["directional_qualifier_spans"] == (
        before["semantic_fidelity_binding"]["directional_qualifier_spans"]
    )
    assert binding["prediction_observation_id"] == "obs:1"
    assert binding["falsification_criterion_id"] == "fals:1"
    assert preview["mutation_audit"]["unexpected_changed_fields"] == []
    assert preview["mutation_audit"]["unexpected_binding_changed_fields"] == []
    assert preview["mutation_audit"]["changed_binding_fields"] == [
        "proposition_basis",
        "scope_qualifier_spans",
    ]
    assert claim.model_dump(mode="json") == before


def test_clean_wrapper_preview_satisfies_authority_shadow() -> None:
    card = _card()
    claim = _claim()
    _, wrapper = _plans(card, claim)
    preview = _preview(card, claim, wrapper)
    authority = _authority(claim, wrapper, preview)

    assert authority["authority_status"] == "AUTHORIZED_SHADOW"
    assert authority["bounded_recompile_contract_satisfied"] is True
    assert authority["authority_reason_codes"] == []
    assert authority["diagnostic_only"] is True
    assert authority["production_authority"] is False
    assert authority["production_recompile_enabled"] is False
    assert authority["recompile_performed"] is False
    assert authority["semantic_scope_synonymy_allowed"] is False
    assert authority["binding_scope"] == BINDING_SCOPE
    assert authority["source_scope"] == SOURCE_SCOPE
    assert authority["nested_binding_changed_fields"] == [
        "proposition_basis",
        "scope_qualifier_spans",
    ]


def test_nonready_wrapper_preview_is_denied() -> None:
    card = _card()
    claim = _claim(prediction_id="obs:missing")
    _, wrapper = _plans(card, claim)
    preview = _preview(card, claim, wrapper)
    authority = _authority(claim, wrapper, preview)

    assert preview["preview_status"] == "REVIEW_REQUIRED_PREVIEW"
    assert preview["ready_for_bounded_recompile"] is False
    assert authority["authority_status"] == "DENIED_SHADOW"
    assert "scope_wrapper_recompile_authority_preview_not_ready" in (
        authority["authority_reason_codes"]
    )


def test_non_candidate_wrapper_plan_has_no_preview_authority() -> None:
    card = _card(bridge=SOURCE.replace("fixed temperature", "higher temperature"))
    claim = _claim()
    _, wrapper = _plans(card, claim)
    preview = _preview(card, claim, wrapper)
    authority = _authority(claim, wrapper, preview)

    assert wrapper["candidate_count"] == 0
    assert preview["preview_status"] == "NOT_ELIGIBLE"
    assert authority["authority_status"] == "DENIED_SHADOW"
    assert "scope_wrapper_recompile_authority_plan_not_unique_candidate" in (
        authority["authority_reason_codes"]
    )


def test_ambiguous_wrapper_plan_is_denied() -> None:
    second = SOURCE.replace("Catalyst A increases", "Catalyst A strongly increases")
    card = _card(assumptions=[second])
    claim = _claim()
    _, wrapper = _plans(card, claim)
    preview = _preview(card, claim, wrapper)
    authority = _authority(claim, wrapper, preview)

    assert wrapper["classification"] == (
        "AMBIGUOUS_SCOPE_WRAPPER_EXACT_SOURCE_RECOMPILE_CANDIDATES"
    )
    assert wrapper["candidate_count"] == 2
    assert preview["preview_status"] == "NOT_ELIGIBLE"
    assert authority["authority_status"] == "DENIED_SHADOW"


def test_preview_source_scope_tamper_is_denied() -> None:
    card = _card()
    claim = _claim()
    _, wrapper = _plans(card, claim)
    preview = deepcopy(_preview(card, claim, wrapper))
    preview["source_scope"] = "Under the condition of higher temperature"

    authority = _authority(claim, wrapper, preview)

    assert authority["authority_status"] == "DENIED_SHADOW"
    assert "scope_wrapper_recompile_authority_source_identity_mismatch" in (
        authority["authority_reason_codes"]
    )


def test_unexpected_top_level_mutation_is_denied() -> None:
    card = _card()
    claim = _claim()
    _, wrapper = _plans(card, claim)
    preview = deepcopy(_preview(card, claim, wrapper))
    preview["preview_claim"]["rationale"] = "tampered"

    authority = _authority(claim, wrapper, preview)

    assert authority["authority_status"] == "DENIED_SHADOW"
    assert "rationale" in authority["unexpected_changed_fields"]
    assert "scope_wrapper_recompile_authority_unexpected_field_mutation" in (
        authority["authority_reason_codes"]
    )


def test_endpoint_binding_tamper_is_denied() -> None:
    card = _card()
    claim = _claim()
    _, wrapper = _plans(card, claim)
    preview = deepcopy(_preview(card, claim, wrapper))
    preview["preview_claim"]["semantic_fidelity_binding"][
        "relation_endpoint_anchors"
    ] = ["Catalyst A", "tampered endpoint"]

    authority = _authority(claim, wrapper, preview)

    assert authority["authority_status"] == "DENIED_SHADOW"
    assert "relation_endpoint_anchors" in (
        authority["unexpected_binding_changed_fields"]
    )
    assert "scope_wrapper_recompile_authority_binding_mutation_out_of_bounds" in (
        authority["authority_reason_codes"]
    )


def test_preview_metadata_tamper_is_denied() -> None:
    card = _card()
    claim = _claim()
    _, wrapper = _plans(card, claim)
    preview = deepcopy(_preview(card, claim, wrapper))
    preview["production_recompile_enabled"] = True

    authority = _authority(claim, wrapper, preview)

    assert authority["authority_status"] == "DENIED_SHADOW"
    assert "scope_wrapper_recompile_authority_preview_metadata_guard_failed" in (
        authority["authority_reason_codes"]
    )
