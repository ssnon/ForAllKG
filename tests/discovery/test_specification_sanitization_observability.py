from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimDecompositionDraft,
    NoveltyClaimDraft,
)
from pipeline_core.discovery.novelty_claim_decomposition import (
    NoveltyClaimDecomposer,
)


def test_raw_specification_is_preserved_only_in_diagnostic_record() -> None:
    raw_bridge = (
        "iCOHP controls hydrogen adsorption free energy."
    )

    hypothesis_bridge = (
        "Metal identity conditions the mapping from M-H iCOHP "
        "to hydrogen adsorption free energy."
    )

    class Backend:
        def decompose(
            self,
            hypothesis,
            *,
            max_claims,
        ):
            return NoveltyClaimDecompositionDraft(
                claims=[
                    NoveltyClaimDraft(
                        local_id="c1",
                        kind="moderator_interaction",
                        importance="core",
                        text=(
                            "Metal identity conditions the "
                            "relationship between iCOHP and "
                            "hydrogen adsorption free energy."
                        ),
                        rationale="test",
                        prior_art_identity_terms=[
                            "metal identity",
                        ],
                        relation_nucleus_terms=[
                            "iCOHP",
                            "hydrogen adsorption free energy",
                        ],
                        required_bridge=raw_bridge,
                    )
                ]
            )

    decomposer = NoveltyClaimDecomposer(
        Backend(),
        max_claims_per_hypothesis=1,
        max_queries_per_claim=2,
    )

    result = decomposer.decompose(
        SimpleNamespace(
            hypothesis_id=(
                "hypothesis:test-sanitization"
            ),
            title="test",
            hypothesis_statement=(
                "Metal identity may moderate "
                "adsorption energetics."
            ),
            inferential_bridge=hypothesis_bridge,
            assumptions=[],
            predicted_observations=[],
            falsification_criteria=[],
        )
    )

    claim = result.claims[0]

    # Canonical scientific contract remains sanitized.
    assert claim.required_bridge == ""

    assert (
        "required_bridge_rejected_branch_identity"
        in claim.specification_sanitization_reason_codes
    )

    # Raw content survives only in diagnostic provenance.
    records = (
        decomposer.specification_sanitization_records
    )

    assert len(records) == 1

    record = records[0]

    assert record["diagnostic_only"] is True
    assert record["claim_id"] == claim.claim_id
    assert (
        record["required_bridge_source"]
        == "draft"
    )
    assert (
        record["raw_required_bridge"]
        == raw_bridge
    )
    assert (
        record["sanitized_required_bridge"]
        == ""
    )
    assert record["prior_art_identity_terms"] == [
        "metal identity",
    ]
    trace = record["source_trace"]
    assert trace["diagnostic_only"] is True
    assert trace["branch_attribution_assessed"] is False
    field = trace["fields"]["required_bridge"]
    assert field["sanitizer_state"] == "REJECTED"
    assert field["accepted_exact_matches"] == []
    assert "source_trace" not in claim.model_dump(mode="json")

    taxonomy = record["semantic_fidelity_taxonomy_shadow"]
    assert taxonomy["diagnostic_only"] is True
    assert taxonomy["production_authority"] is False
    assert taxonomy["flat_reason_codes_are_authority"] is False
    assert "semantic_fidelity_taxonomy_shadow" not in claim.model_dump(mode="json")

    wrapper = record["scope_wrapper_exact_source_recompile_shadow"]
    assert wrapper["diagnostic_only"] is True
    assert wrapper["production_authority"] is False
    assert wrapper["production_recompile_enabled"] is False
    assert wrapper["recompile_performed"] is False
    assert wrapper["semantic_scope_synonymy_allowed"] is False
    assert (
        "scope_wrapper_exact_source_recompile_shadow"
        not in claim.model_dump(mode="json")
    )

    wrapper_preview = record[
        "scope_wrapper_exact_source_recompile_preview_shadow"
    ]
    assert wrapper_preview["diagnostic_only"] is True
    assert wrapper_preview["production_authority"] is False
    assert wrapper_preview["production_recompile_enabled"] is False
    assert wrapper_preview["recompile_performed"] is False
    assert wrapper_preview["semantic_scope_synonymy_allowed"] is False
    assert wrapper_preview["preview_status"] == "NOT_ELIGIBLE"
    assert (
        "scope_wrapper_exact_source_recompile_preview_shadow"
        not in claim.model_dump(mode="json")
    )

    wrapper_authority = record[
        "scope_wrapper_exact_source_recompile_authority_shadow"
    ]
    assert wrapper_authority["diagnostic_only"] is True
    assert wrapper_authority["production_authority"] is False
    assert wrapper_authority["production_recompile_enabled"] is False
    assert wrapper_authority["recompile_performed"] is False
    assert wrapper_authority["semantic_scope_synonymy_allowed"] is False
    assert wrapper_authority["authority_status"] == "DENIED_SHADOW"
    assert wrapper_authority["bounded_recompile_contract_satisfied"] is False
    assert (
        "scope_wrapper_exact_source_recompile_authority_shadow"
        not in claim.model_dump(mode="json")
    )

    preview = record["exact_source_recompile_preview_shadow"]
    assert preview["diagnostic_only"] is True
    assert preview["production_authority"] is False
    assert preview["production_recompile_enabled"] is False
    assert preview["recompile_performed"] is False
    assert preview["preview_status"] == "NOT_ELIGIBLE"
    assert "exact_source_recompile_preview_shadow" not in claim.model_dump(mode="json")

    authority = record["exact_source_recompile_authority_shadow"]
    assert authority["diagnostic_only"] is True
    assert authority["production_authority"] is False
    assert authority["production_recompile_enabled"] is False
    assert authority["recompile_performed"] is False
    assert authority["authority_status"] == "DENIED_SHADOW"
    assert authority["bounded_recompile_contract_satisfied"] is False
    assert "exact_source_recompile_authority_shadow" not in claim.model_dump(mode="json")

    assert (
        "required_bridge_rejected_branch_identity"
        in record["reason_codes"]
    )

    # Raw rejected text must not enter the canonical claim.
    payload = claim.model_dump(mode="json")

    assert "raw_required_bridge" not in payload
    assert "raw_predicted_observation" not in payload
    assert "raw_falsification_condition" not in payload


def test_required_bridge_prompt_requires_self_contained_contiguous_branch_span() -> None:
    from pipeline_core.discovery.external_novelty_llm import (
        _DECOMPOSE_SYSTEM,
    )

    prompt = _DECOMPOSE_SYSTEM

    assert (
        "REQUIRED-BRIDGE RETURN SELF-CHECK"
        in prompt
    )

    assert (
        "RETURNED BRIDGE STRING ITSELF"
        in prompt
    )

    assert (
        "ONE CONTIGUOUS EXTRACTIVE SPAN"
        in prompt
    )

    assert (
        "Do not stitch together non-contiguous fragments"
        in prompt
    )

    assert (
        "Do not expand to a larger umbrella sentence"
        in prompt
    )

    assert (
        "return required_bridge as an empty string"
        in prompt
    )

    assert "ATOMIC CLAIM SOURCE-BINDING CONTRACT:" in prompt
    assert "semantic_fidelity_binding" in prompt
    assert "proposition_basis must be ONE CONTIGUOUS EXACT SOURCE SPAN" in prompt
    assert "prediction_observation_id" in prompt
    assert "falsification_criterion_id" in prompt


def test_empty_atomic_bridge_does_not_fallback_to_hypothesis_bridge() -> None:
    from types import SimpleNamespace

    from pipeline_core.discovery.external_novelty_contracts import (
        NoveltyClaimDecompositionDraft,
        NoveltyClaimDraft,
    )
    from pipeline_core.discovery.novelty_claim_decomposition import (
        NoveltyClaimDecomposer,
    )

    hypothesis_bridge = (
        "Metal identity conditions the relationship between "
        "iCOHP and hydrogen adsorption free energy."
    )

    class Backend:
        def decompose(
            self,
            hypothesis,
            *,
            max_claims,
        ):
            return NoveltyClaimDecompositionDraft(
                claims=[
                    NoveltyClaimDraft(
                        local_id="c1",
                        kind="moderator_interaction",
                        importance="core",
                        text=(
                            "Metal identity moderates the "
                            "relationship between iCOHP and "
                            "hydrogen adsorption free energy."
                        ),
                        rationale="test",
                        prior_art_identity_terms=[
                            "metal identity",
                        ],
                        relation_nucleus_terms=[
                            "iCOHP",
                            "hydrogen adsorption free energy",
                        ],
                        required_bridge="",
                    )
                ]
            )

    decomposer = NoveltyClaimDecomposer(
        Backend(),
        max_claims_per_hypothesis=1,
        max_queries_per_claim=2,
    )

    result = decomposer.decompose(
        SimpleNamespace(
            hypothesis_id=(
                "hypothesis:no-bridge-fallback"
            ),
            title="test",
            hypothesis_statement=(
                "Metal identity may moderate "
                "adsorption energetics."
            ),
            inferential_bridge=hypothesis_bridge,
            assumptions=[],
            predicted_observations=[],
            falsification_criteria=[],
        )
    )

    claim = result.claims[0]

    # Atomic empty is authoritative. The umbrella hypothesis
    # bridge must not be promoted into the claim.
    assert claim.required_bridge == ""

    records = (
        decomposer.specification_sanitization_records
    )

    assert len(records) == 1

    record = records[0]

    assert (
        record["required_bridge_source"]
        == "empty"
    )
    assert record["raw_required_bridge"] == ""
    field = record["source_trace"]["fields"]["required_bridge"]
    assert field["draft_state"] == "EMPTY"
    assert field["sanitizer_state"] == "EMPTY"
    assert field["nonempty_source_paths"] == ["inferential_bridge"]
    assert field["accepted_exact_matches"] == []
    assert (
        record["sanitized_required_bridge"]
        == ""
    )

    assert (
        "required_bridge_source_empty"
        in record["reason_codes"]
    )

    assert (
        "required_bridge_source_hypothesis_fallback"
        not in record["reason_codes"]
    )

    payload = claim.model_dump(mode="json")

    assert (
        payload["required_bridge"]
        == ""
    )
