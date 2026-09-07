from copy import deepcopy

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimDecompositionDraft,
    NoveltyClaimDraft,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisCard
from pipeline_core.discovery.novelty_atomic_semantic_fidelity import (
    assess_atomic_semantic_fidelity,
)
from pipeline_core.discovery.novelty_claim_decomposition import (
    NoveltyClaimDecomposer,
    _authorize_existing_bridge_scope_alignment_shadow,
    _clean_branch_specific_bridge,
    _clean_branch_specific_specification,
    _plan_exact_source_atomic_recompile_shadow,
    _plan_existing_bridge_scope_alignment_shadow,
    _preview_existing_bridge_scope_alignment_shadow,
)
from pipeline_core.discovery.novelty_specification_source_trace import (
    trace_specification_sources,
)


SOURCE = (
    "Under the condition of comparable overall metal–hydrogen coupling, "
    "I further propose that a more balanced bonding–antibonding distribution "
    "promotes compatibility between hydrogen adsorption and H2 desorption."
)
CLAIM_TEXT = (
    "At comparable overall metal–hydrogen coupling, a more balanced "
    "bonding–antibonding distribution promotes compatibility between "
    "hydrogen adsorption and H2 desorption."
)
SOURCE_SCOPE = (
    "Under the condition of comparable overall metal–hydrogen coupling"
)
ASCII_HYPHEN_SOURCE_SCOPE = (
    "Under the condition of comparable overall metal-hydrogen coupling"
)
PREDICTION = (
    "At comparable overall metal–hydrogen coupling, a more balanced "
    "bonding–antibonding distribution is expected to be more compatible "
    "with both hydrogen adsorption and H2 desorption."
)
FALSIFIER = (
    "Changing the bonding–antibonding distribution at comparable overall "
    "metal–hydrogen coupling produces no qualitative difference in hydrogen "
    "adsorption–desorption compatibility."
)


def _card(*, bridge: str = SOURCE, assumptions: list[str] | None = None) -> HypothesisCard:
    return HypothesisCard.model_validate(
        {
            "hypothesis_id": "hypothesis:s15d",
            "domain_profile_id": "domain:s15d",
            "source_context_id": "context:s15d",
            "source_context_sha256": "context-sha",
            "source_report_id": "report:s15d",
            "source_report_sha256": "report-sha",
            "title": "Existing bridge scope alignment fixture",
            "hypothesis_statement": SOURCE,
            "hypothesis_type": "context_dependency",
            "premise_statement_ids": ["premise:1"],
            "gap_statement_ids": ["gap:1"],
            "inferential_bridge": bridge,
            "predicted_observations": [
                {
                    "observation_id": "obs:1",
                    "observable": "hydrogen adsorption–desorption compatibility",
                    "expected_direction": "qualitative_change",
                    "rationale": PREDICTION,
                }
            ],
            "falsification_criteria": [
                {
                    "criterion_id": "fals:1",
                    "observable": "hydrogen adsorption–desorption compatibility",
                    "falsifying_outcome": FALSIFIER,
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
    text: str = CLAIM_TEXT,
    bridge: str = SOURCE,
    binding_scope: str = SOURCE_SCOPE,
) -> NoveltyClaimDraft:
    return NoveltyClaimDraft.model_validate(
        {
            "local_id": "claim_1",
            "kind": "mechanistic_link",
            "importance": "core",
            "novelty_selection_role": "NOVELTY_BEARING",
            "text": text,
            "rationale": "Synthetic alignment fixture only.",
            "search_concepts": ["bonding antibonding distribution"],
            "search_queries": ["bonding antibonding distribution hydrogen"],
            "distinguishing_terms": [
                "comparable overall metal hydrogen coupling"
            ],
            "prior_art_identity_terms": ["bonding antibonding distribution"],
            "relation_nucleus_terms": [
                "overall metal hydrogen coupling",
                "hydrogen adsorption",
                "H2 desorption",
                "compatibility",
            ],
            "semantic_fidelity_binding": {
                "proposition_basis": SOURCE,
                "relation_endpoint_anchors": [
                    "bonding-antibonding distribution",
                    "hydrogen adsorption",
                    "H2 desorption",
                ],
                "scope_qualifier_spans": [binding_scope],
                "directional_qualifier_spans": [
                    "a more balanced bonding-antibonding distribution",
                    "promotes compatibility",
                ],
                "prediction_observation_id": "obs:1",
                "falsification_criterion_id": "fals:1",
            },
            "required_bridge": bridge,
            "predicted_observation": PREDICTION,
            "falsification_condition": FALSIFIER,
        }
    )


def _path(card: HypothesisCard, claim: NoveltyClaimDraft):
    sources = [card.inferential_bridge, *card.assumptions]
    identities = list(claim.prior_art_identity_terms)
    bridge = _clean_branch_specific_bridge(
        claim.required_bridge, identities, sources
    )
    prediction = _clean_branch_specific_specification(
        claim.predicted_observation, identities
    )
    falsifier = _clean_branch_specific_specification(
        claim.falsification_condition, identities
    )
    trace = trace_specification_sources(
        card,
        {
            "required_bridge": claim.required_bridge,
            "predicted_observation": claim.predicted_observation,
            "falsification_condition": claim.falsification_condition,
        },
        {
            "required_bridge": bridge,
            "predicted_observation": prediction,
            "falsification_condition": falsifier,
        },
    )
    semantic = assess_atomic_semantic_fidelity(card, claim)
    exact = _plan_exact_source_atomic_recompile_shadow(
        card, claim, sanitized_required_bridge=bridge
    )
    plan = _plan_existing_bridge_scope_alignment_shadow(
        claim,
        sanitized_required_bridge=bridge,
        exact_source_plan=exact,
        semantic_fidelity_shadow=semantic,
        specification_source_trace=trace,
    )
    preview = _preview_existing_bridge_scope_alignment_shadow(
        card, claim, alignment_plan=plan
    )
    authority = _authorize_existing_bridge_scope_alignment_shadow(
        claim, alignment_plan=plan, alignment_preview=preview
    )
    return exact, semantic, trace, plan, preview, authority


def test_clean_existing_bridge_alignment_is_text_only_and_authorized_shadow():
    card = _card()
    claim = _claim()
    before = claim.model_dump(mode="json")
    exact, semantic, trace, plan, preview, authority = _path(card, claim)

    assert exact["classification"] == "NO_RECOMPILE_NEEDED"
    assert semantic["reason_codes"] == [
        "atomic_claim_scope_qualifier_not_preserved"
    ]
    assert (
        trace["fields"]["required_bridge"]["raw_source_match_state"]
        == "EXACT_UNIQUE"
    )
    assert plan["classification"] == "EXISTING_BRIDGE_SCOPE_ALIGNMENT_CANDIDATE"
    assert plan["candidate_count"] == 1
    assert preview["preview_status"] == "PASS_SHADOW_PREVIEW"
    assert preview["ready_for_bounded_recompile"] is True
    assert preview["preview_reason_codes"] == []
    assert preview["preview_claim"]["text"] == SOURCE
    assert preview["preview_claim"]["required_bridge"] == SOURCE
    assert preview["mutation_audit"]["changed_fields"] == ["text"]
    assert preview["mutation_audit"]["changed_binding_fields"] == []
    assert preview["semantic_fidelity_shadow"]["reason_codes"] == []
    assert (
        preview["semantic_fidelity_taxonomy_shadow"]["overall_review_status"]
        == "PASS_SHADOW"
    )
    assert authority["authority_status"] == "AUTHORIZED_SHADOW"
    assert authority["bounded_recompile_contract_satisfied"] is True
    assert authority["authority_reason_codes"] == []
    assert authority["production_authority"] is False
    assert authority["production_recompile_enabled"] is False
    assert authority["recompile_performed"] is False
    assert claim.model_dump(mode="json") == before


def test_ascii_hyphen_binding_scope_matches_en_dash_source_scope():
    card = _card()
    claim = _claim(binding_scope=ASCII_HYPHEN_SOURCE_SCOPE)
    _, semantic, _, plan, preview, authority = _path(card, claim)

    assert semantic["reason_codes"] == [
        "atomic_claim_scope_qualifier_not_preserved"
    ]
    assert plan["classification"] == (
        "EXISTING_BRIDGE_SCOPE_ALIGNMENT_CANDIDATE"
    )
    assert plan["candidate_count"] == 1
    assert plan["binding_source_scope_match"]["matched"] is True
    assert (
        plan["binding_source_scope_match"]["binding_wrapper"]
        == "under the condition of"
    )
    assert (
        plan["binding_source_scope_match"]["source_wrapper"]
        == "under the condition of"
    )
    assert (
        plan["binding_source_scope_match"]["normalized_binding_core"]
        == "comparable overall metal hydrogen coupling"
    )
    assert (
        plan["binding_source_scope_match"]["normalized_source_core"]
        == "comparable overall metal hydrogen coupling"
    )
    assert preview["preview_status"] == "PASS_SHADOW_PREVIEW"
    assert preview["mutation_audit"]["changed_fields"] == ["text"]
    assert preview["mutation_audit"]["changed_binding_fields"] == []
    assert preview["preview_claim"]["semantic_fidelity_binding"][
        "scope_qualifier_spans"
    ] == [ASCII_HYPHEN_SOURCE_SCOPE]
    assert preview["semantic_fidelity_shadow"]["reason_codes"] == []
    assert (
        preview["semantic_fidelity_taxonomy_shadow"]["overall_review_status"]
        == "PASS_SHADOW"
    )
    assert authority["authority_status"] == "AUTHORIZED_SHADOW"
    assert authority["bounded_recompile_contract_satisfied"] is True


def test_binding_wrapper_change_is_blocked():
    card = _card()
    claim = _claim(
        binding_scope="At comparable overall metal-hydrogen coupling"
    )
    _, _, _, plan, preview, authority = _path(card, claim)

    assert plan["classification"] != (
        "EXISTING_BRIDGE_SCOPE_ALIGNMENT_CANDIDATE"
    )
    assert preview["preview_status"] == "NOT_ELIGIBLE"
    assert authority["authority_status"] == "DENIED_SHADOW"


def test_binding_scientific_core_change_is_blocked():
    card = _card()
    claim = _claim(
        binding_scope=(
            "Under the condition of higher overall "
            "metal-hydrogen coupling"
        )
    )
    _, _, _, plan, preview, authority = _path(card, claim)

    assert plan["classification"] != (
        "EXISTING_BRIDGE_SCOPE_ALIGNMENT_CANDIDATE"
    )
    assert preview["preview_status"] == "NOT_ELIGIBLE"
    assert authority["authority_status"] == "DENIED_SHADOW"


def test_binding_scientific_entity_change_is_blocked():
    card = _card()
    claim = _claim(
        binding_scope=(
            "Under the condition of comparable overall "
            "metal-metal coupling"
        )
    )
    _, _, _, plan, preview, authority = _path(card, claim)

    assert plan["classification"] != (
        "EXISTING_BRIDGE_SCOPE_ALIGNMENT_CANDIDATE"
    )
    assert preview["preview_status"] == "NOT_ELIGIBLE"
    assert authority["authority_status"] == "DENIED_SHADOW"


def test_scope_core_change_is_not_candidate():
    card = _card()
    claim = _claim(text=(
        "At higher overall metal–hydrogen coupling, a more balanced "
        "bonding–antibonding distribution promotes compatibility between "
        "hydrogen adsorption and H2 desorption."
    ))
    _, _, _, plan, preview, authority = _path(card, claim)
    assert plan["classification"] in {
        "NO_EXISTING_BRIDGE_SCOPE_ALIGNMENT_CANDIDATE",
        "NOT_ELIGIBLE_SEMANTIC_REASON_SET",
    }
    assert preview["preview_status"] == "NOT_ELIGIBLE"
    assert authority["authority_status"] == "DENIED_SHADOW"


def test_unapproved_claim_wrapper_is_not_candidate():
    card = _card()
    claim = _claim(text=(
        "Provided comparable overall metal–hydrogen coupling, a more balanced "
        "bonding–antibonding distribution promotes compatibility between "
        "hydrogen adsorption and H2 desorption."
    ))
    _, _, _, plan, preview, authority = _path(card, claim)
    assert plan["classification"] == "CLAIM_SCOPE_WRAPPER_NOT_FOUND"
    assert preview["preview_status"] == "NOT_ELIGIBLE"
    assert authority["authority_status"] == "DENIED_SHADOW"


def test_bridge_must_be_exact_unique():
    card = _card(bridge=SOURCE + " " + SOURCE)
    claim = _claim()
    _, _, _, plan, preview, authority = _path(card, claim)
    assert plan["classification"] in {
        "NOT_ELIGIBLE_REQUIRED_BRIDGE_NOT_EXACT_UNIQUE",
        "NOT_ELIGIBLE_REQUIRED_BRIDGE_TRACE_IDENTITY_MISMATCH",
        "NOT_ELIGIBLE_EXACT_SOURCE_PATH_NOT_STABLE",
    }
    assert preview["preview_status"] == "NOT_ELIGIBLE"
    assert authority["authority_status"] == "DENIED_SHADOW"


def test_extra_semantic_reason_blocks_alignment():
    card = _card()
    claim = _claim()
    sources = [card.inferential_bridge]
    bridge = _clean_branch_specific_bridge(
        claim.required_bridge,
        list(claim.prior_art_identity_terms),
        sources,
    )
    prediction = _clean_branch_specific_specification(
        claim.predicted_observation,
        list(claim.prior_art_identity_terms),
    )
    falsifier = _clean_branch_specific_specification(
        claim.falsification_condition,
        list(claim.prior_art_identity_terms),
    )
    trace = trace_specification_sources(
        card,
        {
            "required_bridge": claim.required_bridge,
            "predicted_observation": claim.predicted_observation,
            "falsification_condition": claim.falsification_condition,
        },
        {
            "required_bridge": bridge,
            "predicted_observation": prediction,
            "falsification_condition": falsifier,
        },
    )
    exact = _plan_exact_source_atomic_recompile_shadow(
        card, claim, sanitized_required_bridge=bridge
    )
    plan = _plan_existing_bridge_scope_alignment_shadow(
        claim,
        sanitized_required_bridge=bridge,
        exact_source_plan=exact,
        semantic_fidelity_shadow={
            "reason_codes": [
                "atomic_claim_scope_qualifier_not_preserved",
                "atomic_claim_relation_endpoint_not_preserved",
            ],
            "bridge_missing_relation_endpoint_anchors": [],
            "bridge_missing_scope_qualifiers": [],
        },
        specification_source_trace=trace,
    )
    assert plan["classification"] == "NOT_ELIGIBLE_SEMANTIC_REASON_SET"


def test_authority_denies_preview_text_tamper():
    card = _card()
    claim = _claim()
    _, _, _, plan, preview, _ = _path(card, claim)
    tampered = deepcopy(preview)
    tampered["preview_claim"]["text"] = "Tampered text."
    authority = _authorize_existing_bridge_scope_alignment_shadow(
        claim, alignment_plan=plan, alignment_preview=tampered
    )
    assert authority["authority_status"] == "DENIED_SHADOW"
    assert authority["authority_reason_codes"]


def test_authority_denies_binding_mutation():
    card = _card()
    claim = _claim()
    _, _, _, plan, preview, _ = _path(card, claim)
    tampered = deepcopy(preview)
    tampered["preview_claim"]["semantic_fidelity_binding"][
        "relation_endpoint_anchors"
    ] = ["bonding-antibonding distribution", "tampered endpoint"]
    authority = _authorize_existing_bridge_scope_alignment_shadow(
        claim, alignment_plan=plan, alignment_preview=tampered
    )
    assert authority["authority_status"] == "DENIED_SHADOW"
    assert authority["authority_reason_codes"]


def test_authority_denies_metadata_tamper():
    card = _card()
    claim = _claim()
    _, _, _, plan, preview, _ = _path(card, claim)
    tampered = deepcopy(preview)
    tampered["production_authority"] = True
    authority = _authorize_existing_bridge_scope_alignment_shadow(
        claim, alignment_plan=plan, alignment_preview=tampered
    )
    assert authority["authority_status"] == "DENIED_SHADOW"
    assert authority["authority_reason_codes"]


def test_decomposer_records_alignment_shadow_without_mutating_canonical_claim():
    card = _card()
    draft = _claim()

    class Backend:
        def decompose(self, hypothesis, *, max_claims):
            return NoveltyClaimDecompositionDraft(
                claims=[draft], decomposition_notes="fixture"
            )

    decomposer = NoveltyClaimDecomposer(
        Backend(),
        max_claims_per_hypothesis=1,
        max_queries_per_claim=1,
    )
    result = decomposer.decompose(card)
    assert len(result.claims) == 1
    assert result.claims[0].text == CLAIM_TEXT

    record = decomposer.specification_sanitization_records[0]
    plan = record["existing_bridge_scope_alignment_shadow"]
    preview = record["existing_bridge_scope_alignment_preview_shadow"]
    authority = record["existing_bridge_scope_alignment_authority_shadow"]
    assert plan["classification"] == "EXISTING_BRIDGE_SCOPE_ALIGNMENT_CANDIDATE"
    assert preview["preview_status"] == "PASS_SHADOW_PREVIEW"
    assert authority["authority_status"] == "AUTHORIZED_SHADOW"
    assert record["source_trace"]["fields"]["required_bridge"][
        "raw_source_match_state"
    ] == "EXACT_UNIQUE"
