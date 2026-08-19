import pytest
from pydantic import ValidationError

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b1_reviewer_contract_v1 import (
    EvidenceLocator,
    FinalEvidenceReference,
    FreshCFinalAdjudication,
    FreshCPaperReview,
    H1,
    H2,
    H3,
    HypothesisPaperAssessment,
)
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b2_scientific_adjudication_v1 import (
    DEFAULT_PROTOCOL_PATH,
    build_target_boundaries,
    canonical_json_sha256,
    openai_strict_transport_schema,
    validate_final_against_reviews,
    validate_protocol,
    validate_review_grounding,
)

def r2_report():
    def row(hid, disposition, title):
        return {
            "hypothesis_id": hid,
            "candidate_disposition": disposition,
            "title": title,
            "interpretation": "Synthetic frozen boundary interpretation.",
            "r2_classification": (
                "BOUNDED_LITERATURE_SUPPORTED_EXTENSION"
                if disposition == "KEEP_BOUNDED_EXTENSION"
                else (
                    "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"
                    if disposition == "KEEP_RELATIONAL_GAP_CANDIDATE"
                    else "KNOWN_MODE_MATCHING_UNSUPPORTED_MONOTONIC_DIRECTION"
                )
            ),
            "scientific_support": "SYNTHETIC_SUPPORT",
            "residual_question": "Synthetic frozen residual boundary?",
            "residual_question_is_new_hypothesis": disposition == "REJECT_AS_FORMULATED",
            "hypothesis_rewrite_performed": False,
        }
    return {
        "hypothesis_decisions": [
            row(H1, "KEEP_BOUNDED_EXTENSION", "H1"),
            row(H2, "REJECT_AS_FORMULATED", "H2"),
            row(H3, "KEEP_RELATIONAL_GAP_CANDIDATE", "H3"),
        ]
    }

def irrelevant(hid):
    return HypothesisPaperAssessment(
        hypothesis_id=hid,
        relation_label="IRRELEVANT",
        scientific_rationale="Synthetic irrelevant rationale.",
    )

def review(index=1, *, mode="DIRECT_ORIGINAL"):
    return FreshCPaperReview(
        reserve_index=index,
        canonical_id=f"doi:synthetic-{index}",
        materialization_mode=mode,
        assessments=[irrelevant(H1), irrelevant(H3)],
    )

def pages(index=1, *, mode="DIRECT_ORIGINAL"):
    return {
        "schema_version": "sers-fresh-c-c1a-r1-pages-v1",
        "reserve_index": index,
        "canonical_id": f"doi:synthetic-{index}",
        "materialization_mode": mode,
        "page_count": 1,
        "pages": [{
            "page_number": 1,
            "text": "Synthetic grounded evidence appears on this page.",
        }],
        "negative_absence_inference_allowed": False,
    }

def record(index=1, *, mode="DIRECT_ORIGINAL"):
    return {
        "reserve_index": index,
        "canonical_id": f"doi:synthetic-{index}",
        "materialization_mode": mode,
    }

def test_protocol_is_one_shot_and_exact_26_calls():
    p = validate_protocol(DEFAULT_PROTOCOL_PATH)
    assert p["paper_review_order"] == list(range(1, 26))
    assert p["paper_review_calls"] == 25
    assert p["final_adjudication_calls"] == 1
    assert p["maximum_scientific_llm_calls"] == 26
    assert p["maximum_scientific_network_calls"] == 26
    assert p["same_epoch_rerun_allowed_after_start"] is False
    assert p["failure_authorizes_tuning_on_fresh_c"] is False

def test_target_boundaries_keep_h1_h3_and_terminal_h2():
    targets = build_target_boundaries(r2_report())
    assert {row["hypothesis_id"] for row in targets} == {H1, H3}
    assert all(row["residual_question_is_new_hypothesis"] is False for row in targets)
    assert all(row["use_as_positive_generation_premise"] is False for row in targets)

def test_target_boundary_rejects_rewrite_or_new_hypothesis():
    report = r2_report()
    report["hypothesis_decisions"][0]["residual_question_is_new_hypothesis"] = True
    with pytest.raises(ValueError, match="new hypothesis"):
        build_target_boundaries(report)
    report = r2_report()
    report["hypothesis_decisions"][2]["hypothesis_rewrite_performed"] = True
    with pytest.raises(ValueError, match="rewrite"):
        build_target_boundaries(report)

def test_verbatim_quote_must_be_grounded_to_cited_page():
    grounded = FreshCPaperReview(
        reserve_index=1,
        canonical_id="doi:synthetic-1",
        materialization_mode="DIRECT_ORIGINAL",
        assessments=[
            HypothesisPaperAssessment(
                hypothesis_id=H1,
                relation_label="DIRECT_PRIOR_ART",
                scientific_rationale="Synthetic.",
                evidence=[EvidenceLocator(
                    page_number=1,
                    evidence_paraphrase="Synthetic paraphrase.",
                    verbatim_quote="grounded evidence appears",
                )],
            ),
            irrelevant(H3),
        ],
    )
    validate_review_grounding(
        grounded,
        expected_record=record(),
        pages_manifest=pages(),
    )
    bad = grounded.model_copy(deep=True)
    bad.assessments[0].evidence[0].verbatim_quote = "words not present"
    with pytest.raises(ValueError, match="Verbatim quote"):
        validate_review_grounding(
            bad,
            expected_record=record(),
            pages_manifest=pages(),
        )

def test_erosion_requires_substantive_support_reference():
    reviews = [review(i, mode=("STRUCTURALLY_REPAIRED_DERIVATIVE" if i == 14 else "DIRECT_ORIGINAL"))
               for i in range(1, 26)]
    final = FreshCFinalAdjudication(
        h1_fresh_c_verdict="FRESH_C_ERODES_PRE_C_BOUNDED_EXTENSION",
        h1_rationale="Synthetic erosion.",
        h3_fresh_c_verdict="FRESH_C_INCONCLUSIVE",
        h3_rationale="Synthetic inconclusive.",
        supporting_evidence=[],
    )
    with pytest.raises(ValueError, match="erosion verdict"):
        validate_final_against_reviews(final, reviews)

def test_erosion_accepts_matching_direct_prior_art_support():
    reviews = [review(i, mode=("STRUCTURALLY_REPAIRED_DERIVATIVE" if i == 14 else "DIRECT_ORIGINAL"))
               for i in range(1, 26)]
    reviews[0] = FreshCPaperReview(
        reserve_index=1,
        canonical_id="doi:synthetic-1",
        materialization_mode="DIRECT_ORIGINAL",
        assessments=[
            HypothesisPaperAssessment(
                hypothesis_id=H1,
                relation_label="DIRECT_PRIOR_ART",
                scientific_rationale="Synthetic direct evidence.",
                evidence=[EvidenceLocator(
                    page_number=1,
                    evidence_paraphrase="Synthetic direct evidence.",
                )],
            ),
            irrelevant(H3),
        ],
    )
    final = FreshCFinalAdjudication(
        h1_fresh_c_verdict="FRESH_C_ERODES_PRE_C_BOUNDED_EXTENSION",
        h1_rationale="Synthetic erosion.",
        h3_fresh_c_verdict="FRESH_C_INCONCLUSIVE",
        h3_rationale="Synthetic inconclusive.",
        supporting_evidence=[FinalEvidenceReference(
            reserve_index=1,
            hypothesis_id=H1,
            relation_label="DIRECT_PRIOR_ART",
            scientific_role="Synthetic substantive support.",
        )],
    )
    validate_final_against_reviews(final, reviews)

def test_repaired_reserve_14_policy_remains_fail_closed():
    r = review(14, mode="STRUCTURALLY_REPAIRED_DERIVATIVE")
    assert r.paper_level_completeness_claim_made is False
    with pytest.raises(ValidationError):
        FreshCPaperReview(
            reserve_index=14,
            canonical_id="doi:synthetic-14",
            materialization_mode="DIRECT_ORIGINAL",
            assessments=[irrelevant(H1), irrelevant(H3)],
        )


def _walk_objects(node):
    if isinstance(node, list):
        for item in node:
            yield from _walk_objects(item)
        return
    if not isinstance(node, dict):
        return
    if isinstance(node.get("properties"), dict):
        yield node
    for value in node.values():
        yield from _walk_objects(value)


def test_openai_strict_transport_schema_requires_every_object_property():
    from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b1_reviewer_contract_v1 import FreshCPaperReview
    schema = openai_strict_transport_schema(FreshCPaperReview)
    for obj in _walk_objects(schema):
        assert obj["required"] == list(obj["properties"].keys())
        assert obj["additionalProperties"] is False
    assert '"default"' not in __import__("json").dumps(schema, sort_keys=True)


def test_optional_verbatim_quote_is_required_key_but_nullable():
    from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b1_reviewer_contract_v1 import FreshCPaperReview
    schema = openai_strict_transport_schema(FreshCPaperReview)
    locator = schema["$defs"]["EvidenceLocator"]
    assert "verbatim_quote" in locator["required"]
    quote_schema = locator["properties"]["verbatim_quote"]
    assert any(branch.get("type") == "null" for branch in quote_schema["anyOf"])


def test_raw_c1b1_schema_hash_is_not_changed_by_transport_adapter():
    from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b1_reviewer_contract_v1 import FreshCPaperReview
    from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b2_scientific_adjudication_v1 import (
        EXPECTED_PAPER_SCHEMA_SHA256,
    )
    assert canonical_json_sha256(FreshCPaperReview.model_json_schema()) == (
        EXPECTED_PAPER_SCHEMA_SHA256
    )
