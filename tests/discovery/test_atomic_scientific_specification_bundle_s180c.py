from __future__ import annotations

from pipeline_core.discovery.atomic_scientific_specification import (
    CompiledAtomicSpecification,
)
from pipeline_core.discovery.atomic_scientific_specification_bundle import (
    AtomicScientificSpecificationBundle,
    build_atomic_scientific_specification_bundle,
)
from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimScientificStructure,
)


def _spec(claim_id: str) -> CompiledAtomicSpecification:
    return CompiledAtomicSpecification(
        local_id="atomic:1",
        claim_id=claim_id,
        kind="mechanistic_link",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text="Factor X affects response Y.",
        rationale="fixture",
        source_candidate_ids=["candidate:1"],
        premise_statement_ids=["statement:1"],
        gap_statement_ids=[],
        prior_art_identity_terms=["Factor X"],
        relation_endpoint_anchors=["Factor X", "response Y"],
        scope_qualifier_spans=[],
        directional_qualifier_spans=[],
        relation_nucleus_terms=["Factor X", "response Y"],
        distinguishing_terms=[],
        required_bridge="Factor X affects response Y.",
        observable="response Y",
        predicted_observation="Response Y changes with Factor X.",
        falsification_condition=(
            "Response Y does not change with Factor X."
        ),
        prediction_observation_id="prediction:1",
        falsification_criterion_id="falsifier:1",
        search_concepts=["Factor X", "response Y"],
        search_queries=["Factor X response Y"],
        scientific_structure=NoveltyClaimScientificStructure(),
        scientific_structure_reason_codes=[],
    )


def test_bundle_is_deterministic_and_preserves_canonical_specifications():
    first = build_atomic_scientific_specification_bundle(
        source_report_id="atomic_report:1",
        source_contract=(
            "atomic-cross-lane-scientific-synthesis-report-v1"
        ),
        hypotheses=[
            ("hypothesis:1", [_spec("claim:1")]),
        ],
    )
    second = build_atomic_scientific_specification_bundle(
        source_report_id="atomic_report:1",
        source_contract=(
            "atomic-cross-lane-scientific-synthesis-report-v1"
        ),
        hypotheses=[
            ("hypothesis:1", [_spec("claim:1")]),
        ],
    )

    assert first.model_dump(mode="json") == second.model_dump(
        mode="json"
    )
    assert first.bundle_id == second.bundle_id
    assert first.bundle_sha256 == second.bundle_sha256
    assert first.hypothesis_count == 1
    assert first.atomic_specification_count == 1
    assert (
        first.hypotheses[0].specifications[0].prediction_observation_id
        == "prediction:1"
    )
    assert first.canonical_scientific_representation is True
    assert first.stable_source_identity_preserved is True
    assert first.exact_text_is_identity_authority is False
    assert first.novelty_authority is False
    assert first.production_authority is False


def test_bundle_rejects_global_duplicate_claim_identity():
    payload = build_atomic_scientific_specification_bundle(
        source_report_id="atomic_report:1",
        source_contract="fixture",
        hypotheses=[
            ("hypothesis:1", [_spec("claim:1")]),
        ],
    ).model_dump(mode="json")

    payload["hypotheses"].append(
        {
            "hypothesis_id": "hypothesis:2",
            "specifications": [
                _spec("claim:1").model_dump(mode="json")
            ],
        }
    )
    payload["hypothesis_count"] = 2
    payload["atomic_specification_count"] = 2

    try:
        AtomicScientificSpecificationBundle.model_validate(payload)
    except Exception as exc:
        assert "duplicate global claim IDs" in str(exc)
    else:
        raise AssertionError("expected duplicate claim rejection")
