from __future__ import annotations

from pipeline_core.discovery.atomic_scientific_source_provenance import (
    build_atomic_scientific_source_binding_bundle,
    build_atomic_scientific_source_binding_record,
    project_atomic_scientific_source_binding_bundle,
)
from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQueryPlan,
    NoveltyClaim,
    NoveltyClaimSemanticFidelityBindingDraft,
)


def _claim(
    claim_id: str,
    *,
    rank: int,
    kind: str = "mechanistic_link",
    bridge: str = "Factor X changes response Y.",
    prediction: str = "response Y",
    falsifier: str = "response Y does not change",
) -> NoveltyClaim:
    return NoveltyClaim(
        claim_id=claim_id,
        hypothesis_id="hypothesis:1",
        claim_rank=rank,
        kind=kind,
        importance="core",
        novelty_selection_role=(
            "NOVELTY_BEARING"
            if rank == 1
            else "REQUIRED_ENABLING_RELATION"
        ),
        text="Factor X changes response Y.",
        rationale="fixture",
        search_concepts=["Factor X", "response Y"],
        search_queries=["Factor X response Y"],
        prior_art_identity_terms=["Factor X"],
        relation_nucleus_terms=["Factor X", "response Y"],
        required_bridge=bridge,
        predicted_observation=prediction,
        falsification_condition=falsifier,
    )


def _plan(
    claims: list[NoveltyClaim],
    *,
    plan_id: str,
    plan_sha: str,
) -> LiteratureQueryPlan:
    return LiteratureQueryPlan(
        plan_id=plan_id,
        plan_sha256=plan_sha,
        source_portfolio_id="portfolio:1",
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id="hypothesis:1",
                title="Fixture",
                claims=claims,
            )
        ],
    )


def _record(
    claim: NoveltyClaim,
    *,
    prediction_id: str,
    falsifier_id: str,
):
    return build_atomic_scientific_source_binding_record(
        hypothesis_id=claim.hypothesis_id,
        claim_local_id="local:" + claim.claim_id,
        claim=claim,
        binding=NoveltyClaimSemanticFidelityBindingDraft(
            proposition_basis="Factor X changes response Y.",
            relation_endpoint_anchors=["Factor X", "response Y"],
            prediction_observation_id=prediction_id,
            falsification_criterion_id=falsifier_id,
        ),
    )


def _source_bundle(
    claims: list[NoveltyClaim],
    plan: LiteratureQueryPlan,
):
    records = [
        _record(
            claim,
            prediction_id="prediction:" + claim.claim_id,
            falsifier_id="falsifier:" + claim.claim_id,
        )
        for claim in claims
    ]
    return build_atomic_scientific_source_binding_bundle(
        source_portfolio_id="portfolio:1",
        query_plan=plan,
        records=records,
    )


def test_specification_surface_rebase_preserves_source_ids() -> None:
    source_claim = _claim(
        "claim:1",
        rank=1,
        bridge="",
    )
    source_plan = _plan(
        [source_claim],
        plan_id="plan:source",
        plan_sha="a" * 64,
    )
    bundle = _source_bundle([source_claim], source_plan)

    repaired_claim = source_claim.model_copy(
        update={"required_bridge": "Factor X changes response Y."}
    )
    output_plan = _plan(
        [repaired_claim],
        plan_id="plan:repaired",
        plan_sha="b" * 64,
    )

    projected = project_atomic_scientific_source_binding_bundle(
        source_bundle=bundle,
        source_query_plan=source_plan,
        output_query_plan=output_plan,
    )

    before = bundle.records[0]
    after = projected.records[0]
    assert after.prediction_observation_id == (
        before.prediction_observation_id
    )
    assert after.falsification_criterion_id == (
        before.falsification_criterion_id
    )
    assert after.source_claim_sha256 != before.source_claim_sha256
    assert projected.source_query_plan_id == output_plan.plan_id


def test_source_alignment_can_replace_ids_from_audited_candidate() -> None:
    source_claim = _claim("claim:1", rank=1)
    source_plan = _plan(
        [source_claim],
        plan_id="plan:source",
        plan_sha="a" * 64,
    )
    bundle = _source_bundle([source_claim], source_plan)

    aligned_claim = source_claim.model_copy(
        update={
            "predicted_observation": "aligned source observable",
            "falsification_condition": "aligned falsifying outcome",
        }
    )
    output_plan = _plan(
        [aligned_claim],
        plan_id="plan:aligned",
        plan_sha="c" * 64,
    )

    projected = project_atomic_scientific_source_binding_bundle(
        source_bundle=bundle,
        source_query_plan=source_plan,
        output_query_plan=output_plan,
        source_id_overrides={
            ("hypothesis:1", "claim:1"): (
                "prediction:selected",
                "falsifier:selected",
            )
        },
    )

    row = projected.records[0]
    assert row.prediction_observation_id == "prediction:selected"
    assert row.falsification_criterion_id == "falsifier:selected"


def test_decomposition_subset_drops_removed_composite_record() -> None:
    atomic = _claim("claim:a", rank=1)
    composite = _claim(
        "claim:c",
        rank=2,
        kind="composite",
    )
    source_plan = _plan(
        [atomic, composite],
        plan_id="plan:source",
        plan_sha="a" * 64,
    )
    bundle = _source_bundle([atomic, composite], source_plan)

    output_plan = _plan(
        [atomic],
        plan_id="plan:decomposed",
        plan_sha="d" * 64,
    )
    projected = project_atomic_scientific_source_binding_bundle(
        source_bundle=bundle,
        source_query_plan=source_plan,
        output_query_plan=output_plan,
    )

    assert projected.claim_count == 1
    assert [row.claim_id for row in projected.records] == ["claim:a"]


def test_projection_rejects_new_claim_identity() -> None:
    source_claim = _claim("claim:1", rank=1)
    source_plan = _plan(
        [source_claim],
        plan_id="plan:source",
        plan_sha="a" * 64,
    )
    bundle = _source_bundle([source_claim], source_plan)

    added = _claim("claim:new", rank=2)
    output_plan = _plan(
        [source_claim, added],
        plan_id="plan:added",
        plan_sha="e" * 64,
    )

    try:
        project_atomic_scientific_source_binding_bundle(
            source_bundle=bundle,
            source_query_plan=source_plan,
            output_query_plan=output_plan,
        )
    except ValueError as exc:
        assert "cannot add claim identities" in str(exc)
    else:
        raise AssertionError("new claim identity must fail closed")


def test_projection_rejects_partial_source_id_override() -> None:
    source_claim = _claim("claim:1", rank=1)
    source_plan = _plan(
        [source_claim],
        plan_id="plan:source",
        plan_sha="a" * 64,
    )
    bundle = _source_bundle([source_claim], source_plan)
    output_plan = _plan(
        [source_claim],
        plan_id="plan:output",
        plan_sha="f" * 64,
    )

    try:
        project_atomic_scientific_source_binding_bundle(
            source_bundle=bundle,
            source_query_plan=source_plan,
            output_query_plan=output_plan,
            source_id_overrides={
                ("hypothesis:1", "claim:1"): (
                    "prediction:selected",
                    "",
                )
            },
        )
    except ValueError as exc:
        assert "requires both IDs" in str(exc)
    else:
        raise AssertionError("partial stable-ID override must fail closed")


def test_projection_rejects_stale_source_claim_sha() -> None:
    source_claim = _claim("claim:1", rank=1)
    source_plan = _plan(
        [source_claim],
        plan_id="plan:source",
        plan_sha="a" * 64,
    )
    bundle = _source_bundle([source_claim], source_plan)

    tampered_source_claim = source_claim.model_copy(
        update={"rationale": "tampered"}
    )
    tampered_source_plan = _plan(
        [tampered_source_claim],
        plan_id=source_plan.plan_id,
        plan_sha=source_plan.plan_sha256,
    )
    output_plan = _plan(
        [tampered_source_claim],
        plan_id="plan:output",
        plan_sha="1" * 64,
    )

    try:
        project_atomic_scientific_source_binding_bundle(
            source_bundle=bundle,
            source_query_plan=tampered_source_plan,
            output_query_plan=output_plan,
        )
    except ValueError as exc:
        assert "source claim SHA mismatch" in str(exc)
    else:
        raise AssertionError("stale source provenance must fail closed")
