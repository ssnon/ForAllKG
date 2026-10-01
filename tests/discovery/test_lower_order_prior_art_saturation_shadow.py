from __future__ import annotations

from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQueryPlan,
    NoveltyClaim,
)
from pipeline_core.discovery.lower_order_prior_art_saturation import (
    _residual_class,
    build_lower_order_saturation_query_plan,
)


def _claim(
    cid: str,
    *,
    kind: str = "mechanistic_link",
    importance: str = "core",
    components: list[str] | None = None,
) -> NoveltyClaim:
    return NoveltyClaim(
        claim_id=cid,
        hypothesis_id="h1",
        claim_rank=1,
        kind=kind,
        importance=importance,
        novelty_selection_role="NOVELTY_BEARING",
        text=f"{cid} atomic relation",
        rationale="test",
        search_concepts=[cid, "Au Al2O3", "SERS"],
        prior_art_identity_terms=[cid],
        relation_nucleus_terms=["SERS", "dependence"],
        higher_order_component_claim_ids=list(components or []),
    )


def test_saturation_plan_uses_only_existing_atomic_claims():
    c1 = _claim("c1")
    c2 = _claim("c2", importance="supporting")
    comp = _claim(
        "c3",
        kind="composite",
        components=["c1", "c2"],
    )
    base = LiteratureQueryPlan(
        plan_id="p",
        plan_sha256="sha",
        source_portfolio_id="portfolio",
        queries=[],
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id="h1",
                title="H",
                claims=[c1, c2, comp],
            )
        ],
    )

    plan, targets = build_lower_order_saturation_query_plan(base)

    target_ids = {row.claim_id for row in targets}
    assert target_ids == {"c1", "c2"}
    assert all(row.query_kind == "claim_exact_verification" for row in plan.queries)
    assert all(row.claim_id in {"c1", "c2"} for row in plan.queries)
    assert "c3" not in {row.claim_id for row in plan.queries}


def test_same_work_component_closure_preserves_only_residual_gap():
    residual, closure, preserved, _ = _residual_class(
        original_composite_status="COMPONENTS_ONLY",
        component_ids=["c1", "c2"],
        relation_backed_by_component={
            "c1": {"w1", "w2"},
            "c2": {"w1", "w3"},
        },
        unresolved_component_ids=set(),
    )
    assert residual == "SAME_WORK_COMPONENT_CLOSURE_WITH_RESIDUAL_GAP"
    assert closure == ["w1"]
    assert preserved is True


def test_distributed_known_components_do_not_fake_same_work_closure():
    residual, closure, preserved, _ = _residual_class(
        original_composite_status="COMPONENTS_ONLY",
        component_ids=["c1", "c2"],
        relation_backed_by_component={
            "c1": {"w1"},
            "c2": {"w2"},
        },
        unresolved_component_ids=set(),
    )
    assert residual == "DISTRIBUTED_KNOWN_COMPONENTS_WITH_RESIDUAL_GAP"
    assert closure == []
    assert preserved is True


def test_full_relation_prior_art_removes_residual_claim():
    residual, closure, preserved, _ = _residual_class(
        original_composite_status="DIRECT_PRIOR_ART",
        component_ids=["c1", "c2"],
        relation_backed_by_component={
            "c1": {"w1"},
            "c2": {"w1"},
        },
        unresolved_component_ids=set(),
    )
    assert residual == "FULL_RELATION_ALREADY_BACKED"
    assert closure == []
    assert preserved is False

def test_artifact_path_appends_suffix_without_clobbering_dotted_prefix(tmp_path):
    from scripts.discovery.run_lower_order_prior_art_saturation_shadow import (
        _artifact_path,
    )

    prefix = tmp_path / "B.lower_order_saturation"
    assert _artifact_path(prefix, "report.json") == (
        tmp_path / "B.lower_order_saturation.report.json"
    )
    assert _artifact_path(prefix, ".prior_art.json") == (
        tmp_path / "B.lower_order_saturation.prior_art.json"
    )

