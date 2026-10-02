
from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQuery,
    LiteratureQueryPlan,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)
from pipeline_core.discovery.sers_effective_gen1_certification_closeout import (
    build_effective_subset_query_plan,
)


def test_subset_query_plan_rebinds_to_effective_portfolio():
    plan = LiteratureQueryPlan(
        plan_id="old",
        plan_sha256="oldsha",
        source_portfolio_id="oldp",
        queries=[
            LiteratureQuery(
                query_id="q1",
                hypothesis_id="h1",
                claim_id=None,
                query_kind="hypothesis_composite",
                query_text="one",
            ),
            LiteratureQuery(
                query_id="q2",
                hypothesis_id="h2",
                claim_id=None,
                query_kind="hypothesis_composite",
                query_text="two",
            ),
        ],
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id="h1",
                title="one",
                claims=[],
            ),
            HypothesisNoveltyClaims(
                hypothesis_id="h2",
                title="two",
                claims=[],
            ),
        ],
    )
    portfolio = HypothesisPortfolio.model_construct(
        schema_version="hypothesis-portfolio-v1",
        portfolio_id="effective",
        domain_profile_id="sers_au_ag",
        source_context_id="c",
        source_context_sha256="s",
        source_report_id="r",
        source_report_sha256="rs",
        hypotheses=[
            type(
                "Card",
                (),
                {"hypothesis_id": "h2"},
            )()
        ],
        abstention_reason=None,
    )

    subset = build_effective_subset_query_plan(
        source_plan=plan,
        effective_portfolio=portfolio,
    )
    assert subset.source_portfolio_id == "effective"
    assert [x.hypothesis_id for x in subset.claims] == ["h2"]
    assert [x.hypothesis_id for x in subset.queries] == ["h2"]
    assert subset.plan_id != plan.plan_id
    assert subset.plan_sha256 != plan.plan_sha256
