from __future__ import annotations

import numpy as np

from dac_her.external_novelty_contracts import (
    LiteratureQuery,
    LiteratureQueryPlan,
    NoveltyClaim,
    PriorArtPacket,
    PriorArtWork,
)
from dac_her.prior_art_matching import PriorArtRanker
from dac_her.domains import get_domain_profile
from dac_her.standard2_ranker_dev_validation import sha256_json


class FakeEncoder:
    def encode_query(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    def encode_documents(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        rows = []
        for i, _ in enumerate(texts):
            rows.append(
                [1.0 - i * 0.1, i * 0.1]
            )
        return np.asarray(rows, dtype=np.float32)


def _packet() -> PriorArtPacket:
    works = [
        PriorArtWork(
            work_id=f"w{i}",
            title=f"SERS Au Ag nanogap work {i}",
            abstract="SERS Au Ag plasmon nanogap abstract",
            providers=["openalex"],
            provider_ids={"openalex": f"w{i}"},
            retrieval_query_ids=["q1"],
            retrieval_claim_ids=["c1"],
        )
        for i in range(3)
    ]
    body = {
        "schema_version": "prior-art-packet-v1",
        "packet_id": "p1",
        "source_portfolio_id": "portfolio1",
        "source_query_plan_id": "plan1",
        "searched_at_utc": "2026-08-16T00:00:00+00:00",
        "providers_requested": ["openalex"],
        "works": [w.model_dump(mode="json") for w in works],
        "executions": [],
        "raw_work_count": 3,
        "canonical_work_count": 3,
        "deduplicated_work_count": 0,
        "supplementary_records_collapsed": 0,
        "epistemic_usage": "prior_art_only_not_positive_premise",
    }
    return PriorArtPacket(
        **body,
        packet_sha256=sha256_json(body),
    )


def test_production_ranker_is_deterministic_with_fixed_encoder():
    claim = NoveltyClaim(
        claim_id="c1",
        hypothesis_id="h1",
        claim_rank=1,
        kind="mechanistic_link",
        importance="core",
        text="Au Ag nanogap increases SERS electromagnetic enhancement",
        rationale="test",
        search_concepts=["Au Ag", "nanogap", "SERS"],
        search_queries=[],
    )
    plan_body = {
        "schema_version": "literature-query-plan-v1",
        "plan_id": "plan1",
        "plan_sha256": "x",
        "source_portfolio_id": "portfolio1",
        "queries": [
            LiteratureQuery(
                query_id="q1",
                hypothesis_id="h1",
                claim_id="c1",
                query_kind="claim_primary",
                query_text="SERS Au Ag nanogap",
            )
        ],
        "claims": [],
        "policy_version": "external-novelty-query-policy-v1",
    }
    plan = LiteratureQueryPlan(**plan_body)
    ranker = PriorArtRanker(
        FakeEncoder(),
        max_ranked_works_per_claim=2,
        domain_profile=get_domain_profile("sers"),
    )
    left = ranker.rank(claim, _packet(), plan)
    right = ranker.rank(claim, _packet(), plan)
    assert [
        row.work_id for row in left.ranked_works
    ] == [
        row.work_id for row in right.ranked_works
    ]
    assert len(left.ranked_works) == 2
