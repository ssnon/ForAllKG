from __future__ import annotations

import hashlib
import json

from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQuery,
    LiteratureQueryPlan,
)


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(
        str(value)
        for value in parts
    ).encode("utf-8")

    return (
        f"{prefix}:"
        f"{hashlib.sha256(raw).hexdigest()[:length]}"
    )


def build_diagnostic_query_plan(
    base: LiteratureQueryPlan,
) -> LiteratureQueryPlan:
    """Materialize diagnostic candidates into a delta-only query plan.

    The ordinary LiteratureQueryPlan is intentionally left unchanged.

    Only canonical claim-level diagnostic_execution_query values are
    materialized. This pass does not add, infer, expand, or rewrite
    scientific concepts.

    The resulting claim_diagnostic queries are intended for a bounded
    diagnostic retrieval pass. They are not first-pass novelty queries
    and must not be interpreted as positive evidence for the full claim.
    """

    queries: list[LiteratureQuery] = []
    seen: set[
        tuple[str, str, str]
    ] = set()

    for group in base.claims:
        for claim in group.claims:
            if (
                claim.diagnostic_query_kind
                == "NONE"
            ):
                continue

            query_text = " ".join(
                str(
                    claim.diagnostic_execution_query
                    or ""
                ).split()
            )

            if not query_text:
                continue

            key = (
                claim.hypothesis_id,
                claim.claim_id,
                query_text.lower(),
            )

            if key in seen:
                continue

            seen.add(key)

            queries.append(
                LiteratureQuery(
                    query_id=_stable_id(
                        "literature_query",
                        base.plan_id,
                        claim.hypothesis_id,
                        claim.claim_id,
                        "claim_diagnostic",
                        claim.diagnostic_query_kind,
                        query_text,
                    ),
                    hypothesis_id=(
                        claim.hypothesis_id
                    ),
                    claim_id=claim.claim_id,
                    query_kind=(
                        "claim_diagnostic"
                    ),
                    query_text=query_text,
                )
            )

    plan_id = _stable_id(
        "literature_query_plan",
        base.source_portfolio_id,
        base.plan_id,
        "diagnostic_delta",
        *[
            query.query_id
            for query in queries
        ],
    )

    body = {
        "schema_version":
            "literature-query-plan-v1",
        "plan_id": plan_id,
        "source_portfolio_id":
            base.source_portfolio_id,
        "queries": [
            query.model_dump(mode="json")
            for query in queries
        ],
        # Preserve the canonical atomic claim
        # provenance needed to interpret the
        # diagnostic query.
        "claims": [
            group.model_dump(mode="json")
            for group in base.claims
        ],
        "policy_version":
            "external-novelty-query-policy-v1",
    }

    return LiteratureQueryPlan(
        **body,
        plan_sha256=_sha256_json(body),
    )
