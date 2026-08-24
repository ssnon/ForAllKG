from __future__ import annotations

import hashlib
import json

from pipeline_core.discovery.hypothesis_action_contracts import (
    G1FindingAuthority,
    G1FindingRef,
    G1FindingScope,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)
from pipeline_core.discovery.hypothesis_semantic_contracts import (
    HypothesisSemanticReview,
    SEMANTIC_DIMENSIONS,
)


_POLICY_VERSION = (
    "hypothesis-semantic-g1-normalization-v1"
)


class HypothesisSemanticActionBindingError(
    ValueError
):
    pass


def _canonical_json(
    value: object,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_id(
    prefix: str,
    payload: object,
) -> str:
    digest = hashlib.sha256(
        _canonical_json(
            payload
        ).encode("utf-8")
    ).hexdigest()[:20]

    return f"{prefix}:{digest}"


def _authority_for_verdict(
    verdict: str,
) -> G1FindingAuthority:
    if verdict in {
        "pass",
        "not_applicable",
    }:
        return "informational"

    if verdict == "warning":
        return "advisory"

    if verdict == "fail":
        # Semantic critic failure is diagnostic repair pressure.
        # It is deliberately NOT terminal authority by itself.
        return "actionable"

    raise HypothesisSemanticActionBindingError(
        "unsupported semantic verdict: "
        f"{verdict!r}"
    )


class HypothesisSemanticFindingActionAdapter:
    """Normalize a final semantic review into hypothesis-level G1 refs.

    Boundary:
      * review must directly target the final portfolio;
      * no cross-generation rebound is performed;
      * warning is advisory only;
      * fail is actionable but never terminal by itself;
      * informational portfolio-level dimensions are not broadcast
        across hypotheses;
      * non-informational portfolio-level dimensions require a
        separate portfolio policy and are rejected here.
    """

    policy_version = (
        _POLICY_VERSION
    )

    def normalize(
        self,
        *,
        review:
            HypothesisSemanticReview,

        final_portfolio:
            HypothesisPortfolio,

        source_artifact_id:
            str | None = None,
    ) -> tuple[
        G1FindingRef,
        ...
    ]:
        if (
            review.source_portfolio_id
            != final_portfolio.portfolio_id
        ):
            raise HypothesisSemanticActionBindingError(
                "semantic review must be directly "
                "bound to target final portfolio"
            )

        dimensions = [
            row.dimension
            for row in review.dimensions
        ]

        if (
            len(dimensions)
            != len(set(dimensions))
        ):
            raise HypothesisSemanticActionBindingError(
                "semantic review contains duplicate dimensions"
            )

        if (
            set(dimensions)
            != set(SEMANTIC_DIMENSIONS)
        ):
            raise HypothesisSemanticActionBindingError(
                "semantic review dimension set mismatch"
            )

        final_ids = {
            row.hypothesis_id
            for row
            in final_portfolio.hypotheses
        }

        artifact_id = (
            source_artifact_id
            or review.review_id
        )

        output: list[
            G1FindingRef
        ] = []

        for row in review.dimensions:
            unknown = (
                set(row.hypothesis_ids)
                - final_ids
            )

            if unknown:
                raise HypothesisSemanticActionBindingError(
                    "semantic review references "
                    "non-final hypothesis IDs: "
                    f"{sorted(unknown)}"
                )

            authority = (
                _authority_for_verdict(
                    row.verdict
                )
            )

            # ------------------------------------------------------
            # Portfolio-level semantic dimensions are not silently
            # duplicated into every hypothesis.
            # ------------------------------------------------------

            if not row.hypothesis_ids:
                if authority != "informational":
                    raise HypothesisSemanticActionBindingError(
                        "non-informational portfolio-level "
                        "semantic finding requires separate "
                        "portfolio action policy: "
                        + row.dimension
                    )

                continue

            for hypothesis_id in sorted(
                row.hypothesis_ids
            ):
                finding_ref_id = _stable_id(
                    "g1_finding_ref",
                    {
                        "policy":
                            self.policy_version,

                        "review_id":
                            review.review_id,

                        "dimension":
                            row.dimension,

                        "verdict":
                            row.verdict,

                        "target_portfolio_id":
                            final_portfolio
                            .portfolio_id,

                        "target_hypothesis_id":
                            hypothesis_id,
                    },
                )

                output.append(
                    G1FindingRef(
                        finding_ref_id=
                            finding_ref_id,

                        source_kind=
                            "semantic_review",

                        source_artifact_id=
                            artifact_id,

                        source_finding_id=(
                            "semantic_dimension:"
                            + row.dimension
                        ),

                        source_status=
                            row.verdict,

                        source_attributes={
                            "semantic_dimension":
                                row.dimension,

                            "assessment_level":
                                "hypothesis",

                            "statement_ids_json":
                                _canonical_json(
                                    row.statement_ids
                                ),

                            "critic_prompt_version":
                                review
                                .critic_prompt_version,

                            "direct_final_binding":
                                "true",
                        },

                        authority=
                            authority,

                        source_portfolio_id=
                            review
                            .source_portfolio_id,

                        source_hypothesis_ids=[
                            hypothesis_id
                        ],

                        source_scope=
                            G1FindingScope(
                                kind=
                                    "hypothesis",

                                hypothesis_ids=[
                                    hypothesis_id
                                ],
                            ),

                        target_portfolio_id=
                            final_portfolio
                            .portfolio_id,

                        target_hypothesis_id=
                            hypothesis_id,

                        target_scope=
                            G1FindingScope(
                                kind=
                                    "hypothesis",

                                hypothesis_ids=[
                                    hypothesis_id
                                ],
                            ),

                        # Same-generation direct binding.
                        lineage_ref_ids=[],

                        rationale=
                            row.rationale,
                    )
                )

        ids = [
            row.finding_ref_id
            for row in output
        ]

        if (
            len(ids)
            != len(set(ids))
        ):
            raise HypothesisSemanticActionBindingError(
                "duplicate normalized semantic "
                "finding_ref_id"
            )

        return tuple(
            sorted(
                output,
                key=lambda row:
                    row.finding_ref_id,
            )
        )
