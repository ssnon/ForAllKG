from __future__ import annotations

from collections import defaultdict
import hashlib
import json

from domains.sers.context_contracts import (
    SERSContextCompatibilityStatus,
    SERSContextFinding,
    SERSContextReview,
)
from domains.sers.hypothesis_context_contracts import (
    expected_hypothesis_context_assertions,
)
from pipeline_core.discovery.hypothesis_action_contracts import (
    G1FindingAuthority,
    G1FindingRef,
    G1FindingScope,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
)


_POLICY_VERSION = (
    "sers-g1-context-normalization-v1"
)


class SERSActionBindingError(
    ValueError
):
    pass


_AUTHORITY_BY_STATUS: dict[
    str,
    G1FindingAuthority,
] = {
    "match":
        "informational",

    "intentional_variation":
        "informational",

    "compatible_extension":
        "informational",

    "unknown":
        "advisory",

    "role_mismatch":
        "actionable",

    "context_conflation":
        "actionable",

    "conflict":
        "actionable",
}


_SCOPE_ORDER = {
    "central": 0,
    "bridge": 1,
    "prediction": 2,
    "assumption": 3,
}


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
    *,
    length: int = 20,
) -> str:
    digest = hashlib.sha256(
        _canonical_json(
            payload
        ).encode("utf-8")
    ).hexdigest()[:length]

    return f"{prefix}:{digest}"


def _authority(
    status:
        SERSContextCompatibilityStatus,
) -> G1FindingAuthority:
    try:
        return _AUTHORITY_BY_STATUS[
            status
        ]
    except KeyError as exc:
        raise SERSActionBindingError(
            "unsupported SERS context "
            f"status for G1: {status!r}"
        ) from exc


def _assertion_ids_from_tags(
    tags: list[str],
) -> list[str]:
    prefix = "assertion:"

    return sorted({
        tag[len(prefix):]
        for tag in tags
        if tag.startswith(prefix)
        and len(tag) > len(prefix)
    })


def _source_assertion_kind(
    *,
    hypothesis_id: str,
    assertion_id: str,
) -> str:
    if (
        assertion_id
        == f"central:{hypothesis_id}"
    ):
        return "central"

    if (
        assertion_id
        == f"bridge:{hypothesis_id}"
    ):
        return "bridge"

    if assertion_id.startswith(
        f"assumption:{hypothesis_id}:"
    ):
        suffix = assertion_id[
            len(
                f"assumption:{hypothesis_id}:"
            ):
        ]

        if not suffix.isdigit():
            raise SERSActionBindingError(
                "malformed assumption assertion ID: "
                + assertion_id
            )

        return "assumption"

    if assertion_id.startswith(
        "prediction:"
    ):
        return "prediction"

    raise SERSActionBindingError(
        "unrecognized SERS hypothesis "
        "assertion ID: "
        + assertion_id
    )


def _source_assertion_texts(
    review: SERSContextReview,
) -> dict[str, str]:
    """Recover exact assertion text from S1 provenance.

    Do not reconstruct source scientific content from IDs.
    The hypothesis-side context signature stores the original
    assertion excerpt used by the S1 interpretation/compiler.
    """

    candidates = [
        signature
        for signature
        in review.signatures
        if (
            signature.scope
            == "hypothesis"
            and signature.source_ref_id
            == review.hypothesis_id
        )
    ]

    if len(candidates) != 1:
        raise SERSActionBindingError(
            "S1 review must contain exactly one "
            "hypothesis-side source signature; "
            f"found={len(candidates)}"
        )

    signature = candidates[0]

    result: dict[
        str,
        str,
    ] = {}

    for fact in signature.facts:
        assertion_ids = (
            _assertion_ids_from_tags(
                fact.tags
            )
        )

        if not assertion_ids:
            continue

        excerpts = sorted({
            str(provenance.excerpt)
            for provenance
            in fact.provenance
            if (
                provenance.kind
                == "hypothesis_assertion"
                and (
                    provenance.excerpt
                    or ""
                ).strip()
            )
        })

        if len(excerpts) != 1:
            raise SERSActionBindingError(
                "hypothesis-side S1 context fact "
                "does not have exactly one assertion "
                f"excerpt: fact_id={fact.fact_id}, "
                f"excerpt_count={len(excerpts)}"
            )

        excerpt = excerpts[0]

        for assertion_id in (
            assertion_ids
        ):
            previous = result.get(
                assertion_id
            )

            if (
                previous is not None
                and previous != excerpt
            ):
                raise SERSActionBindingError(
                    "conflicting source assertion "
                    "excerpts in S1 review: "
                    + assertion_id
                )

            result[
                assertion_id
            ] = excerpt

    if not result:
        raise SERSActionBindingError(
            "S1 review contains no recoverable "
            "hypothesis assertion provenance"
        )

    return result


def _target_assertion_catalog(
    card: HypothesisCard,
) -> dict[
    tuple[str, str],
    list[str],
]:
    """Index final assertions by kind + exact scientific text."""

    result: dict[
        tuple[str, str],
        list[str],
    ] = defaultdict(
        list
    )

    for row in (
        expected_hypothesis_context_assertions(
            card
        )
    ):
        key = (
            row[
                "assertion_kind"
            ],
            row[
                "assertion_text"
            ],
        )

        result[key].append(
            row[
                "assertion_id"
            ]
        )

    return dict(result)


def _rebound_assertion_id(
    *,
    review: SERSContextReview,
    source_assertion_id: str,
    source_assertion_texts:
        dict[str, str],
    target_catalog:
        dict[
            tuple[str, str],
            list[str],
        ],
) -> tuple[str, str]:
    source_text = (
        source_assertion_texts.get(
            source_assertion_id
        )
    )

    if source_text is None:
        raise SERSActionBindingError(
            "S1 finding references an assertion "
            "whose source text is absent from "
            "hypothesis context provenance: "
            + source_assertion_id
        )

    kind = _source_assertion_kind(
        hypothesis_id=
            review.hypothesis_id,
        assertion_id=
            source_assertion_id,
    )

    matches = target_catalog.get(
        (
            kind,
            source_text,
        ),
        [],
    )

    if not matches:
        raise SERSActionBindingError(
            "source S1 assertion has no exact "
            "scientific-content match in target "
            "hypothesis; S1 re-review is required: "
            f"kind={kind}, "
            f"source_assertion_id="
            f"{source_assertion_id}"
        )

    if len(matches) != 1:
        raise SERSActionBindingError(
            "source S1 assertion maps ambiguously "
            "to multiple target assertions; "
            "implicit choice is forbidden: "
            f"kind={kind}, "
            f"source_assertion_id="
            f"{source_assertion_id}, "
            f"matches={matches}"
        )

    return (
        kind,
        matches[0],
    )


class SERSContextFindingActionAdapter:
    """Normalize SERS S1 findings into generic G1 finding refs.

    This adapter:
      * does not mutate a hypothesis,
      * does not decide final G1 disposition,
      * does not call an LLM,
      * does not infer missing scientific context,
      * does not transfer a finding across changed assertion text.

    One SERS semantic finding may span central/bridge/prediction/
    assumption assertions. Such a finding is exploded into one
    G1FindingRef per scope kind so local lifecycle action can remain
    bounded.
    """

    policy_version = (
        _POLICY_VERSION
    )

    def normalize(
        self,
        review: SERSContextReview,
        *,
        target_card: HypothesisCard,
        target_portfolio_id: str,
        source_portfolio_id:
            str | None = None,
        source_artifact_id:
            str | None = None,
        lineage_ref_ids:
            list[str] | None = None,
    ) -> tuple[
        G1FindingRef,
        ...
    ]:
        domain_ids = {
            signature.domain_profile_id
            for signature
            in review.signatures
        }

        if len(domain_ids) != 1:
            raise SERSActionBindingError(
                "S1 review contains mixed "
                "domain_profile_id values"
            )

        source_domain = next(
            iter(domain_ids)
        )

        if (
            source_domain
            != target_card.domain_profile_id
        ):
            raise SERSActionBindingError(
                "S1 review / target hypothesis "
                "domain mismatch"
            )

        lineage = sorted(
            set(
                lineage_ref_ids
                or []
            )
        )

        crosses_identity = (
            review.hypothesis_id
            != target_card.hypothesis_id
        )

        crosses_portfolio = (
            source_portfolio_id
            is not None
            and source_portfolio_id
            != target_portfolio_id
        )

        if (
            crosses_identity
            or crosses_portfolio
        ) and not lineage:
            raise SERSActionBindingError(
                "cross-generation S1 finding "
                "normalization requires explicit "
                "lineage_ref_ids"
            )

        source_texts = (
            _source_assertion_texts(
                review
            )
        )

        target_catalog = (
            _target_assertion_catalog(
                target_card
            )
        )

        output: list[
            G1FindingRef
        ] = []

        artifact_id = (
            source_artifact_id
            or review.review_id
        )

        for finding in (
            review.findings
        ):
            assertion_ids = (
                _assertion_ids_from_tags(
                    finding.tags
                )
            )

            if not assertion_ids:
                raise SERSActionBindingError(
                    "S1 finding has no assertion "
                    "scope tags: "
                    + finding.finding_id
                )

            grouped_source: dict[
                str,
                list[str],
            ] = defaultdict(
                list
            )

            grouped_target: dict[
                str,
                list[str],
            ] = defaultdict(
                list
            )

            for source_id in (
                assertion_ids
            ):
                (
                    kind,
                    target_id,
                ) = _rebound_assertion_id(
                    review=review,
                    source_assertion_id=
                        source_id,
                    source_assertion_texts=
                        source_texts,
                    target_catalog=
                        target_catalog,
                )

                grouped_source[
                    kind
                ].append(
                    source_id
                )

                grouped_target[
                    kind
                ].append(
                    target_id
                )

            if (
                set(grouped_source)
                != set(grouped_target)
            ):
                raise SERSActionBindingError(
                    "internal S1 scope rebound "
                    "group mismatch"
                )

            for kind in sorted(
                grouped_source,
                key=lambda value:
                    _SCOPE_ORDER[value],
            ):
                source_ids = sorted(
                    set(
                        grouped_source[
                            kind
                        ]
                    )
                )

                target_ids = sorted(
                    set(
                        grouped_target[
                            kind
                        ]
                    )
                )

                if (
                    len(source_ids)
                    != len(
                        grouped_source[
                            kind
                        ]
                    )
                ):
                    raise SERSActionBindingError(
                        "duplicate source assertion "
                        "binding inside S1 finding"
                    )

                if (
                    len(target_ids)
                    != len(
                        grouped_target[
                            kind
                        ]
                    )
                ):
                    raise SERSActionBindingError(
                        "multiple source assertions "
                        "collapsed onto one target "
                        "assertion"
                    )

                finding_ref_id = (
                    _stable_id(
                        "g1_finding_ref",
                        {
                            "policy":
                                self.policy_version,

                            "source_review_id":
                                review.review_id,

                            "source_finding_id":
                                finding.finding_id,

                            "scope_kind":
                                kind,

                            "source_assertion_ids":
                                source_ids,

                            "target_portfolio_id":
                                target_portfolio_id,

                            "target_hypothesis_id":
                                target_card
                                .hypothesis_id,

                            "target_assertion_ids":
                                target_ids,

                            "authority":
                                _authority(
                                    finding.status
                                ),
                        },
                    )
                )

                output.append(
                    G1FindingRef(
                        finding_ref_id=
                            finding_ref_id,

                        source_kind=
                            "context_review",

                        source_artifact_id=
                            artifact_id,

                        source_finding_id=
                            finding.finding_id,

                        source_status=
                            finding.status,

                        authority=
                            _authority(
                                finding.status
                            ),

                        source_portfolio_id=
                            source_portfolio_id,

                        source_hypothesis_ids=[
                            review.hypothesis_id
                        ],

                        source_scope=
                            G1FindingScope(
                                kind=kind,
                                hypothesis_ids=[
                                    review
                                    .hypothesis_id
                                ],
                                assertion_ids=
                                    source_ids,
                            ),

                        target_portfolio_id=
                            target_portfolio_id,

                        target_hypothesis_id=
                            target_card
                            .hypothesis_id,

                        target_scope=
                            G1FindingScope(
                                kind=kind,
                                hypothesis_ids=[
                                    target_card
                                    .hypothesis_id
                                ],
                                assertion_ids=
                                    target_ids,
                            ),

                        lineage_ref_ids=
                            lineage,

                        rationale=
                            finding.rationale,
                    )
                )

        ids = [
            row.finding_ref_id
            for row in output
        ]

        if len(ids) != len(set(ids)):
            raise SERSActionBindingError(
                "duplicate normalized G1 "
                "finding_ref_id"
            )

        return tuple(
            sorted(
                output,
                key=lambda row:
                    row.finding_ref_id,
            )
        )
