from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import re
from typing import Any

from domains.sers.context_contracts import (
    SERSContextBinding,
    SERSContextCompatibilityStatus,
    SERSContextFact,
    SERSContextFinding,
    SERSContextProvenance,
    SERSContextReview,
    SERSContextSignature,
    expected_context_finding_severity,
    expected_context_review_status,
)
from domains.sers.hypothesis_context_contracts import (
    HypothesisContextInterpretation,
    HypothesisContextMentionDraft,
)


_COMPARATOR_POLICY_VERSION = (
    "sers-context-comparator-v1"
)


class SERSContextComparatorError(
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


def _digest(
    prefix: str,
    payload: object,
) -> str:
    sha = hashlib.sha256(
        _canonical_json(
            payload
        ).encode("utf-8")
    ).hexdigest()[:20]

    return f"{prefix}:{sha}"


def _normalize_text(
    value: str | None,
) -> str:
    if value is None:
        return ""

    text = str(value).strip().lower()

    text = re.sub(
        r"[\s\-_]+",
        " ",
        text,
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    # Only a narrow normalization for obvious English plural
    # attachment labels. This is identity normalization, not
    # scientific semantic inference.
    if (
        text.endswith("s")
        and not text.endswith("ss")
        and len(text) > 4
    ):
        text = text[:-1]

    return text


def _hypothesis_signature_id(
    hypothesis_id: str,
) -> str:
    return _digest(
        "sers_context_signature",
        {
            "kind":
                "hypothesis_context",
            "policy":
                _COMPARATOR_POLICY_VERSION,
            "hypothesis_id":
                hypothesis_id,
        },
    )


def _coverage_signature_id(
    hypothesis_id: str,
) -> str:
    return _digest(
        "sers_context_signature",
        {
            "kind":
                "typed_context_coverage",
            "policy":
                _COMPARATOR_POLICY_VERSION,
            "hypothesis_id":
                hypothesis_id,
        },
    )


def _hypothesis_fact_id(
    *,
    hypothesis_id: str,
    assertion_id: str,
    mention: HypothesisContextMentionDraft,
) -> str:
    return _digest(
        "sers_context_fact",
        {
            "kind":
                "hypothesis_assertion",
            "hypothesis_id":
                hypothesis_id,
            "assertion_id":
                assertion_id,
            "mention_id":
                mention.mention_id,
            "dimension":
                mention.asserted_dimension,
            "role":
                mention.asserted_role,
            "owner":
                _normalize_text(
                    mention.asserted_owner_label
                ),
            "treatment":
                mention.treatment,
        },
    )


def _coverage_fact_id(
    *,
    hypothesis_id: str,
    mention: HypothesisContextMentionDraft,
) -> str:
    return _digest(
        "sers_context_fact",
        {
            "kind":
                "typed_context_coverage_unknown",
            "hypothesis_id":
                hypothesis_id,
            "dimension":
                mention.asserted_dimension,
            "role":
                mention.asserted_role,
            "owner":
                _normalize_text(
                    mention.asserted_owner_label
                ),
        },
    )


def _build_hypothesis_signature(
    interpretation:
        HypothesisContextInterpretation,
    *,
    domain_profile_id: str,
) -> tuple[
    SERSContextSignature,
    dict[str, str],
]:
    facts = []
    mention_fact_ids: dict[
        str,
        str,
    ] = {}

    for assertion in (
        interpretation.assertions
    ):
        for mention in (
            assertion.mentions
        ):
            fact_id = (
                _hypothesis_fact_id(
                    hypothesis_id=
                        interpretation.hypothesis_id,
                    assertion_id=
                        assertion.assertion_id,
                    mention=mention,
                )
            )

            if (
                mention.mention_id
                in mention_fact_ids
            ):
                raise SERSContextComparatorError(
                    "duplicate hypothesis context "
                    f"mention_id: {mention.mention_id}"
                )

            mention_fact_ids[
                mention.mention_id
            ] = fact_id

            binding = None

            if (
                mention.asserted_owner_label
                is not None
            ):
                binding = SERSContextBinding(
                    basis=
                        "hypothesis_assertion",
                    owner_ref_id=
                        assertion.assertion_id,
                    owner_label=
                        mention.asserted_owner_label,
                    owner_type=(
                        mention.asserted_owner_type
                        or "HypothesisContextOwner"
                    ),
                    relation=
                        mention.asserted_relation,
                )

            facts.append(
                SERSContextFact(
                    fact_id=fact_id,
                    dimension=
                        mention.asserted_dimension,
                    scientific_role=
                        mention.asserted_role,
                    knowledge_state=
                        "explicit",
                    value=
                        mention.mention_text,
                    normalized_value=
                        _normalize_text(
                            mention.mention_text
                        ),
                    binding=binding,
                    provenance=[
                        SERSContextProvenance(
                            kind=
                                "hypothesis_assertion",
                            hypothesis_ids=[
                                interpretation.hypothesis_id
                            ],
                            excerpt=
                                assertion.assertion_text,
                        )
                    ],
                    tags=sorted({
                        "hypothesis_context",
                        (
                            "assertion:"
                            + assertion.assertion_id
                        ),
                        (
                            "mention:"
                            + mention.mention_id
                        ),
                        (
                            "treatment:"
                            + mention.treatment
                        ),
                        (
                            "experimental_role:"
                            + mention.experimental_role
                        ),
                    }),
                )
            )

    if not facts:
        raise SERSContextComparatorError(
            "hypothesis interpretation contains "
            "no context facts"
        )

    signature = SERSContextSignature(
        signature_id=
            _hypothesis_signature_id(
                interpretation.hypothesis_id
            ),
        domain_profile_id=
            domain_profile_id,
        scope="hypothesis",
        source_ref_id=
            interpretation.hypothesis_id,
        facts=sorted(
            facts,
            key=lambda row:
                row.fact_id,
        ),
    )

    return (
        signature,
        mention_fact_ids,
    )


def _source_fact_index(
    source_signatures: list[
        SERSContextSignature
    ],
) -> tuple[
    dict[str, SERSContextFact],
    dict[str, str],
]:
    fact_by_id: dict[
        str,
        SERSContextFact,
    ] = {}

    signature_by_fact_id: dict[
        str,
        str,
    ] = {}

    for signature in (
        source_signatures
    ):
        for fact in signature.facts:
            if fact.fact_id in fact_by_id:
                raise SERSContextComparatorError(
                    "duplicate source context fact_id: "
                    + fact.fact_id
                )

            fact_by_id[
                fact.fact_id
            ] = fact

            signature_by_fact_id[
                fact.fact_id
            ] = signature.signature_id

    return (
        fact_by_id,
        signature_by_fact_id,
    )


def _classify_source_backed_mention(
    *,
    mention:
        HypothesisContextMentionDraft,
    source_facts: list[
        SERSContextFact
    ],
) -> SERSContextCompatibilityStatus:
    if not source_facts:
        return "unknown"

    treatment = mention.treatment

    if treatment == "uncertain":
        return "unknown"

    dimensions_match = all(
        fact.dimension
        == mention.asserted_dimension
        for fact in source_facts
    )

    roles_match = all(
        fact.scientific_role
        == mention.asserted_role
        for fact in source_facts
    )

    if treatment == "reattach":
        if (
            not dimensions_match
            or not roles_match
        ):
            return "role_mismatch"

        # A same-role attachment has still moved.
        # Treat it as actionable only when that moved
        # attachment is scientifically active in the
        # hypothesis. Controlled/comparison-only attachment
        # changes remain conservative UNKNOWN in v1.
        if mention.experimental_role in {
            "experimental_variable",
            "moderator",
            "response",
        }:
            return "context_conflation"

        return "unknown"

    if treatment in {
        "preserve",
        "generalize",
        "intentionally_vary",
        "combine",
    }:
        if (
            not dimensions_match
            or not roles_match
        ):
            return "role_mismatch"

        if any(
            fact.knowledge_state
            == "unknown"
            for fact in source_facts
        ):
            return "unknown"

        if treatment == "preserve":
            return "match"

        if treatment == "generalize":
            return "compatible_extension"

        if treatment == "intentionally_vary":
            return "intentional_variation"

        if treatment == "combine":
            return "compatible_extension"

    if treatment == "reference_only":
        if (
            dimensions_match
            and roles_match
        ):
            return "match"

        return "unknown"

    return "unknown"


def _rationale(
    *,
    status:
        SERSContextCompatibilityStatus,
    treatment: str,
    dimension: str,
    source_facts: list[
        SERSContextFact
    ],
    mention:
        HypothesisContextMentionDraft,
    occurrence_count: int,
) -> str:
    source_summary = ", ".join(
        sorted({
            (
                f"{fact.dimension}/"
                f"{fact.scientific_role}"
            )
            for fact in source_facts
        })
    )

    if not source_summary:
        source_summary = (
            "no typed source-context fact"
        )

    owner = (
        mention.asserted_owner_label
        or "unspecified owner"
    )

    base = (
        f"Hypothesis context treatment "
        f"{treatment!r} maps source context "
        f"[{source_summary}] to "
        f"{dimension}/{mention.asserted_role} "
        f"at {owner!r}; deterministic status "
        f"is {status!r}."
    )

    if occurrence_count > 1:
        base += (
            f" The same semantic transformation "
            f"appears in {occurrence_count} "
            f"hypothesis assertions/mentions."
        )

    return base


class SERSHypothesisContextComparator:
    """Deterministic source↔hypothesis context comparator.

    This component does not call an LLM, assess novelty, or decide
    downstream repair actions. It maps the validated interpretation
    contract onto the existing SERSContextFinding/SERSContextReview
    contract.
    """

    policy_version = (
        _COMPARATOR_POLICY_VERSION
    )

    def compare(
        self,
        *,
        interpretation:
            HypothesisContextInterpretation,
        source_signatures: list[
            SERSContextSignature
        ],
        domain_profile_id: str,
    ) -> SERSContextReview:
        if not source_signatures:
            raise SERSContextComparatorError(
                "source_signatures must not be empty"
            )

        if len({
            row.signature_id
            for row in source_signatures
        }) != len(source_signatures):
            raise SERSContextComparatorError(
                "duplicate source signature_id"
            )

        if any(
            row.domain_profile_id
            != domain_profile_id
            for row in source_signatures
        ):
            raise SERSContextComparatorError(
                "source signature domain mismatch"
            )

        supplied_ids = sorted(
            row.signature_id
            for row in source_signatures
        )

        interpreted_ids = sorted(
            interpretation.source_signature_ids
        )

        if supplied_ids != interpreted_ids:
            raise SERSContextComparatorError(
                "interpretation source_signature_ids "
                "do not match supplied source signatures"
            )

        (
            source_fact_by_id,
            source_signature_by_fact_id,
        ) = _source_fact_index(
            source_signatures
        )

        (
            hypothesis_signature,
            hypothesis_fact_by_mention_id,
        ) = _build_hypothesis_signature(
            interpretation,
            domain_profile_id=
                domain_profile_id,
        )

        coverage_signature_id = (
            _coverage_signature_id(
                interpretation.hypothesis_id
            )
        )

        coverage_facts: dict[
            str,
            SERSContextFact,
        ] = {}

        # Each raw observation is later folded into a semantic
        # finding family. The family key deliberately excludes
        # assertion_id and mention_id.
        observations: list[
            dict[str, Any]
        ] = []

        for assertion in (
            interpretation.assertions
        ):
            for mention in (
                assertion.mentions
            ):
                right_fact_id = (
                    hypothesis_fact_by_mention_id[
                        mention.mention_id
                    ]
                )

                if mention.source_fact_ids:
                    unknown_ids = [
                        fact_id
                        for fact_id
                        in mention.source_fact_ids
                        if fact_id
                        not in source_fact_by_id
                    ]

                    if unknown_ids:
                        raise SERSContextComparatorError(
                            "interpretation references "
                            "unknown source context facts: "
                            + ", ".join(
                                sorted(
                                    unknown_ids
                                )
                            )
                        )

                    grouped_ids: dict[
                        str,
                        list[str],
                    ] = defaultdict(
                        list
                    )

                    for fact_id in (
                        mention.source_fact_ids
                    ):
                        grouped_ids[
                            source_signature_by_fact_id[
                                fact_id
                            ]
                        ].append(
                            fact_id
                        )

                    # Pairwise source-signature comparison is
                    # required by SERSContextFinding's existing
                    # left_signature_id contract.
                    for (
                        left_signature_id,
                        fact_ids,
                    ) in sorted(
                        grouped_ids.items()
                    ):
                        source_facts = [
                            source_fact_by_id[
                                fact_id
                            ]
                            for fact_id
                            in sorted(
                                fact_ids
                            )
                        ]

                        status = (
                            _classify_source_backed_mention(
                                mention=mention,
                                source_facts=
                                    source_facts,
                            )
                        )

                        observations.append({
                            "left_signature_id":
                                left_signature_id,
                            "left_fact_ids":
                                sorted(
                                    fact_ids
                                ),
                            "right_fact_id":
                                right_fact_id,
                            "dimension":
                                mention.asserted_dimension,
                            "status":
                                status,
                            "treatment":
                                mention.treatment,
                            "asserted_role":
                                mention.asserted_role,
                            "owner":
                                _normalize_text(
                                    mention
                                    .asserted_owner_label
                                ),
                            "mention":
                                mention,
                            "assertion_id":
                                assertion.assertion_id,
                            "source_facts":
                                source_facts,
                        })

                    continue

                # No typed source fact exists. This does NOT mean
                # the broader scientific premise is unsupported.
                # Represent only the typed-context coverage state.
                coverage_fact_id = (
                    _coverage_fact_id(
                        hypothesis_id=
                            interpretation.hypothesis_id,
                        mention=mention,
                    )
                )

                if (
                    coverage_fact_id
                    not in coverage_facts
                ):
                    coverage_facts[
                        coverage_fact_id
                    ] = SERSContextFact(
                        fact_id=
                            coverage_fact_id,
                        dimension=
                            mention.asserted_dimension,
                        scientific_role=
                            mention.asserted_role,
                        knowledge_state=
                            "unknown",
                        value=None,
                        normalized_value=None,
                        binding=None,
                        provenance=[
                            SERSContextProvenance(
                                kind="question",
                                hypothesis_ids=[
                                    interpretation
                                    .hypothesis_id
                                ],
                                excerpt=(
                                    "Typed source-context "
                                    "coverage is unknown for "
                                    "this hypothesis context."
                                ),
                            )
                        ],
                        tags=[
                            "typed_context_coverage_unknown"
                        ],
                    )

                observations.append({
                    "left_signature_id":
                        coverage_signature_id,
                    "left_fact_ids": [
                        coverage_fact_id
                    ],
                    "right_fact_id":
                        right_fact_id,
                    "dimension":
                        mention.asserted_dimension,
                    "status":
                        "unknown",
                    "treatment":
                        mention.treatment,
                    "asserted_role":
                        mention.asserted_role,
                    "owner":
                        _normalize_text(
                            mention
                            .asserted_owner_label
                        ),
                    "mention":
                        mention,
                    "assertion_id":
                        assertion.assertion_id,
                    "source_facts": [],
                })

        if not observations:
            raise SERSContextComparatorError(
                "interpretation produced no "
                "context comparison observations"
            )

        # -------------------------------------------------------------
        # SEMANTIC FINDING FAMILIES
        #
        # Repeated central/bridge/prediction/assumption expressions
        # of the same source→hypothesis transformation become one
        # finding with multiple right_fact_ids and assertion tags.
        # -------------------------------------------------------------

        grouped: dict[
            tuple[Any, ...],
            list[dict[str, Any]],
        ] = defaultdict(
            list
        )

        for row in observations:
            # ROLE_MISMATCH identity is the scientific role/dimension
            # transition itself. Repeated central/bridge/prediction/
            # assumption phrasings may attach that same promoted role
            # to slightly different hypothesis owner labels; those are
            # evidence occurrences, not independent defects.
            #
            # For CONTEXT_CONFLATION, by contrast, the moved attachment
            # target is itself the scientific defect, so owner remains
            # part of the semantic family identity.
            if (
                row["status"]
                == "role_mismatch"
            ):
                attachment_key = (
                    "role_transition"
                )
            else:
                attachment_key = (
                    "owner:"
                    + row["owner"]
                )

            family_key = (
                row[
                    "left_signature_id"
                ],
                tuple(
                    row[
                        "left_fact_ids"
                    ]
                ),
                row[
                    "dimension"
                ],
                row[
                    "status"
                ],
                row[
                    "treatment"
                ],
                row[
                    "asserted_role"
                ],
                attachment_key,
            )

            grouped[
                family_key
            ].append(
                row
            )

        findings = []

        for family_key, rows in sorted(
            grouped.items(),
            key=lambda item:
                repr(item[0]),
        ):
            exemplar = rows[0]

            right_fact_ids = sorted({
                row[
                    "right_fact_id"
                ]
                for row in rows
            })

            assertion_ids = sorted({
                row[
                    "assertion_id"
                ]
                for row in rows
            })

            status = exemplar[
                "status"
            ]

            finding_id = _digest(
                "sers_context_finding",
                {
                    "policy":
                        self.policy_version,
                    "hypothesis_id":
                        interpretation
                        .hypothesis_id,
                    "family":
                        family_key,
                    "right_fact_ids":
                        right_fact_ids,
                },
            )

            findings.append(
                SERSContextFinding(
                    finding_id=
                        finding_id,
                    dimension=
                        exemplar[
                            "dimension"
                        ],
                    status=
                        status,
                    severity=
                        expected_context_finding_severity(
                            status
                        ),
                    left_signature_id=
                        exemplar[
                            "left_signature_id"
                        ],
                    right_signature_id=
                        hypothesis_signature
                        .signature_id,
                    left_fact_ids=
                        exemplar[
                            "left_fact_ids"
                        ],
                    right_fact_ids=
                        right_fact_ids,
                    rationale=
                        _rationale(
                            status=status,
                            treatment=
                                exemplar[
                                    "treatment"
                                ],
                            dimension=
                                exemplar[
                                    "dimension"
                                ],
                            source_facts=
                                exemplar[
                                    "source_facts"
                                ],
                            mention=
                                exemplar[
                                    "mention"
                                ],
                            occurrence_count=
                                len(rows),
                        ),
                    tags=sorted({
                        (
                            "treatment:"
                            + exemplar[
                                "treatment"
                            ]
                        ),
                        (
                            "asserted_role:"
                            + exemplar[
                                "asserted_role"
                            ]
                        ),
                        *(
                            "assertion:"
                            + assertion_id
                            for assertion_id
                            in assertion_ids
                        ),
                        *(
                            "experimental_role:"
                            + row[
                                "mention"
                            ].experimental_role
                            for row in rows
                        ),
                    }),
                )
            )

        signatures = [
            *sorted(
                source_signatures,
                key=lambda row:
                    row.signature_id,
            ),
            hypothesis_signature,
        ]

        if coverage_facts:
            signatures.append(
                SERSContextSignature(
                    signature_id=
                        coverage_signature_id,
                    domain_profile_id=
                        domain_profile_id,
                    scope="question",
                    source_ref_id=(
                        "typed-context-coverage:"
                        + interpretation
                        .hypothesis_id
                    ),
                    facts=sorted(
                        coverage_facts.values(),
                        key=lambda row:
                            row.fact_id,
                    ),
                )
            )

        findings = sorted(
            findings,
            key=lambda row:
                row.finding_id,
        )

        review_status = (
            expected_context_review_status(
                findings
            )
        )

        review_id = _digest(
            "sers_context_review",
            {
                "policy":
                    self.policy_version,
                "hypothesis_id":
                    interpretation.hypothesis_id,
                "signature_ids": [
                    row.signature_id
                    for row in signatures
                ],
                "finding_ids": [
                    row.finding_id
                    for row in findings
                ],
                "status":
                    review_status,
            },
        )

        return SERSContextReview(
            review_id=
                review_id,
            hypothesis_id=
                interpretation.hypothesis_id,
            signatures=
                signatures,
            findings=
                findings,
            status=
                review_status,
        )
