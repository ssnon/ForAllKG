from __future__ import annotations

import re
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from domains.sers.context_contracts import (
    SERSContextDimension,
    SERSContextRole,
    SERSContextSignature,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


HypothesisContextAssertionKind = Literal[
    "central",
    "bridge",
    "prediction",
    "assumption",
]


HypothesisContextTreatment = Literal[
    "preserve",
    "generalize",
    "intentionally_vary",
    "reattach",
    "combine",
    "introduce",
    "reference_only",
    "uncertain",
]


HypothesisContextExperimentalRole = Literal[
    "controlled_constant",
    "experimental_variable",
    "moderator",
    "response",
    "comparison_context",
    "unspecified",
]


def _compact_ws(
    value: str,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(value or "").strip(),
    )


def expected_hypothesis_context_assertions(
    card: HypothesisCard,
) -> tuple[dict[str, str], ...]:
    rows: list[
        dict[str, str]
    ] = [
        {
            "assertion_id":
                f"central:{card.hypothesis_id}",
            "assertion_kind":
                "central",
            "assertion_text":
                card.hypothesis_statement,
        },
        {
            "assertion_id":
                f"bridge:{card.hypothesis_id}",
            "assertion_kind":
                "bridge",
            "assertion_text":
                card.inferential_bridge,
        },
    ]

    for observation in (
        card.predicted_observations
    ):
        rows.append({
            "assertion_id":
                observation.observation_id,

            "assertion_kind":
                "prediction",

            "assertion_text":
                (
                    "Observable: "
                    + observation.observable
                    + "\nExpected direction: "
                    + observation.expected_direction
                    + "\nRationale: "
                    + observation.rationale
                ),
        })

    for index, assumption in enumerate(
        card.assumptions
    ):
        rows.append({
            "assertion_id":
                (
                    f"assumption:"
                    f"{card.hypothesis_id}:"
                    f"{index}"
                ),

            "assertion_kind":
                "assumption",

            "assertion_text":
                assumption,
        })

    return tuple(rows)


class HypothesisContextMentionDraft(
    StrictModel
):
    mention_id: str = Field(
        min_length=1
    )

    mention_text: str = Field(
        min_length=1
    )

    source_fact_ids: list[str] = Field(
        default_factory=list
    )

    asserted_dimension: SERSContextDimension

    asserted_role: SERSContextRole

    asserted_owner_label: str | None = None

    asserted_owner_type: str | None = None

    asserted_relation: str | None = None

    treatment: HypothesisContextTreatment

    experimental_role: HypothesisContextExperimentalRole = (
        "unspecified"
    )

    rationale: str = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def _treatment_shape(
        self,
    ) -> "HypothesisContextMentionDraft":
        count = len(
            set(self.source_fact_ids)
        )

        if (
            count
            != len(self.source_fact_ids)
        ):
            raise ValueError(
                "duplicate source_fact_ids"
            )

        needs_source = {
            "preserve",
            "generalize",
            "intentionally_vary",
            "reattach",
            "reference_only",
        }

        if (
            self.treatment
            in needs_source
            and count < 1
        ):
            raise ValueError(
                f"{self.treatment} requires "
                "source_fact_ids"
            )

        if (
            self.treatment == "combine"
            and count < 2
        ):
            raise ValueError(
                "combine requires at least "
                "two source facts"
            )

        if (
            self.treatment == "introduce"
            and count != 0
        ):
            raise ValueError(
                "introduce must not claim "
                "source fact support"
            )

        if (
            self.treatment == "reattach"
            and not (
                self.asserted_owner_label
                or ""
            ).strip()
        ):
            raise ValueError(
                "reattach requires an asserted owner"
            )

        return self


class HypothesisContextAssertionDraft(
    StrictModel
):
    assertion_id: str = Field(
        min_length=1
    )

    assertion_kind: HypothesisContextAssertionKind

    assertion_text: str = Field(
        min_length=1
    )

    mentions: list[
        HypothesisContextMentionDraft
    ] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def _unique_mentions(
        self,
    ) -> "HypothesisContextAssertionDraft":
        ids = [
            row.mention_id
            for row in self.mentions
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "duplicate mention_id"
            )

        return self


class HypothesisContextInterpretationDraft(
    StrictModel
):
    schema_version: Literal[
        "sers-hypothesis-context-draft-v1"
    ] = "sers-hypothesis-context-draft-v1"

    hypothesis_id: str = Field(
        min_length=1
    )

    source_signature_ids: list[
        str
    ] = Field(
        min_length=1
    )

    assertions: list[
        HypothesisContextAssertionDraft
    ] = Field(
        min_length=1
    )


class HypothesisContextInterpretation(
    StrictModel
):
    schema_version: Literal[
        "sers-hypothesis-context-interpretation-v1"
    ] = (
        "sers-hypothesis-context-interpretation-v1"
    )

    policy_version: Literal[
        "sers-hypothesis-context-interpretation-policy-v1"
    ] = (
        "sers-hypothesis-context-interpretation-policy-v1"
    )

    hypothesis_id: str = Field(
        min_length=1
    )

    source_signature_ids: list[
        str
    ] = Field(
        min_length=1
    )

    assertions: list[
        HypothesisContextAssertionDraft
    ] = Field(
        min_length=1
    )


class HypothesisContextInterpretationValidationError(
    ValueError
):
    def __init__(
        self,
        issues: list[str],
    ) -> None:
        self.issues = tuple(
            issues
        )

        super().__init__(
            "invalid SERS hypothesis-context "
            "interpretation: "
            + "; ".join(issues)
        )


class HypothesisContextInterpretationCompiler:
    """Validate LLM interpretation against immutable source facts.

    The compiler does not decide context compatibility.
    It only ensures that the interpretation is traceable,
    complete, and structurally consistent with the supplied
    source-context signatures.
    """

    def compile(
        self,
        *,
        card: HypothesisCard,
        source_signatures: list[
            SERSContextSignature
        ],
        draft: HypothesisContextInterpretationDraft,
    ) -> HypothesisContextInterpretation:
        issues: list[str] = []

        if (
            draft.hypothesis_id
            != card.hypothesis_id
        ):
            issues.append(
                "hypothesis_id mismatch"
            )

        expected_signatures = sorted({
            row.signature_id
            for row in source_signatures
        })

        if (
            sorted(
                draft.source_signature_ids
            )
            != expected_signatures
        ):
            issues.append(
                "source_signature_ids do not "
                "exactly match supplied signatures"
            )

        allowed_facts = {
            fact.fact_id: fact
            for signature
            in source_signatures
            for fact in signature.facts
        }

        expected = (
            expected_hypothesis_context_assertions(
                card
            )
        )

        expected_by_id = {
            row["assertion_id"]: row
            for row in expected
        }

        actual_by_id = {
            row.assertion_id: row
            for row in draft.assertions
        }

        if (
            len(actual_by_id)
            != len(draft.assertions)
        ):
            issues.append(
                "duplicate assertion_id"
            )

        if (
            set(actual_by_id)
            != set(expected_by_id)
        ):
            missing = sorted(
                set(expected_by_id)
                - set(actual_by_id)
            )

            extra = sorted(
                set(actual_by_id)
                - set(expected_by_id)
            )

            if missing:
                issues.append(
                    "missing assertions: "
                    + ", ".join(missing)
                )

            if extra:
                issues.append(
                    "unexpected assertions: "
                    + ", ".join(extra)
                )

        global_mention_ids: set[str] = set()

        for assertion_id, expected_row in (
            expected_by_id.items()
        ):
            actual = actual_by_id.get(
                assertion_id
            )

            if actual is None:
                continue

            if (
                actual.assertion_kind
                != expected_row[
                    "assertion_kind"
                ]
            ):
                issues.append(
                    f"{assertion_id}: "
                    "assertion_kind mismatch"
                )

            if (
                actual.assertion_text
                != expected_row[
                    "assertion_text"
                ]
            ):
                issues.append(
                    f"{assertion_id}: "
                    "assertion_text mismatch"
                )

            assertion_text_norm = (
                _compact_ws(
                    actual.assertion_text
                ).lower()
            )

            for mention in (
                actual.mentions
            ):
                if (
                    mention.mention_id
                    in global_mention_ids
                ):
                    issues.append(
                        "duplicate global mention_id: "
                        + mention.mention_id
                    )
                else:
                    global_mention_ids.add(
                        mention.mention_id
                    )

                mention_norm = (
                    _compact_ws(
                        mention.mention_text
                    ).lower()
                )

                if (
                    mention_norm
                    not in assertion_text_norm
                ):
                    issues.append(
                        f"{mention.mention_id}: "
                        "mention_text is not an exact "
                        "assertion span after whitespace "
                        "normalization"
                    )

                referenced = []

                for fact_id in (
                    mention.source_fact_ids
                ):
                    fact = allowed_facts.get(
                        fact_id
                    )

                    if fact is None:
                        issues.append(
                            f"{mention.mention_id}: "
                            f"unknown source fact "
                            f"{fact_id}"
                        )
                        continue

                    referenced.append(
                        fact
                    )

                same_dimension_treatments = {
                    "preserve",
                    "generalize",
                    "intentionally_vary",
                }

                if (
                    mention.treatment
                    in same_dimension_treatments
                    and referenced
                    and any(
                        fact.dimension
                        != mention.asserted_dimension
                        for fact in referenced
                    )
                ):
                    issues.append(
                        f"{mention.mention_id}: "
                        f"{mention.treatment} cannot "
                        "silently change context dimension"
                    )

                if (
                    mention.treatment
                    == "reattach"
                    and referenced
                    and not any(
                        fact.binding
                        is not None
                        for fact in referenced
                    )
                ):
                    issues.append(
                        f"{mention.mention_id}: "
                        "reattach requires a source "
                        "fact with attachment binding"
                    )

        if issues:
            raise (
                HypothesisContextInterpretationValidationError(
                    issues
                )
            )

        return HypothesisContextInterpretation(
            hypothesis_id=(
                card.hypothesis_id
            ),
            source_signature_ids=(
                expected_signatures
            ),
            assertions=[
                actual_by_id[
                    row["assertion_id"]
                ]
                for row in expected
            ],
        )
