from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Literal, Mapping, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from pipeline_core.discovery.nonobviousness_mechanism_semantic_prompt import (
    MechanismSemanticPrompt,
    MechanismSemanticPromptAssembler,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


def _stable_id(
    prefix: str,
    payload: object,
) -> str:
    body = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

    digest = hashlib.sha256(
        body.encode("utf-8")
    ).hexdigest()[:20]

    return f"{prefix}:{digest}"


def _unique(
    values: Sequence[str],
) -> list[str]:
    out = []
    seen = set()

    for value in values:
        value = str(value).strip()

        if (
            not value
            or value in seen
        ):
            continue

        seen.add(value)
        out.append(value)

    return out


class N11AlternateMechanismSemanticCase(
    StrictModel
):
    case_id: str

    source_claim_node_id: str

    source_supply_candidate_ids: list[str] = Field(
        default_factory=list
    )

    source_factor_node_ids: list[str] = Field(
        default_factory=list
    )

    source_paper_ids: list[str] = Field(
        default_factory=list
    )

    factor_local_text_segments: list[str] = Field(
        min_length=1
    )

    factor_local_scope_features: list[str] = Field(
        default_factory=list
    )

    whole_claim_text: str

    supply_geometry: Literal[
        "DIRECT_SCIENTIFIC_CHAIN"
    ] = "DIRECT_SCIENTIFIC_CHAIN"

    semantic_distinctness_assessed: Literal[
        False
    ] = False

    eligible_for_semantic_review: Literal[
        True
    ] = True

    eligible_as_positive_hypothesis_premise: Literal[
        False
    ] = False

    production_authority: Literal[
        False
    ] = False


class N11AlternateMechanismSemanticCohort(
    StrictModel
):
    schema_version: Literal[
        "n11-alternate-mechanism-semantic-cohort-v1"
    ] = (
        "n11-alternate-mechanism-semantic-cohort-v1"
    )

    status: Literal[
        "FOUND_SEMANTIC_REVIEW_COHORT",
        "ABSTAIN_NO_REVIEWABLE_FACTOR_LOCAL_CLAIMS",
        "NOT_ELIGIBLE_FROM_D3A",
    ]

    reviewed_supply_candidate_count: int = 0

    unique_claim_count: int = 0

    case_count: int = 0

    cases: list[
        N11AlternateMechanismSemanticCase
    ] = Field(
        default_factory=list
    )

    rejection_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict
    )

    production_authority: Literal[
        False
    ] = False


def build_alternate_mechanism_semantic_cohort(
    supply_result: Any,
) -> N11AlternateMechanismSemanticCohort:
    status = str(
        getattr(
            supply_result,
            "status",
            "",
        )
    )

    candidates = list(
        getattr(
            supply_result,
            "candidates",
            [],
        )
        or []
    )

    if status != (
        "FOUND_FACTOR_GROUNDED_MECHANISM_SUPPLY"
    ):
        return (
            N11AlternateMechanismSemanticCohort(
                status="NOT_ELIGIBLE_FROM_D3A",
                reviewed_supply_candidate_count=(
                    len(candidates)
                ),
            )
        )

    grouped = defaultdict(list)

    for candidate in candidates:
        claim_node_id = str(
            getattr(
                candidate,
                "claim_node_id",
                "",
            )
        ).strip()

        if not claim_node_id:
            continue

        grouped[
            claim_node_id
        ].append(
            candidate
        )

    cases = []

    rejection_counts = defaultdict(int)

    for claim_node_id in sorted(grouped):
        rows = grouped[
            claim_node_id
        ]

        segments = _unique(
            [
                segment
                for row in rows
                for segment in list(
                    getattr(
                        row,
                        "factor_local_text_segments",
                        [],
                    )
                    or []
                )
            ]
        )

        if not segments:
            rejection_counts[
                "missing_factor_local_text"
            ] += 1
            continue

        whole_claim_texts = _unique(
            [
                str(
                    getattr(
                        row,
                        "claim_text",
                        "",
                    )
                )
                for row in rows
            ]
        )

        whole_claim_text = (
            whole_claim_texts[0]
            if whole_claim_texts
            else ""
        )

        case_payload = {
            "claim_node_id":
                claim_node_id,

            "factor_node_ids":
                sorted(
                    {
                        str(
                            getattr(
                                row,
                                "factor_node_id",
                                "",
                            )
                        )
                        for row in rows
                        if str(
                            getattr(
                                row,
                                "factor_node_id",
                                "",
                            )
                        ).strip()
                    }
                ),

            "factor_local_text_segments":
                segments,
        }

        cases.append(
            N11AlternateMechanismSemanticCase(
                case_id=_stable_id(
                    "n11_alt_mech_semantic_case",
                    case_payload,
                ),

                source_claim_node_id=
                    claim_node_id,

                source_supply_candidate_ids=
                    sorted(
                        {
                            str(
                                getattr(
                                    row,
                                    "supply_candidate_id",
                                    "",
                                )
                            )
                            for row in rows
                            if str(
                                getattr(
                                    row,
                                    "supply_candidate_id",
                                    "",
                                )
                            ).strip()
                        }
                    ),

                source_factor_node_ids=
                    case_payload[
                        "factor_node_ids"
                    ],

                source_paper_ids=
                    sorted(
                        {
                            str(paper_id)
                            for row in rows
                            for paper_id in list(
                                getattr(
                                    row,
                                    "source_paper_ids",
                                    [],
                                )
                                or []
                            )
                            if str(
                                paper_id
                            ).strip()
                        }
                    ),

                factor_local_text_segments=
                    segments,

                factor_local_scope_features=
                    sorted(
                        {
                            str(scope)
                            for row in rows
                            for scope in list(
                                getattr(
                                    row,
                                    "mechanism_scope_features",
                                    [],
                                )
                                or []
                            )
                            if str(
                                scope
                            ).strip()
                        }
                    ),

                whole_claim_text=
                    whole_claim_text,
            )
        )

    if not cases:
        return (
            N11AlternateMechanismSemanticCohort(
                status=(
                    "ABSTAIN_NO_REVIEWABLE_"
                    "FACTOR_LOCAL_CLAIMS"
                ),
                reviewed_supply_candidate_count=(
                    len(candidates)
                ),
                unique_claim_count=(
                    len(grouped)
                ),
                rejection_counts=dict(
                    rejection_counts
                ),
            )
        )

    return (
        N11AlternateMechanismSemanticCohort(
            status=(
                "FOUND_SEMANTIC_REVIEW_COHORT"
            ),
            reviewed_supply_candidate_count=(
                len(candidates)
            ),
            unique_claim_count=(
                len(grouped)
            ),
            case_count=len(cases),
            cases=cases,
            rejection_counts=dict(
                rejection_counts
            ),
        )
    )


def build_semantic_prompt_for_case(
    *,
    case:
        N11AlternateMechanismSemanticCase,
    scientific_task: str,
    canonical_task_feature: str,
    baseline_mechanism_statements:
        Sequence[
            Mapping[str, Any]
        ],
    factor_node_text_by_id:
        Mapping[str, str],
    assembler:
        MechanismSemanticPromptAssembler
        | None = None,
) -> MechanismSemanticPrompt:
    """
    Build a B1 semantic-review prompt from factor-local evidence.

    APPLIES_TO is navigation/provenance attachment only.
    It is deliberately NOT represented as a scientific step.

    The scientific step is the exact factor-local clause extracted
    from the grounded MechanismClaim.
    """

    assembler = (
        assembler
        or MechanismSemanticPromptAssembler()
    )

    factor_nodes = []

    for node_id in (
        case.source_factor_node_ids
    ):
        factor_nodes.append(
            {
                "node_id":
                    node_id,

                "node_text":
                    str(
                        factor_node_text_by_id.get(
                            node_id,
                            "",
                        )
                    ),
            }
        )

    task_feature = {
        "canonical_task_feature":
            str(
                canonical_task_feature
            ).strip(),

        "grounded_factor_nodes":
            factor_nodes,

        "epistemic_note": (
            "Factor-family attachment alone does not establish "
            "that interparticle spacing controls every mechanism "
            "reported in the source claim."
        ),
    }

    supplemental_text = "\n".join(
        case.factor_local_text_segments
    )

    supplemental_nodes = [
        {
            "node_id":
                case.source_claim_node_id,

            "label":
                "Factor-local excerpt from grounded MechanismClaim",

            "node_text":
                supplemental_text,

            "source_paper_id":
                (
                    case.source_paper_ids[0]
                    if case.source_paper_ids
                    else None
                ),
        }
    ]

    scientific_steps = [
        {
            "step_type":
                "FACTOR_LOCAL_CLAIM_RELATION",

            "relation_text":
                segment,

            "source_claim_node_id":
                case.source_claim_node_id,

            "source_factor_node_ids":
                list(
                    case.source_factor_node_ids
                ),

            "source_paper_ids":
                list(
                    case.source_paper_ids
                ),

            "evidence_basis":
                (
                    "exact_factor_local_clause_from_"
                    "grounded_mechanism_claim"
                ),

            "applies_to_edge_used_as_scientific_relation":
                False,
        }
        for segment in (
            case.factor_local_text_segments
        )
    ]

    return assembler.build(
        hypothesis_id=
            case.case_id,

        scientific_task=
            scientific_task,

        supply_geometry=
            case.supply_geometry,

        baseline_mechanism_statements=
            baseline_mechanism_statements,

        task_feature=
            task_feature,

        supplemental_mechanism_nodes=
            supplemental_nodes,

        scientific_steps=
            scientific_steps,
    )
