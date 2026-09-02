from __future__ import annotations

import hashlib
from typing import Literal, Mapping, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from pipeline_core.discovery.hypothesis_contracts import (
    ExpectedDirection,
    HypothesisContext,
    HypothesisType,
)
from pipeline_core.discovery.nonobviousness_mechanism_semantics_contracts import (
    MechanismSearchOperator,
)
from pipeline_core.discovery.nonobviousness_operator_generation_contracts import (
    N11OperatorGenerationDraft,
)
from pipeline_core.discovery.nonobviousness_operator_generation_validation import (
    N11OperatorGenerationValidation,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


class N11CanonicalSupplementalEvidence(
    StrictModel
):
    node_id: str = Field(
        min_length=1
    )

    label: str | None

    node_text: str = Field(
        min_length=1
    )

    source_paper_id: str | None

    epistemic_usage: Literal[
        "grounded_component_not_positive_premise"
    ] = (
        "grounded_component_not_positive_premise"
    )

    eligible_as_positive_premise: Literal[
        False
    ] = False


class N11CanonicalPrediction(
    StrictModel
):
    prediction_id: str
    source_local_id: str

    observable: str
    expected_direction: ExpectedDirection
    rationale: str

    discriminating: bool


class N11CanonicalFalsifier(
    StrictModel
):
    falsifier_id: str
    source_local_id: str

    prediction_id: str

    falsifying_outcome: str


class N11CanonicalCandidate(
    StrictModel
):
    schema_version: Literal[
        "n11-canonical-candidate-v1"
    ] = "n11-canonical-candidate-v1"

    candidate_id: str

    source_hypothesis_id: str

    source_context_id: str
    source_context_sha256: str

    source_report_id: str
    source_report_sha256: str

    domain_profile_id: str
    task_id: str
    question: str

    operator: MechanismSearchOperator

    title: str
    hypothesis_statement: str
    hypothesis_type: HypothesisType

    # Positive evidence lane only.
    baseline_premise_statement_ids: list[str]

    # Supplemental grounded component lane only.
    supplemental_evidence: list[
        N11CanonicalSupplementalEvidence
    ]

    # Unresolved lane only.
    gap_statement_ids: list[str]

    shared_component_ids: list[str]

    supplemental_only_component_ids: list[str]

    relative_contribution_claim: str

    inferential_bridge: str

    predictions: list[
        N11CanonicalPrediction
    ]

    discriminating_prediction_id: str

    falsifiers: list[
        N11CanonicalFalsifier
    ]

    assumptions: list[str]

    baseline_source_paper_ids: list[str]
    supplemental_source_paper_ids: list[str]
    gap_source_paper_ids: list[str]

    cross_source_grounding: bool

    generated_relation_status: Literal[
        "INFERENCE_NOT_REPORTED"
    ] = "INFERENCE_NOT_REPORTED"

    task_to_supplemental_relation_grounded: Literal[
        False
    ] = False

    supplemental_promoted_to_positive_premise: Literal[
        False
    ] = False

    gap_promoted_to_positive_premise: Literal[
        False
    ] = False

    operator_authority_source: Literal[
        "deterministic_n11_mechanism_operator_policy"
    ] = (
        "deterministic_n11_mechanism_operator_policy"
    )

    generation_validation_passed: Literal[
        True
    ] = True

    novelty_status: Literal[
        "NOT_ASSESSED"
    ] = "NOT_ASSESSED"

    n10_status: Literal[
        "NOT_ASSESSED"
    ] = "NOT_ASSESSED"

    production_authority: Literal[
        False
    ] = False


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(
        str(part)
        for part in parts
    ).encode(
        "utf-8"
    )

    return (
        f"{prefix}:"
        f"{hashlib.sha256(raw).hexdigest()[:length]}"
    )


def _sorted_unique(
    values: Sequence[str],
) -> list[str]:
    return sorted(
        {
            str(value)
            for value in values
            if str(value).strip()
        }
    )


class N11CanonicalCandidateCompiler:
    def compile(
        self,
        *,
        context: HypothesisContext,
        source_hypothesis_id: str,
        draft: N11OperatorGenerationDraft,
        validation: N11OperatorGenerationValidation,
        supplemental_nodes: Sequence[
            Mapping[str, object]
        ],
    ) -> N11CanonicalCandidate:
        if not validation.passes:
            raise ValueError(
                "N11 generation validation must pass "
                "before canonical compilation"
            )

        if validation.issues:
            raise ValueError(
                "passing N11 validation must contain no issues"
            )

        candidate = draft.candidate

        if candidate is None:
            raise ValueError(
                "cannot compile an abstention as "
                "a canonical N11 candidate"
            )

        statement_index = {
            row.statement_id: row
            for row
            in context.evidence_statements
        }

        baseline_ids = _sorted_unique(
            candidate
            .baseline_premise_statement_ids
        )

        gap_ids = _sorted_unique(
            candidate
            .gap_statement_ids
        )

        supplemental_ids = _sorted_unique(
            candidate
            .supplemental_mechanism_node_ids
        )

        # --------------------------------------------------------
        # Positive baseline lane
        # --------------------------------------------------------

        baseline_statements = []

        for statement_id in baseline_ids:
            row = statement_index.get(
                statement_id
            )

            if row is None:
                raise ValueError(
                    "unknown baseline premise: "
                    + statement_id
                )

            if not row.eligible_as_premise:
                raise ValueError(
                    "baseline statement is not "
                    "premise eligible: "
                    + statement_id
                )

            if row.epistemic_role not in {
                "reported",
                "evidence_synthesis",
            }:
                raise ValueError(
                    "unsafe baseline epistemic role: "
                    + statement_id
                )

            if row.alignment_path_ids:
                raise ValueError(
                    "alignment-dependent statement cannot "
                    "be a canonical N11 positive premise: "
                    + statement_id
                )

            baseline_statements.append(
                row
            )

        # --------------------------------------------------------
        # Unresolved gap lane
        # --------------------------------------------------------

        gap_statements = []

        for statement_id in gap_ids:
            row = statement_index.get(
                statement_id
            )

            if row is None:
                raise ValueError(
                    "unknown gap statement: "
                    + statement_id
                )

            if not row.eligible_as_gap:
                raise ValueError(
                    "statement is not gap eligible: "
                    + statement_id
                )

            if (
                row.epistemic_role
                != "unresolved"
            ):
                raise ValueError(
                    "gap statement must remain unresolved: "
                    + statement_id
                )

            gap_statements.append(
                row
            )

        # --------------------------------------------------------
        # Supplemental grounded-component lane
        # --------------------------------------------------------

        supplemental_index = {}

        for raw in supplemental_nodes:
            node_id = str(
                raw.get(
                    "node_id",
                    "",
                )
            ).strip()

            if not node_id:
                raise ValueError(
                    "supplemental node missing node_id"
                )

            if node_id in supplemental_index:
                raise ValueError(
                    "duplicate supplemental node_id: "
                    + node_id
                )

            supplemental_index[
                node_id
            ] = raw

        if (
            set(supplemental_ids)
            != set(supplemental_index)
        ):
            raise ValueError(
                "canonical supplemental node set must "
                "exactly match generated candidate"
            )

        supplemental_records = []

        for node_id in supplemental_ids:
            raw = supplemental_index[
                node_id
            ]

            node_text = str(
                raw.get(
                    "node_text",
                    "",
                )
            ).strip()

            if not node_text:
                raise ValueError(
                    "supplemental node missing scientific text: "
                    + node_id
                )

            label_raw = raw.get(
                "label"
            )

            paper_raw = raw.get(
                "source_paper_id"
            )

            supplemental_records.append(
                N11CanonicalSupplementalEvidence(
                    node_id=node_id,
                    label=(
                        str(label_raw)
                        if label_raw is not None
                        else None
                    ),
                    node_text=node_text,
                    source_paper_id=(
                        str(paper_raw)
                        if paper_raw is not None
                        else None
                    ),
                )
            )

        # --------------------------------------------------------
        # Stable canonical candidate identity
        # --------------------------------------------------------

        candidate_id = _stable_id(
            "n11_candidate",
            context.domain_profile_id,
            context.context_sha256,
            source_hypothesis_id,
            candidate.operator,
            candidate.local_id,
            candidate.hypothesis_statement,
            ",".join(
                baseline_ids
            ),
            ",".join(
                supplemental_ids
            ),
            ",".join(
                gap_ids
            ),
        )

        # --------------------------------------------------------
        # Predictions / falsifiers
        # --------------------------------------------------------

        predictions = []

        prediction_id_by_local = {}

        for row in (
            candidate.predicted_observations
        ):
            prediction_id = _stable_id(
                "n11_prediction",
                candidate_id,
                row.local_id,
                row.observable,
                row.expected_direction,
            )

            prediction_id_by_local[
                row.local_id
            ] = prediction_id

            predictions.append(
                N11CanonicalPrediction(
                    prediction_id=
                        prediction_id,

                    source_local_id=
                        row.local_id,

                    observable=
                        row.observable,

                    expected_direction=
                        row.expected_direction,

                    rationale=
                        row.rationale,

                    discriminating=(
                        row.local_id
                        == candidate
                        .discriminating_observation_local_id
                    ),
                )
            )

        discriminating_prediction_id = (
            prediction_id_by_local[
                candidate
                .discriminating_observation_local_id
            ]
        )

        falsifiers = []

        for row in (
            candidate.falsification_criteria
        ):
            prediction_id = (
                prediction_id_by_local.get(
                    row.prediction_local_id
                )
            )

            if prediction_id is None:
                raise ValueError(
                    "falsifier references unknown prediction: "
                    + row.prediction_local_id
                )

            falsifiers.append(
                N11CanonicalFalsifier(
                    falsifier_id=_stable_id(
                        "n11_falsifier",
                        candidate_id,
                        row.local_id,
                        prediction_id,
                        row.falsifying_outcome,
                    ),
                    source_local_id=
                        row.local_id,
                    prediction_id=
                        prediction_id,
                    falsifying_outcome=
                        row.falsifying_outcome,
                )
            )

        baseline_papers = _sorted_unique(
            [
                paper_id
                for row in baseline_statements
                for paper_id in row.paper_ids
            ]
        )

        gap_papers = _sorted_unique(
            [
                paper_id
                for row in gap_statements
                for paper_id in row.paper_ids
            ]
        )

        supplemental_papers = _sorted_unique(
            [
                row.source_paper_id
                for row in supplemental_records
                if row.source_paper_id
                is not None
            ]
        )

        grounded_papers = set(
            baseline_papers
        ) | set(
            supplemental_papers
        )

        return N11CanonicalCandidate(
            candidate_id=
                candidate_id,

            source_hypothesis_id=
                str(
                    source_hypothesis_id
                ),

            source_context_id=
                context.context_id,

            source_context_sha256=
                context.context_sha256,

            source_report_id=
                context.source_report_id,

            source_report_sha256=
                context.source_report_sha256,

            domain_profile_id=
                context.domain_profile_id,

            task_id=
                context.task_id,

            question=
                context.question,

            operator=
                candidate.operator,

            title=
                candidate.title,

            hypothesis_statement=
                candidate.hypothesis_statement,

            hypothesis_type=
                candidate.hypothesis_type,

            baseline_premise_statement_ids=
                baseline_ids,

            supplemental_evidence=
                supplemental_records,

            gap_statement_ids=
                gap_ids,

            shared_component_ids=_sorted_unique(
                candidate
                .shared_component_ids
            ),

            supplemental_only_component_ids=
                _sorted_unique(
                    candidate
                    .supplemental_only_component_ids
                ),

            relative_contribution_claim=
                candidate
                .relative_contribution_claim,

            inferential_bridge=
                candidate
                .inferential_bridge,

            predictions=
                predictions,

            discriminating_prediction_id=
                discriminating_prediction_id,

            falsifiers=
                falsifiers,

            assumptions=list(
                candidate.assumptions
            ),

            baseline_source_paper_ids=
                baseline_papers,

            supplemental_source_paper_ids=
                supplemental_papers,

            gap_source_paper_ids=
                gap_papers,

            cross_source_grounding=(
                len(
                    grounded_papers
                )
                >= 2
            ),
        )
