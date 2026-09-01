from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxis,
    DiscoveryAxisPlannerPolicy,
)
from pipeline_core.discovery.discovery_axis_planner import (
    DiscoveryAxisPlanner,
)
from pipeline_core.discovery.dual_hypothesis_context import (
    DualHypothesisContext,
)
from pipeline_core.discovery.question_axis_responsiveness import (
    summarize_question_axis_two_pass,
)
from pipeline_core.discovery.question_axis_responsiveness_prompt import (
    QuestionAxisResponsivenessPromptAssembler,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResponsivenessBackendProtocol(Protocol):
    def review(
        self,
        prompt: Any,
        *,
        review_pass_index: int,
        debug_path: str | None = None,
    ) -> Any:
        ...


class TaskResponsiveAxisAssessment(StrictModel):
    axis_id: str
    axis_rank: int

    pass_1_status: str
    pass_2_status: str

    pass_1_role: str
    pass_2_role: str

    decision_stable: bool
    stable_status: str | None = None
    stable_role: str | None = None

    task_eligible: bool

    planner_score: float

    signature_tokens: list[str] = Field(
        default_factory=list
    )


class TaskResponsiveAxisSelection(StrictModel):
    axis_id: str
    source_axis_rank: int
    selected_rank: int

    stable_status: str
    stable_role: str

    planner_score: float
    min_diversity_distance: float

    reason_codes: list[str] = Field(
        default_factory=list
    )


class TaskResponsiveAxisShadowReport(StrictModel):
    schema_version: str = (
        "task-responsive-axis-shadow-v1"
    )

    report_id: str
    report_sha256: str

    source_dual_context_id: str
    source_dual_context_sha256: str

    question: str

    generic_pool_size_requested: int
    generic_pool_size_actual: int
    retained_axis_count_requested: int

    generic_axis_ids: list[str]
    assessments: list[
        TaskResponsiveAxisAssessment
    ]

    task_eligible_axis_count: int

    selected: list[
        TaskResponsiveAxisSelection
    ]

    selected_axis_ids: list[str]

    shadow_only: bool = True
    production_selection_changed: bool = False
    llm_review_passes_per_axis: int = 2


_TOKEN_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_+\-@]*"
)


def _sha256_json(
    payload: object,
) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        raw
    ).hexdigest()


def _stable_id(
    prefix: str,
    *parts: object,
) -> str:
    raw = "|".join(
        str(x)
        for x in parts
    ).encode("utf-8")

    return (
        prefix
        + ":"
        + hashlib.sha256(
            raw
        ).hexdigest()[:20]
    )


def axis_signature_tokens(
    axis: DiscoveryAxis,
) -> tuple[str, ...]:
    """
    Deterministic lexical signature used only to discourage
    near-duplicate S/R/O axes inside the already task-eligible set.

    This is not a scientific-quality score and does not inspect
    semantic-distinctiveness outcomes.
    """

    text = " ".join(
        [
            axis.proposed_subject,
            axis.proposed_relation,
            axis.proposed_object,
            axis.entry_anchor_label,
            axis.exit_anchor_label,
            axis.label,
        ]
    )

    tokens = {
        x.lower()
        for x in _TOKEN_RE.findall(
            text
        )
        if len(x) >= 2
    }

    return tuple(
        sorted(tokens)
    )


def _jaccard_distance(
    left: Sequence[str],
    right: Sequence[str],
) -> float:
    a = set(left)
    b = set(right)

    if not a and not b:
        return 0.0

    union = a | b

    if not union:
        return 0.0

    return 1.0 - (
        len(a & b)
        / len(union)
    )


def _role_rank(
    role: str | None,
) -> int:
    normalized = str(
        role or ""
    ).strip().upper()

    if normalized in {
        "DIRECT_ANSWER",
        "DIRECT",
    }:
        return 2

    if normalized in {
        "SUBORDINATE_EXTENSION",
        "SUBORDINATE",
    }:
        return 1

    return 0


def _status_rank(
    status: str | None,
) -> int:
    normalized = str(
        status or ""
    ).strip().upper()

    if normalized == "PASS":
        return 2

    if normalized == "WARNING":
        return 1

    return 0


def task_axis_is_eligible(
    assessment: (
        TaskResponsiveAxisAssessment
    ),
) -> bool:
    return (
        assessment.decision_stable
        and
        _role_rank(
            assessment.stable_role
        ) > 0
        and
        _status_rank(
            assessment.stable_status
        ) > 0
    )


@dataclass(frozen=True)
class _EligibleRow:
    axis: DiscoveryAxis
    assessment: (
        TaskResponsiveAxisAssessment
    )


def select_task_responsive_diverse_axes(
    *,
    axes: Sequence[DiscoveryAxis],
    assessments: Sequence[
        TaskResponsiveAxisAssessment
    ],
    retained_axes: int,
) -> tuple[
    TaskResponsiveAxisSelection,
    ...,
]:
    """
    Greedy deterministic selection inside the task-eligible set.

    Priority:
      1. DIRECT over SUBORDINATE
      2. PASS over WARNING
      3. greater lexical S/R/O diversity from already selected axes
      4. original planner score
      5. earlier frozen generic axis rank

    No semantic-distinctiveness score is consumed here.
    """

    if retained_axes < 1:
        raise ValueError(
            "retained_axes must be >= 1"
        )

    by_id = {
        row.axis_id: row
        for row in assessments
    }

    if len(by_id) != len(
        assessments
    ):
        raise ValueError(
            "duplicate axis assessment"
        )

    eligible: list[
        _EligibleRow
    ] = []

    for axis in axes:
        row = by_id.get(
            axis.axis_id
        )

        if row is None:
            raise ValueError(
                "missing responsiveness "
                f"assessment for {axis.axis_id}"
            )

        if task_axis_is_eligible(
            row
        ):
            eligible.append(
                _EligibleRow(
                    axis=axis,
                    assessment=row,
                )
            )

    selected: list[
        _EligibleRow
    ] = []

    out: list[
        TaskResponsiveAxisSelection
    ] = []

    remaining = list(
        eligible
    )

    while (
        remaining
        and
        len(selected) < retained_axes
    ):
        ranked = []

        for row in remaining:
            sig = (
                row.assessment
                .signature_tokens
            )

            if not selected:
                min_distance = 1.0
            else:
                min_distance = min(
                    _jaccard_distance(
                        sig,
                        selected_row
                        .assessment
                        .signature_tokens,
                    )
                    for selected_row
                    in selected
                )

            key = (
                _role_rank(
                    row.assessment
                    .stable_role
                ),
                _status_rank(
                    row.assessment
                    .stable_status
                ),
                min_distance,
                float(
                    row.axis.planner_score
                ),
                -int(
                    row.axis.axis_rank
                ),
            )

            ranked.append(
                (
                    key,
                    row,
                    min_distance,
                )
            )

        ranked.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        _key, winner, distance = (
            ranked[0]
        )

        selected.append(
            winner
        )

        remaining = [
            row
            for row in remaining
            if (
                row.axis.axis_id
                != winner.axis.axis_id
            )
        ]

        out.append(
            TaskResponsiveAxisSelection(
                axis_id=(
                    winner.axis.axis_id
                ),
                source_axis_rank=(
                    winner.axis.axis_rank
                ),
                selected_rank=len(
                    out
                ) + 1,
                stable_status=str(
                    winner.assessment
                    .stable_status
                ),
                stable_role=str(
                    winner.assessment
                    .stable_role
                ),
                planner_score=float(
                    winner.axis.planner_score
                ),
                min_diversity_distance=float(
                    distance
                ),
                reason_codes=[
                    "TASK_DECISION_STABLE",
                    "TASK_ROLE_ELIGIBLE",
                    "TASK_STATUS_ELIGIBLE",
                    (
                        "DIVERSITY_WITHIN_"
                        "TASK_ELIGIBLE_SET"
                    ),
                    (
                        "PLANNER_SCORE_"
                        "PRESERVED_AS_TIEBREAK"
                    ),
                ],
            )
        )

    return tuple(
        out
    )


def evaluate_task_responsive_axis_shadow(
    *,
    dual: DualHypothesisContext,
    question: str,
    backend: ResponsivenessBackendProtocol,
    generic_pool_size: int = 9,
    retained_axes: int = 3,
    debug_dir: str | Path | None = None,
) -> TaskResponsiveAxisShadowReport:
    """
    Shadow-only N8 selector.

    1. Reuse the existing generic DiscoveryAxisPlanner to form a
       bounded top-K candidate pool.
    2. Two-pass review each candidate axis against the user question.
    3. Retain only stable DIRECT/SUBORDINATE PASS/WARNING axes.
    4. Select up to retained_axes with deterministic diversity.

    Production selection is never modified.
    """

    question = str(
        question
    ).strip()

    if not question:
        raise ValueError(
            "question must be non-empty"
        )

    if generic_pool_size < 1:
        raise ValueError(
            "generic_pool_size must be >= 1"
        )

    if retained_axes < 1:
        raise ValueError(
            "retained_axes must be >= 1"
        )

    planner = DiscoveryAxisPlanner(
        DiscoveryAxisPlannerPolicy(
            max_axes=generic_pool_size
        )
    )

    generic_plan = planner.build(
        dual
    )

    assembler = (
        QuestionAxisResponsivenessPromptAssembler()
    )

    debug_root = (
        Path(debug_dir)
        if debug_dir is not None
        else None
    )

    if debug_root is not None:
        debug_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    assessments = []

    for index, axis in enumerate(
        generic_plan.axes,
        start=1,
    ):
        prompt = assembler.build(
            question=question,
            axis=axis,
        )

        debug_1 = (
            str(
                debug_root
                / (
                    f"axis_{index:02d}"
                    ".pass_1.json"
                )
            )
            if debug_root is not None
            else None
        )

        debug_2 = (
            str(
                debug_root
                / (
                    f"axis_{index:02d}"
                    ".pass_2.json"
                )
            )
            if debug_root is not None
            else None
        )

        generation_1 = backend.review(
            prompt,
            review_pass_index=1,
            debug_path=debug_1,
        )

        generation_2 = backend.review(
            prompt,
            review_pass_index=2,
            debug_path=debug_2,
        )

        draft_1 = generation_1.draft
        draft_2 = generation_2.draft

        stability = (
            summarize_question_axis_two_pass(
                draft_1,
                draft_2,
            )
        )

        preliminary = (
            TaskResponsiveAxisAssessment(
                axis_id=axis.axis_id,
                axis_rank=axis.axis_rank,
                pass_1_status=(
                    draft_1.overall_status
                ),
                pass_2_status=(
                    draft_2.overall_status
                ),
                pass_1_role=(
                    draft_1.axis_role
                ),
                pass_2_role=(
                    draft_2.axis_role
                ),
                decision_stable=(
                    stability.decision_stable
                ),
                stable_status=(
                    stability.stable_status
                ),
                stable_role=(
                    stability.stable_role
                ),
                task_eligible=False,
                planner_score=float(
                    axis.planner_score
                ),
                signature_tokens=list(
                    axis_signature_tokens(
                        axis
                    )
                ),
            )
        )

        assessments.append(
            preliminary.model_copy(
                update={
                    "task_eligible":
                        task_axis_is_eligible(
                            preliminary
                        )
                }
            )
        )

    selected = (
        select_task_responsive_diverse_axes(
            axes=generic_plan.axes,
            assessments=assessments,
            retained_axes=retained_axes,
        )
    )

    report_id = _stable_id(
        "task_responsive_axis_shadow",
        dual.dual_context_sha256,
        question,
        generic_pool_size,
        retained_axes,
        *[
            row.axis_id
            for row in selected
        ],
    )

    payload = {
        "schema_version":
            "task-responsive-axis-shadow-v1",

        "report_id":
            report_id,

        "source_dual_context_id":
            dual.dual_context_id,

        "source_dual_context_sha256":
            dual.dual_context_sha256,

        "question":
            question,

        "generic_pool_size_requested":
            generic_pool_size,

        "generic_pool_size_actual":
            len(
                generic_plan.axes
            ),

        "retained_axis_count_requested":
            retained_axes,

        "generic_axis_ids":
            [
                axis.axis_id
                for axis in generic_plan.axes
            ],

        "assessments":
            [
                row.model_dump(
                    mode="json"
                )
                for row in assessments
            ],

        "task_eligible_axis_count":
            sum(
                1
                for row in assessments
                if row.task_eligible
            ),

        "selected":
            [
                row.model_dump(
                    mode="json"
                )
                for row in selected
            ],

        "selected_axis_ids":
            [
                row.axis_id
                for row in selected
            ],

        "shadow_only":
            True,

        "production_selection_changed":
            False,

        "llm_review_passes_per_axis":
            2,
    }

    return (
        TaskResponsiveAxisShadowReport(
            **payload,
            report_sha256=(
                _sha256_json(
                    payload
                )
            ),
        )
    )
