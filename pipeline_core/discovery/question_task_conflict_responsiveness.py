from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxis,
)
from pipeline_core.discovery.question_axis_responsiveness import (
    summarize_question_axis_two_pass,
)
from pipeline_core.discovery.question_axis_responsiveness_contracts import (
    QuestionAxisTwoPassStability,
)
from pipeline_core.discovery.question_axis_responsiveness_prompt import (
    QuestionAxisResponsivenessPromptAssembler,
)


class ResponsivenessBackendProtocol(Protocol):
    def review(
        self,
        prompt: Any,
        *,
        review_pass_index: int,
        debug_path: str | None = None,
    ) -> Any:
        ...


@dataclass(frozen=True)
class ConflictCandidateUnit:
    unit_id: str
    label: str
    proposed_subject: str
    proposed_relation: str
    proposed_object: str
    source_path_id: str = ""


@dataclass(frozen=True)
class ConflictCandidateResponsivenessResult:
    group: str
    unit_id: str
    label: str

    pass_1_status: str
    pass_2_status: str

    pass_1_role: str
    pass_2_role: str

    decision_stable: bool
    stable_status: str | None
    stable_role: str | None

    pass_1_rationale: str
    pass_2_rationale: str

    shadow_only: bool = True
    production_selection_changed: bool = False


def conflict_candidate_ids(
    raw_conflict_payload: Mapping[str, Any],
) -> tuple[str, ...]:
    """
    Return unique candidate-unit IDs in first-observation order.
    """

    rows = raw_conflict_payload.get(
        "observations",
        [],
    )

    if not isinstance(rows, list):
        raise ValueError(
            "raw conflict observations must be a list"
        )

    ordered: dict[str, None] = {}

    for row in rows:
        if not isinstance(row, dict):
            continue

        for key in (
            "incumbent_id",
            "challenger_id",
        ):
            value = str(
                row.get(
                    key,
                    "",
                )
            ).strip()

            if value.startswith(
                "candidate_unit:"
            ):
                ordered.setdefault(
                    value,
                    None,
                )

    return tuple(
        ordered.keys()
    )


def recover_candidate_units(
    *,
    traversal_payloads: Sequence[
        Mapping[str, Any]
    ],
    wanted_ids: Sequence[str],
) -> dict[
    str,
    ConflictCandidateUnit,
]:
    """
    Recover candidate S/R/O from traversal candidate_paths.

    If the same unit appears many times, its scientific identity must be
    consistent. Path duplication is expected; conflicting S/R/O is not.
    """

    wanted = set(
        wanted_ids
    )

    recovered: dict[
        str,
        ConflictCandidateUnit,
    ] = {}

    for payload in traversal_payloads:
        rows = payload.get(
            "candidate_paths",
            [],
        )

        if not isinstance(rows, list):
            continue

        for row in rows:
            if not isinstance(row, dict):
                continue

            unit = row.get(
                "candidate_unit"
            )

            if not isinstance(unit, dict):
                continue

            unit_id = str(
                unit.get(
                    "unit_id",
                    "",
                )
            ).strip()

            if (
                not unit_id
                or unit_id not in wanted
            ):
                continue

            candidate = ConflictCandidateUnit(
                unit_id=unit_id,
                label=str(
                    unit.get(
                        "label",
                        "",
                    )
                ).strip(),
                proposed_subject=str(
                    unit.get(
                        "proposed_subject",
                        "",
                    )
                ).strip(),
                proposed_relation=str(
                    unit.get(
                        "proposed_relation",
                        "",
                    )
                ).strip(),
                proposed_object=str(
                    unit.get(
                        "proposed_object",
                        "",
                    )
                ).strip(),
                source_path_id=str(
                    row.get(
                        "path_id",
                        "",
                    )
                ).strip(),
            )

            previous = recovered.get(
                unit_id
            )

            if previous is not None:
                previous_identity = (
                    previous.label,
                    previous.proposed_subject,
                    previous.proposed_relation,
                    previous.proposed_object,
                )

                candidate_identity = (
                    candidate.label,
                    candidate.proposed_subject,
                    candidate.proposed_relation,
                    candidate.proposed_object,
                )

                if (
                    previous_identity
                    != candidate_identity
                ):
                    raise ValueError(
                        "candidate unit identity mismatch "
                        f"for {unit_id}"
                    )

                continue

            recovered[
                unit_id
            ] = candidate

    return recovered


def candidate_unit_to_shadow_axis(
    candidate: ConflictCandidateUnit,
) -> DiscoveryAxis:
    """
    Represent one candidate-unit scientific relation as a shadow axis.

    Planner/exploration fields are neutral placeholders because this
    evaluator is judging only Question ↔ candidate relation preservation.
    """

    return DiscoveryAxis(
        axis_id=(
            "question-task-shadow:"
            + candidate.unit_id
        ),
        axis_rank=1,
        inspiration_id="",
        source_path_id=(
            candidate.source_path_id
        ),
        candidate_unit_id=(
            candidate.unit_id
        ),
        label=candidate.label,
        proposed_subject=(
            candidate.proposed_subject
        ),
        proposed_relation=(
            candidate.proposed_relation
        ),
        proposed_object=(
            candidate.proposed_object
        ),
        rendered_path=(
            candidate.label
        ),
        source_mode="shadow_candidate_unit",
        exploration_score=0.0,
        candidate_unit_score=0.0,
        planner_score=0.0,
        mechanistic_continuity_band="unknown",
        requires_verification=True,
        reason_codes=[
            "question_task_preservation_shadow",
            "candidate_unit_proxy_axis",
        ],
    )


def evaluate_conflict_candidates_shadow(
    *,
    group: str,
    question: str,
    raw_conflict_payload: Mapping[
        str,
        Any,
    ],
    traversal_payloads: Sequence[
        Mapping[str, Any]
    ],
    backend: ResponsivenessBackendProtocol,
    debug_path_prefix: str | None = None,
) -> tuple[
    ConflictCandidateResponsivenessResult,
    ...,
]:
    """
    Two-pass responsiveness review only for candidate units that actually
    participate in observed Bundle semantic conflicts.

    No Bundle selection is modified.
    """

    wanted_ids = conflict_candidate_ids(
        raw_conflict_payload
    )

    recovered = recover_candidate_units(
        traversal_payloads=(
            traversal_payloads
        ),
        wanted_ids=wanted_ids,
    )

    missing = [
        unit_id
        for unit_id in wanted_ids
        if unit_id not in recovered
    ]

    if missing:
        raise ValueError(
            "candidate units missing from traversal artifacts: "
            + ", ".join(missing)
        )

    assembler = (
        QuestionAxisResponsivenessPromptAssembler()
    )

    results = []

    for index, unit_id in enumerate(
        wanted_ids,
        start=1,
    ):
        candidate = recovered[
            unit_id
        ]

        axis = candidate_unit_to_shadow_axis(
            candidate
        )

        prompt = assembler.build(
            question=question,
            axis=axis,
        )

        debug_1 = (
            f"{debug_path_prefix}."
            f"{index:03d}.pass1.json"
            if debug_path_prefix
            else None
        )

        debug_2 = (
            f"{debug_path_prefix}."
            f"{index:03d}.pass2.json"
            if debug_path_prefix
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

        stability: QuestionAxisTwoPassStability = (
            summarize_question_axis_two_pass(
                draft_1,
                draft_2,
            )
        )

        results.append(
            ConflictCandidateResponsivenessResult(
                group=group,
                unit_id=unit_id,
                label=candidate.label,
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
                pass_1_rationale=(
                    draft_1.rationale
                ),
                pass_2_rationale=(
                    draft_2.rationale
                ),
            )
        )

    return tuple(
        results
    )


def conflict_responsiveness_artifact(
    *,
    group: str,
    question: str,
    results: Sequence[
        ConflictCandidateResponsivenessResult
    ],
    raw_conflict_source: str | None = None,
    traversal_sources: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "schema_version":
            "question-task-conflict-responsiveness-shadow-v1",

        "group":
            group,

        "question":
            question,

        "source_raw_conflict_artifact":
            raw_conflict_source,

        "source_traversal_artifacts":
            list(
                traversal_sources
            ),

        "candidate_count":
            len(results),

        # Keep this key compatible with the generic C7C artifact joiner.
        "candidate_results": [
            {
                "group":
                    row.group,

                "unit_id":
                    row.unit_id,

                "label":
                    row.label,

                "pass_1_status":
                    row.pass_1_status,

                "pass_2_status":
                    row.pass_2_status,

                "pass_1_role":
                    row.pass_1_role,

                "pass_2_role":
                    row.pass_2_role,

                "decision_stable":
                    row.decision_stable,

                "stable_status":
                    row.stable_status,

                "stable_role":
                    row.stable_role,

                "pass_1_rationale":
                    row.pass_1_rationale,

                "pass_2_rationale":
                    row.pass_2_rationale,
            }
            for row in results
        ],

        "shadow_only":
            True,

        "production_selection_changed":
            False,
    }
