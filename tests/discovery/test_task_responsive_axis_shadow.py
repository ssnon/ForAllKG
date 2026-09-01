from __future__ import annotations

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxis,
)
from pipeline_core.discovery.task_responsive_axis_shadow import (
    TaskResponsiveAxisAssessment,
    axis_signature_tokens,
    select_task_responsive_diverse_axes,
    task_axis_is_eligible,
)


def _axis(
    *,
    axis_id: str,
    rank: int,
    subject: str,
    relation: str,
    object_: str,
    score: float,
) -> DiscoveryAxis:
    return DiscoveryAxis(
        axis_id=axis_id,
        axis_rank=rank,
        inspiration_id=f"inspiration:{axis_id}",
        source_path_id=f"path:{axis_id}",
        candidate_unit_id=f"unit:{axis_id}",
        label=(
            f"{subject} {relation} {object_}"
        ),
        entry_anchor_id="entry",
        entry_anchor_label=subject,
        exit_anchor_id="exit",
        exit_anchor_label=object_,
        proposed_subject=subject,
        proposed_relation=relation,
        proposed_object=object_,
        rendered_path=(
            f"{subject} -> "
            f"{relation} -> "
            f"{object_}"
        ),
        source_mode="test",
        exploration_score=0.5,
        candidate_unit_score=0.5,
        planner_score=score,
        mechanistic_continuity_band="high",
        generic_entity_fraction=0.0,
        registry_hop_fraction=0.0,
        grounding_semantic_overlap=0.0,
        reaction_domain_switch_penalty=0.0,
        requires_verification=True,
        reason_codes=[],
    )


def _assessment(
    axis: DiscoveryAxis,
    *,
    status: str,
    role: str,
    stable: bool = True,
) -> TaskResponsiveAxisAssessment:
    provisional = TaskResponsiveAxisAssessment(
        axis_id=axis.axis_id,
        axis_rank=axis.axis_rank,
        pass_1_status=status,
        pass_2_status=status,
        pass_1_role=role,
        pass_2_role=role,
        decision_stable=stable,
        stable_status=(
            status
            if stable
            else None
        ),
        stable_role=(
            role
            if stable
            else None
        ),
        task_eligible=False,
        planner_score=axis.planner_score,
        signature_tokens=list(
            axis_signature_tokens(axis)
        ),
    )

    return provisional.model_copy(
        update={
            "task_eligible":
                task_axis_is_eligible(
                    provisional
                )
        }
    )


def test_task_eligibility_requires_stable_direct_or_subordinate():
    direct = _axis(
        axis_id="a",
        rank=1,
        subject="binding sites",
        relation="AFFECTS",
        object_="SERS response",
        score=0.8,
    )

    replacement = _axis(
        axis_id="b",
        rank=2,
        subject="laser power",
        relation="VARIES_WITH",
        object_="SERS spectra",
        score=0.9,
    )

    assert task_axis_is_eligible(
        _assessment(
            direct,
            status="PASS",
            role="DIRECT_ANSWER",
        )
    )

    assert not task_axis_is_eligible(
        _assessment(
            replacement,
            status="PASS",
            role="TASK_REPLACEMENT",
        )
    )

    assert not task_axis_is_eligible(
        _assessment(
            direct,
            status="PASS",
            role="DIRECT_ANSWER",
            stable=False,
        )
    )


def test_direct_answer_is_prioritized_over_subordinate():
    subordinate = _axis(
        axis_id="sub",
        rank=1,
        subject="binding sites",
        relation="MEDIATES",
        object_="adsorption",
        score=0.95,
    )

    direct = _axis(
        axis_id="direct",
        rank=2,
        subject="binding sites",
        relation="AFFECTS",
        object_="SERS response",
        score=0.70,
    )

    selected = (
        select_task_responsive_diverse_axes(
            axes=[
                subordinate,
                direct,
            ],
            assessments=[
                _assessment(
                    subordinate,
                    status="PASS",
                    role="SUBORDINATE_EXTENSION",
                ),
                _assessment(
                    direct,
                    status="PASS",
                    role="DIRECT_ANSWER",
                ),
            ],
            retained_axes=1,
        )
    )

    assert [
        row.axis_id
        for row in selected
    ] == ["direct"]


def test_diversity_breaks_same_role_status_before_planner_score():
    first = _axis(
        axis_id="first",
        rank=1,
        subject="binding sites",
        relation="AFFECTS",
        object_="SERS response",
        score=0.90,
    )

    near_duplicate = _axis(
        axis_id="near",
        rank=2,
        subject="binding sites",
        relation="AFFECTS",
        object_="SERS intensity",
        score=0.89,
    )

    diverse = _axis(
        axis_id="diverse",
        rank=3,
        subject="binding sites",
        relation="MODULATES",
        object_="charge transfer pathway",
        score=0.75,
    )

    axes = [
        first,
        near_duplicate,
        diverse,
    ]

    assessments = [
        _assessment(
            axis,
            status="PASS",
            role="DIRECT_ANSWER",
        )
        for axis in axes
    ]

    selected = (
        select_task_responsive_diverse_axes(
            axes=axes,
            assessments=assessments,
            retained_axes=2,
        )
    )

    assert selected[0].axis_id == "first"
    assert selected[1].axis_id == "diverse"


def test_ineligible_axis_never_selected_even_with_higher_score():
    off_task = _axis(
        axis_id="off",
        rank=1,
        subject="laser power",
        relation="VARIES_WITH",
        object_="spectral intensity",
        score=0.99,
    )

    direct = _axis(
        axis_id="direct",
        rank=2,
        subject="binding sites",
        relation="AFFECTS",
        object_="SERS behavior",
        score=0.60,
    )

    selected = (
        select_task_responsive_diverse_axes(
            axes=[
                off_task,
                direct,
            ],
            assessments=[
                _assessment(
                    off_task,
                    status="FAIL",
                    role="TASK_REPLACEMENT",
                ),
                _assessment(
                    direct,
                    status="PASS",
                    role="DIRECT_ANSWER",
                ),
            ],
            retained_axes=3,
        )
    )

    assert [
        row.axis_id
        for row in selected
    ] == ["direct"]


def test_underfill_is_allowed_when_task_eligible_supply_is_small():
    a = _axis(
        axis_id="a",
        rank=1,
        subject="binding sites",
        relation="AFFECTS",
        object_="SERS",
        score=0.8,
    )

    b = _axis(
        axis_id="b",
        rank=2,
        subject="laser",
        relation="VARIES_WITH",
        object_="power",
        score=0.9,
    )

    selected = (
        select_task_responsive_diverse_axes(
            axes=[a, b],
            assessments=[
                _assessment(
                    a,
                    status="PASS",
                    role="DIRECT_ANSWER",
                ),
                _assessment(
                    b,
                    status="FAIL",
                    role="TASK_REPLACEMENT",
                ),
            ],
            retained_axes=3,
        )
    )

    assert len(selected) == 1
    assert selected[0].axis_id == "a"
