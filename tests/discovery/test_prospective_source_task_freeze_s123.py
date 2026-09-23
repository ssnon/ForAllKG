from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.prospective_source_task_freeze import (
    ProspectiveSourceTaskCampaignFreeze,
    ProspectiveSourceTaskCampaignSpec,
    build_prospective_source_task_campaign_freeze,
)


def _spec() -> ProspectiveSourceTaskCampaignSpec:
    rows = [
        (
            "P06",
            "geometry_field_coupling",
            "interparticle gap",
            "electromagnetic enhancement",
            (
                "How does interparticle gap relate to electromagnetic "
                "enhancement in plasmonic SERS nanostructures?"
            ),
        ),
        (
            "P07",
            "size_resonance_coupling",
            "particle size",
            "plasmon resonance wavelength",
            (
                "How does particle size relate to plasmon resonance "
                "wavelength in SERS-active nanostructures?"
            ),
        ),
        (
            "P08",
            "surface_chemistry_selectivity",
            "surface functionalization",
            "analyte adsorption selectivity",
            (
                "How does surface functionalization relate to analyte "
                "adsorption selectivity in SERS measurements?"
            ),
        ),
        (
            "P09",
            "excitation_stability",
            "laser power",
            "spectral stability",
            (
                "How does laser power relate to spectral stability or "
                "degradation during SERS measurements?"
            ),
        ),
        (
            "P10",
            "concentration_response",
            "analyte concentration",
            "SERS signal linearity",
            (
                "How does analyte concentration relate to SERS signal "
                "linearity across the usable quantitative range?"
            ),
        ),
    ]
    return ProspectiveSourceTaskCampaignSpec(
        campaign_name="fresh_relational_verifier_prospective_p06_p10",
        campaign_root="/tmp/prospective",
        domain_profile_id="sers_au_ag",
        corpus_id="sers500_final_v2",
        data_root="/tmp/data",
        semantic_roots=["/tmp/corpus"],
        generation_model="openai/gpt-5.6-luna",
        critic_model="openai/gpt-5.6-luna",
        tasks=[
            {
                "case_id": case_id,
                "relation_family": family,
                "source": source,
                "target": target,
                "question": question,
            }
            for case_id, family, source, target, question in rows
        ],
    )


def test_freeze_binds_exactly_p06_p10_before_generation() -> None:
    freeze = build_prospective_source_task_campaign_freeze(
        spec=_spec(),
        source_spec_sha256="a" * 64,
        repository_head_sha="b" * 40,
        repository_tracked_worktree_dirty=False,
    )

    assert freeze.case_ids == ["P06", "P07", "P08", "P09", "P10"]
    assert freeze.task_count == 5
    assert freeze.source_tasks_frozen_before_generation is True
    assert freeze.hypotheses_generated_before_freeze is False
    assert freeze.scientific_reframing_generated_before_freeze is False
    assert freeze.prior_art_observed_before_freeze is False
    assert freeze.old_n10_observed_before_freeze is False
    assert freeze.new_verifier_observed_before_freeze is False
    assert freeze.post_freeze_task_replacement_allowed is False
    assert freeze.failed_or_abstained_case_replacement_allowed is False


def test_freeze_is_deterministic() -> None:
    kwargs = dict(
        spec=_spec(),
        source_spec_sha256="c" * 64,
        repository_head_sha="d" * 40,
        repository_tracked_worktree_dirty=False,
    )
    first = build_prospective_source_task_campaign_freeze(**kwargs)
    second = build_prospective_source_task_campaign_freeze(**kwargs)

    assert first.freeze_id == second.freeze_id
    assert first.freeze_sha256 == second.freeze_sha256
    assert [row.task_id for row in first.tasks] == [
        row.task_id for row in second.tasks
    ]


def test_freeze_rejects_tracked_worktree_mutation() -> None:
    with pytest.raises(
        ValueError,
        match="clean tracked worktree",
    ):
        build_prospective_source_task_campaign_freeze(
            spec=_spec(),
            source_spec_sha256="e" * 64,
            repository_head_sha="f" * 40,
            repository_tracked_worktree_dirty=True,
        )


def test_campaign_spec_rejects_case_reordering() -> None:
    payload = _spec().model_dump(mode="json")
    payload["tasks"][0], payload["tasks"][1] = (
        payload["tasks"][1],
        payload["tasks"][0],
    )

    with pytest.raises(
        ValidationError,
        match="ordered exactly P06-P10",
    ):
        ProspectiveSourceTaskCampaignSpec.model_validate(payload)


def test_campaign_freeze_hash_detects_mutation() -> None:
    freeze = build_prospective_source_task_campaign_freeze(
        spec=_spec(),
        source_spec_sha256="1" * 64,
        repository_head_sha="2" * 40,
        repository_tracked_worktree_dirty=False,
    )
    payload = freeze.model_dump(mode="json")
    payload["tasks"][0]["question"] = "mutated after freeze"

    with pytest.raises(
        ValidationError,
        match="prospective source task SHA mismatch",
    ):
        ProspectiveSourceTaskCampaignFreeze.model_validate(payload)


def test_campaign_level_hash_detects_campaign_metadata_mutation() -> None:
    freeze = build_prospective_source_task_campaign_freeze(
        spec=_spec(),
        source_spec_sha256="3" * 64,
        repository_head_sha="4" * 40,
        repository_tracked_worktree_dirty=False,
    )
    payload = freeze.model_dump(mode="json")
    payload["campaign_name"] = "mutated campaign metadata"

    with pytest.raises(
        ValidationError,
        match="prospective source campaign freeze SHA mismatch",
    ):
        ProspectiveSourceTaskCampaignFreeze.model_validate(payload)
