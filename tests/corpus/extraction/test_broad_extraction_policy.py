from __future__ import annotations

from pathlib import Path

from domains.catalysis_mechanism.extraction_policy import (
    BROAD_ABSTRACT_RECOVERY_POLICY_ID,
    broad_abstract_extraction_policy,
)
from pipeline_core.corpus.extraction.extraction_policy import ExtractionPolicy
from domains.extraction_registry import get_extraction_adapter


def test_broad_abstract_policy_limits_recovery_without_relaxing_acceptance():
    base = ExtractionPolicy()
    policy = broad_abstract_extraction_policy(base)

    assert BROAD_ABSTRACT_RECOVERY_POLICY_ID
    assert policy.max_generation_attempts == 1
    assert policy.max_patch_attempts == 1
    assert policy.max_patch_operations <= 6
    assert policy.max_micro_reextract_attempts == 0
    assert policy.max_post_micro_patch_attempts == 0
    assert policy.max_split_depth == 0
    assert policy.allow_destructive_patches is False


def test_broad_policy_capability_is_domain_scoped_and_script_generic():
    broad = get_extraction_adapter(
        "catalysis_mechanism"
    )
    dac = get_extraction_adapter(
        "dac_her"
    )
    sers = get_extraction_adapter(
        "sers_au_ag"
    )

    assert (
        broad.extraction_policy_transform
        is broad_abstract_extraction_policy
    )
    assert (
        broad.extraction_policy_id
        == BROAD_ABSTRACT_RECOVERY_POLICY_ID
    )

    for adapter in (
        dac,
        sers,
    ):
        assert (
            adapter.extraction_policy_transform
            is None
        )
        assert (
            adapter.extraction_policy_id
            is None
        )

    root = Path(__file__).resolve().parents[3]

    source = (
        root
        / "scripts"
        / "corpus"
        / "extract_paper.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "extraction_adapter.extraction_policy_transform"
        in source
    )

    assert (
        "broad_abstract_extraction_policy"
        not in source
    )

    assert (
        "BROAD_ABSTRACT_RECOVERY_POLICY_ID"
        not in source
    )
