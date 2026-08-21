from __future__ import annotations

from pathlib import Path

from pipeline_core.corpus.broad_extraction_policy import (
    BROAD_ABSTRACT_RECOVERY_POLICY_ID,
    broad_abstract_extraction_policy,
)
from pipeline_core.corpus.extraction.extraction_policy import ExtractionPolicy


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


def test_extract_paper_source_applies_policy_only_to_broad_domain():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "corpus/extract_paper.py").read_text(encoding="utf-8")
    assert "broad_abstract_extraction_policy" in source
    assert 'domain_profile.profile_id == "catalysis_mechanism"' in source
    assert "BROAD_ABSTRACT_RECOVERY_POLICY_ID" in source
