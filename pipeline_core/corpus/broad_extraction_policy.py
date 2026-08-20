from __future__ import annotations

from dataclasses import replace

from pipeline_core.extraction_policy import ExtractionPolicy


BROAD_ABSTRACT_RECOVERY_POLICY_ID = (
    "catalysis-mechanism-abstract-recovery-v1-low-cost"
)


def broad_abstract_extraction_policy(
    base: ExtractionPolicy | None = None,
) -> ExtractionPolicy:
    """Return the strict-but-low-cost recovery policy for Broad abstracts.

    Scientific validators and finalization rules are unchanged. The Broad
    corpus treats one abstract as replaceable discovery evidence, so it spends
    at most one generation and one semantic patch before excluding the paper.
    Tiny-leaf micro recovery and rechunking are disabled for this profile.
    """
    policy = base or ExtractionPolicy()
    return replace(
        policy,
        max_generation_attempts=1,
        max_patch_attempts=1,
        max_patch_operations=min(policy.max_patch_operations, 6),
        max_micro_reextract_attempts=0,
        max_post_micro_patch_attempts=0,
        max_split_depth=0,
    )
