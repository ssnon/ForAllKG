from __future__ import annotations

import pytest

from dac_her.external_novelty_contracts import (
    ClaimPriorArtReviewDraft,
    PriorArtMatchDraft,
)
from dac_her.external_novelty_llm import _REVIEW_SYSTEM
from dac_her.standard2_claim_review_dev_validation_v3 import (
    REQUIRED_PROMPT_SENTINELS,
    validate_draft_work_ids,
    validate_hardened_prompt,
)


def _draft(ids: list[str]) -> ClaimPriorArtReviewDraft:
    return ClaimPriorArtReviewDraft(
        matches=[
            PriorArtMatchDraft(
                work_id=work_id,
                relationship="COMPONENT_ONLY",
                confidence=0.8,
                rationale="test",
            )
            for work_id in ids
        ],
        interpretation="bounded",
    )


def test_work_id_copy_prompt_contract_present():
    validate_hardened_prompt()
    for sentinel in REQUIRED_PROMPT_SENTINELS:
        assert sentinel in _REVIEW_SYSTEM


def test_exact_allowed_ids_pass():
    validate_draft_work_ids(
        claim_id="c1",
        draft=_draft(["prior_art_work:abc", "prior_art_work:def"]),
        allowed_work_ids=[
            "prior_art_work:abc",
            "prior_art_work:def",
        ],
    )


def test_index_style_hallucination_fails():
    with pytest.raises(RuntimeError):
        validate_draft_work_ids(
            claim_id="c1",
            draft=_draft(["prior_art_work:6"]),
            allowed_work_ids=["prior_art_work:abcdef"],
        )


def test_duplicate_allowed_id_fails():
    with pytest.raises(RuntimeError):
        validate_draft_work_ids(
            claim_id="c1",
            draft=_draft([
                "prior_art_work:abc",
                "prior_art_work:abc",
            ]),
            allowed_work_ids=["prior_art_work:abc"],
        )
