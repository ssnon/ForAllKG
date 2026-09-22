from __future__ import annotations

import inspect

from pipeline_core.discovery.projection_relation_adjudication import (
    _ADJUDICATION_SYSTEM,
    _prompt_for_claim,
)


def test_adjudicator_requests_exhaustive_presented_work_classification():
    normalized = " ".join(_ADJUDICATION_SYSTEM.split())

    assert (
        "Return exactly one record for every work in ALLOWED_WORK_IDS."
        in normalized
    )
    assert "Do not omit any allowed work." in normalized
    assert (
        "If a record has no material bearing, use UNRELATED"
        in normalized
    )
    assert (
        "if metadata is insufficient, use INSUFFICIENT_METADATA"
        in normalized
    )
    assert (
        "Omit records that do not need discussion."
        not in normalized
    )


def test_claim_prompt_repeats_exhaustive_coverage_contract():
    source = inspect.getsource(_prompt_for_claim)

    assert "CLASSIFICATION COVERAGE CONTRACT" in source
    assert (
        "Return exactly one match record for every ALLOWED_WORK_ID."
        in source
    )
    assert (
        "Every presented work must receive one allowed relationship label."
        in source
    )
