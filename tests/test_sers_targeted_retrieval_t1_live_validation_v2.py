from __future__ import annotations

from pydantic import BaseModel

from campaigns.sers_novelty_gap.sers_targeted_retrieval_t1_live_validation_v2 import (
    _canonical_json,
)


class _TinyModel(BaseModel):
    value: int


def test_canonical_json_recursively_serializes_model_lists() -> None:
    text = _canonical_json(
        [_TinyModel(value=2), _TinyModel(value=1)]
    )
    assert text == '[{"value":2},{"value":1}]'


def test_canonical_json_recursively_serializes_nested_models() -> None:
    text = _canonical_json(
        {"rows": [_TinyModel(value=3)]}
    )
    assert text == '{"rows":[{"value":3}]}'
