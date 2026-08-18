from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import BaseModel

from dac_her.evaluation_runtime.artifacts import (
    canonical_json,
    load_json_object,
    sha256_file,
    sha256_json,
    sha256_json_without_fields,
)
from dac_her.fresh_c_acquisition import (
    canonical_json as legacy_canonical_json,
    sha256_file as legacy_sha256_file,
    sha256_json as legacy_sha256_json,
)


class ExampleModel(BaseModel):
    z: int
    text: str
    values: list[int]


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"b": 2, "a": 1},
        {"unicode": "한글", "nested": {"z": 3, "a": 1}},
        [3, 2, 1],
        ExampleModel(z=7, text="SERS", values=[3, 1, 2]),
    ],
)
def test_canonical_json_is_legacy_equivalent(value):
    assert canonical_json(value) == legacy_canonical_json(value)
    assert sha256_json(value) == legacy_sha256_json(value)


def test_canonical_json_exact_encoding_contract():
    value = {"b": "한글", "a": 1}
    expected = '{"a":1,"b":"한글"}'
    assert canonical_json(value) == expected
    assert sha256_json(value) == hashlib.sha256(
        expected.encode("utf-8")
    ).hexdigest()


def test_sha_without_fields_matches_historical_payload_rule():
    payload = {
        "protocol_id": "prefix:deadbeef",
        "protocol_sha256": "f" * 64,
        "b": 2,
        "a": 1,
    }

    payload_body = dict(payload)
    payload_body.pop("protocol_sha256")

    identity_body = dict(payload)
    identity_body.pop("protocol_id")
    identity_body.pop("protocol_sha256")

    assert sha256_json_without_fields(
        payload,
        "protocol_sha256",
    ) == legacy_sha256_json(payload_body)

    assert sha256_json_without_fields(
        payload,
        "protocol_id",
        "protocol_sha256",
    ) == legacy_sha256_json(identity_body)


def test_sha256_file_is_legacy_equivalent(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"GraphAgentsDAC\x00Fresh-C\n")

    assert sha256_file(path) == legacy_sha256_file(path)


def test_load_json_object(tmp_path):
    path = tmp_path / "object.json"
    path.write_text(
        json.dumps({"a": 1}),
        encoding="utf-8",
    )
    assert load_json_object(path) == {"a": 1}


def test_load_json_object_rejects_non_object(tmp_path):
    path = tmp_path / "array.json"
    path.write_text("[1,2,3]", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected JSON object"):
        load_json_object(path)
