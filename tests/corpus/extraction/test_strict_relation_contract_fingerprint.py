from __future__ import annotations

from domains.extraction_registry import (
    get_extraction_adapter,
)


def test_sers_strict_relation_contract_payload_is_deterministic():
    adapter = get_extraction_adapter(
        "sers_au_ag"
    )

    payload = (
        adapter
        .strict_relation_contract_payload()
    )

    assert len(payload) == 7

    assert [
        row["relation"]
        for row in payload
    ][-1] == "USES_PRECURSOR"

    uses_precursor = payload[-1]

    assert uses_precursor == {
        "relation": "USES_PRECURSOR",
        "source_types": [
            "SynthesisMethod"
        ],
        "target_types": [
            "Precursor"
        ],
        "severity": "warning",
    }


def test_sers_strict_contract_payload_contains_no_graph_only_promotions():
    adapter = get_extraction_adapter(
        "sers_au_ag"
    )

    relations = {
        row["relation"]
        for row
        in adapter
        .strict_relation_contract_payload()
    }

    assert relations == {
        "HAS_MEASUREMENT",
        "MEASURED_FOR",
        "IN_MEASUREMENT_GROUP",
        "SUPPORTS_CLAIM",
        "INTERPRETED_AS",
        "APPLIES_TO",
        "USES_PRECURSOR",
    }
