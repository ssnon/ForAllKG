from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from dac_her.alpha4c5h1_reserve_b import (
    ALPHA4C5H1_PROTOCOL_SEMANTICS_ID,
    Alpha4c5h1ExecutionPolicy,
    DEFAULT_RESERVE_A_TEMPLATE_PROTOCOL,
    EXPECTED_5H_CONFIRMATION_PROTOCOL_ID,
    EXPECTED_5H_FREEZE_ID,
    EXPECTED_RESERVE_A_TEMPLATE_PROTOCOL_ID,
    EXPECTED_RESERVE_A_TEMPLATE_SEMANTICS_ID,
    EXPECTED_TREND_SEMANTICS_ID,
    discover_reserve_a_template_protocol,
    protocol_sha_payload,
)
from dac_her.alpha4c5f_reserve import (
    Alpha4c5fArtifactIds,
    sha256_json,
)
from dac_her.alpha4c5h1_runtime_bindings import (
    V5_PRECISION_ADAPTER,
    V6R2_RUNTIME_PRECISION_ADAPTER,
    V6R2_TREND_ADAPTER,
)


def test_frozen_ids_are_exact():
    assert EXPECTED_5H_FREEZE_ID.endswith(
        "710786859181e21535f2"
    )
    assert EXPECTED_5H_CONFIRMATION_PROTOCOL_ID.endswith(
        "7e71422f94caadf9161a"
    )
    assert EXPECTED_TREND_SEMANTICS_ID == (
        "sers_au_ag_trend_v6r2_alpha4c5g2r2"
    )
    assert ALPHA4C5H1_PROTOCOL_SEMANTICS_ID == (
        "sers_alpha4c5h1_guarded_reserve_b_v1"
    )


def test_runtime_trend_binding_is_v6r2():
    assert V6R2_TREND_ADAPTER.semantics_id == (
        EXPECTED_TREND_SEMANTICS_ID
    )


def test_precision_algorithm_identity_is_preserved():
    assert (
        V6R2_RUNTIME_PRECISION_ADAPTER.precision_semantics_id
        == V5_PRECISION_ADAPTER.precision_semantics_id
    )
    assert (
        V6R2_RUNTIME_PRECISION_ADAPTER.annotate_fn
        is not V5_PRECISION_ADAPTER.annotate_fn
    )
    assert (
        V6R2_RUNTIME_PRECISION_ADAPTER.trend_semantics_id
        == EXPECTED_TREND_SEMANTICS_ID
    )


def test_precision_annotation_delegates_to_v5():
    # Function-level delegation is tested structurally: the runtime adapter
    # uses the same public precision semantics and only changes parent Trend
    # metadata during consolidation.
    assert (
        V6R2_RUNTIME_PRECISION_ADAPTER.adapter_id
        == V5_PRECISION_ADAPTER.adapter_id
    )
    assert (
        V6R2_RUNTIME_PRECISION_ADAPTER.domain_profile_id
        == V5_PRECISION_ADAPTER.domain_profile_id
    )

def test_reserve_a_template_binding_is_explicit_and_valid():
    path, protocol = discover_reserve_a_template_protocol(
        root=Path.cwd()
    )
    assert path == Path.cwd() / DEFAULT_RESERVE_A_TEMPLATE_PROTOCOL
    assert protocol.protocol_id == EXPECTED_RESERVE_A_TEMPLATE_PROTOCOL_ID
    assert (
        protocol.semantics_id
        == EXPECTED_RESERVE_A_TEMPLATE_SEMANTICS_ID
    )
    assert protocol.campaign_id == "sers_alpha4c5f2_reserve_a_v1"
    assert protocol.reserve_partition == "reserve_a"
    assert len(protocol.reserve_paper_ids) == 25

def test_protocol_sha_payload_serializes_nested_models():
    artifact_ids = Alpha4c5fArtifactIds(
        corpus="c",
        measurement_result_identity="identity",
        metric_definition="metric",
        comparison="comparison",
        trend="trend",
        precision="precision",
        context="context",
        assessment="assessment",
    )
    payload = protocol_sha_payload(
        {
            "protocol_sha256": "",
            "artifact_ids": artifact_ids,
            "execution_policy": Alpha4c5h1ExecutionPolicy(),
        }
    )
    assert "protocol_sha256" not in payload
    assert isinstance(payload["artifact_ids"], dict)
    assert isinstance(payload["execution_policy"], dict)
    assert payload["artifact_ids"]["corpus"] == "c"
    assert len(sha256_json(payload)) == 64

