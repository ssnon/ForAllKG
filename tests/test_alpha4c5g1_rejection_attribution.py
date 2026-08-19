from __future__ import annotations

from types import SimpleNamespace

import networkx as nx

from campaigns.sers_alpha4_epoch.support.trend_rejection_attribution import (
    attribute_claim_miss,
    build_stratified_sample,
)


class StubTrend:
    _NUMERIC_RESPONSE_KEYS = {
        "raman_intensity",
    }

    @staticmethod
    def _claim_control(text):
        if "shell" in text:
            return ("shell_thickness", "shell")
        return None

    @staticmethod
    def _claim_response(text):
        if "sers" in text.lower():
            return ("raman_intensity", "signal")
        return None

    @staticmethod
    def _claim_direction_shape(control_key, text):
        if "increased" in text:
            return ("positive", "monotonic")
        return None

    @staticmethod
    def _control_key_from_name(name):
        return "shell_thickness" if "shell" in name else ""

    @staticmethod
    def _measurement_control(*args):
        return None

    @staticmethod
    def _lineage(*args):
        return None

    @staticmethod
    def _methods_compatible(rows):
        return True


def graph_with_claim(text: str) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "c1",
        type="ObservationClaim",
        statement=text,
        source_chunk_id="chunk-1",
    )
    return graph


def test_claim_control_reason_is_first():
    row = attribute_claim_miss(
        candidate={
            "paper_id": "P1",
            "claim_id": "c1",
            "text": "SERS increased.",
            "admitted_by_current_trend": False,
        },
        graph=graph_with_claim("SERS increased."),
        trend_module=StubTrend,
    )
    assert row["primary_reason"] == (
        "claim_control_not_normalized"
    )


def test_claim_direction_reason_after_control_response():
    text = "SERS signal with shell thickness."
    row = attribute_claim_miss(
        candidate={
            "paper_id": "P1",
            "claim_id": "c1",
            "text": text,
            "admitted_by_current_trend": False,
        },
        graph=graph_with_claim(text),
        trend_module=StubTrend,
    )
    assert row["primary_reason"] == (
        "claim_direction_not_normalized"
    )


def test_claim_unexplained_bucket_is_preserved():
    text = "SERS increased with shell thickness."
    row = attribute_claim_miss(
        candidate={
            "paper_id": "P1",
            "claim_id": "c1",
            "text": text,
            "admitted_by_current_trend": False,
        },
        graph=graph_with_claim(text),
        trend_module=StubTrend,
    )
    assert row["primary_reason"] == (
        "claim_unexplained_post_normalization_miss"
    )


def test_stratified_sample_is_deterministic():
    rows = [
        {
            "paper_id": f"P{i}",
            "claim_id": f"c{i}",
            "primary_reason": "r1",
            "text": str(i),
        }
        for i in range(20)
    ]
    a = build_stratified_sample(
        claim_rows=rows,
        numeric_rows=[],
        per_bucket=5,
    )
    b = build_stratified_sample(
        claim_rows=list(reversed(rows)),
        numeric_rows=[],
        per_bucket=5,
    )
    assert a == b
    assert len(a) == 5
