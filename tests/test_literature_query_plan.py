from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pipeline_core.literature.discovery.query_plan import load_query_plan


def test_query_plan_loads_bucket_targets_and_queries(tmp_path: Path):
    path = tmp_path / "plan.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "v1",
                "plan_id": "pilot",
                "description": "test plan",
                "buckets": {
                    "working_state": {
                        "label": "Working state",
                        "target": 20,
                        "queries": ["operando active site", "catalyst reconstruction"],
                    },
                    "kinetics": {
                        "label": "Kinetics",
                        "target": 30,
                        "queries": ["microkinetic electrocatalysis"],
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    plan = load_query_plan(path)
    assert plan.plan_id == "pilot"
    assert plan.target_paper_count == 50
    assert plan.query_count == 3
    assert [bucket.bucket_id for bucket in plan.buckets] == ["working_state", "kinetics"]


def test_query_plan_rejects_non_positive_target(tmp_path: Path):
    path = tmp_path / "invalid.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "v1",
                "plan_id": "invalid",
                "buckets": {
                    "bad": {
                        "label": "Bad",
                        "target": 0,
                        "queries": ["query"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target must be positive"):
        load_query_plan(path)
