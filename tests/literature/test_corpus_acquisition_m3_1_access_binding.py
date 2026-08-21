from __future__ import annotations

import json

from pipeline_core.literature.acquisition.access_contracts import AccessResolution
from pipeline_core.literature.acquisition.supplementary_state import access_resolution_sha256, state_matches_main_access


def test_access_resolution_hash_changes_when_locations_change(tmp_path):
    one = AccessResolution(
        work_id="w",
        status="unresolved",
    )
    two = AccessResolution(
        work_id="w",
        status="resolved_landing_only",
    )
    assert access_resolution_sha256(one) != access_resolution_sha256(two)


def test_old_state_without_access_hash_forces_refresh(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "work_id": "w",
                "supplementary_discovery": {},
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )
    assert (
        state_matches_main_access(
            path=path,
            main_access_sha256="abc",
        )
        is False
    )
