import json

from scripts.discovery.run_adaptive_discovery_controller_v2_shadow import (
    _eligible_request,
)


def test_v2_synthesizes_retraversal_after_known_axis_exhaustion(
    tmp_path,
):
    local = tmp_path / "local"
    local.mkdir()

    history = {
        "schema_version": "adaptive-discovery-history-v1",
        "history_by_root": {
            "root": [
                {
                    "round_index": 1,
                    "current_hypothesis_id": "h1",
                    "action": "SAME_PREMISE_SHARPEN",
                    "epistemic_state": "PRIOR_ART_BACKED_OR_NO_RESIDUAL",
                },
                {
                    "round_index": 2,
                    "current_hypothesis_id": "h2",
                    "action": "AXIS_MUTATION",
                    "epistemic_state": "PRIOR_ART_BACKED_OR_NO_RESIDUAL",
                },
                {
                    "round_index": 3,
                    "current_hypothesis_id": "h2",
                    "action": "STOP",
                    "epistemic_state": "PRIOR_ART_BACKED_OR_NO_RESIDUAL",
                },
            ]
        },
    }
    (local / "adaptive_search.history.json").write_text(
        json.dumps(history),
        encoding="utf-8",
    )
    (local / "graph_retraversal.handoff.json").write_text(
        json.dumps({"requests": []}),
        encoding="utf-8",
    )
    round_dir = local / "round_03"
    round_dir.mkdir()
    (round_dir / "controller.plan.json").write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "root_hypothesis_id": "root",
                        "current_hypothesis_id": "h2",
                        "current_epistemic_state": "PRIOR_ART_BACKED_OR_NO_RESIDUAL",
                        "current_external_status": "WELL_ESTABLISHED",
                        "action": "STOP",
                        "already_known_boundary": ["known"],
                        "unresolved_boundary": [],
                        "target_claim_ids": ["c"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    request, deferred = _eligible_request(
        local_dir=local,
        effective_ids=set(),
    )
    assert deferred == []
    assert request is not None
    assert request["current_hypothesis_id"] == "h2"
    assert (
        request["v2_request_reason"]
        == "known_region_local_search_exhausted"
    )


def test_v2_defers_multiple_graph_requests(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    (local / "graph_retraversal.handoff.json").write_text(
        json.dumps(
            {
                "requests": [
                    {"current_hypothesis_id": "h1"},
                    {"current_hypothesis_id": "h2"},
                ]
            }
        ),
        encoding="utf-8",
    )
    request, deferred = _eligible_request(
        local_dir=local,
        effective_ids=set(),
    )
    assert request is None
    assert len(deferred) == 2
