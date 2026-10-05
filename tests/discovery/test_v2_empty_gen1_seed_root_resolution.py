import json

from scripts.discovery.run_adaptive_discovery_controller_shadow import (
    _resolve_stage775_seed_root,
)


def test_v2_initial_seed_root_uses_empty_gen1_fallback(tmp_path):
    closed = tmp_path / "closed_loop"
    fallback = closed / "adaptive_seed"
    fallback.mkdir(parents=True)

    (closed / "closed_loop.summary.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE_EMPTY_GEN1",
                "adaptive_seed_available": True,
            }
        ),
        encoding="utf-8",
    )

    root, mode = _resolve_stage775_seed_root(closed)

    assert root == fallback
    assert mode == "STAGE_775_GEN0_EMPTY_GEN1_FALLBACK"


def test_v2_initial_seed_root_keeps_normal_seed(tmp_path):
    closed = tmp_path / "closed_loop"
    closed.mkdir()

    (closed / "closed_loop.summary.json").write_text(
        json.dumps({"status": "COMPLETE"}),
        encoding="utf-8",
    )

    root, mode = _resolve_stage775_seed_root(closed)

    assert root == closed
    assert mode == "STAGE_775_GEN1"
