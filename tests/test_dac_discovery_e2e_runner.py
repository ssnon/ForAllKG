import inspect
import sys
from types import SimpleNamespace

from scripts.run_dac_discovery_e2e import (
    _data_root_args,
    _returned_path_count,
    parse_args,
    run_pipeline,
)


def test_returned_path_count_supports_current_and_fallback_shapes():
    assert _returned_path_count({"returned_path_count": 4}) == 4
    assert _returned_path_count({"paths": [{}, {}]}) == 2
    assert _returned_path_count({"summary": {"returned_path_count": 3}}) == 3
    assert _returned_path_count({}) == 0


def test_data_root_args_preserve_default_behavior():
    assert (
        _data_root_args(
            SimpleNamespace(
                data_root=None,
            )
        )
        == []
    )

    assert (
        _data_root_args(
            SimpleNamespace(
                data_root="",
            )
        )
        == []
    )


def test_data_root_args_forward_explicit_override():
    assert _data_root_args(
        SimpleNamespace(
            data_root="/tmp/sers-scratch",
        )
    ) == [
        "--data-root",
        "/tmp/sers-scratch",
    ]


def test_e2e_parser_accepts_data_root_override(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dac_discovery_e2e",
            "--run-dir",
            "/tmp/e2e-run",
            "--source",
            "source",
            "--target",
            "target",
            "--question",
            "question",
            "--data-root",
            "/tmp/sers-scratch",
        ],
    )

    args = parse_args()

    assert (
        args.data_root
        == "/tmp/sers-scratch"
    )


def test_e2e_forwards_data_root_to_all_traversal_lanes():
    source = inspect.getsource(
        run_pipeline
    )

    assert (
        source.count(
            "*_data_root_args(args)"
        )
        == 3
    )

    assert (
        source.count(
            '"scripts.run_graph_traversal"'
        )
        == 2
    )

    assert (
        source.count(
            '"scripts.run_candidate_unit_traversal"'
        )
        == 1
    )
