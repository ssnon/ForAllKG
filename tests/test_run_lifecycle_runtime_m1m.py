from __future__ import annotations

from datetime import datetime as RealDateTime, timezone
import inspect
from pathlib import Path

import dac_her.run_state as legacy
import pipeline_core.run_lifecycle as shared


def _metadata() -> dict[str, str]:
    return {
        "run_id": "run-A",
        "run_fingerprint": "fp-A",
    }


def test_shared_runtime_requires_explicit_policy_inputs():
    for name in (
        "paper_output_root",
        "run_directory",
        "attempt_directory",
        "resolve_run_directory",
    ):
        signature = inspect.signature(
            getattr(shared, name)
        )
        assert (
            signature.parameters["data_root"].default
            is inspect.Parameter.empty
        )

    for name in (
        "write_latest_attempt_pointer",
        "write_latest_run_pointer",
    ):
        signature = inspect.signature(
            getattr(shared, name)
        )
        for parameter in (
            "data_root",
            "attempt_layout_version",
            "updated_at_utc",
        ):
            assert (
                signature.parameters[parameter].default
                is inspect.Parameter.empty
            )


def test_legacy_api_keeps_dac_ownership_and_default():
    for name in (
        "paper_output_root",
        "run_directory",
        "attempt_directory",
        "write_latest_attempt_pointer",
        "write_latest_run_pointer",
        "resolve_run_directory",
    ):
        assert (
            getattr(legacy, name).__module__
            == "dac_her.run_state"
        )

    assert (
        inspect.signature(
            legacy.paper_output_root
        ).parameters["data_root"].default
        == "data_dac"
    )


def test_shared_and_legacy_paths_match(
    tmp_path: Path,
):
    project_root = tmp_path / "project"
    project_root.mkdir()
    data_root = tmp_path / "data"

    assert (
        legacy.paper_output_root(
            project_root,
            "paper-A",
            data_root=data_root,
        )
        == shared.paper_output_root(
            project_root,
            "paper-A",
            data_root=data_root,
        )
    )

    assert (
        legacy.attempt_directory(
            project_root,
            "paper-A",
            "run-A",
            "attempt-A",
            data_root=data_root,
        )
        == shared.attempt_directory(
            project_root,
            "paper-A",
            "run-A",
            "attempt-A",
            data_root=data_root,
        )
    )


def test_shared_pointer_policy_is_injected(
    tmp_path: Path,
):
    project_root = tmp_path / "project"
    project_root.mkdir()
    data_root = tmp_path / "data"

    concrete = shared.attempt_directory(
        project_root,
        "paper-A",
        "run-A",
        "attempt-A",
        data_root=data_root,
    )
    concrete.mkdir(parents=True)

    path = shared.write_latest_run_pointer(
        project_root=project_root,
        paper_id="paper-A",
        run_metadata=_metadata(),
        data_root=data_root,
        attempt_layout_version="test-layout-v1",
        updated_at_utc="2026-08-19T01:02:03+00:00",
        attempt_id="attempt-A",
    )

    payload = legacy.read_json(path)

    assert (
        payload["attempt_layout_version"]
        == "test-layout-v1"
    )
    assert (
        payload["updated_at_utc"]
        == "2026-08-19T01:02:03+00:00"
    )

    assert (
        shared.resolve_run_directory(
            project_root=project_root,
            paper_id="paper-A",
            run_id=None,
            data_root=data_root,
        )
        == concrete.resolve()
    )


def test_legacy_datetime_policy_remains_wrapper_owned(
    tmp_path: Path,
    monkeypatch,
):
    class FrozenDateTime:
        @classmethod
        def now(cls, tz=None):
            return RealDateTime(
                2026,
                8,
                19,
                1,
                2,
                3,
                tzinfo=timezone.utc,
            )

    monkeypatch.setattr(
        legacy,
        "datetime",
        FrozenDateTime,
    )

    project_root = tmp_path / "project"
    project_root.mkdir()
    data_root = tmp_path / "data"

    concrete = legacy.attempt_directory(
        project_root,
        "paper-A",
        "run-A",
        "attempt-A",
        data_root=data_root,
    )
    concrete.mkdir(parents=True)

    path = legacy.write_latest_run_pointer(
        project_root=project_root,
        paper_id="paper-A",
        run_metadata=_metadata(),
        data_root=data_root,
        attempt_id="attempt-A",
    )

    payload = legacy.read_json(path)

    assert (
        payload["attempt_layout_version"]
        == legacy.ATTEMPT_LAYOUT_VERSION
    )
    assert (
        payload["updated_at_utc"]
        == "2026-08-19T01:02:03+00:00"
    )


def test_legacy_flat_layout_remains_supported(
    tmp_path: Path,
):
    project_root = tmp_path / "project"
    project_root.mkdir()
    data_root = tmp_path / "data"

    family = legacy.run_directory(
        project_root,
        "paper-A",
        "legacy-run",
        data_root=data_root,
    )
    family.mkdir(parents=True)

    legacy.write_json(
        family / "active_chunks.json",
        {"run_id": "legacy-run"},
    )

    legacy.write_latest_run_pointer(
        project_root=project_root,
        paper_id="paper-A",
        run_metadata={
            "run_id": "legacy-run",
            "run_fingerprint": "legacy-fp",
        },
        data_root=data_root,
    )

    assert (
        shared.resolve_run_directory(
            project_root=project_root,
            paper_id="paper-A",
            run_id=None,
            data_root=data_root,
        )
        == family.resolve()
    )
