from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import pipeline_core.run_lifecycle as runtime
from pipeline_core.serialization_primitives import (
    read_json,
    write_json,
)


LAYOUT_VERSION = "run-attempt-provenance-v1"
UPDATED_AT_UTC = "2026-08-21T00:00:00+00:00"


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
            getattr(runtime, name)
        )

        assert (
            signature.parameters[
                "data_root"
            ].default
            is inspect.Parameter.empty
        )

    for name in (
        "write_latest_attempt_pointer",
        "write_latest_run_pointer",
    ):
        signature = inspect.signature(
            getattr(runtime, name)
        )

        for parameter in (
            "data_root",
            "attempt_layout_version",
            "updated_at_utc",
        ):
            assert (
                signature.parameters[
                    parameter
                ].default
                is inspect.Parameter.empty
            )


def test_pointer_policy_and_resolution_are_explicit(
    tmp_path: Path,
):
    project_root = tmp_path / "project"
    project_root.mkdir()

    data_root = tmp_path / "data"

    concrete = runtime.attempt_directory(
        project_root,
        "paper-A",
        "run-A",
        "attempt-A",
        data_root=data_root,
    )
    concrete.mkdir(parents=True)

    runtime.write_latest_attempt_pointer(
        project_root=project_root,
        paper_id="paper-A",
        run_metadata=_metadata(),
        attempt_id="attempt-A",
        data_root=data_root,
        attempt_layout_version=LAYOUT_VERSION,
        updated_at_utc=UPDATED_AT_UTC,
    )

    latest_run = runtime.write_latest_run_pointer(
        project_root=project_root,
        paper_id="paper-A",
        run_metadata=_metadata(),
        data_root=data_root,
        attempt_layout_version=LAYOUT_VERSION,
        updated_at_utc=UPDATED_AT_UTC,
        attempt_id="attempt-A",
    )

    payload = read_json(
        latest_run
    )

    assert (
        payload["attempt_layout_version"]
        == LAYOUT_VERSION
    )
    assert (
        payload["updated_at_utc"]
        == UPDATED_AT_UTC
    )

    assert (
        runtime.resolve_run_directory(
            project_root=project_root,
            paper_id="paper-A",
            run_id=None,
            data_root=data_root,
        )
        == concrete.resolve()
    )


def test_flat_run_layout_remains_supported(
    tmp_path: Path,
):
    project_root = tmp_path / "project"
    project_root.mkdir()

    data_root = tmp_path / "data"

    family = runtime.run_directory(
        project_root,
        "paper-A",
        "legacy-run",
        data_root=data_root,
    )
    family.mkdir(parents=True)

    write_json(
        family / "active_chunks.json",
        {
            "run_id": "legacy-run",
        },
    )

    runtime.write_latest_run_pointer(
        project_root=project_root,
        paper_id="paper-A",
        run_metadata={
            "run_id": "legacy-run",
            "run_fingerprint": "legacy-fp",
        },
        data_root=data_root,
        attempt_layout_version=LAYOUT_VERSION,
        updated_at_utc=UPDATED_AT_UTC,
    )

    assert (
        runtime.resolve_run_directory(
            project_root=project_root,
            paper_id="paper-A",
            run_id=None,
            data_root=data_root,
        )
        == family.resolve()
    )


def test_stale_latest_run_attempt_falls_back_to_latest_attempt(
    tmp_path: Path,
):
    project_root = tmp_path / "project"
    project_root.mkdir()

    data_root = tmp_path / "data"

    latest = runtime.attempt_directory(
        project_root,
        "paper-A",
        "run-A",
        "latest-A",
        data_root=data_root,
    )
    latest.mkdir(parents=True)

    runtime.write_latest_attempt_pointer(
        project_root=project_root,
        paper_id="paper-A",
        run_metadata=_metadata(),
        attempt_id="latest-A",
        data_root=data_root,
        attempt_layout_version=LAYOUT_VERSION,
        updated_at_utc=UPDATED_AT_UTC,
    )

    # Points latest_run at an attempt that does not exist.
    runtime.write_latest_run_pointer(
        project_root=project_root,
        paper_id="paper-A",
        run_metadata=_metadata(),
        data_root=data_root,
        attempt_layout_version=LAYOUT_VERSION,
        updated_at_utc=UPDATED_AT_UTC,
        attempt_id="stale-A",
    )

    assert (
        runtime.resolve_run_directory(
            project_root=project_root,
            paper_id="paper-A",
            run_id=None,
            data_root=data_root,
        )
        == latest.resolve()
    )


def test_resolver_missing_paths_raise(
    tmp_path: Path,
):
    project_root = tmp_path / "project"
    project_root.mkdir()

    data_root = tmp_path / "data"

    with pytest.raises(
        FileNotFoundError
    ):
        runtime.resolve_run_directory(
            project_root=project_root,
            paper_id="paper-A",
            run_id=None,
            data_root=data_root,
        )

    with pytest.raises(
        FileNotFoundError
    ):
        runtime.resolve_run_directory(
            project_root=project_root,
            paper_id="paper-A",
            run_id="missing-run",
            data_root=data_root,
        )
