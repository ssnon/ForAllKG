from __future__ import annotations

import os
from pathlib import Path

import pipeline_core.literature.acquisition.materializers as module
from pipeline_core.literature.acquisition.materialization_contracts import (
    MaterializationPolicy,
)
from pipeline_core.literature.acquisition.materializers import MarkerPdfMaterializer


class _Completed:
    returncode = 1
    stdout = "synthetic marker failure"


def test_default_policy_uses_gnu_mkl_threading_for_marker_only():
    policy = MaterializationPolicy(policy_id="p")
    assert policy.marker_environment_overrides["MKL_THREADING_LAYER"] == "GNU"
    assert "MKL_SERVICE_FORCE_INTEL" in policy.marker_environment_unset


def test_marker_subprocess_overrides_inherited_intel_mkl(monkeypatch, tmp_path):
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.7\nsynthetic")

    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/marker_single")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = dict(kwargs["env"])
        return _Completed()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setenv("MKL_THREADING_LAYER", "INTEL")
    monkeypatch.setenv("MKL_SERVICE_FORCE_INTEL", "1")

    policy = MaterializationPolicy(policy_id="p")
    try:
        MarkerPdfMaterializer().materialize(
            source_path=source,
            policy=policy,
        )
    except RuntimeError as exc:
        assert "marker_failed" in str(exc)

    assert captured["env"]["MKL_THREADING_LAYER"] == "GNU"
    assert "MKL_SERVICE_FORCE_INTEL" not in captured["env"]
    # Parent process environment is not mutated by the materializer.
    assert os.environ["MKL_THREADING_LAYER"] == "INTEL"
    assert os.environ["MKL_SERVICE_FORCE_INTEL"] == "1"
