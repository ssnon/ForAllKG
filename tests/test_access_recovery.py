from __future__ import annotations

import json
from pathlib import Path

import yaml

from pipeline_core.literature.acquisition.access_recovery import build_resolver_capability_context, prepare_access_recovery, suppressed_download_urls


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _policy(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "source-acquisition-policy-v1",
                "policy_id": "p",
                "unpaywall_email_env": "UNPAYWALL_EMAIL",
                "fallback_email_env": "CROSSREF_MAILTO",
                "openalex_api_key_env": "OPENALEX_API_KEY",
                "openalex_mailto_env": "OPENALEX_MAILTO",
                "openalex_require_api_key": True,
                "use_unpaywall": True,
                "use_openalex": True,
                "use_catalog_open_access_url": True,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _state(
    work_id: str,
    *,
    access_status: str,
    artifact_status: str,
    attempts=None,
    resolver="openalex",
):
    return {
        "work_id": work_id,
        "access_resolution": {
            "work_id": work_id,
            "status": access_status,
            "locations": [
                {
                    "resolver": resolver,
                    "url": "https://example.org/paper.pdf",
                    "url_for_pdf": "https://example.org/paper.pdf",
                }
            ],
        },
        "main_artifact": {
            "work_id": work_id,
            "status": artifact_status,
            "download_attempts": attempts or [],
        },
    }


def test_context_is_secret_free_and_tracks_capability(tmp_path: Path):
    policy = _policy(tmp_path / "policy.yaml")
    env = {
        "UNPAYWALL_EMAIL": "secret@example.org",
        "OPENALEX_API_KEY": "super-secret-key",
    }
    context = build_resolver_capability_context(policy, environ=env)
    text = json.dumps(context)
    assert context["unpaywall_contact_available"] is True
    assert context["openalex_lookup_available"] is True
    assert "secret@example.org" not in text
    assert "super-secret-key" not in text


def test_retry_access_miss_archives_state_and_preserves_downloaded(tmp_path: Path):
    policy = _policy(tmp_path / "policy.yaml")
    source = tmp_path / "source"
    output = tmp_path / "out"
    _write_json(
        source / "state" / "miss.json",
        _state("miss", access_status="unresolved", artifact_status="not_attempted"),
    )
    _write_json(
        source / "state" / "good.json",
        _state("good", access_status="resolved_direct_pdf", artifact_status="downloaded"),
    )
    report = prepare_access_recovery(
        source_policy_path=policy,
        source_m3_dir=source,
        output_m3_dir=output,
        retry_access_misses=True,
        environ={"UNPAYWALL_EMAIL": "x@y.z", "OPENALEX_API_KEY": "k"},
    )
    assert report["copied_state_count"] == 2
    assert report["refreshed_state_count"] == 1
    assert not (output / "state" / "miss.json").exists()
    assert (output / "state" / "good.json").exists()
    assert report["preserved_downloaded_state_count"] == 1


def test_retry_failed_learns_hard_urls_but_not_transient_urls(tmp_path: Path):
    policy = _policy(tmp_path / "policy.yaml")
    source = tmp_path / "source"
    output = tmp_path / "out"
    hard = _state(
        "hard",
        access_status="resolved_direct_pdf",
        artifact_status="download_failed",
        attempts=[
            {
                "status": "failed",
                "url": "https://example.org/paper.pdf",
                "error_code": "http_403",
            },
            {
                "status": "failed",
                "url": "https://example.org/html",
                "error_code": "not_pdf",
            },
        ],
    )
    transient = _state(
        "transient",
        access_status="resolved_direct_pdf",
        artifact_status="download_failed",
        attempts=[
            {
                "status": "failed",
                "url": "https://example.org/timeout.pdf",
                "error_code": "url_error",
            }
        ],
    )
    _write_json(source / "state" / "hard.json", hard)
    _write_json(source / "state" / "transient.json", transient)
    report = prepare_access_recovery(
        source_policy_path=policy,
        source_m3_dir=source,
        output_m3_dir=output,
        retry_failed=True,
        environ={"UNPAYWALL_EMAIL": "x@y.z", "OPENALEX_API_KEY": "k"},
    )
    assert report["refreshed_state_count"] == 2
    assert report["hard_failure_code_counts"] == {"http_403": 1, "not_pdf": 1}
    assert suppressed_download_urls(output, "hard") == {
        "https://example.org/paper.pdf",
        "https://example.org/html",
    }
    assert suppressed_download_urls(output, "transient") == set()


def test_capability_change_auto_invalidates_non_downloaded_state(tmp_path: Path):
    policy = _policy(tmp_path / "policy.yaml")
    source = tmp_path / "source"
    output = tmp_path / "out"
    _write_json(
        source / "state" / "miss.json",
        _state("miss", access_status="unresolved", artifact_status="not_attempted"),
    )
    first = prepare_access_recovery(
        source_policy_path=policy,
        source_m3_dir=source,
        output_m3_dir=output,
        environ={},
    )
    assert first["refreshed_state_count"] == 0
    second = prepare_access_recovery(
        source_policy_path=policy,
        source_m3_dir=source,
        output_m3_dir=output,
        environ={"UNPAYWALL_EMAIL": "x@y.z", "OPENALEX_API_KEY": "k"},
    )
    assert second["context_changed"] is True
    assert second["refreshed_state_count"] == 1
    assert second["refresh_reason_counts"]["resolver_capability_changed"] == 1


def test_legacy_state_does_not_trigger_network_without_explicit_retry(tmp_path: Path):
    policy = _policy(tmp_path / "policy.yaml")
    source = tmp_path / "source"
    output = tmp_path / "out"
    _write_json(
        source / "state" / "miss.json",
        _state("miss", access_status="unresolved", artifact_status="not_attempted"),
    )
    report = prepare_access_recovery(
        source_policy_path=policy,
        source_m3_dir=source,
        output_m3_dir=output,
        environ={"UNPAYWALL_EMAIL": "x@y.z", "OPENALEX_API_KEY": "k"},
    )
    assert report["previous_context_known"] is False
    assert report["refreshed_state_count"] == 0
    assert (output / "state" / "miss.json").exists()
