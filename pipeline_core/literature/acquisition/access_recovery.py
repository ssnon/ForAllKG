from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml


CONTEXT_SCHEMA = "access-recovery-context-v1"
LEDGER_SCHEMA = "access-recovery-bad-locations-v1"
REPORT_SCHEMA = "access-recovery-report-v1"

HARD_LOCATION_FAILURE_CODES = frozenset(
    {
        "http_401",
        "http_403",
        "http_404",
        "http_410",
        "not_pdf",
    }
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def build_resolver_capability_context(
    source_policy_path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a secret-free fingerprint of resolver capability and policy.

    Environment *values* are never persisted.  Only booleans describing
    whether a configured credential/contact capability is available are used.
    """
    policy_path = Path(source_policy_path)
    payload = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Source policy must be a mapping: {policy_path}")
    env = os.environ if environ is None else environ

    fallback_email_env = str(payload.get("fallback_email_env") or "")
    unpaywall_email_env = str(payload.get("unpaywall_email_env") or "")
    openalex_api_key_env = str(payload.get("openalex_api_key_env") or "")
    openalex_mailto_env = str(payload.get("openalex_mailto_env") or "")

    def present(name: str) -> bool:
        return bool(name and str(env.get(name) or "").strip())

    use_unpaywall = bool(payload.get("use_unpaywall", True))
    use_openalex = bool(payload.get("use_openalex", True))
    require_openalex_key = bool(payload.get("openalex_require_api_key", True))
    context: dict[str, Any] = {
        "schema_version": CONTEXT_SCHEMA,
        "policy_id": str(payload.get("policy_id") or ""),
        "source_policy_sha256": _sha256_file(policy_path),
        "unpaywall_enabled": use_unpaywall,
        "unpaywall_contact_available": bool(
            present(unpaywall_email_env) or present(fallback_email_env)
        ),
        "openalex_enabled": use_openalex,
        "openalex_api_key_required": require_openalex_key,
        "openalex_api_key_available": present(openalex_api_key_env),
        "openalex_lookup_available": bool(
            use_openalex
            and (not require_openalex_key or present(openalex_api_key_env))
        ),
        "openalex_contact_available": bool(
            present(openalex_mailto_env) or present(fallback_email_env)
        ),
        "catalog_oa_fallback_enabled": bool(
            payload.get("use_catalog_open_access_url", True)
        ),
    }
    context["fingerprint"] = _canonical_sha256(context)
    return context


def _attempt_error_code(row: Mapping[str, Any]) -> str:
    return str(row.get("error_code") or "").strip().lower()


def hard_failure_urls(
    state: Mapping[str, Any],
) -> dict[str, set[str]]:
    """Return URL -> hard-failure codes from one work-state artifact."""
    artifact = state.get("main_artifact")
    if not isinstance(artifact, dict):
        return {}
    rows: dict[str, set[str]] = {}
    for attempt in artifact.get("download_attempts") or []:
        if not isinstance(attempt, dict):
            continue
        if str(attempt.get("status") or "") != "failed":
            continue
        code = _attempt_error_code(attempt)
        if code not in HARD_LOCATION_FAILURE_CODES:
            continue
        url = str(attempt.get("url") or "").strip()
        if url:
            rows.setdefault(url, set()).add(code)
    return rows


def _resolver_by_url(state: Mapping[str, Any]) -> dict[str, str]:
    resolution = state.get("access_resolution")
    if not isinstance(resolution, dict):
        return {}
    rows: dict[str, str] = {}
    for location in resolution.get("locations") or []:
        if not isinstance(location, dict):
            continue
        url = str(location.get("url_for_pdf") or location.get("url") or "").strip()
        if url:
            rows[url] = str(location.get("resolver") or "")
    return rows


def _refresh_reason(
    state: Mapping[str, Any],
    *,
    retry_failed: bool,
    retry_access_misses: bool,
    context_changed: bool,
) -> str | None:
    artifact = state.get("main_artifact")
    resolution = state.get("access_resolution")
    if not isinstance(artifact, dict) or not isinstance(resolution, dict):
        return "invalid_legacy_state"
    artifact_status = str(artifact.get("status") or "")
    if artifact_status == "downloaded":
        return None
    if context_changed:
        return "resolver_capability_changed"
    if artifact_status == "download_failed" and retry_failed:
        return "explicit_retry_failed"
    if artifact_status == "not_attempted" and retry_access_misses:
        access_status = str(resolution.get("status") or "")
        if access_status in {
            "unresolved",
            "resolved_landing_only",
            "resolved_direct_pdf",
        }:
            return "explicit_retry_access_miss"
    return None


def _copy_state_seed(source_state_root: Path, target_state_root: Path) -> int:
    if not source_state_root.is_dir():
        return 0
    target_state_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source in sorted(source_state_root.glob("*.json")):
        target = target_state_root / source.name
        if target.exists():
            continue
        shutil.copy2(source, target)
        copied += 1
    return copied


def _load_ledger(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload is None or payload.get("schema_version") != LEDGER_SCHEMA:
        return {"schema_version": LEDGER_SCHEMA, "works": {}}
    if not isinstance(payload.get("works"), dict):
        payload["works"] = {}
    return payload


def _merge_source_ledger(source_m3_dir: Path, target_ledger: dict[str, Any]) -> None:
    source = _load_json(source_m3_dir / "access_recovery_bad_locations.json")
    if not source or source.get("schema_version") != LEDGER_SCHEMA:
        return
    source_works = source.get("works")
    if not isinstance(source_works, dict):
        return
    target_works = target_ledger.setdefault("works", {})
    for work_id, source_work in source_works.items():
        if not isinstance(source_work, dict):
            continue
        target_work = target_works.setdefault(str(work_id), {"urls": {}})
        target_urls = target_work.setdefault("urls", {})
        for url, info in (source_work.get("urls") or {}).items():
            if isinstance(info, dict):
                target_urls.setdefault(str(url), dict(info))


def suppressed_download_urls(
    output_root: str | Path,
    work_id: str,
) -> set[str]:
    """Read recovery-generation hard-failure URLs for the downloader."""
    payload = _load_json(Path(output_root) / "access_recovery_bad_locations.json")
    if not payload or payload.get("schema_version") != LEDGER_SCHEMA:
        return set()
    works = payload.get("works")
    if not isinstance(works, dict):
        return set()
    work = works.get(str(work_id))
    if not isinstance(work, dict):
        return set()
    urls = work.get("urls")
    if not isinstance(urls, dict):
        return set()
    return {str(url).strip() for url in urls if str(url).strip()}


def prepare_access_recovery(
    *,
    source_policy_path: str | Path,
    source_m3_dir: str | Path,
    output_m3_dir: str | Path,
    retry_failed: bool = False,
    retry_access_misses: bool = False,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    source_dir = Path(source_m3_dir)
    output_dir = Path(output_m3_dir)
    source_state_root = source_dir / "state"
    target_state_root = output_dir / "state"
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_state_count = _copy_state_seed(source_state_root, target_state_root)
    current_context = build_resolver_capability_context(
        source_policy_path,
        environ=environ,
    )
    context_path = output_dir / "access_recovery_context.json"
    previous_context = _load_json(context_path)
    if previous_context is None:
        previous_context = _load_json(source_dir / "access_recovery_context.json")
    previous_fingerprint = str((previous_context or {}).get("fingerprint") or "")
    current_fingerprint = str(current_context["fingerprint"])
    context_changed = bool(
        previous_fingerprint and previous_fingerprint != current_fingerprint
    )

    ledger_path = output_dir / "access_recovery_bad_locations.json"
    # A capability/policy generation change gets a fresh bad-location ledger.
    # Old hard failures may be retried once under the new resolver generation.
    ledger = (
        {"schema_version": LEDGER_SCHEMA, "works": {}}
        if context_changed
        else _load_ledger(ledger_path)
    )
    if not context_changed:
        _merge_source_ledger(source_dir, ledger)
    ledger["context_fingerprint"] = current_fingerprint

    reason_counts: Counter[str] = Counter()
    hard_code_counts: Counter[str] = Counter()
    hard_resolver_counts: Counter[str] = Counter()
    suppressed_url_count = 0
    refreshed_work_ids: list[str] = []
    preserved_downloaded = 0
    legacy_unknown_context = previous_context is None

    stale_root = target_state_root / ".recovery_stale" / current_fingerprint[:12]
    if target_state_root.is_dir():
        for state_path in sorted(target_state_root.glob("*.json")):
            state = _load_json(state_path)
            if state is None:
                continue
            work_id = str(state.get("work_id") or "").strip()
            artifact = state.get("main_artifact")
            if isinstance(artifact, dict) and artifact.get("status") == "downloaded":
                preserved_downloaded += 1
                continue

            reason = _refresh_reason(
                state,
                retry_failed=retry_failed,
                retry_access_misses=retry_access_misses,
                context_changed=context_changed,
            )

            # Known hard endpoints are suppressed only within the current
            # recovery generation.  On a capability change we intentionally
            # let them be tried once again before re-learning failures.
            if not context_changed:
                resolver_by_url = _resolver_by_url(state)
                for url, codes in hard_failure_urls(state).items():
                    work = ledger.setdefault("works", {}).setdefault(
                        work_id,
                        {"urls": {}},
                    )
                    info = work.setdefault("urls", {}).setdefault(
                        url,
                        {"error_codes": [], "resolvers": []},
                    )
                    existing_codes = set(info.get("error_codes") or [])
                    existing_codes.update(codes)
                    info["error_codes"] = sorted(existing_codes)
                    resolver = resolver_by_url.get(url, "")
                    existing_resolvers = set(info.get("resolvers") or [])
                    if resolver:
                        existing_resolvers.add(resolver)
                    info["resolvers"] = sorted(existing_resolvers)
                    suppressed_url_count += 1
                    for code in codes:
                        hard_code_counts[code] += 1
                    if resolver:
                        hard_resolver_counts[resolver] += 1

            if reason is None:
                reason_counts["resume_preserved"] += 1
                continue

            stale_root.mkdir(parents=True, exist_ok=True)
            archive = stale_root / state_path.name
            if archive.exists():
                archive.unlink()
            shutil.move(str(state_path), str(archive))
            reason_counts[reason] += 1
            if work_id:
                refreshed_work_ids.append(work_id)

    ledger["suppressed_url_count"] = sum(
        len((work or {}).get("urls") or {})
        for work in (ledger.get("works") or {}).values()
        if isinstance(work, dict)
    )
    _write_json_atomic(ledger_path, ledger)
    _write_json_atomic(context_path, current_context)

    report = {
        "schema_version": REPORT_SCHEMA,
        "source_m3_dir": str(source_dir),
        "output_m3_dir": str(output_dir),
        "copied_state_count": copied_state_count,
        "preserved_downloaded_state_count": preserved_downloaded,
        "refreshed_state_count": len(refreshed_work_ids),
        "refreshed_work_ids": sorted(set(refreshed_work_ids)),
        "refresh_reason_counts": dict(sorted(reason_counts.items())),
        "hard_failure_code_counts": dict(sorted(hard_code_counts.items())),
        "hard_failure_resolver_counts": dict(sorted(hard_resolver_counts.items())),
        "suppressed_url_count": int(ledger["suppressed_url_count"]),
        "retry_failed": bool(retry_failed),
        "retry_access_misses": bool(retry_access_misses),
        "previous_context_known": not legacy_unknown_context,
        "context_changed": context_changed,
        "resolver_context": current_context,
        "secrets_persisted": False,
        "paywall_bypass_enabled": False,
    }
    _write_json_atomic(output_dir / "access_recovery_report.json", report)
    return report
