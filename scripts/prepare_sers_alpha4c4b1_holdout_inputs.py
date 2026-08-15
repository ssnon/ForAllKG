from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import networkx as nx

from dac_her.extraction_policy import ExtractionPolicy
from dac_her.extraction_quality import (
    QUALITY_PARTIAL_CRITICAL,
    QUALITY_REJECTED,
    quality_from_active_payload,
)
from dac_her.measurement_merge_invariants import (
    MEASUREMENT_MERGE_INVARIANT_ID,
    measurement_value_payload_issues,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "configs"
    / "heldout"
    / "sers_alpha4c4b21_input_refreeze.json"
)


class CanonicalInputRefreezeError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CanonicalInputRefreezeError(
            f"Expected JSON object: {path}"
        )
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise CanonicalInputRefreezeError(
                f"Expected JSON object: {path}:{line_number}"
            )
        rows.append(value)
    return rows


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob(path: Path) -> str:
    result = subprocess.run(
        ["git", "hash-object", str(path)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise CanonicalInputRefreezeError(
            f"Cannot hash {path}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _require_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise CanonicalInputRefreezeError(
            f"{label} mismatch: {observed!r} != {expected!r}"
        )


def _snapshot_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "present": False,
            "sha256": "",
        }
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "present": True,
        "sha256": _sha256(path),
    }


def _manual_decisions(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    manual: dict[str, dict[str, Any]] = {}
    for row in rows:
        reviewer = str(row.get("reviewer") or "").strip()
        decision = str(
            row.get("decision") or "unreviewed"
        ).strip()
        approved = bool(row.get("approved", False))
        automatic = reviewer == "automatic_registry_rule"
        untouched = (
            not reviewer
            and decision == "unreviewed"
            and not approved
        )
        if automatic or untouched:
            continue

        candidate_id = str(
            row.get("candidate_id") or ""
        ).strip()
        if not candidate_id:
            raise CanonicalInputRefreezeError(
                "Manual resolution decision lacks candidate_id."
            )
        manual[candidate_id] = {
            "candidate_id": candidate_id,
            "decision": decision,
            "approved": approved,
            "reviewer": reviewer,
            "canonical_id": str(
                row.get("canonical_id") or ""
            ),
        }
    return manual


def _verify_manual_decisions_preserved(
    *,
    paper_id: str,
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
) -> None:
    before = _manual_decisions(before_rows)
    after = _manual_decisions(after_rows)
    _require_equal(
        f"{paper_id} manual resolution decisions",
        after,
        before,
    )


def _quality_snapshot(
    active: Mapping[str, Any],
) -> dict[str, Any]:
    quality = quality_from_active_payload(
        active,
        policy=ExtractionPolicy(),
    )
    return {
        "graph_materialization_status": str(
            quality.get("graph_materialization_status", "")
        ),
        "strict_complete": bool(
            quality.get("strict_complete", False)
        ),
        "graph_usable_by_default": bool(
            quality.get("graph_usable_by_default", False)
        ),
        "graph_usable_with_explicit_override": bool(
            quality.get(
                "graph_usable_with_explicit_override",
                False,
            )
        ),
        "positive_evidence_queries_allowed": bool(
            quality.get(
                "positive_evidence_queries_allowed",
                False,
            )
        ),
        "coverage_sensitive_queries_allowed": bool(
            quality.get(
                "coverage_sensitive_queries_allowed",
                False,
            )
        ),
        "absence_claims_allowed": bool(
            quality.get("absence_claims_allowed", False)
        ),
        "classification_reason": str(
            quality.get("classification_reason", "")
        ),
        "active_chunk_count": int(
            quality.get("active_chunk_count", 0)
        ),
        "quarantined_chunk_count": int(
            quality.get("quarantined_chunk_count", 0)
        ),
        "failed_chunk_count": int(
            quality.get("failed_chunk_count", 0)
        ),
        "source_token_coverage": quality.get(
            "source_token_coverage"
        ),
        "quarantine_token_fraction": quality.get(
            "quarantine_token_fraction"
        ),
        "coverage_exact": bool(
            quality.get("coverage_exact", False)
        ),
    }


def _resolve_strict_source(
    *,
    paper_id: str,
    data_root: Path,
) -> dict[str, Any]:
    paper_root = data_root / "extracted" / paper_id
    pointer_path = paper_root / "latest_run.json"
    if not pointer_path.exists():
        raise CanonicalInputRefreezeError(
            f"Frozen Strict extraction missing for {paper_id}: "
            f"{pointer_path}"
        )

    pointer = _read_json(pointer_path)
    run_directory_raw = str(
        pointer.get("run_directory") or ""
    ).strip()
    if not run_directory_raw:
        raise CanonicalInputRefreezeError(
            f"{paper_id} latest_run.json lacks run_directory."
        )
    run_dir = Path(run_directory_raw).expanduser()
    if not run_dir.is_absolute():
        run_dir = (PROJECT_ROOT / run_dir).resolve()
    if not run_dir.exists():
        raise CanonicalInputRefreezeError(
            f"{paper_id} frozen Strict run directory missing: {run_dir}"
        )

    run_json_path = run_dir / "run.json"
    active_path = run_dir / "active_chunks.json"
    if not run_json_path.exists() or not active_path.exists():
        raise CanonicalInputRefreezeError(
            f"{paper_id} Strict run lacks run.json/active_chunks.json."
        )

    run_json = _read_json(run_json_path)
    active = _read_json(active_path)
    quality = _quality_snapshot(active)
    status = quality["graph_materialization_status"]

    # The active_chunks.complete field is diagnostic only. The frozen
    # extraction-quality contract is authoritative for graph materialization.
    if status == QUALITY_REJECTED:
        raise CanonicalInputRefreezeError(
            f"{paper_id} frozen Strict source is REJECTED by the "
            "frozen extraction-quality contract. "
            f"Reason: {quality['classification_reason']} "
            f"(active={quality['active_chunk_count']}, "
            f"quarantined={quality['quarantined_chunk_count']}, "
            f"failed={quality['failed_chunk_count']}, "
            f"coverage={quality['source_token_coverage']!r}). "
            "Do not force canonical refreeze from a rejected source."
        )

    if not quality["positive_evidence_queries_allowed"]:
        raise CanonicalInputRefreezeError(
            f"{paper_id} frozen Strict source does not permit "
            "positive-evidence queries."
        )
    _require_equal(
        f"{paper_id} active paper_id",
        active.get("paper_id"),
        paper_id,
    )
    _require_equal(
        f"{paper_id} active/run run_id",
        active.get("run_id"),
        run_json.get("run_id"),
    )

    run_id = str(
        run_json.get("run_id")
        or pointer.get("run_id")
        or ""
    ).strip()
    if not run_id:
        raise CanonicalInputRefreezeError(
            f"{paper_id} cannot resolve Strict run_id."
        )
    attempt_id = str(
        active.get("attempt_id")
        or run_json.get("attempt_id")
        or ""
    ).strip()

    chunks = active.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise CanonicalInputRefreezeError(
            f"{paper_id} active_chunks has no chunk records."
        )

    chunk_inputs: list[dict[str, Any]] = []
    for row in chunks:
        if not isinstance(row, dict):
            raise CanonicalInputRefreezeError(
                f"{paper_id} has invalid active chunk record."
            )
        output = Path(str(row.get("output_path") or "")).expanduser()
        if not output.is_absolute():
            output = (PROJECT_ROOT / output).resolve()
        if not output.exists():
            raise CanonicalInputRefreezeError(
                f"{paper_id} active chunk JSON missing: {output}"
            )
        chunk_inputs.append(
            {
                "chunk_id": str(row.get("chunk_id") or ""),
                "path": str(output),
                "sha256": _sha256(output),
            }
        )

    source = {
        "paper_id": paper_id,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "run_directory": str(run_dir),
        "active_payload_complete_flag": bool(
            active.get("complete", False)
        ),
        "extraction_quality": quality,
        "requires_allow_incomplete": (
            status == QUALITY_PARTIAL_CRITICAL
        ),
        "latest_run": {
            "path": str(pointer_path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(pointer_path),
        },
        "run_json": {
            "path": str(run_json_path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(run_json_path),
        },
        "active_chunks": {
            "path": str(active_path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(active_path),
        },
        "chunk_inputs": chunk_inputs,
    }

    locator = run_dir / "locator_index.json"
    source["locator_index"] = _snapshot_optional(locator)
    return source


def _verify_strict_source_unchanged(
    source: Mapping[str, Any],
) -> None:
    for key in ("latest_run", "run_json", "active_chunks"):
        row = source[key]
        path = PROJECT_ROOT / str(row["path"])
        _require_equal(
            f"{source['paper_id']} {key} SHA256",
            _sha256(path),
            row["sha256"],
        )

    locator = source.get("locator_index", {})
    locator_path = PROJECT_ROOT / str(locator.get("path", ""))
    if bool(locator.get("present", False)):
        if not locator_path.exists():
            raise CanonicalInputRefreezeError(
                f"{source['paper_id']} locator_index disappeared."
            )
        _require_equal(
            f"{source['paper_id']} locator_index SHA256",
            _sha256(locator_path),
            locator.get("sha256"),
        )
    else:
        if locator_path.exists():
            raise CanonicalInputRefreezeError(
                f"{source['paper_id']} locator_index appeared during refreeze."
            )

    for row in source["chunk_inputs"]:
        path = Path(str(row["path"]))
        if not path.exists():
            raise CanonicalInputRefreezeError(
                f"{source['paper_id']} chunk input disappeared: {path}"
            )
        _require_equal(
            f"{source['paper_id']} chunk {row['chunk_id']} SHA256",
            _sha256(path),
            row["sha256"],
        )


def _verify_protocol(protocol: Mapping[str, Any]) -> None:
    _require_equal(
        "phase",
        protocol.get("phase"),
        "alpha4c.4b.2.1",
    )
    _require_equal(
        "refreeze semantics",
        protocol.get("refreeze_semantics_id"),
        "canonical_input_epoch_refreeze_v2_alpha4c4b21",
    )
    _require_equal(
        "new LLM extraction allowed",
        protocol.get("new_llm_extraction_allowed"),
        False,
    )
    _require_equal(
        "trend outputs allowed",
        protocol.get("trend_outputs_allowed"),
        False,
    )

    for rel, expected in protocol[
        "frozen_implementation_blobs"
    ].items():
        path = PROJECT_ROOT / str(rel)
        if not path.exists():
            raise CanonicalInputRefreezeError(
                f"Frozen implementation file missing: {rel}"
            )
        _require_equal(
            f"implementation blob {rel}",
            _git_blob(path),
            expected,
        )

    b1_path = PROJECT_ROOT / str(
        protocol["source_holdout_protocol"]["path"]
    )
    _require_equal(
        "alpha4c.4b.1 protocol SHA256",
        _sha256(b1_path),
        protocol["source_holdout_protocol"]["sha256"],
    )
    b1 = _read_json(b1_path)
    _require_equal(
        "holdout papers",
        b1.get("holdout_papers"),
        protocol.get("holdout_papers"),
    )
    _require_equal(
        "split SHA256",
        b1.get("source_split_protocol", {}).get("split_sha256"),
        protocol.get("source_split_sha256"),
    )
    _require_equal(
        "measurement merge semantics",
        MEASUREMENT_MERGE_INVARIANT_ID,
        protocol["target_measurement_merge_invariant_id"],
    )

    # Refreeze is only legal before the first successful 4c.4b.1 input lock
    # and before any Trend holdout scientific output.
    input_lock = PROJECT_ROOT / str(
        protocol["forbidden_if_exists"]["canonical_input_lock"]
    )
    manifest = PROJECT_ROOT / str(
        protocol["forbidden_if_exists"]["holdout_manifest"]
    )
    report = PROJECT_ROOT / str(
        protocol["forbidden_if_exists"]["holdout_report"]
    )
    forbidden = [
        path
        for path in (input_lock, manifest, report)
        if path.exists()
    ]
    if forbidden:
        raise CanonicalInputRefreezeError(
            "Refreeze is forbidden after holdout input lock/scientific "
            f"campaign start: {forbidden!r}"
        )


def _copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refreeze the alpha4c.4b.1 holdout canonical GraphML inputs "
            "from their existing frozen Strict runs under the already-frozen "
            "Measurement merge invariant. No LLM or Trend stage is invoked."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Verify all 10 frozen Strict sources and implementation hashes "
            "without rebuilding any canonical graph."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_path = (
        args.protocol
        if args.protocol.is_absolute()
        else PROJECT_ROOT / args.protocol
    ).resolve()
    protocol = _read_json(protocol_path)
    _verify_protocol(protocol)

    data_root = PROJECT_ROOT / str(protocol["data_root"])
    sources = {
        paper_id: _resolve_strict_source(
            paper_id=str(paper_id),
            data_root=data_root,
        )
        for paper_id in protocol["holdout_papers"]
    }
    print("alpha4c.4b.2 frozen Strict source preflight: PASS")
    print("Papers:", len(sources))
    for paper_id, source in sources.items():
        suffix = (
            f" attempt={source['attempt_id']}"
            if source["attempt_id"]
            else ""
        )
        quality = source["extraction_quality"]
        print(
            f" - {paper_id}: run={source['run_id']}{suffix} "
            f"status={quality['graph_materialization_status']} "
            f"active_complete_flag="
            f"{source['active_payload_complete_flag']} "
            f"active={quality['active_chunk_count']} "
            f"quarantined={quality['quarantined_chunk_count']} "
            f"failed={quality['failed_chunk_count']} "
            f"coverage={quality['source_token_coverage']!r}"
            + (
                " [--allow-incomplete]"
                if source["requires_allow_incomplete"]
                else ""
            )
        )

    if args.preflight_only:
        print(
            "No canonical graph, resolution file, or Trend output was "
            "modified in --preflight-only mode."
        )
        return 0

    evaluation_root = PROJECT_ROOT / str(protocol["evaluation_root"])
    pre_root = evaluation_root / "pre_refreeze"
    post_root = evaluation_root / "post_refreeze"
    report_path = evaluation_root / "canonical_input_refreeze_report.json"

    if report_path.exists():
        existing = _read_json(report_path)
        if existing.get("status") == "complete":
            raise CanonicalInputRefreezeError(
                f"Refreeze report already complete: {report_path}. "
                "Do not rerun the migration."
            )

    records: list[dict[str, Any]] = []

    for index, paper_id in enumerate(
        map(str, protocol["holdout_papers"]),
        start=1,
    ):
        source = sources[paper_id]
        _verify_strict_source_unchanged(source)

        paper_root = data_root / "extracted" / paper_id
        canonical = paper_root / f"{paper_id}.graphml"
        decisions = paper_root / "resolution" / "decisions.jsonl"

        before_decision_rows = _read_jsonl(decisions)
        before_manual = _manual_decisions(before_decision_rows)

        before = {
            "canonical": _snapshot_optional(canonical),
            "resolution_decisions": _snapshot_optional(decisions),
            "manual_resolution_decision_count": len(before_manual),
        }

        _copy_if_exists(
            canonical,
            pre_root / "canonical" / f"{paper_id}.graphml",
        )
        _copy_if_exists(
            decisions,
            pre_root / "resolution" / f"{paper_id}.decisions.jsonl",
        )

        command = [
            sys.executable,
            "-m",
            "scripts.build_paper_graph",
            "--paper-id",
            paper_id,
            "--config",
            str(protocol["paper_config"]),
            "--domain-profile",
            str(protocol["domain_profile"]),
            "--data-root",
            str(protocol["data_root"]),
            "--run-id",
            str(source["run_id"]),
        ]
        if source["attempt_id"]:
            command.extend(
                ["--attempt-id", str(source["attempt_id"])]
            )
        if source["requires_allow_incomplete"]:
            command.append("--allow-incomplete")

        print()
        print(
            f"[{index}/{len(sources)}] {paper_id} canonical refreeze"
        )
        print("$", " ".join(command), flush=True)
        result = subprocess.run(command, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            raise CanonicalInputRefreezeError(
                f"{paper_id} build_paper_graph failed with "
                f"exit code {result.returncode}."
            )

        _verify_strict_source_unchanged(source)

        if not canonical.exists():
            raise CanonicalInputRefreezeError(
                f"{paper_id} canonical GraphML missing after refreeze."
            )
        graph = nx.read_graphml(
            canonical,
            force_multigraph=True,
        )
        invariant = str(
            graph.graph.get("measurement_merge_invariant_id", "")
        )
        _require_equal(
            f"{paper_id} Measurement merge invariant",
            invariant,
            protocol["target_measurement_merge_invariant_id"],
        )

        xor_issues = measurement_value_payload_issues(graph)
        if xor_issues:
            raise CanonicalInputRefreezeError(
                f"{paper_id} canonical Measurement XOR failure: "
                f"{xor_issues[:5]!r}"
            )

        after_decision_rows = _read_jsonl(decisions)
        _verify_manual_decisions_preserved(
            paper_id=paper_id,
            before_rows=before_decision_rows,
            after_rows=after_decision_rows,
        )
        after_manual = _manual_decisions(after_decision_rows)

        after = {
            "canonical": _snapshot_optional(canonical),
            "resolution_decisions": _snapshot_optional(decisions),
            "manual_resolution_decision_count": len(after_manual),
            "canonical_nodes": graph.number_of_nodes(),
            "canonical_edges": graph.number_of_edges(),
            "measurement_merge_invariant_id": invariant,
            "measurement_xor_issue_count": 0,
        }

        _copy_if_exists(
            canonical,
            post_root / "canonical" / f"{paper_id}.graphml",
        )
        _copy_if_exists(
            decisions,
            post_root / "resolution" / f"{paper_id}.decisions.jsonl",
        )

        records.append(
            {
                "paper_id": paper_id,
                "strict_source": source,
                "before": before,
                "after": after,
                "build_command": command,
                "strict_source_unchanged": True,
                "manual_resolution_decisions_preserved": True,
            }
        )

    # One final source check after all graph materializations.
    for source in sources.values():
        _verify_strict_source_unchanged(source)

    report = {
        "phase": "alpha4c.4b.2.1",
        "refreeze_id": protocol["refreeze_id"],
        "refreeze_semantics_id": protocol["refreeze_semantics_id"],
        "status": "complete",
        "completed_at_utc": _now(),
        "llm_calls_performed": False,
        "trend_outputs_generated": False,
        "source_split_sha256": protocol["source_split_sha256"],
        "holdout_papers": list(protocol["holdout_papers"]),
        "target_measurement_merge_invariant_id":
            protocol["target_measurement_merge_invariant_id"],
        "paper_count": len(records),
        "all_strict_sources_unchanged": True,
        "all_manual_resolution_decisions_preserved": True,
        "all_measurement_xor_clean": True,
        "records": records,
        "next_step": (
            "Run scripts.run_sers_alpha4c4b1_trend_holdout "
            "--preflight-only to create the persistent canonical input lock."
        ),
    }
    _atomic_json(report_path, report)

    print()
    print("alpha4c.4b.2 canonical input refreeze: PASS")
    print("Papers:", len(records))
    print("LLM calls performed: False")
    print("Trend outputs generated: False")
    print("Strict sources unchanged: True")
    print("Manual resolution decisions preserved: True")
    print("Measurement XOR clean: True")
    print("Report:", report_path)
    print(
        "Next: python -m scripts.run_sers_alpha4c4b1_trend_holdout "
        "--preflight-only"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
