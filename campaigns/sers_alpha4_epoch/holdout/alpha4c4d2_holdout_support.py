from __future__ import annotations

import hashlib
import json
import subprocess
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


from campaigns.sers_alpha4_epoch.paths import PROJECT_ROOT as ROOT
DATA_ROOT = ROOT / "data_sers"


class Alpha4c4d2Error(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise Alpha4c4d2Error(f"Required JSON missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Alpha4c4d2Error(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise Alpha4c4d2Error(
                f"Expected JSONL object: {path}:{line_no}"
            )
        rows.append(value)
    return rows


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_blob(path: Path) -> str:
    proc = subprocess.run(
        ["git", "hash-object", str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise Alpha4c4d2Error(proc.stderr.strip())
    return proc.stdout.strip()


def require_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise Alpha4c4d2Error(
            f"{label}: {observed!r} != {expected!r}"
        )


def quality_snapshot(active: Mapping[str, Any]) -> dict[str, Any]:
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


def manual_decisions(
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
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id:
            raise Alpha4c4d2Error(
                "Manual resolution decision lacks candidate_id."
            )
        manual[candidate_id] = {
            "candidate_id": candidate_id,
            "decision": decision,
            "approved": approved,
            "reviewer": reviewer,
            "canonical_id": str(row.get("canonical_id") or ""),
        }
    return manual


def snapshot_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path.relative_to(ROOT)),
            "present": False,
            "sha256": "",
        }
    return {
        "path": str(path.relative_to(ROOT)),
        "present": True,
        "sha256": sha256(path),
    }


def resolve_strict_source(paper_id: str) -> dict[str, Any]:
    paper_root = DATA_ROOT / "extracted" / paper_id
    pointer_path = paper_root / "latest_run.json"
    if not pointer_path.exists():
        raise Alpha4c4d2Error(
            f"{paper_id}: latest_run.json missing."
        )

    pointer = read_json(pointer_path)
    raw_run_dir = str(pointer.get("run_directory") or "").strip()
    if not raw_run_dir:
        raise Alpha4c4d2Error(
            f"{paper_id}: latest_run.json lacks run_directory."
        )
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()

    run_json_path = run_dir / "run.json"
    active_path = run_dir / "active_chunks.json"
    run_json = read_json(run_json_path)
    active = read_json(active_path)

    require_equal(
        f"{paper_id} active paper_id",
        active.get("paper_id"),
        paper_id,
    )
    require_equal(
        f"{paper_id} run_id consistency",
        active.get("run_id"),
        run_json.get("run_id"),
    )

    quality = quality_snapshot(active)
    status = quality["graph_materialization_status"]
    if status == QUALITY_REJECTED:
        raise Alpha4c4d2Error(
            f"{paper_id}: frozen Strict source is REJECTED. "
            f"Reason={quality['classification_reason']}; "
            f"active={quality['active_chunk_count']}; "
            f"quarantined={quality['quarantined_chunk_count']}; "
            f"failed={quality['failed_chunk_count']}; "
            f"coverage={quality['source_token_coverage']!r}"
        )
    if not quality["positive_evidence_queries_allowed"]:
        raise Alpha4c4d2Error(
            f"{paper_id}: positive-evidence queries are not allowed."
        )

    run_id = str(
        run_json.get("run_id") or pointer.get("run_id") or ""
    ).strip()
    if not run_id:
        raise Alpha4c4d2Error(f"{paper_id}: run_id unresolved.")
    attempt_id = str(
        active.get("attempt_id")
        or run_json.get("attempt_id")
        or ""
    ).strip()

    chunks = active.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise Alpha4c4d2Error(
            f"{paper_id}: no active strict-valid chunks."
        )

    chunk_inputs: list[dict[str, Any]] = []
    for row in chunks:
        output = Path(str(row.get("output_path") or "")).expanduser()
        if not output.is_absolute():
            output = (ROOT / output).resolve()
        if not output.exists():
            raise Alpha4c4d2Error(
                f"{paper_id}: active chunk missing: {output}"
            )
        chunk_inputs.append(
            {
                "chunk_id": str(row.get("chunk_id") or ""),
                "path": str(output),
                "sha256": sha256(output),
            }
        )

    locator = run_dir / "locator_index.json"
    return {
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
            "path": str(pointer_path.relative_to(ROOT)),
            "sha256": sha256(pointer_path),
        },
        "run_json": {
            "path": str(run_json_path.relative_to(ROOT)),
            "sha256": sha256(run_json_path),
        },
        "active_chunks": {
            "path": str(active_path.relative_to(ROOT)),
            "sha256": sha256(active_path),
        },
        "locator_index": snapshot_optional(locator),
        "chunk_inputs": chunk_inputs,
    }


def verify_strict_source_unchanged(
    source: Mapping[str, Any],
) -> None:
    for key in ("latest_run", "run_json", "active_chunks"):
        row = source[key]
        path = ROOT / str(row["path"])
        require_equal(
            f"{source['paper_id']} {key} SHA256",
            sha256(path),
            row["sha256"],
        )

    # locator_index.json is deliberately NOT part of the immutable Strict
    # source set. scripts.build_paper_graph refreshes document assets and may
    # create/rewrite this run-level index deterministically before loading it
    # for provenance backfill.  The immutable inputs are latest_run/run.json,
    # active_chunks.json, and the active strict-valid chunk JSONs.  Downstream
    # holdout reproducibility is frozen by the final canonical GraphML and
    # resolution-decision SHA lock.
    for row in source["chunk_inputs"]:
        path = Path(str(row["path"]))
        if not path.exists():
            raise Alpha4c4d2Error(
                f"{source['paper_id']}: chunk disappeared: {path}"
            )
        require_equal(
            f"{source['paper_id']} chunk {row['chunk_id']} SHA256",
            sha256(path),
            row["sha256"],
        )

    active = read_json(
        ROOT / str(source["active_chunks"]["path"])
    )
    require_equal(
        f"{source['paper_id']} extraction quality",
        quality_snapshot(active),
        source["extraction_quality"],
    )


def canonical_snapshot(paper_id: str) -> dict[str, Any]:
    paper_root = DATA_ROOT / "extracted" / paper_id
    graph_path = paper_root / f"{paper_id}.graphml"
    decisions_path = paper_root / "resolution" / "decisions.jsonl"

    if not graph_path.exists():
        return {
            "paper_id": paper_id,
            "canonical_present": False,
            "canonical_path": str(graph_path.relative_to(ROOT)),
            "measurement_merge_invariant_id": "",
            "measurement_xor_issue_count": None,
            "canonical_sha256": "",
            "resolution_decisions": snapshot_optional(decisions_path),
        }

    graph = nx.read_graphml(graph_path, force_multigraph=True)
    return {
        "paper_id": paper_id,
        "canonical_present": True,
        "canonical_path": str(graph_path.relative_to(ROOT)),
        "canonical_sha256": sha256(graph_path),
        "canonical_nodes": graph.number_of_nodes(),
        "canonical_edges": graph.number_of_edges(),
        "domain_profile_id": str(
            graph.graph.get("domain_profile_id", "")
        ),
        "measurement_merge_invariant_id": str(
            graph.graph.get("measurement_merge_invariant_id", "")
        ),
        "measurement_xor_issue_count": len(
            measurement_value_payload_issues(graph)
        ),
        "extraction_quality_status": str(
            graph.graph.get("extraction_quality_status", "")
        ),
        "resolution_decisions": snapshot_optional(decisions_path),
    }


def verify_locked_input_record(
    paper_id: str,
    record: Mapping[str, Any],
) -> None:
    canonical_path = ROOT / str(record["canonical_path"])
    if not canonical_path.exists():
        raise Alpha4c4d2Error(
            f"{paper_id}: locked canonical disappeared."
        )
    require_equal(
        f"{paper_id} canonical SHA256",
        sha256(canonical_path),
        record["canonical_sha256"],
    )

    graph = nx.read_graphml(canonical_path, force_multigraph=True)
    require_equal(
        f"{paper_id} Measurement merge invariant",
        str(graph.graph.get("measurement_merge_invariant_id", "")),
        MEASUREMENT_MERGE_INVARIANT_ID,
    )
    issues = measurement_value_payload_issues(graph)
    if issues:
        raise Alpha4c4d2Error(
            f"{paper_id}: locked canonical Measurement XOR issues: "
            f"{issues[:5]!r}"
        )

    decision = record["resolution_decisions"]
    path = ROOT / str(decision["path"])
    observed_present = path.exists()
    require_equal(
        f"{paper_id} resolution decisions presence",
        observed_present,
        bool(decision["present"]),
    )
    if observed_present:
        require_equal(
            f"{paper_id} resolution decisions SHA256",
            sha256(path),
            decision["sha256"],
        )
