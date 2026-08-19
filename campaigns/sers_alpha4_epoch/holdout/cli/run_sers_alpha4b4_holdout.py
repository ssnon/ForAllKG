from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import networkx as nx


from campaigns.sers_alpha4_epoch.paths import PROJECT_ROOT
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "heldout" / "sers_alpha4b4_protocol.json"


class FrozenContractViolation(RuntimeError):
    pass


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob(root: Path, relative_path: str) -> str:
    path = root / relative_path
    if not path.exists():
        raise FrozenContractViolation(f"Frozen file is missing: {relative_path}")
    completed = subprocess.run(
        ["git", "hash-object", str(path)],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(value) for value in command)


def run_command(
    *,
    command: list[str],
    root: Path,
    log_path: Path,
    dry_run: bool,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$", command_text(command))
    if dry_run:
        log_path.write_text("$ " + command_text(command) + "\n[DRY RUN]\n", encoding="utf-8")
        return 0
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + command_text(command) + "\n\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        return process.wait()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def assert_no_manual_resolution(paper_root: Path) -> None:
    decisions = paper_root / "resolution" / "decisions.jsonl"
    manual: list[dict[str, Any]] = []
    for row in read_jsonl(decisions):
        reviewer = str(row.get("reviewer") or "").strip()
        decision = str(row.get("decision") or "unreviewed").strip()
        approved = bool(row.get("approved", False))
        automatic = reviewer == "automatic_registry_rule"
        untouched = not reviewer and decision == "unreviewed" and not approved
        if not automatic and not untouched:
            manual.append({
                "candidate_id": row.get("candidate_id"),
                "decision": decision,
                "approved": approved,
                "reviewer": reviewer,
            })
    if manual:
        raise FrozenContractViolation(
            "Held-out canonical graph has manual resolution decisions. "
            f"Examples: {manual[:5]!r}"
        )


def resolve_latest_strict_run(root: Path, paper_root: Path) -> tuple[str, Path, dict[str, Any]]:
    pointer_path = paper_root / "latest_run.json"
    if not pointer_path.exists():
        raise FileNotFoundError(
            f"Frozen strict extraction is missing for {paper_root.name}: {pointer_path}. "
            "alpha4b.4 does not create a new strict extraction; use the frozen upstream run."
        )
    pointer = read_json(pointer_path)
    run_directory = Path(str(pointer.get("run_directory", ""))).expanduser()
    if not run_directory.is_absolute():
        run_directory = (root / run_directory).resolve()
    if not run_directory.exists():
        raise FileNotFoundError(run_directory)
    run_json_path = run_directory / "run.json"
    active_path = run_directory / "active_chunks.json"
    run_json = read_json(run_json_path)
    if active_path.exists():
        active = read_json(active_path)
        if not bool(active.get("complete", False)):
            raise FrozenContractViolation(
                f"Frozen strict extraction is not complete for {paper_root.name}."
            )
    run_id = str(run_json.get("run_id") or pointer.get("run_id") or "").strip()
    if not run_id:
        raise FrozenContractViolation(f"Cannot resolve strict run_id for {paper_root.name}.")
    snapshot = {
        "latest_run": {"path": str(pointer_path.relative_to(root)), "sha256": sha256_file(pointer_path)},
        "run_json": {"path": str(run_json_path.relative_to(root)), "sha256": sha256_file(run_json_path)},
        "active_chunks": (
            {"path": str(active_path.relative_to(root)), "sha256": sha256_file(active_path)}
            if active_path.exists() else None
        ),
    }
    return run_id, run_directory, snapshot


def validate_protocol(protocol: dict[str, Any]) -> None:
    calibration = [str(v) for v in protocol.get("calibration_papers", [])]
    holdout = [str(v) for v in protocol.get("holdout_papers", [])]
    if not calibration or not holdout:
        raise ValueError("Calibration and holdout papers must both be non-empty.")
    overlap = sorted(set(calibration) & set(holdout))
    if overlap:
        raise ValueError(f"Calibration/holdout overlap is forbidden: {overlap!r}")
    acceptance = protocol.get("acceptance_policy", {})
    forbidden_targets = (
        "minimum_numeric_ranking_allowed",
        "minimum_same_protocol_pairs",
        "maximum_unknown_contexts",
        "maximum_different_protocol_pairs",
        "minimum_metric_definition_known",
    )
    for key in forbidden_targets:
        if acceptance.get(key) is not None:
            raise ValueError(
                f"Holdout distribution target {key!r} must remain null. "
                "Counts are observations, not optimization targets."
            )


def verify_frozen_blobs(root: Path, protocol: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative_path, expected in sorted(protocol["frozen_git_blobs"].items()):
        actual = git_blob(root, relative_path)
        observed[relative_path] = actual
        if actual != expected:
            raise FrozenContractViolation(
                "Frozen implementation drift detected before/during holdout: "
                f"{relative_path}: expected {expected}, observed {actual}. "
                "Do not patch in place. Diagnose the invariant, rerun calibration, "
                "then start a new holdout campaign from the beginning."
            )
    return observed


def verify_runtime_semantics(protocol: dict[str, Any]) -> dict[str, str]:
    from dac_her.domains.registry import get_domain_profile
    from dac_her.domains.comparison_registry import get_comparison_adapter
    from dac_her.domains.reproducibility_registry import get_reproducibility_adapter
    from dac_her.domains.metric_definition_registry import get_metric_definition_adapter
    from dac_her.quality_aware_comparison import QUALITY_AWARE_NUMERIC_GATE_SEMANTICS_ID

    profile = get_domain_profile(str(protocol["domain_profile"]))
    comparison = get_comparison_adapter(profile)
    reproducibility = get_reproducibility_adapter(profile)
    metric_definition = get_metric_definition_adapter(profile)
    observed = {
        "projection": str(profile.projection.semantics_id),
        "corpus": str(profile.corpus.semantics_id),
        "comparison": str(comparison.semantics_id),
        "method": str(comparison.method_semantics.semantics_id if comparison.method_semantics else ""),
        "reproducibility": str(reproducibility.semantics_id),
        "metric_definition": str(metric_definition.semantics_id),
        "quality_gate": str(QUALITY_AWARE_NUMERIC_GATE_SEMANTICS_ID),
    }
    expected = {str(k): str(v) for k, v in protocol["frozen_semantics"].items()}
    if observed != expected:
        raise FrozenContractViolation(
            f"Frozen semantic IDs drifted: expected={expected!r}, observed={observed!r}."
        )
    return observed


def _assert_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise FrozenContractViolation(
            f"Calibration freeze mismatch for {label}: {observed!r} != {expected!r}."
        )


def verify_calibration_freeze(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    freeze = protocol["calibration_freeze"]
    expected = freeze["expected"]
    paths = {
        name: (root / relative).resolve()
        for name, relative in freeze.items()
        if name != "expected"
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Frozen calibration artifact is missing ({name}): {path}"
            )
    corpus = read_json(paths["corpus_manifest"])
    repro = read_json(paths["reproducibility_summary"])
    metric = read_json(paths["metric_definition_summary"])
    comparison = read_json(paths["comparison_summary"])

    _assert_equal("calibration papers/corpus", corpus.get("paper_ids"), expected["calibration_papers"])
    _assert_equal("corpus semantics", corpus.get("corpus_semantics_id"), expected["corpus_semantics_id"])
    _assert_equal("repro papers", repro.get("paper_ids"), expected["calibration_papers"])
    _assert_equal("repro semantics", repro.get("reproducibility_semantics_id"), expected["reproducibility_semantics_id"])
    _assert_equal("repro evidence count", repro.get("evidence_count"), expected["reproducibility_evidence_count"])
    _assert_equal("repro structural gate", repro.get("structural_gate"), expected["reproducibility_structural_gate"])
    _assert_equal("metric papers", metric.get("paper_ids"), expected["calibration_papers"])
    _assert_equal("metric semantics", metric.get("metric_definition_semantics_id"), expected["metric_definition_semantics_id"])
    _assert_equal("metric context count", metric.get("context_count"), expected["metric_definition_context_count"])
    statuses = metric.get("definition_status_counts", {})
    _assert_equal("metric known count", statuses.get("known", 0), expected["metric_definition_known_count"])
    _assert_equal("metric unknown count", statuses.get("unknown", 0), expected["metric_definition_unknown_count"])
    _assert_equal("metric structural gate", metric.get("structural_gate"), expected["metric_definition_structural_gate"])
    _assert_equal("comparison papers", comparison.get("paper_ids"), expected["calibration_papers"])
    _assert_equal("comparison semantics", comparison.get("comparison_semantics_id"), expected["comparison_semantics_id"])
    _assert_equal("quality gate semantics", comparison.get("quality_gate_semantics_id"), expected["quality_gate_semantics_id"])
    _assert_equal("comparison context count", comparison.get("context_count"), expected["comparison_context_count"])
    _assert_equal("comparison assessment count", comparison.get("assessment_count"), expected["comparison_assessment_count"])
    _assert_equal("protocol counts", comparison.get("protocol_comparability_counts"), expected["protocol_comparability_counts"])
    _assert_equal("metric compatibility counts", comparison.get("metric_definition_compatibility_counts"), expected["metric_definition_compatibility_counts"])
    _assert_equal("ranking-relevant metric count", comparison.get("metric_definition_ranking_relevant_assessment_count"), expected["metric_definition_ranking_relevant_assessment_count"])
    _assert_equal("ranking-relevant metric passes", comparison.get("metric_definition_ranking_relevant_gate_pass_count"), expected["metric_definition_ranking_relevant_gate_pass_count"])
    _assert_equal("numeric ranking allowed", comparison.get("numeric_ranking_allowed_count"), expected["numeric_ranking_allowed_count"])
    _assert_equal("global concentration leak guard", comparison.get("global_entity_concentration_consumed"), expected["global_entity_concentration_consumed"])
    _assert_equal("missing context semantics", comparison.get("missing_context_is_not_quarantine"), expected["missing_context_is_not_quarantine"])
    _assert_equal("comparison structural gate", comparison.get("passes_structural_gate"), expected["comparison_structural_gate"])

    return {
        name: {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }


def snapshot_holdout_inputs(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    data_root = root / str(protocol["data_root"])
    snapshots: dict[str, Any] = {}
    for paper_id in protocol["holdout_papers"]:
        paper_root = data_root / "extracted" / str(paper_id)
        canonical = paper_root / f"{paper_id}.graphml"
        if not canonical.exists():
            raise FileNotFoundError(
                f"Held-out canonical graph is missing: {canonical}. "
                "This holdout intentionally adopts the pre-existing frozen strict extraction."
            )
        assert_no_manual_resolution(paper_root)
        graph = nx.read_graphml(canonical, force_multigraph=True)
        domain = str(graph.graph.get("domain_profile_id", ""))
        if domain and domain != protocol["domain_profile"]:
            raise FrozenContractViolation(
                f"{paper_id} canonical domain mismatch: {domain!r}."
            )
        run_id, run_dir, strict_snapshot = resolve_latest_strict_run(root, paper_root)
        decisions = paper_root / "resolution" / "decisions.jsonl"
        snapshots[str(paper_id)] = {
            "canonical_graph": {
                "path": str(canonical.relative_to(root)),
                "sha256": sha256_file(canonical),
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
            },
            "strict_run_id": run_id,
            "strict_run_directory": str(run_dir.relative_to(root)),
            "strict_run_files": strict_snapshot,
            "resolution_decisions": (
                {
                    "exists": True,
                    "path": str(decisions.relative_to(root)),
                    "sha256": sha256_file(decisions),
                }
                if decisions.exists()
                else {
                    "exists": False,
                    "path": str(decisions.relative_to(root)),
                }
            ),
        }
    return snapshots


def verify_snapshot_unchanged(root: Path, snapshot: dict[str, Any]) -> None:
    for paper_id, record in snapshot.items():
        canonical = root / record["canonical_graph"]["path"]
        if not canonical.exists() or sha256_file(canonical) != record["canonical_graph"]["sha256"]:
            raise FrozenContractViolation(
                f"Held-out canonical input changed after campaign start: {paper_id}. "
                "Restart calibration and holdout rather than continuing."
            )
        for item in record["strict_run_files"].values():
            if not item:
                continue
            path = root / item["path"]
            if not path.exists() or sha256_file(path) != item["sha256"]:
                raise FrozenContractViolation(
                    f"Frozen strict-run input changed after campaign start: {path}."
                )
        resolution = record.get("resolution_decisions", {"exists": False})
        resolution_path = root / resolution.get(
            "path",
            f"data_sers/extracted/{paper_id}/resolution/decisions.jsonl",
        )
        if resolution.get("exists"):
            if not resolution_path.exists() or sha256_file(resolution_path) != resolution["sha256"]:
                raise FrozenContractViolation(
                    f"Held-out resolution decisions changed after campaign start: {paper_id}."
                )
        elif resolution_path.exists():
            raise FrozenContractViolation(
                f"A resolution decisions file appeared after holdout start: {paper_id}."
            )


def verify_bridge_pair(bridge: Path, candidate: Path, paper_id: str) -> None:
    if not bridge.exists() or not candidate.exists():
        raise FileNotFoundError(
            f"Bridge materialization incomplete for {paper_id}: {bridge}, {candidate}"
        )
    confirmed_graph = nx.read_graphml(bridge, force_multigraph=True)
    candidate_graph = nx.read_graphml(candidate, force_multigraph=True)
    for key in ("bridge_extraction_id", "bridge_policy_run_id", "bridge_policy_version"):
        left = str(confirmed_graph.graph.get(key, ""))
        right = str(candidate_graph.graph.get(key, ""))
        if not left or left != right:
            raise FrozenContractViolation(
                f"Bridge pair is not from one materialization for {paper_id}: "
                f"{key} {left!r}/{right!r}."
            )


def lock_or_verify_derived_files(
    *,
    manifest: dict[str, Any],
    save_manifest,
    root: Path,
    lock_id: str,
    paths: Iterable[Path],
) -> None:
    rows = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Expected holdout artifact is missing: {path}")
        rows.append({
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })
    locks = manifest.setdefault("derived_artifact_locks", {})
    prior = locks.get(lock_id)
    if prior is None:
        locks[lock_id] = rows
        save_manifest()
        return
    if prior != rows:
        raise FrozenContractViolation(
            f"Derived holdout artifact changed after its stage completed: {lock_id}. "
            "Do not continue the same campaign after modifying results."
        )


def ensure_stage(manifest: dict[str, Any], stage: str) -> dict[str, Any]:
    return manifest.setdefault("stages", {}).setdefault(stage, {"status": "pending", "attempts": []})


def run_stage(
    *,
    manifest: dict[str, Any],
    save_manifest,
    stage: str,
    command: list[str],
    root: Path,
    logs_root: Path,
    dry_run: bool,
    max_passes: int = 1,
    backoffs: list[int] | None = None,
) -> None:
    record = ensure_stage(manifest, stage)
    if record.get("status") == "complete":
        print(f"[SKIP COMPLETE] {stage}")
        return
    backoffs = backoffs or [0]
    completed = len(record.get("attempts", []))
    for index in range(completed, max_passes):
        delay = backoffs[index] if index < len(backoffs) else backoffs[-1]
        if delay and not dry_run:
            print(f"[BACKOFF] {stage}: {delay}s")
            time.sleep(delay)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = logs_root / f"{stage}_pass_{index + 1}_{stamp}.log"
        attempt = {
            "pass": index + 1,
            "started_at_utc": now_utc(),
            "command": command,
            "log_path": str(log_path.relative_to(root)),
            "status": "running",
        }
        record.setdefault("attempts", []).append(attempt)
        record["status"] = "running"
        save_manifest()
        code = run_command(command=command, root=root, log_path=log_path, dry_run=dry_run)
        attempt["finished_at_utc"] = now_utc()
        attempt["exit_code"] = code
        if code == 0:
            attempt["status"] = "complete"
            record["status"] = "complete"
            record["completed_at_utc"] = now_utc()
            save_manifest()
            return
        attempt["status"] = "failed"
        record["status"] = "failed"
        save_manifest()
    raise RuntimeError(f"Stage failed after {max_passes} pass(es): {stage}")


def build_holdout_report(root: Path, protocol: dict[str, Any], ids: dict[str, str]) -> dict[str, Any]:
    data_root = root / str(protocol["data_root"])
    mode = str(protocol["mode"])
    corpus_root = data_root / "corpus" / ids["corpus"] / mode
    corpus = read_json(corpus_root / "manifest.json")
    corpus_audit = read_json(corpus_root / "audit.json")
    repro = read_json(corpus_root / "reproducibility" / ids["reproducibility"] / "summary.json")
    repro_audit = read_json(corpus_root / "reproducibility" / ids["reproducibility"] / "audit.json")
    metric = read_json(corpus_root / "metric_definition" / ids["metric_definition"] / "summary.json")
    metric_audit = read_json(corpus_root / "metric_definition" / ids["metric_definition"] / "audit.json")
    comparison = read_json(corpus_root / "comparison" / ids["comparison"] / "summary.json")
    comparison_audit = read_json(corpus_root / "comparison" / ids["comparison"] / "audit.json")

    violations: list[dict[str, Any]] = []
    def violation(code: str, observed: Any, expected: Any) -> None:
        violations.append({
            "category": "spec_violation",
            "code": code,
            "observed": observed,
            "expected": expected,
        })

    if not bool(corpus.get("passes_structural_gate", False)):
        violation("corpus_structural_gate", corpus.get("passes_structural_gate"), True)
    if not bool(corpus_audit.get("passes_structural_gate", False)):
        violation("corpus_audit_structural_gate", corpus_audit.get("passes_structural_gate"), True)
    if int(corpus.get("destructive_cross_paper_merges", -1)) != 0:
        violation("destructive_cross_paper_merges", corpus.get("destructive_cross_paper_merges"), 0)
    if not bool(repro.get("structural_gate", False)) or not bool(repro_audit.get("structural_gate", False)):
        violation("reproducibility_structural_gate", [repro.get("structural_gate"), repro_audit.get("structural_gate")], [True, True])
    if not bool(metric.get("structural_gate", False)) or not bool(metric_audit.get("structural_gate", False)):
        violation("metric_definition_structural_gate", [metric.get("structural_gate"), metric_audit.get("structural_gate")], [True, True])
    if not bool(comparison.get("passes_structural_gate", False)) or not bool(comparison_audit.get("passes_structural_gate", False)):
        violation("comparison_structural_gate", [comparison.get("passes_structural_gate"), comparison_audit.get("passes_structural_gate")], [True, True])
    if comparison.get("global_entity_concentration_consumed") is not False:
        violation("global_entity_concentration_consumed", comparison.get("global_entity_concentration_consumed"), False)
    if comparison.get("missing_context_is_not_quarantine") is not True:
        violation("missing_context_is_not_quarantine", comparison.get("missing_context_is_not_quarantine"), True)

    semantic_expected = protocol["frozen_semantics"]
    semantic_observed = {
        "corpus": corpus.get("corpus_semantics_id"),
        "reproducibility": repro.get("reproducibility_semantics_id"),
        "metric_definition": metric.get("metric_definition_semantics_id"),
        "comparison": comparison.get("comparison_semantics_id"),
        "method": comparison.get("method_semantics_id"),
        "quality_gate": comparison.get("quality_gate_semantics_id"),
    }
    for key, observed in semantic_observed.items():
        if observed != semantic_expected[key]:
            violation(f"semantic_drift:{key}", observed, semantic_expected[key])

    nonfatal: list[dict[str, Any]] = []
    protocol_counts = comparison.get("protocol_comparability_counts", {})
    if protocol_counts.get("different_protocol", 0):
        nonfatal.append({"category": "unsupported_context", "code": "different_protocol_pairs", "count": protocol_counts.get("different_protocol", 0), "action": "observe_not_tune"})
    if protocol_counts.get("unknown", 0):
        nonfatal.append({"category": "unsupported_context", "code": "unknown_protocol_pairs", "count": protocol_counts.get("unknown", 0), "action": "observe_not_tune"})
    metric_statuses = metric.get("definition_status_counts", {})
    if metric_statuses.get("unknown", 0):
        nonfatal.append({"category": "unsupported_context", "code": "unknown_metric_definitions", "count": metric_statuses.get("unknown", 0), "action": "observe_not_tune"})
    if metric_statuses.get("partial", 0):
        nonfatal.append({"category": "unsupported_context", "code": "partial_metric_definitions", "count": metric_statuses.get("partial", 0), "action": "observe_not_tune"})
    metric_compat = comparison.get("metric_definition_compatibility_counts", {})
    if metric_compat.get("different_definition", 0):
        nonfatal.append({"category": "unsupported_context", "code": "different_metric_definitions", "count": metric_compat.get("different_definition", 0), "action": "observe_not_tune"})
    unregistered = int(comparison.get("unregistered_observable_assessment_count", 0))
    if unregistered:
        nonfatal.append({"category": "expected_new_content", "code": "unregistered_observable_assessments", "count": unregistered, "action": "review_without_policy_tuning"})
    duplicate_pairs = int(repro.get("possible_duplicate_result_pair_count", 0))
    if duplicate_pairs:
        nonfatal.append({"category": "unsupported_context", "code": "possible_duplicate_reproducibility_results", "count": duplicate_pairs, "action": "preserve_fail_closed"})

    distribution = {
        "paper_ids": corpus.get("paper_ids", []),
        "corpus_nodes": corpus.get("corpus_nodes"),
        "corpus_edges": corpus.get("corpus_edges"),
        "comparison_context_count": comparison.get("context_count"),
        "comparison_assessment_count": comparison.get("assessment_count"),
        "compatibility_counts": comparison.get("compatibility_counts", {}),
        "protocol_comparability_counts": protocol_counts,
        "method_dimension_status_counts": comparison.get("method_dimension_status_counts", {}),
        "reproducibility_evidence_count": repro.get("evidence_count"),
        "reproducibility_kind_counts": repro.get("evidence_kind_counts", {}),
        "metric_definition_context_count": metric.get("context_count"),
        "metric_definition_status_counts": metric_statuses,
        "metric_definition_compatibility_counts": metric_compat,
        "metric_definition_ranking_relevant_assessment_count": comparison.get("metric_definition_ranking_relevant_assessment_count"),
        "metric_definition_ranking_relevant_gate_pass_count": comparison.get("metric_definition_ranking_relevant_gate_pass_count"),
        "numeric_ranking_allowed_count": comparison.get("numeric_ranking_allowed_count"),
        "observable_family_counts": comparison.get("observable_family_counts", {}),
        "unregistered_observable_assessment_count": unregistered,
    }
    return {
        "holdout_protocol_version": protocol["protocol_version"],
        "holdout_scope": protocol["holdout_scope"],
        "verdict": "pass" if not violations else "fail",
        "passes_holdout_invariants": not violations,
        "count_thresholds_used_for_acceptance": False,
        "heterogeneity_is_first_class": True,
        "distribution_observations": distribution,
        "nonfatal_classifications": nonfatal,
        "violations": violations,
        "interpretation": (
            "Unknown, different-protocol, different-definition, and zero-rankable outcomes are valid holdout observations. "
            "Only structural/provenance/frozen-contract violations fail this holdout."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run alpha4b.4 frozen SERS evidence-substrate holdout validation.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--adopt-existing-bridge", action="store_true", help="Explicitly adopt pre-existing confirmed/candidate Bridge graphs instead of rerunning frozen Bridge extraction.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = PROJECT_ROOT.resolve()
    protocol_path = args.protocol if args.protocol.is_absolute() else (root / args.protocol)
    protocol_path = protocol_path.resolve()
    protocol = read_json(protocol_path)
    validate_protocol(protocol)

    holdout_state = str(
        protocol.get("holdout_execution_state", "enabled")
    )
    if holdout_state != "enabled":
        reason = str(
            protocol.get(
                "holdout_execution_pause_reason",
                "calibration replay/refreeze is required",
            )
        )
        raise FrozenContractViolation(
            "SERS alpha4b.4 holdout execution is paused: "
            f"{holdout_state}. {reason}"
        )

    branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, text=True, capture_output=True).stdout.strip()
    if branch and branch != "feat/SERS-specification-v2.9.1":
        print(f"[WARNING] branch={branch!r}; frozen file hashes, not branch name, are authoritative.")

    frozen_blobs = verify_frozen_blobs(root, protocol)
    runtime_semantics = verify_runtime_semantics(protocol)
    calibration_artifacts = verify_calibration_freeze(root, protocol)
    input_snapshot = snapshot_holdout_inputs(root, protocol)

    evaluation_root = root / "evaluation" / "sers_alpha4b4" / args.campaign_id
    manifest_path = evaluation_root / "manifest.json"
    logs_root = evaluation_root / "logs"
    ids = {
        "corpus": f"{args.campaign_id}_corpus",
        "reproducibility": f"{args.campaign_id}_reproducibility",
        "metric_definition": f"{args.campaign_id}_metric_definition",
        "comparison": f"{args.campaign_id}_comparison",
    }

    protocol_hash = sha256_file(protocol_path)
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("protocol_sha256") != protocol_hash:
            raise FrozenContractViolation("Holdout protocol changed after campaign start.")
        if manifest.get("frozen_git_blobs") != frozen_blobs:
            raise FrozenContractViolation("Frozen implementation changed after campaign start.")
        if manifest.get("runtime_semantics") != runtime_semantics:
            raise FrozenContractViolation("Runtime semantics changed after campaign start.")
        if manifest.get("calibration_artifacts") != calibration_artifacts:
            raise FrozenContractViolation("Calibration artifacts changed after holdout start. Rerun calibration and start a new campaign.")
        if manifest.get("input_snapshot") != input_snapshot:
            raise FrozenContractViolation("Held-out input snapshot changed after campaign start.")
        if bool(manifest.get("adopt_existing_bridge")) != bool(args.adopt_existing_bridge):
            raise FrozenContractViolation("Bridge adoption mode changed on campaign resume.")
        if bool(manifest.get("dry_run")) != bool(args.dry_run):
            raise FrozenContractViolation("Dry-run and real campaign cannot share a campaign-id.")
    else:
        manifest = {
            "campaign_id": args.campaign_id,
            "status": "running",
            "dry_run": bool(args.dry_run),
            "created_at_utc": now_utc(),
            "updated_at_utc": now_utc(),
            "protocol_path": str(protocol_path.relative_to(root)),
            "protocol_sha256": protocol_hash,
            "protocol": protocol,
            "frozen_git_blobs": frozen_blobs,
            "runtime_semantics": runtime_semantics,
            "calibration_artifacts": calibration_artifacts,
            "input_snapshot": input_snapshot,
            "adopt_existing_bridge": bool(args.adopt_existing_bridge),
            "ids": ids,
            "stages": {},
        }

    def save_manifest() -> None:
        manifest["updated_at_utc"] = now_utc()
        atomic_write_json(manifest_path, manifest)

    save_manifest()
    verify_snapshot_unchanged(root, input_snapshot)

    data_root = root / str(protocol["data_root"])
    config_path = root / str(protocol["config_path"])
    bridge_model = (
        os.environ.get(protocol["upstream_policy"]["bridge_model_environment"], "").strip()
        or os.environ.get(protocol["upstream_policy"]["strict_model_fallback_environment"], "").strip()
    )
    provider = os.environ.get(protocol["upstream_policy"]["provider_environment"], "").strip()
    resolved_runtime = {
        "bridge_mode": (
            "explicit_existing_adoption"
            if args.adopt_existing_bridge
            else "fresh_frozen_policy"
        ),
        "bridge_model": "" if args.adopt_existing_bridge else bridge_model,
        "provider": "" if args.adopt_existing_bridge else provider,
        "bridge_concurrency": protocol["upstream_policy"]["bridge_concurrency"],
    }
    prior_runtime = manifest.get("resolved_runtime")
    if prior_runtime is None:
        manifest["resolved_runtime"] = resolved_runtime
        save_manifest()
    elif prior_runtime != resolved_runtime:
        raise FrozenContractViolation(
            "Resolved Bridge model/provider/mode changed after campaign creation."
        )

    for paper_id in protocol["holdout_papers"]:
        paper_root = data_root / "extracted" / paper_id
        canonical = paper_root / f"{paper_id}.graphml"
        bridge = paper_root / f"{paper_id}.bridge.graphml"
        candidate = paper_root / f"{paper_id}.bridge.candidates.graphml"
        strict_run_id = input_snapshot[paper_id]["strict_run_id"]

        audit_dir = evaluation_root / "canonical_audit" / paper_id
        run_stage(
            manifest=manifest,
            save_manifest=save_manifest,
            stage=f"{paper_id}:canonical_audit",
            command=[sys.executable, "-m", "scripts.inspect_graphml", "--graphml", str(canonical), "--output-dir", str(audit_dir)],
            root=root,
            logs_root=logs_root,
            dry_run=args.dry_run,
        )

        if args.adopt_existing_bridge:
            stage = ensure_stage(manifest, f"{paper_id}:bridge")
            if stage.get("status") != "complete":
                if not bridge.exists() or not candidate.exists():
                    raise FileNotFoundError(
                        f"--adopt-existing-bridge requires both {bridge} and {candidate}."
                    )
                verify_bridge_pair(bridge, candidate, paper_id)
                stage.update({
                    "status": "complete",
                    "mode": "explicit_existing_adoption",
                    "confirmed_sha256": sha256_file(bridge),
                    "candidate_sha256": sha256_file(candidate),
                    "completed_at_utc": now_utc(),
                })
                save_manifest()
        else:
            if not args.dry_run:
                if not os.environ.get("OPENROUTER_API_KEY"):
                    raise RuntimeError("OPENROUTER_API_KEY is required unless --adopt-existing-bridge is used.")
                if not bridge_model:
                    raise RuntimeError("OPENROUTER_BRIDGE_MODEL or OPENROUTER_EXTRACT_MODEL is required for fresh Bridge extraction.")
            bridge_command = [
                sys.executable, "-m", "scripts.extract_bridge_graph",
                "--paper-id", paper_id,
                "--config", str(config_path),
                "--domain-profile", str(protocol["domain_profile"]),
                "--data-root", str(protocol["data_root"]),
                "--run-id", strict_run_id,
                "--concurrency", str(protocol["upstream_policy"]["bridge_concurrency"]),
            ]
            if bridge_model:
                bridge_command.extend(["--model", bridge_model])
            if provider:
                bridge_command.extend(["--provider", provider])
            run_stage(
                manifest=manifest,
                save_manifest=save_manifest,
                stage=f"{paper_id}:bridge",
                command=bridge_command,
                root=root,
                logs_root=logs_root,
                dry_run=args.dry_run,
                max_passes=int(protocol["upstream_policy"]["bridge_max_outer_passes"]),
                backoffs=[int(v) for v in protocol["upstream_policy"]["bridge_backoff_seconds"]],
            )

        if not args.dry_run:
            verify_bridge_pair(bridge, candidate, paper_id)
            lock_or_verify_derived_files(
                manifest=manifest,
                save_manifest=save_manifest,
                root=root,
                lock_id=f"{paper_id}:bridge",
                paths=[bridge, candidate],
            )

        for mode in protocol["upstream_policy"]["projection_modes"]:
            command = [
                sys.executable, "-m", "scripts.build_graphagents_projection",
                "--paper-id", paper_id,
                "--domain-profile", str(protocol["domain_profile"]),
                "--data-root", str(protocol["data_root"]),
                "--mode", str(mode),
                "--canonical-graphml", str(canonical),
            ]
            if mode in {"mechanism", "exploratory"}:
                command.extend(["--bridge-graphml", str(bridge)])
            if mode == "exploratory":
                command.extend(["--candidate-bridge-graphml", str(candidate)])
            run_stage(
                manifest=manifest,
                save_manifest=save_manifest,
                stage=f"{paper_id}:projection:{mode}",
                command=command,
                root=root,
                logs_root=logs_root,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                projection_root = paper_root / "graphagents" / str(mode)
                lock_or_verify_derived_files(
                    manifest=manifest,
                    save_manifest=save_manifest,
                    root=root,
                    lock_id=f"{paper_id}:projection:{mode}",
                    paths=[
                        projection_root / "graph.graphml",
                        projection_root / "summary.json",
                        projection_root / "node_text.jsonl",
                        projection_root / "edge_evidence.jsonl",
                    ],
                )

    paper_args: list[str] = []
    for paper_id in protocol["holdout_papers"]:
        paper_args.append(str(paper_id))
    run_stage(
        manifest=manifest,
        save_manifest=save_manifest,
        stage="holdout_corpus",
        command=[
            sys.executable, "-m", "scripts.build_corpus_graph",
            "--corpus-id", ids["corpus"],
            "--domain-profile", str(protocol["domain_profile"]),
            "--data-root", str(protocol["data_root"]),
            "--paper-ids", *paper_args,
            "--mode", str(protocol["mode"]),
        ],
        root=root,
        logs_root=logs_root,
        dry_run=args.dry_run,
    )

    run_stage(
        manifest=manifest,
        save_manifest=save_manifest,
        stage="holdout_reproducibility",
        command=[
            sys.executable, "-m", "scripts.build_reproducibility_evidence",
            "--domain-profile", str(protocol["domain_profile"]),
            "--data-root", str(protocol["data_root"]),
            "--corpus-id", ids["corpus"],
            "--mode", str(protocol["mode"]),
            "--reproducibility-id", ids["reproducibility"],
        ],
        root=root,
        logs_root=logs_root,
        dry_run=args.dry_run,
    )

    run_stage(
        manifest=manifest,
        save_manifest=save_manifest,
        stage="holdout_metric_definition",
        command=[
            sys.executable, "-m", "scripts.build_metric_definition_contexts",
            "--domain-profile", str(protocol["domain_profile"]),
            "--data-root", str(protocol["data_root"]),
            "--corpus-id", ids["corpus"],
            "--mode", str(protocol["mode"]),
            "--metric-definition-id", ids["metric_definition"],
        ],
        root=root,
        logs_root=logs_root,
        dry_run=args.dry_run,
    )

    run_stage(
        manifest=manifest,
        save_manifest=save_manifest,
        stage="holdout_comparison",
        command=[
            sys.executable, "-m", "scripts.build_comparison_contexts",
            "--domain-profile", str(protocol["domain_profile"]),
            "--data-root", str(protocol["data_root"]),
            "--corpus-id", ids["corpus"],
            "--mode", str(protocol["mode"]),
            "--comparison-id", ids["comparison"],
            "--metric-definition-id", ids["metric_definition"],
        ],
        root=root,
        logs_root=logs_root,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        corpus_root = data_root / "corpus" / ids["corpus"] / str(protocol["mode"])
        lock_or_verify_derived_files(
            manifest=manifest, save_manifest=save_manifest, root=root,
            lock_id="holdout_corpus",
            paths=[corpus_root / "manifest.json", corpus_root / "audit.json", corpus_root / "graph.graphml"],
        )
        repro_root = corpus_root / "reproducibility" / ids["reproducibility"]
        lock_or_verify_derived_files(
            manifest=manifest, save_manifest=save_manifest, root=root,
            lock_id="holdout_reproducibility",
            paths=[repro_root / "summary.json", repro_root / "audit.json", repro_root / "evidence.jsonl"],
        )
        metric_root = corpus_root / "metric_definition" / ids["metric_definition"]
        lock_or_verify_derived_files(
            manifest=manifest, save_manifest=save_manifest, root=root,
            lock_id="holdout_metric_definition",
            paths=[metric_root / "summary.json", metric_root / "audit.json", metric_root / "contexts.jsonl"],
        )
        comparison_root = corpus_root / "comparison" / ids["comparison"]
        lock_or_verify_derived_files(
            manifest=manifest, save_manifest=save_manifest, root=root,
            lock_id="holdout_comparison",
            paths=[
                comparison_root / "summary.json",
                comparison_root / "audit.json",
                comparison_root / "contexts.jsonl",
                comparison_root / "assessments.jsonl",
                comparison_root / "protocol_assessments.jsonl",
                comparison_root / "metric_definition_assessments.jsonl",
            ],
        )

    if args.dry_run:
        manifest["status"] = "dry_run_complete"
        save_manifest()
        print("Dry-run complete. No holdout outputs were evaluated.")
        print("Manifest:", manifest_path)
        return 0

    verify_frozen_blobs(root, protocol)
    verify_snapshot_unchanged(root, input_snapshot)
    report = build_holdout_report(root, protocol, ids)
    report_path = evaluation_root / "holdout_report.json"
    atomic_write_json(report_path, report)
    manifest["holdout_report"] = str(report_path.relative_to(root))
    manifest["status"] = "complete" if report["passes_holdout_invariants"] else "failed"
    manifest["finished_at_utc"] = now_utc()
    save_manifest()

    print()
    print("alpha4b.4 holdout verdict:", report["verdict"])
    print("Count thresholds used for acceptance:", report["count_thresholds_used_for_acceptance"])
    print("Heterogeneity is first-class:", report["heterogeneity_is_first_class"])
    print("Report:", report_path)
    print("Manifest:", manifest_path)
    if report["nonfatal_classifications"]:
        print("Non-fatal holdout observations:")
        for item in report["nonfatal_classifications"]:
            print(" -", item["category"], item["code"], item.get("count", ""))
    if report["violations"]:
        print("Holdout invariant violations:")
        for item in report["violations"]:
            print(" -", item["code"], "observed=", item["observed"], "expected=", item["expected"])
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
