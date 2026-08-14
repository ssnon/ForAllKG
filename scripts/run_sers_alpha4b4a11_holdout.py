from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import networkx as nx

from scripts.run_sers_alpha4b4_holdout import (
    FrozenContractViolation,
    assert_no_manual_resolution,
    atomic_write_json,
    ensure_stage,
    git_blob,
    lock_or_verify_derived_files,
    now_utc,
    read_json,
    read_jsonl,
    resolve_latest_strict_run,
    run_stage,
    sha256_file,
    verify_bridge_pair,
    verify_snapshot_unchanged,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "configs" / "heldout" / "sers_alpha4b4a11_protocol.json"
)


def _assert_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise FrozenContractViolation(
            f"Calibration refreeze mismatch for {label}: "
            f"{observed!r} != {expected!r}."
        )


def validate_protocol(protocol: dict[str, Any]) -> None:
    if str(protocol.get("holdout_execution_state", "")) != "enabled":
        raise FrozenContractViolation(
            "alpha4b.4a11 holdout protocol is not enabled."
        )

    calibration = [str(value) for value in protocol.get("calibration_papers", [])]
    holdout = [str(value) for value in protocol.get("holdout_papers", [])]
    if not calibration or not holdout:
        raise ValueError("Calibration and holdout papers must both be non-empty.")
    overlap = sorted(set(calibration) & set(holdout))
    if overlap:
        raise ValueError(
            f"Calibration/holdout overlap is forbidden: {overlap!r}"
        )

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

    prefix = str(protocol.get("required_campaign_id_prefix", "")).strip()
    if not prefix:
        raise ValueError("required_campaign_id_prefix must be non-empty.")

    retired = protocol.get("retired_epoch", {})
    if not isinstance(retired, dict) or not bool(
        retired.get("old_holdout_campaign_must_not_resume", False)
    ):
        raise ValueError("Retired holdout epoch must remain non-resumable.")

    refreeze = protocol.get("holdout_input_refreeze", {})
    if not isinstance(refreeze, dict) or not bool(refreeze.get("required", False)):
        raise ValueError("Holdout canonical-input refreeze must be required.")


def validate_campaign_id(protocol: dict[str, Any], campaign_id: str) -> None:
    prefix = str(protocol["required_campaign_id_prefix"])
    if not campaign_id.startswith(prefix):
        raise FrozenContractViolation(
            f"Campaign ID must start with {prefix!r}; "
            "old alpha4b.4 campaign IDs are retired."
        )
    retired_ids = {
        str(value)
        for value in protocol.get("retired_epoch", {}).get(
            "known_retired_campaign_ids", []
        )
    }
    if campaign_id in retired_ids:
        raise FrozenContractViolation(
            f"Campaign ID {campaign_id!r} belongs to the retired epoch."
        )


def verify_frozen_blobs(
    root: Path,
    protocol: dict[str, Any],
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative_path, expected in sorted(
        protocol["frozen_git_blobs"].items()
    ):
        actual = git_blob(root, relative_path)
        observed[relative_path] = actual
        if actual != expected:
            raise FrozenContractViolation(
                "Frozen implementation drift detected before/during "
                f"alpha4b.4a11 holdout: {relative_path}: "
                f"expected {expected}, observed {actual}. "
                "Do not patch in place; diagnose, replay calibration, and "
                "start another holdout epoch."
            )
    return observed


def verify_runtime_semantics(
    protocol: dict[str, Any],
) -> dict[str, str]:
    from dac_her.domains.registry import get_domain_profile
    from dac_her.domains.comparison_registry import get_comparison_adapter
    from dac_her.domains.reproducibility_registry import (
        get_reproducibility_adapter,
    )
    from dac_her.domains.metric_definition_registry import (
        get_metric_definition_adapter,
    )
    from dac_her.measurement_merge_invariants import (
        MEASUREMENT_MERGE_INVARIANT_ID,
    )
    from dac_her.measurement_result_identity import (
        IDENTITY_AWARE_DOMAIN_RECONSTRUCTION_ID,
        MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID,
    )
    from dac_her.quality_aware_comparison import (
        QUALITY_AWARE_NUMERIC_GATE_SEMANTICS_ID,
    )

    profile = get_domain_profile(str(protocol["domain_profile"]))
    comparison = get_comparison_adapter(profile)
    reproducibility = get_reproducibility_adapter(profile)
    metric_definition = get_metric_definition_adapter(profile)
    observed = {
        "projection": str(profile.projection.semantics_id),
        "corpus": str(profile.corpus.semantics_id),
        "comparison": str(comparison.semantics_id),
        "method": str(
            comparison.method_semantics.semantics_id
            if comparison.method_semantics
            else ""
        ),
        "reproducibility": str(reproducibility.semantics_id),
        "metric_definition": str(metric_definition.semantics_id),
        "quality_gate": str(QUALITY_AWARE_NUMERIC_GATE_SEMANTICS_ID),
        "measurement_result_identity": str(
            MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID
        ),
        "identity_aware_domain_reconstruction": str(
            IDENTITY_AWARE_DOMAIN_RECONSTRUCTION_ID
        ),
    }
    expected = {
        str(key): str(value)
        for key, value in protocol["frozen_semantics"].items()
    }
    if observed != expected:
        raise FrozenContractViolation(
            f"Frozen semantic IDs drifted: "
            f"expected={expected!r}, observed={observed!r}."
        )

    invariant_expected = str(
        protocol["frozen_invariants"]["measurement_merge"]
    )
    if str(MEASUREMENT_MERGE_INVARIANT_ID) != invariant_expected:
        raise FrozenContractViolation(
            "Measurement merge invariant ID drifted: "
            f"{MEASUREMENT_MERGE_INVARIANT_ID!r} != "
            f"{invariant_expected!r}."
        )
    return observed


def verify_calibration_freeze(
    root: Path,
    protocol: dict[str, Any],
) -> dict[str, dict[str, str]]:
    freeze = protocol["calibration_freeze"]
    expected = freeze["expected"]
    path_keys = (
        "calibration_replay_report",
        "corpus_manifest",
        "measurement_result_identity_summary",
        "reproducibility_summary",
        "metric_definition_summary",
        "comparison_summary",
    )
    paths = {
        key: (root / str(freeze[key])).resolve()
        for key in path_keys
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Refrozen calibration artifact is missing ({name}): {path}"
            )

    replay = read_json(paths["calibration_replay_report"])
    corpus = read_json(paths["corpus_manifest"])
    identity = read_json(paths["measurement_result_identity_summary"])
    repro = read_json(paths["reproducibility_summary"])
    metric = read_json(paths["metric_definition_summary"])
    comparison = read_json(paths["comparison_summary"])

    _assert_equal(
        "replay id",
        replay.get("replay_id"),
        expected["calibration_replay_id"],
    )
    _assert_equal(
        "replay LLM calls",
        replay.get("llm_calls_performed"),
        False,
    )
    _assert_equal(
        "replay merge invariant",
        replay.get("measurement_merge_invariant_id"),
        expected["measurement_merge_invariant_id"],
    )
    _assert_equal(
        "replay identity semantics",
        replay.get("measurement_result_identity_semantics_id"),
        expected["measurement_result_identity_semantics_id"],
    )

    _assert_equal(
        "calibration papers/corpus",
        corpus.get("paper_ids"),
        expected["calibration_papers"],
    )
    _assert_equal(
        "corpus semantics",
        corpus.get("corpus_semantics_id"),
        expected["corpus_semantics_id"],
    )

    _assert_equal(
        "identity papers",
        identity.get("paper_ids"),
        expected["calibration_papers"],
    )
    _assert_equal(
        "identity semantics",
        identity.get("measurement_result_identity_semantics_id"),
        expected["measurement_result_identity_semantics_id"],
    )
    _assert_equal(
        "identity source mentions",
        identity.get("source_mention_count"),
        expected["measurement_source_mention_count"],
    )
    _assert_equal(
        "scientific result count",
        identity.get("scientific_result_count"),
        expected["scientific_result_count"],
    )
    _assert_equal(
        "consolidated exact result count",
        identity.get("consolidated_exact_result_count"),
        expected["consolidated_exact_result_count"],
    )
    _assert_equal(
        "unresolved same-lineage groups",
        identity.get("unresolved_same_lineage_group_count"),
        expected["unresolved_same_lineage_group_count"],
    )
    _assert_equal(
        "identity issues",
        identity.get("issues"),
        [],
    )
    _assert_equal(
        "identity structural gate",
        identity.get("structural_gate"),
        True,
    )
    _assert_equal(
        "same-value-alone guard",
        identity.get("policy", {}).get("same_value_alone_never_merges"),
        True,
    )

    _assert_equal(
        "repro papers",
        repro.get("paper_ids"),
        expected["calibration_papers"],
    )
    _assert_equal(
        "repro semantics",
        repro.get("reproducibility_semantics_id"),
        expected["reproducibility_semantics_id"],
    )
    _assert_equal(
        "repro evidence count",
        repro.get("evidence_count"),
        expected["reproducibility_evidence_count"],
    )
    _assert_equal(
        "repro structural gate",
        repro.get("structural_gate"),
        True,
    )

    _assert_equal(
        "metric papers",
        metric.get("paper_ids"),
        expected["calibration_papers"],
    )
    _assert_equal(
        "metric semantics",
        metric.get("metric_definition_semantics_id"),
        expected["metric_definition_semantics_id"],
    )
    _assert_equal(
        "metric identity semantics",
        metric.get("measurement_result_identity_semantics_id"),
        expected["measurement_result_identity_semantics_id"],
    )
    _assert_equal(
        "metric context count",
        metric.get("context_count"),
        expected["metric_definition_context_count"],
    )
    statuses = metric.get("definition_status_counts", {})
    _assert_equal(
        "metric known count",
        statuses.get("known", 0),
        expected["metric_definition_known_count"],
    )
    _assert_equal(
        "metric unknown count",
        statuses.get("unknown", 0),
        expected["metric_definition_unknown_count"],
    )
    _assert_equal(
        "metric structural gate",
        metric.get("structural_gate"),
        True,
    )

    _assert_equal(
        "comparison papers",
        comparison.get("paper_ids"),
        expected["calibration_papers"],
    )
    _assert_equal(
        "comparison semantics",
        comparison.get("comparison_semantics_id"),
        expected["comparison_semantics_id"],
    )
    _assert_equal(
        "method semantics",
        comparison.get("method_semantics_id"),
        expected["method_semantics_id"],
    )
    _assert_equal(
        "comparison identity semantics",
        comparison.get("measurement_result_identity_semantics_id"),
        expected["measurement_result_identity_semantics_id"],
    )
    _assert_equal(
        "identity-aware domain reconstruction",
        comparison.get("identity_aware_domain_reconstruction_id"),
        expected["identity_aware_domain_reconstruction_id"],
    )
    _assert_equal(
        "quality gate semantics",
        comparison.get("quality_gate_semantics_id"),
        expected["quality_gate_semantics_id"],
    )
    _assert_equal(
        "comparison context count",
        comparison.get("context_count"),
        expected["comparison_context_count"],
    )
    _assert_equal(
        "comparison assessment count",
        comparison.get("assessment_count"),
        expected["comparison_assessment_count"],
    )
    _assert_equal(
        "protocol counts",
        comparison.get("protocol_comparability_counts"),
        expected["protocol_comparability_counts"],
    )
    _assert_equal(
        "metric compatibility counts",
        comparison.get("metric_definition_compatibility_counts"),
        expected["metric_definition_compatibility_counts"],
    )
    _assert_equal(
        "ranking-relevant metric count",
        comparison.get(
            "metric_definition_ranking_relevant_assessment_count"
        ),
        expected["metric_definition_ranking_relevant_assessment_count"],
    )
    _assert_equal(
        "ranking-relevant metric passes",
        comparison.get(
            "metric_definition_ranking_relevant_gate_pass_count"
        ),
        expected["metric_definition_ranking_relevant_gate_pass_count"],
    )
    _assert_equal(
        "numeric ranking allowed",
        comparison.get("numeric_ranking_allowed_count"),
        expected["numeric_ranking_allowed_count"],
    )
    _assert_equal(
        "global concentration leak guard",
        comparison.get("global_entity_concentration_consumed"),
        False,
    )
    _assert_equal(
        "missing context semantics",
        comparison.get("missing_context_is_not_quarantine"),
        True,
    )
    _assert_equal(
        "comparison structural gate",
        comparison.get("passes_structural_gate"),
        True,
    )

    sample_preparation = (
        comparison.get("method_dimension_status_counts", {})
        .get("sample_preparation", {})
    )
    _assert_equal(
        "sample_preparation status counts",
        sample_preparation,
        expected["sample_preparation_status_counts"],
    )
    sample_preparation_mismatch = (
        comparison.get("protocol_mismatched_dimension_counts", {})
        .get("sample_preparation", 0)
    )
    _assert_equal(
        "sample_preparation protocol mismatches",
        sample_preparation_mismatch,
        expected["sample_preparation_protocol_mismatch_count"],
    )

    return {
        name: {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }


def verify_holdout_input_refreeze(
    root: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    from dac_her.measurement_merge_invariants import (
        measurement_value_payload_issues,
    )

    refreeze = protocol["holdout_input_refreeze"]
    report_path = (root / str(refreeze["report"])).resolve()
    if not report_path.exists():
        command = (
            "python -m scripts.prepare_sers_alpha4b4a11_holdout_inputs"
        )
        raise FrozenContractViolation(
            "Refrozen holdout canonical-input report is missing. "
            f"Run `{command}` first."
        )

    report = read_json(report_path)
    _assert_equal(
        "holdout input refreeze id",
        report.get("refreeze_id"),
        refreeze["refreeze_id"],
    )
    _assert_equal(
        "holdout input refreeze papers",
        report.get("paper_ids"),
        protocol["holdout_papers"],
    )
    _assert_equal(
        "holdout input refreeze invariant",
        report.get("measurement_merge_invariant_id"),
        protocol["frozen_invariants"]["measurement_merge"],
    )
    _assert_equal(
        "holdout input refreeze LLM calls",
        report.get("llm_calls_performed"),
        False,
    )
    _assert_equal(
        "holdout input refreeze gate",
        report.get("passes_input_refreeze"),
        True,
    )

    data_root = root / str(protocol["data_root"])
    paper_records = report.get("paper_records", {})
    for paper_id in protocol["holdout_papers"]:
        record = paper_records.get(paper_id)
        if not isinstance(record, dict):
            raise FrozenContractViolation(
                f"Input refreeze record missing for {paper_id}."
            )
        canonical = data_root / "extracted" / paper_id / f"{paper_id}.graphml"
        if not canonical.exists():
            raise FileNotFoundError(canonical)
        observed_hash = sha256_file(canonical)
        _assert_equal(
            f"{paper_id} refrozen canonical hash",
            observed_hash,
            record.get("post_canonical_sha256"),
        )
        graph = nx.read_graphml(canonical, force_multigraph=True)
        _assert_equal(
            f"{paper_id} measurement merge invariant",
            str(graph.graph.get("measurement_merge_invariant_id", "")),
            protocol["frozen_invariants"]["measurement_merge"],
        )
        issues = measurement_value_payload_issues(graph)
        if issues:
            raise FrozenContractViolation(
                f"{paper_id} refrozen canonical graph violates Measurement "
                f"numeric/text XOR: {issues[:5]!r}"
            )

        for item in record.get("strict_run_files", {}).values():
            if not item:
                continue
            path = root / str(item["path"])
            if not path.exists() or sha256_file(path) != item["sha256"]:
                raise FrozenContractViolation(
                    f"Frozen strict input changed after holdout refreeze: {path}"
                )

    return {
        "path": str(report_path.relative_to(root)),
        "sha256": sha256_file(report_path),
        "refreeze_id": report["refreeze_id"],
        "paper_canonical_hashes": {
            paper_id: paper_records[paper_id]["post_canonical_sha256"]
            for paper_id in protocol["holdout_papers"]
        },
    }


def snapshot_holdout_inputs(
    root: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    from dac_her.measurement_merge_invariants import (
        measurement_value_payload_issues,
    )

    data_root = root / str(protocol["data_root"])
    expected_invariant = str(
        protocol["frozen_invariants"]["measurement_merge"]
    )
    snapshots: dict[str, Any] = {}
    for paper_id in protocol["holdout_papers"]:
        paper_root = data_root / "extracted" / str(paper_id)
        canonical = paper_root / f"{paper_id}.graphml"
        if not canonical.exists():
            raise FileNotFoundError(canonical)
        assert_no_manual_resolution(paper_root)

        graph = nx.read_graphml(canonical, force_multigraph=True)
        domain = str(graph.graph.get("domain_profile_id", ""))
        if domain and domain != protocol["domain_profile"]:
            raise FrozenContractViolation(
                f"{paper_id} canonical domain mismatch: {domain!r}."
            )
        observed_invariant = str(
            graph.graph.get("measurement_merge_invariant_id", "")
        )
        if observed_invariant != expected_invariant:
            raise FrozenContractViolation(
                f"{paper_id} canonical graph is not from the refrozen "
                f"Measurement merge epoch: {observed_invariant!r}."
            )
        xor_issues = measurement_value_payload_issues(graph)
        if xor_issues:
            raise FrozenContractViolation(
                f"{paper_id} canonical graph violates Measurement XOR: "
                f"{xor_issues[:5]!r}"
            )

        run_id, run_dir, strict_snapshot = resolve_latest_strict_run(
            root,
            paper_root,
        )
        decisions = paper_root / "resolution" / "decisions.jsonl"
        snapshots[str(paper_id)] = {
            "canonical_graph": {
                "path": str(canonical.relative_to(root)),
                "sha256": sha256_file(canonical),
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "measurement_merge_invariant_id": observed_invariant,
                "measurement_xor_issue_count": 0,
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


def detection_limit_ranking_violations(
    assessments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for row in assessments:
        if str(row.get("observable_key", "")) != "detection_limit":
            continue
        if bool(row.get("numeric_ranking_allowed", False)):
            rows.append(row)
    return rows


def build_holdout_report(
    root: Path,
    protocol: dict[str, Any],
    ids: dict[str, str],
) -> dict[str, Any]:
    data_root = root / str(protocol["data_root"])
    mode = str(protocol["mode"])
    corpus_root = data_root / "corpus" / ids["corpus"] / mode

    corpus = read_json(corpus_root / "manifest.json")
    corpus_audit = read_json(corpus_root / "audit.json")
    identity_root = (
        corpus_root
        / "measurement_result_identity"
        / ids["measurement_result_identity"]
    )
    identity = read_json(identity_root / "summary.json")
    identity_audit = read_json(identity_root / "audit.json")
    repro = read_json(
        corpus_root
        / "reproducibility"
        / ids["reproducibility"]
        / "summary.json"
    )
    repro_audit = read_json(
        corpus_root
        / "reproducibility"
        / ids["reproducibility"]
        / "audit.json"
    )
    metric = read_json(
        corpus_root
        / "metric_definition"
        / ids["metric_definition"]
        / "summary.json"
    )
    metric_audit = read_json(
        corpus_root
        / "metric_definition"
        / ids["metric_definition"]
        / "audit.json"
    )
    comparison_root = (
        corpus_root / "comparison" / ids["comparison"]
    )
    comparison = read_json(comparison_root / "summary.json")
    comparison_audit = read_json(comparison_root / "audit.json")
    comparison_assessments = read_jsonl(
        comparison_root / "assessments.jsonl"
    )

    violations: list[dict[str, Any]] = []

    def violation(code: str, observed: Any, expected: Any) -> None:
        violations.append(
            {
                "category": "spec_violation",
                "code": code,
                "observed": observed,
                "expected": expected,
            }
        )

    if not bool(corpus.get("passes_structural_gate", False)):
        violation(
            "corpus_structural_gate",
            corpus.get("passes_structural_gate"),
            True,
        )
    if not bool(corpus_audit.get("passes_structural_gate", False)):
        violation(
            "corpus_audit_structural_gate",
            corpus_audit.get("passes_structural_gate"),
            True,
        )
    if int(corpus.get("destructive_cross_paper_merges", -1)) != 0:
        violation(
            "destructive_cross_paper_merges",
            corpus.get("destructive_cross_paper_merges"),
            0,
        )

    if not bool(identity.get("structural_gate", False)) or not bool(
        identity_audit.get("structural_gate", False)
    ):
        violation(
            "measurement_result_identity_structural_gate",
            [
                identity.get("structural_gate"),
                identity_audit.get("structural_gate"),
            ],
            [True, True],
        )
    if identity.get("issues") not in ([], None):
        violation(
            "measurement_result_identity_issues",
            identity.get("issues"),
            [],
        )

    if not bool(repro.get("structural_gate", False)) or not bool(
        repro_audit.get("structural_gate", False)
    ):
        violation(
            "reproducibility_structural_gate",
            [
                repro.get("structural_gate"),
                repro_audit.get("structural_gate"),
            ],
            [True, True],
        )
    if not bool(metric.get("structural_gate", False)) or not bool(
        metric_audit.get("structural_gate", False)
    ):
        violation(
            "metric_definition_structural_gate",
            [
                metric.get("structural_gate"),
                metric_audit.get("structural_gate"),
            ],
            [True, True],
        )
    if not bool(comparison.get("passes_structural_gate", False)) or not bool(
        comparison_audit.get("passes_structural_gate", False)
    ):
        violation(
            "comparison_structural_gate",
            [
                comparison.get("passes_structural_gate"),
                comparison_audit.get("passes_structural_gate"),
            ],
            [True, True],
        )

    if comparison.get("global_entity_concentration_consumed") is not False:
        violation(
            "global_entity_concentration_consumed",
            comparison.get("global_entity_concentration_consumed"),
            False,
        )
    if comparison.get("missing_context_is_not_quarantine") is not True:
        violation(
            "missing_context_is_not_quarantine",
            comparison.get("missing_context_is_not_quarantine"),
            True,
        )

    semantic_expected = protocol["frozen_semantics"]
    semantic_observed = {
        "corpus": corpus.get("corpus_semantics_id"),
        "reproducibility": repro.get("reproducibility_semantics_id"),
        "metric_definition": metric.get("metric_definition_semantics_id"),
        "comparison": comparison.get("comparison_semantics_id"),
        "method": comparison.get("method_semantics_id"),
        "quality_gate": comparison.get("quality_gate_semantics_id"),
        "measurement_result_identity": identity.get(
            "measurement_result_identity_semantics_id"
        ),
        "identity_aware_domain_reconstruction": comparison.get(
            "identity_aware_domain_reconstruction_id"
        ),
    }
    for key, observed in semantic_observed.items():
        if observed != semantic_expected[key]:
            violation(
                f"semantic_drift:{key}",
                observed,
                semantic_expected[key],
            )

    if metric.get("measurement_result_identity_id") != ids[
        "measurement_result_identity"
    ]:
        violation(
            "metric_definition_identity_binding",
            metric.get("measurement_result_identity_id"),
            ids["measurement_result_identity"],
        )
    if comparison.get("measurement_result_identity_id") != ids[
        "measurement_result_identity"
    ]:
        violation(
            "comparison_identity_binding",
            comparison.get("measurement_result_identity_id"),
            ids["measurement_result_identity"],
        )

    detection_limit_violations = detection_limit_ranking_violations(
        comparison_assessments
    )
    if detection_limit_violations:
        violation(
            "detection_limit_numeric_ranking_must_remain_disabled",
            len(detection_limit_violations),
            0,
        )

    nonfatal: list[dict[str, Any]] = []
    protocol_counts = comparison.get("protocol_comparability_counts", {})
    if protocol_counts.get("different_protocol", 0):
        nonfatal.append(
            {
                "category": "unsupported_context",
                "code": "different_protocol_pairs",
                "count": protocol_counts.get("different_protocol", 0),
                "action": "observe_not_tune",
            }
        )
    if protocol_counts.get("unknown", 0):
        nonfatal.append(
            {
                "category": "unsupported_context",
                "code": "unknown_protocol_pairs",
                "count": protocol_counts.get("unknown", 0),
                "action": "observe_not_tune",
            }
        )

    metric_statuses = metric.get("definition_status_counts", {})
    if metric_statuses.get("unknown", 0):
        nonfatal.append(
            {
                "category": "unsupported_context",
                "code": "unknown_metric_definitions",
                "count": metric_statuses.get("unknown", 0),
                "action": "observe_not_tune",
            }
        )
    if metric_statuses.get("partial", 0):
        nonfatal.append(
            {
                "category": "unsupported_context",
                "code": "partial_metric_definitions",
                "count": metric_statuses.get("partial", 0),
                "action": "observe_not_tune",
            }
        )

    metric_compat = comparison.get(
        "metric_definition_compatibility_counts", {}
    )
    if metric_compat.get("different_definition", 0):
        nonfatal.append(
            {
                "category": "unsupported_context",
                "code": "different_metric_definitions",
                "count": metric_compat.get("different_definition", 0),
                "action": "observe_not_tune",
            }
        )

    unresolved_identity = int(
        identity.get("unresolved_same_lineage_group_count", 0)
    )
    if unresolved_identity:
        nonfatal.append(
            {
                "category": "unsupported_context",
                "code": "unresolved_same_lineage_measurement_results",
                "count": unresolved_identity,
                "action": "preserve_separate_results_fail_closed",
            }
        )

    unregistered = int(
        comparison.get("unregistered_observable_assessment_count", 0)
    )
    if unregistered:
        nonfatal.append(
            {
                "category": "expected_new_content",
                "code": "unregistered_observable_assessments",
                "count": unregistered,
                "action": "review_without_policy_tuning",
            }
        )

    duplicate_pairs = int(
        repro.get("possible_duplicate_result_pair_count", 0)
    )
    if duplicate_pairs:
        nonfatal.append(
            {
                "category": "unsupported_context",
                "code": "possible_duplicate_reproducibility_results",
                "count": duplicate_pairs,
                "action": "preserve_fail_closed",
            }
        )

    distribution = {
        "paper_ids": corpus.get("paper_ids", []),
        "corpus_nodes": corpus.get("corpus_nodes"),
        "corpus_edges": corpus.get("corpus_edges"),
        "measurement_source_mention_count": identity.get(
            "source_mention_count"
        ),
        "scientific_result_count": identity.get("scientific_result_count"),
        "consolidated_exact_result_count": identity.get(
            "consolidated_exact_result_count"
        ),
        "unresolved_same_lineage_group_count": unresolved_identity,
        "comparison_context_count": comparison.get("context_count"),
        "comparison_assessment_count": comparison.get("assessment_count"),
        "compatibility_counts": comparison.get("compatibility_counts", {}),
        "protocol_comparability_counts": protocol_counts,
        "method_dimension_status_counts": comparison.get(
            "method_dimension_status_counts", {}
        ),
        "reproducibility_evidence_count": repro.get("evidence_count"),
        "reproducibility_kind_counts": repro.get(
            "evidence_kind_counts", {}
        ),
        "metric_definition_context_count": metric.get("context_count"),
        "metric_definition_status_counts": metric_statuses,
        "metric_definition_compatibility_counts": metric_compat,
        "metric_definition_ranking_relevant_assessment_count": (
            comparison.get(
                "metric_definition_ranking_relevant_assessment_count"
            )
        ),
        "metric_definition_ranking_relevant_gate_pass_count": (
            comparison.get(
                "metric_definition_ranking_relevant_gate_pass_count"
            )
        ),
        "numeric_ranking_allowed_count": comparison.get(
            "numeric_ranking_allowed_count"
        ),
        "observable_family_counts": comparison.get(
            "observable_family_counts", {}
        ),
        "unregistered_observable_assessment_count": unregistered,
    }

    return {
        "holdout_protocol_version": protocol["protocol_version"],
        "holdout_epoch": protocol["holdout_epoch"],
        "holdout_scope": protocol["holdout_scope"],
        "verdict": "pass" if not violations else "fail",
        "passes_holdout_invariants": not violations,
        "count_thresholds_used_for_acceptance": False,
        "heterogeneity_is_first_class": True,
        "distribution_observations": distribution,
        "nonfatal_classifications": nonfatal,
        "violations": violations,
        "interpretation": (
            "Unknown, ambiguous, different-protocol, different-definition, "
            "unresolved identity, and zero-rankable outcomes are valid "
            "holdout observations. Only structural/provenance/frozen-"
            "contract violations fail this holdout."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run refrozen alpha4b.4a11 SERS evidence-substrate holdout "
            "validation after the Measurement merge/result-identity epoch."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument(
        "--adopt-existing-bridge",
        action="store_true",
        help=(
            "Explicitly adopt pre-existing confirmed/candidate Bridge "
            "graphs. Recommended for this invariant-fix restart when the "
            "failed epoch already produced complete Bridge pairs."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = PROJECT_ROOT.resolve()
    protocol_path = (
        args.protocol
        if args.protocol.is_absolute()
        else (root / args.protocol)
    ).resolve()
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    validate_campaign_id(protocol, args.campaign_id)

    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if branch and branch != "feat/SERS-specification-v2.9.1":
        print(
            f"[WARNING] branch={branch!r}; exact frozen hashes, "
            "not branch name, are authoritative."
        )

    frozen_blobs = verify_frozen_blobs(root, protocol)
    runtime_semantics = verify_runtime_semantics(protocol)
    calibration_artifacts = verify_calibration_freeze(root, protocol)
    input_refreeze = verify_holdout_input_refreeze(root, protocol)
    input_snapshot = snapshot_holdout_inputs(root, protocol)

    evaluation_root = (
        root / "evaluation" / "sers_alpha4b4a11" / args.campaign_id
    )
    manifest_path = evaluation_root / "manifest.json"
    logs_root = evaluation_root / "logs"
    ids = {
        "corpus": f"{args.campaign_id}_corpus",
        "measurement_result_identity": (
            f"{args.campaign_id}_measurement_identity"
        ),
        "reproducibility": f"{args.campaign_id}_reproducibility",
        "metric_definition": f"{args.campaign_id}_metric_definition",
        "comparison": f"{args.campaign_id}_comparison",
    }

    protocol_hash = sha256_file(protocol_path)
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("protocol_sha256") != protocol_hash:
            raise FrozenContractViolation(
                "Holdout protocol changed after campaign start."
            )
        if manifest.get("frozen_git_blobs") != frozen_blobs:
            raise FrozenContractViolation(
                "Frozen implementation changed after campaign start."
            )
        if manifest.get("runtime_semantics") != runtime_semantics:
            raise FrozenContractViolation(
                "Runtime semantics changed after campaign start."
            )
        if manifest.get("calibration_artifacts") != calibration_artifacts:
            raise FrozenContractViolation(
                "Refrozen calibration artifacts changed after holdout start."
            )
        if manifest.get("input_refreeze") != input_refreeze:
            raise FrozenContractViolation(
                "Prepared holdout canonical inputs changed after campaign "
                "start."
            )
        if manifest.get("input_snapshot") != input_snapshot:
            raise FrozenContractViolation(
                "Held-out input snapshot changed after campaign start."
            )
        if bool(manifest.get("adopt_existing_bridge")) != bool(
            args.adopt_existing_bridge
        ):
            raise FrozenContractViolation(
                "Bridge adoption mode changed on campaign resume."
            )
        if bool(manifest.get("dry_run")) != bool(args.dry_run):
            raise FrozenContractViolation(
                "Dry-run and real campaign cannot share a campaign-id."
            )
    else:
        manifest = {
            "campaign_id": args.campaign_id,
            "holdout_epoch": protocol["holdout_epoch"],
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
            "input_refreeze": input_refreeze,
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
        os.environ.get(
            protocol["upstream_policy"]["bridge_model_environment"],
            "",
        ).strip()
        or os.environ.get(
            protocol["upstream_policy"][
                "strict_model_fallback_environment"
            ],
            "",
        ).strip()
    )
    provider = os.environ.get(
        protocol["upstream_policy"]["provider_environment"],
        "",
    ).strip()

    resolved_runtime = {
        "bridge_mode": (
            "explicit_existing_adoption"
            if args.adopt_existing_bridge
            else "fresh_frozen_policy"
        ),
        "bridge_model": (
            "" if args.adopt_existing_bridge else bridge_model
        ),
        "provider": "" if args.adopt_existing_bridge else provider,
        "bridge_concurrency": protocol["upstream_policy"][
            "bridge_concurrency"
        ],
    }
    prior_runtime = manifest.get("resolved_runtime")
    if prior_runtime is None:
        manifest["resolved_runtime"] = resolved_runtime
        save_manifest()
    elif prior_runtime != resolved_runtime:
        raise FrozenContractViolation(
            "Resolved Bridge model/provider/mode changed after campaign "
            "creation."
        )

    for paper_id in protocol["holdout_papers"]:
        paper_root = data_root / "extracted" / paper_id
        canonical = paper_root / f"{paper_id}.graphml"
        bridge = paper_root / f"{paper_id}.bridge.graphml"
        candidate = (
            paper_root / f"{paper_id}.bridge.candidates.graphml"
        )
        strict_run_id = input_snapshot[paper_id]["strict_run_id"]

        audit_dir = (
            evaluation_root / "canonical_audit" / paper_id
        )
        run_stage(
            manifest=manifest,
            save_manifest=save_manifest,
            stage=f"{paper_id}:canonical_audit",
            command=[
                sys.executable,
                "-m",
                "scripts.inspect_graphml",
                "--graphml",
                str(canonical),
                "--output-dir",
                str(audit_dir),
            ],
            root=root,
            logs_root=logs_root,
            dry_run=args.dry_run,
        )

        if args.adopt_existing_bridge:
            stage = ensure_stage(manifest, f"{paper_id}:bridge")
            if stage.get("status") != "complete":
                if not bridge.exists() or not candidate.exists():
                    raise FileNotFoundError(
                        "--adopt-existing-bridge requires both "
                        f"{bridge} and {candidate}."
                    )
                verify_bridge_pair(bridge, candidate, paper_id)
                stage.update(
                    {
                        "status": "complete",
                        "mode": "explicit_existing_adoption",
                        "confirmed_sha256": sha256_file(bridge),
                        "candidate_sha256": sha256_file(candidate),
                        "completed_at_utc": now_utc(),
                    }
                )
                save_manifest()
        else:
            if not args.dry_run:
                if not os.environ.get("OPENROUTER_API_KEY"):
                    raise RuntimeError(
                        "OPENROUTER_API_KEY is required unless "
                        "--adopt-existing-bridge is used."
                    )
                if not bridge_model:
                    raise RuntimeError(
                        "OPENROUTER_BRIDGE_MODEL or "
                        "OPENROUTER_EXTRACT_MODEL is required for fresh "
                        "Bridge extraction."
                    )
            bridge_command = [
                sys.executable,
                "-m",
                "scripts.extract_bridge_graph",
                "--paper-id",
                paper_id,
                "--config",
                str(config_path),
                "--domain-profile",
                str(protocol["domain_profile"]),
                "--data-root",
                str(protocol["data_root"]),
                "--run-id",
                strict_run_id,
                "--concurrency",
                str(
                    protocol["upstream_policy"]["bridge_concurrency"]
                ),
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
                max_passes=int(
                    protocol["upstream_policy"][
                        "bridge_max_outer_passes"
                    ]
                ),
                backoffs=[
                    int(value)
                    for value in protocol["upstream_policy"][
                        "bridge_backoff_seconds"
                    ]
                ],
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
                sys.executable,
                "-m",
                "scripts.build_graphagents_projection",
                "--paper-id",
                paper_id,
                "--domain-profile",
                str(protocol["domain_profile"]),
                "--data-root",
                str(protocol["data_root"]),
                "--mode",
                str(mode),
                "--canonical-graphml",
                str(canonical),
            ]
            if mode in {"mechanism", "exploratory"}:
                command.extend(
                    ["--bridge-graphml", str(bridge)]
                )
            if mode == "exploratory":
                command.extend(
                    ["--candidate-bridge-graphml", str(candidate)]
                )
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
                projection_root = (
                    paper_root / "graphagents" / str(mode)
                )
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

    paper_args = [
        str(paper_id)
        for paper_id in protocol["holdout_papers"]
    ]
    run_stage(
        manifest=manifest,
        save_manifest=save_manifest,
        stage="holdout_corpus",
        command=[
            sys.executable,
            "-m",
            "scripts.build_corpus_graph",
            "--corpus-id",
            ids["corpus"],
            "--domain-profile",
            str(protocol["domain_profile"]),
            "--data-root",
            str(protocol["data_root"]),
            "--paper-ids",
            *paper_args,
            "--mode",
            str(protocol["mode"]),
        ],
        root=root,
        logs_root=logs_root,
        dry_run=args.dry_run,
    )

    run_stage(
        manifest=manifest,
        save_manifest=save_manifest,
        stage="holdout_measurement_result_identity",
        command=[
            sys.executable,
            "-m",
            "scripts.build_measurement_result_identities",
            "--domain-profile",
            str(protocol["domain_profile"]),
            "--data-root",
            str(protocol["data_root"]),
            "--corpus-id",
            ids["corpus"],
            "--mode",
            str(protocol["mode"]),
            "--measurement-result-identity-id",
            ids["measurement_result_identity"],
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
            sys.executable,
            "-m",
            "scripts.build_reproducibility_evidence",
            "--domain-profile",
            str(protocol["domain_profile"]),
            "--data-root",
            str(protocol["data_root"]),
            "--corpus-id",
            ids["corpus"],
            "--mode",
            str(protocol["mode"]),
            "--reproducibility-id",
            ids["reproducibility"],
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
            sys.executable,
            "-m",
            "scripts.build_metric_definition_contexts",
            "--domain-profile",
            str(protocol["domain_profile"]),
            "--data-root",
            str(protocol["data_root"]),
            "--corpus-id",
            ids["corpus"],
            "--mode",
            str(protocol["mode"]),
            "--metric-definition-id",
            ids["metric_definition"],
            "--measurement-result-identity-id",
            ids["measurement_result_identity"],
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
            sys.executable,
            "-m",
            "scripts.build_comparison_contexts",
            "--domain-profile",
            str(protocol["domain_profile"]),
            "--data-root",
            str(protocol["data_root"]),
            "--corpus-id",
            ids["corpus"],
            "--mode",
            str(protocol["mode"]),
            "--comparison-id",
            ids["comparison"],
            "--metric-definition-id",
            ids["metric_definition"],
            "--measurement-result-identity-id",
            ids["measurement_result_identity"],
        ],
        root=root,
        logs_root=logs_root,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        corpus_root = (
            data_root / "corpus" / ids["corpus"] / str(protocol["mode"])
        )
        lock_or_verify_derived_files(
            manifest=manifest,
            save_manifest=save_manifest,
            root=root,
            lock_id="holdout_corpus",
            paths=[
                corpus_root / "manifest.json",
                corpus_root / "audit.json",
                corpus_root / "graph.graphml",
            ],
        )

        identity_root = (
            corpus_root
            / "measurement_result_identity"
            / ids["measurement_result_identity"]
        )
        lock_or_verify_derived_files(
            manifest=manifest,
            save_manifest=save_manifest,
            root=root,
            lock_id="holdout_measurement_result_identity",
            paths=[
                identity_root / "summary.json",
                identity_root / "audit.json",
                identity_root / "identities.jsonl",
                identity_root / "same_lineage_candidates.jsonl",
            ],
        )

        repro_root = (
            corpus_root
            / "reproducibility"
            / ids["reproducibility"]
        )
        lock_or_verify_derived_files(
            manifest=manifest,
            save_manifest=save_manifest,
            root=root,
            lock_id="holdout_reproducibility",
            paths=[
                repro_root / "summary.json",
                repro_root / "audit.json",
                repro_root / "evidence.jsonl",
            ],
        )

        metric_root = (
            corpus_root
            / "metric_definition"
            / ids["metric_definition"]
        )
        lock_or_verify_derived_files(
            manifest=manifest,
            save_manifest=save_manifest,
            root=root,
            lock_id="holdout_metric_definition",
            paths=[
                metric_root / "summary.json",
                metric_root / "audit.json",
                metric_root / "contexts.jsonl",
            ],
        )

        comparison_root = (
            corpus_root / "comparison" / ids["comparison"]
        )
        lock_or_verify_derived_files(
            manifest=manifest,
            save_manifest=save_manifest,
            root=root,
            lock_id="holdout_comparison",
            paths=[
                comparison_root / "summary.json",
                comparison_root / "audit.json",
                comparison_root / "contexts.jsonl",
                comparison_root / "assessments.jsonl",
                comparison_root / "protocol_assessments.jsonl",
                comparison_root / "metric_definition_assessments.jsonl",
                comparison_root / "method_contexts.jsonl",
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
    verify_holdout_input_refreeze(root, protocol)

    report = build_holdout_report(root, protocol, ids)
    report_path = evaluation_root / "holdout_report.json"
    atomic_write_json(report_path, report)
    manifest["holdout_report"] = str(report_path.relative_to(root))
    manifest["status"] = (
        "complete"
        if report["passes_holdout_invariants"]
        else "failed"
    )
    manifest["finished_at_utc"] = now_utc()
    save_manifest()

    print()
    print("alpha4b.4a11 holdout verdict:", report["verdict"])
    print(
        "Count thresholds used for acceptance:",
        report["count_thresholds_used_for_acceptance"],
    )
    print(
        "Heterogeneity is first-class:",
        report["heterogeneity_is_first_class"],
    )
    print("Report:", report_path)
    print("Manifest:", manifest_path)
    if report["nonfatal_classifications"]:
        print("Non-fatal holdout observations:")
        for item in report["nonfatal_classifications"]:
            print(
                " -",
                item["category"],
                item["code"],
                item.get("count", ""),
            )
    if report["violations"]:
        print("Holdout invariant violations:")
        for item in report["violations"]:
            print(
                " -",
                item["code"],
                "observed=",
                item["observed"],
                "expected=",
                item["expected"],
            )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
