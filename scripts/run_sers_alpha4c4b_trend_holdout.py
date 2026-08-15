from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import networkx as nx

from dac_her.measurement_merge_invariants import (
    MEASUREMENT_MERGE_INVARIANT_ID,
    measurement_value_payload_issues,
)
from dac_her.trend_holdout import validate_protocol_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "configs"
    / "heldout"
    / "sers_alpha4c4b_trend_holdout_run.json"
)


class FrozenTrendHoldoutError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FrozenTrendHoldoutError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise FrozenTrendHoldoutError(
                    f"JSONL row must be an object: {path}:{line_number}"
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
    os.replace(temp, path)


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
        raise FrozenTrendHoldoutError(
            f"Cannot hash implementation file {path}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def _require_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise FrozenTrendHoldoutError(
            f"{label} mismatch: {observed!r} != {expected!r}"
        )


def _verify_semantics(protocol: Mapping[str, Any]) -> dict[str, str]:
    from dac_her.cross_context_trend import (
        CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
    )
    from dac_her.cross_context_trend_assessment import (
        CROSS_CONTEXT_TREND_ASSESSMENT_SEMANTICS_ID,
    )
    from dac_her.domains.comparison_registry import get_comparison_adapter
    from dac_her.domains.metric_definition_registry import (
        get_metric_definition_adapter,
    )
    from dac_her.domains.registry import get_domain_profile
    from dac_her.domains.sers_au_ag_cross_context_trend import (
        SERS_AU_AG_TREND_CONTEXT_SEMANTICS_ID,
    )
    from dac_her.domains.trend_precision_registry import (
        get_trend_precision_adapter,
    )
    from dac_her.domains.trend_registry import get_trend_adapter
    from dac_her.measurement_result_identity import (
        MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID,
    )
    from dac_her.quality_aware_comparison import (
        QUALITY_AWARE_NUMERIC_GATE_SEMANTICS_ID,
    )
    from dac_her.trend_domain import (
        TREND_EVIDENCE_CONTRACT_SEMANTICS_ID,
    )

    profile = get_domain_profile("sers_au_ag")
    comparison = get_comparison_adapter(profile)
    metric = get_metric_definition_adapter(profile)
    trend = get_trend_adapter(profile)
    precision = get_trend_precision_adapter(profile)

    observed = {
        "projection": str(profile.projection.semantics_id),
        "corpus": str(profile.corpus.semantics_id),
        "measurement_merge_invariant":
            str(MEASUREMENT_MERGE_INVARIANT_ID),
        "measurement_result_identity":
            str(MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID),
        "metric_definition": str(metric.semantics_id),
        "comparison": str(comparison.semantics_id),
        "method": str(
            comparison.method_semantics.semantics_id
            if comparison.method_semantics
            else ""
        ),
        "quality_gate": str(
            QUALITY_AWARE_NUMERIC_GATE_SEMANTICS_ID
        ),
        "trend_contract": str(
            TREND_EVIDENCE_CONTRACT_SEMANTICS_ID
        ),
        "trend": str(trend.semantics_id),
        "trend_precision": str(
            precision.precision_semantics_id
        ),
        "cross_context_contract": str(
            CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID
        ),
        "trend_context": str(
            SERS_AU_AG_TREND_CONTEXT_SEMANTICS_ID
        ),
        "cross_context_assessment": str(
            CROSS_CONTEXT_TREND_ASSESSMENT_SEMANTICS_ID
        ),
    }
    expected = {
        str(key): str(value)
        for key, value in protocol["frozen_semantics"].items()
    }
    _require_equal("frozen semantics", observed, expected)
    return observed


def _verify_frozen_implementation(
    protocol: Mapping[str, Any],
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative_path, expected_blob in sorted(
        protocol["frozen_implementation_blobs"].items()
    ):
        path = PROJECT_ROOT / str(relative_path)
        if not path.exists():
            raise FrozenTrendHoldoutError(
                f"Frozen implementation file is missing: {relative_path}"
            )
        blob = _git_blob(path)
        observed[str(relative_path)] = blob
        if blob != expected_blob:
            raise FrozenTrendHoldoutError(
                "Frozen implementation drift detected: "
                f"{relative_path}: {blob} != {expected_blob}"
            )
    return observed


def _verify_source_split_protocol(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    source = protocol["source_split_protocol"]
    path = PROJECT_ROOT / str(source["path"])
    if not path.exists():
        raise FrozenTrendHoldoutError(
            f"alpha4c.4a source split protocol is missing: {path}"
        )
    _require_equal(
        "alpha4c.4a source protocol SHA256",
        _sha256(path),
        source["sha256"],
    )

    split_protocol = _read_json(path)
    split = validate_protocol_split(split_protocol)
    _require_equal(
        "alpha4c.4a split SHA256",
        split.split_sha256,
        source["split_sha256"],
    )
    _require_equal(
        "frozen holdout paper list",
        list(split.holdout_papers),
        list(protocol["holdout_papers"]),
    )
    _require_equal(
        "future reserve paper list",
        list(split.reserved_future_papers),
        list(protocol["future_reserve_papers"]),
    )
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": _sha256(path),
        "split_sha256": split.split_sha256,
        "holdout_papers": list(split.holdout_papers),
        "reserve_count": len(split.reserved_future_papers),
    }


def verify_protocol(
    protocol_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _read_json(protocol_path)
    _require_equal("phase", protocol.get("phase"), "alpha4c.4b")
    _require_equal(
        "state",
        protocol.get("state"),
        "enabled_frozen_trend_holdout_v1",
    )
    _require_equal("domain profile", protocol.get("domain_profile"), "sers_au_ag")
    _require_equal("projection mode", protocol.get("mode"), "evidence")
    _require_equal("LLM calls allowed", protocol.get("llm_calls_allowed"), False)
    _require_equal("Bridge used", protocol.get("bridge_used"), False)
    _require_equal(
        "arbitrary paper override allowed",
        protocol.get("arbitrary_paper_override_allowed"),
        False,
    )

    holdout = list(protocol.get("holdout_papers", []))
    if len(holdout) != 10 or len(holdout) != len(set(holdout)):
        raise FrozenTrendHoldoutError(
            "alpha4c.4b must contain exactly 10 unique frozen holdout papers."
        )

    source_split = _verify_source_split_protocol(protocol)
    semantics = _verify_semantics(protocol)
    blobs = _verify_frozen_implementation(protocol)

    acceptance = protocol["acceptance_policy"]
    _require_equal(
        "count thresholds used",
        acceptance.get("count_thresholds_used"),
        False,
    )
    for key in (
        "minimum_trend_evidence_count",
        "minimum_cross_paper_pair_count",
        "minimum_repeated_count",
        "minimum_reversed_count",
        "minimum_context_specific_count",
        "maximum_insufficient_count",
    ):
        if acceptance.get(key) is not None:
            raise FrozenTrendHoldoutError(
                f"Forbidden holdout distribution target is populated: {key}"
            )

    return protocol, {
        "source_split": source_split,
        "semantics": semantics,
        "implementation_blobs": blobs,
    }


def _nonempty_resolution_decisions(paper_root: Path) -> list[str]:
    path = paper_root / "resolution" / "decisions.jsonl"
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def snapshot_canonical_inputs(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    data_root = PROJECT_ROOT / str(protocol["data_root"])
    expected_invariant = str(
        protocol["frozen_semantics"]["measurement_merge_invariant"]
    )

    missing: list[str] = []
    records: dict[str, Any] = {}
    for paper_id in protocol["holdout_papers"]:
        paper_root = data_root / "extracted" / str(paper_id)
        canonical = paper_root / f"{paper_id}.graphml"
        if not canonical.exists():
            missing.append(str(paper_id))
            continue

        decisions = _nonempty_resolution_decisions(paper_root)
        if decisions:
            raise FrozenTrendHoldoutError(
                f"Manual resolution decisions are forbidden for frozen "
                f"Trend holdout paper {paper_id}: "
                f"{paper_root / 'resolution' / 'decisions.jsonl'}"
            )

        graph = nx.read_graphml(canonical, force_multigraph=True)
        domain = str(graph.graph.get("domain_profile_id", ""))
        if domain and domain != "sers_au_ag":
            raise FrozenTrendHoldoutError(
                f"{paper_id} canonical domain mismatch: {domain!r}"
            )
        invariant = str(
            graph.graph.get("measurement_merge_invariant_id", "")
        )
        if invariant != expected_invariant:
            raise FrozenTrendHoldoutError(
                f"{paper_id} canonical graph is not in the frozen "
                f"Measurement merge epoch: {invariant!r} != "
                f"{expected_invariant!r}"
            )
        xor_issues = measurement_value_payload_issues(graph)
        if xor_issues:
            raise FrozenTrendHoldoutError(
                f"{paper_id} canonical graph violates Measurement "
                f"numeric/text XOR: {xor_issues[:5]!r}"
            )

        records[str(paper_id)] = {
            "path": str(canonical.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(canonical),
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "measurement_merge_invariant_id": invariant,
            "measurement_xor_issue_count": 0,
            "manual_resolution_decision_count": 0,
        }

    if missing:
        raise FrozenTrendHoldoutError(
            "Frozen alpha4c.4b canonical KG substrate is not ready for "
            "all holdout papers. Missing canonical GraphML for: "
            + ", ".join(missing)
            + ". Prepare these papers with the existing frozen Strict/"
            "paper-graph workflow without changing Trend semantics, then "
            "rerun --preflight-only. The 4c.4b runner intentionally does "
            "not invoke LLM extraction."
        )

    _require_equal(
        "canonical snapshot paper set",
        sorted(records),
        sorted(map(str, protocol["holdout_papers"])),
    )
    return records


def _verify_canonical_snapshot_unchanged(
    snapshot: Mapping[str, Any],
) -> None:
    for paper_id, record in snapshot.items():
        path = PROJECT_ROOT / str(record["path"])
        if not path.exists():
            raise FrozenTrendHoldoutError(
                f"Canonical input disappeared: {paper_id}: {path}"
            )
        _require_equal(
            f"{paper_id} canonical SHA256",
            _sha256(path),
            record["sha256"],
        )


def _run_stage(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    stage: str,
    command: list[str],
    output_paths: Iterable[Path],
    dry_run: bool,
) -> None:
    stages = manifest.setdefault("stages", {})
    existing = stages.get(stage)
    outputs = [Path(path) for path in output_paths]

    if isinstance(existing, dict) and existing.get("status") == "complete":
        locked = existing.get("outputs", {})
        for path in outputs:
            rel = str(path.relative_to(PROJECT_ROOT))
            expected = locked.get(rel)
            if not path.exists() or not expected:
                raise FrozenTrendHoldoutError(
                    f"Completed stage output is missing: {stage}: {rel}"
                )
            _require_equal(
                f"locked stage output {stage}/{rel}",
                _sha256(path),
                expected,
            )
        print(f"[SKIP VERIFIED] {stage}")
        return

    print("$", " ".join(command), flush=True)
    if dry_run:
        stages[stage] = {
            "status": "planned",
            "command": command,
            "outputs": [
                str(path.relative_to(PROJECT_ROOT))
                for path in outputs
            ],
        }
        _atomic_json(manifest_path, manifest)
        return

    stages[stage] = {
        "status": "running",
        "command": command,
        "started_at_utc": _now(),
    }
    _atomic_json(manifest_path, manifest)

    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        stages[stage]["status"] = "failed"
        stages[stage]["exit_code"] = result.returncode
        stages[stage]["finished_at_utc"] = _now()
        _atomic_json(manifest_path, manifest)
        raise FrozenTrendHoldoutError(
            f"Stage {stage!r} failed with exit code {result.returncode}."
        )

    locked: dict[str, str] = {}
    for path in outputs:
        if not path.exists():
            raise FrozenTrendHoldoutError(
                f"Stage {stage!r} did not produce required output: {path}"
            )
        locked[str(path.relative_to(PROJECT_ROOT))] = _sha256(path)

    stages[stage] = {
        "status": "complete",
        "command": command,
        "started_at_utc": stages[stage].get("started_at_utc"),
        "finished_at_utc": _now(),
        "outputs": locked,
    }
    _atomic_json(manifest_path, manifest)


def _ids(protocol: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in protocol["artifact_ids"].items()
    }


def _corpus_root(
    protocol: Mapping[str, Any],
    ids: Mapping[str, str],
) -> Path:
    return (
        PROJECT_ROOT
        / str(protocol["data_root"])
        / "corpus"
        / ids["corpus"]
        / str(protocol["mode"])
    )


def _run_pipeline(
    *,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
    dry_run: bool,
) -> None:
    ids = _ids(protocol)
    data_root = PROJECT_ROOT / str(protocol["data_root"])
    mode = str(protocol["mode"])
    paper_args = [str(p) for p in protocol["holdout_papers"]]

    # Evidence projection deliberately excludes Bridge: the Trend stack uses
    # canonical + frozen sidecars, and Bridge is not a Trend evidence input.
    for paper_id in paper_args:
        canonical = (
            data_root
            / "extracted"
            / paper_id
            / f"{paper_id}.graphml"
        )
        projection_root = (
            data_root
            / "extracted"
            / paper_id
            / "graphagents"
            / mode
        )
        _run_stage(
            manifest=manifest,
            manifest_path=manifest_path,
            stage=f"projection:{paper_id}",
            command=[
                sys.executable,
                "-m",
                "scripts.build_graphagents_projection",
                "--paper-id",
                paper_id,
                "--domain-profile",
                "sers_au_ag",
                "--data-root",
                str(protocol["data_root"]),
                "--mode",
                mode,
                "--canonical-graphml",
                str(canonical),
            ],
            output_paths=[
                projection_root / "graph.graphml",
                projection_root / "summary.json",
                projection_root / "node_text.jsonl",
                projection_root / "edge_evidence.jsonl",
            ],
            dry_run=dry_run,
        )

    corpus_root = _corpus_root(protocol, ids)
    _run_stage(
        manifest=manifest,
        manifest_path=manifest_path,
        stage="corpus",
        command=[
            sys.executable,
            "-m",
            "scripts.build_corpus_graph",
            "--corpus-id",
            ids["corpus"],
            "--domain-profile",
            "sers_au_ag",
            "--data-root",
            str(protocol["data_root"]),
            "--paper-ids",
            *paper_args,
            "--mode",
            mode,
        ],
        output_paths=[
            corpus_root / "manifest.json",
            corpus_root / "audit.json",
            corpus_root / "graph.graphml",
        ],
        dry_run=dry_run,
    )

    identity_root = (
        corpus_root
        / "measurement_result_identity"
        / ids["measurement_result_identity"]
    )
    _run_stage(
        manifest=manifest,
        manifest_path=manifest_path,
        stage="measurement_result_identity",
        command=[
            sys.executable,
            "-m",
            "scripts.build_measurement_result_identities",
            "--domain-profile",
            "sers_au_ag",
            "--data-root",
            str(protocol["data_root"]),
            "--corpus-id",
            ids["corpus"],
            "--mode",
            mode,
            "--measurement-result-identity-id",
            ids["measurement_result_identity"],
        ],
        output_paths=[
            identity_root / "summary.json",
            identity_root / "audit.json",
            identity_root / "identities.jsonl",
            identity_root / "same_lineage_candidates.jsonl",
        ],
        dry_run=dry_run,
    )

    metric_root = (
        corpus_root
        / "metric_definition"
        / ids["metric_definition"]
    )
    _run_stage(
        manifest=manifest,
        manifest_path=manifest_path,
        stage="metric_definition",
        command=[
            sys.executable,
            "-m",
            "scripts.build_metric_definition_contexts",
            "--domain-profile",
            "sers_au_ag",
            "--data-root",
            str(protocol["data_root"]),
            "--corpus-id",
            ids["corpus"],
            "--mode",
            mode,
            "--metric-definition-id",
            ids["metric_definition"],
            "--measurement-result-identity-id",
            ids["measurement_result_identity"],
        ],
        output_paths=[
            metric_root / "summary.json",
            metric_root / "audit.json",
            metric_root / "contexts.jsonl",
        ],
        dry_run=dry_run,
    )

    comparison_root = (
        corpus_root
        / "comparison"
        / ids["comparison"]
    )
    _run_stage(
        manifest=manifest,
        manifest_path=manifest_path,
        stage="comparison",
        command=[
            sys.executable,
            "-m",
            "scripts.build_comparison_contexts",
            "--domain-profile",
            "sers_au_ag",
            "--data-root",
            str(protocol["data_root"]),
            "--corpus-id",
            ids["corpus"],
            "--mode",
            mode,
            "--comparison-id",
            ids["comparison"],
            "--metric-definition-id",
            ids["metric_definition"],
            "--measurement-result-identity-id",
            ids["measurement_result_identity"],
        ],
        output_paths=[
            comparison_root / "summary.json",
            comparison_root / "audit.json",
            comparison_root / "contexts.jsonl",
            comparison_root / "method_contexts.jsonl",
            comparison_root / "assessments.jsonl",
            comparison_root / "protocol_assessments.jsonl",
            comparison_root / "metric_definition_assessments.jsonl",
        ],
        dry_run=dry_run,
    )

    trend_root = (
        corpus_root / "trend" / ids["trend"]
    )
    _run_stage(
        manifest=manifest,
        manifest_path=manifest_path,
        stage="trend_evidence",
        command=[
            sys.executable,
            "-m",
            "scripts.build_trend_evidence",
            "--domain-profile",
            "sers_au_ag",
            "--data-root",
            str(protocol["data_root"]),
            "--corpus-id",
            ids["corpus"],
            "--mode",
            mode,
            "--trend-id",
            ids["trend"],
            "--comparison-id",
            ids["comparison"],
            "--measurement-result-identity-id",
            ids["measurement_result_identity"],
        ],
        output_paths=[
            trend_root / "summary.json",
            trend_root / "audit.json",
            trend_root / "evidence.jsonl",
        ],
        dry_run=dry_run,
    )

    precision_root = (
        trend_root / "precision" / ids["precision"]
    )
    _run_stage(
        manifest=manifest,
        manifest_path=manifest_path,
        stage="trend_precision",
        command=[
            sys.executable,
            "-m",
            "scripts.build_trend_precision",
            "--domain-profile",
            "sers_au_ag",
            "--data-root",
            str(protocol["data_root"]),
            "--corpus-id",
            ids["corpus"],
            "--mode",
            mode,
            "--trend-id",
            ids["trend"],
            "--precision-id",
            ids["precision"],
        ],
        output_paths=[
            precision_root / "summary.json",
            precision_root / "audit.json",
            precision_root / "annotations.jsonl",
            precision_root / "local_results.jsonl",
        ],
        dry_run=dry_run,
    )

    context_root = (
        precision_root
        / "cross_context"
        / ids["context"]
    )
    _run_stage(
        manifest=manifest,
        manifest_path=manifest_path,
        stage="trend_context",
        command=[
            sys.executable,
            "-m",
            "scripts.build_cross_context_profiles",
            "--domain-profile",
            "sers_au_ag",
            "--data-root",
            str(protocol["data_root"]),
            "--corpus-id",
            ids["corpus"],
            "--mode",
            mode,
            "--trend-id",
            ids["trend"],
            "--precision-id",
            ids["precision"],
            "--context-id",
            ids["context"],
        ],
        output_paths=[
            context_root / "summary.json",
            context_root / "audit.json",
            context_root / "context_profiles.jsonl",
        ],
        dry_run=dry_run,
    )

    assessment_root = (
        context_root
        / "assessment"
        / ids["assessment"]
    )
    _run_stage(
        manifest=manifest,
        manifest_path=manifest_path,
        stage="cross_context_assessment",
        command=[
            sys.executable,
            "-m",
            "scripts.build_cross_context_assessments",
            "--domain-profile",
            "sers_au_ag",
            "--data-root",
            str(protocol["data_root"]),
            "--corpus-id",
            ids["corpus"],
            "--mode",
            mode,
            "--trend-id",
            ids["trend"],
            "--precision-id",
            ids["precision"],
            "--context-id",
            ids["context"],
            "--assessment-id",
            ids["assessment"],
        ],
        output_paths=[
            assessment_root / "summary.json",
            assessment_root / "audit.json",
            assessment_root / "pairwise_contrasts.jsonl",
            assessment_root / "assessments.jsonl",
        ],
        dry_run=dry_run,
    )


def _build_report(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    ids = _ids(protocol)
    corpus_root = _corpus_root(protocol, ids)

    corpus = _read_json(corpus_root / "manifest.json")
    corpus_audit = _read_json(corpus_root / "audit.json")

    identity_root = (
        corpus_root
        / "measurement_result_identity"
        / ids["measurement_result_identity"]
    )
    identity = _read_json(identity_root / "summary.json")
    identity_audit = _read_json(identity_root / "audit.json")

    metric_root = (
        corpus_root
        / "metric_definition"
        / ids["metric_definition"]
    )
    metric = _read_json(metric_root / "summary.json")
    metric_audit = _read_json(metric_root / "audit.json")

    comparison_root = (
        corpus_root / "comparison" / ids["comparison"]
    )
    comparison = _read_json(comparison_root / "summary.json")
    comparison_audit = _read_json(comparison_root / "audit.json")

    trend_root = corpus_root / "trend" / ids["trend"]
    trend = _read_json(trend_root / "summary.json")
    trend_audit = _read_json(trend_root / "audit.json")
    trend_rows = _read_jsonl(trend_root / "evidence.jsonl")

    precision_root = trend_root / "precision" / ids["precision"]
    precision = _read_json(precision_root / "summary.json")
    precision_audit = _read_json(precision_root / "audit.json")
    local_results = _read_jsonl(precision_root / "local_results.jsonl")

    context_root = (
        precision_root
        / "cross_context"
        / ids["context"]
    )
    context = _read_json(context_root / "summary.json")
    context_audit = _read_json(context_root / "audit.json")
    profiles = _read_jsonl(context_root / "context_profiles.jsonl")

    assessment_root = (
        context_root / "assessment" / ids["assessment"]
    )
    assessment = _read_json(assessment_root / "summary.json")
    assessment_audit = _read_json(assessment_root / "audit.json")
    contrasts = _read_jsonl(
        assessment_root / "pairwise_contrasts.jsonl"
    )
    assessments = _read_jsonl(
        assessment_root / "assessments.jsonl"
    )

    violations: list[dict[str, Any]] = []

    def fail(code: str, observed: Any, expected: Any) -> None:
        violations.append(
            {
                "code": code,
                "observed": observed,
                "expected": expected,
            }
        )

    expected_papers = list(protocol["holdout_papers"])
    if sorted(map(str, corpus.get("paper_ids", []))) != sorted(expected_papers):
        fail(
            "corpus_paper_set",
            sorted(map(str, corpus.get("paper_ids", []))),
            sorted(expected_papers),
        )
    if not bool(corpus.get("passes_structural_gate", False)):
        fail(
            "corpus_structural_gate",
            corpus.get("passes_structural_gate"),
            True,
        )
    if not bool(corpus_audit.get("passes_structural_gate", False)):
        fail(
            "corpus_audit_structural_gate",
            corpus_audit.get("passes_structural_gate"),
            True,
        )
    if int(corpus.get("destructive_cross_paper_merges", -1)) != 0:
        fail(
            "destructive_cross_paper_merges",
            corpus.get("destructive_cross_paper_merges"),
            0,
        )

    for name, summary, audit, gate_key in (
        ("identity", identity, identity_audit, "structural_gate"),
        ("metric_definition", metric, metric_audit, "structural_gate"),
        ("trend", trend, trend_audit, "structural_gate"),
        ("precision", precision, precision_audit, "structural_gate"),
        ("context", context, context_audit, "structural_gate"),
        ("assessment", assessment, assessment_audit, "structural_gate"),
    ):
        if summary.get(gate_key) is not True:
            fail(
                f"{name}_structural_gate",
                summary.get(gate_key),
                True,
            )
        if audit.get("structural_gate") is not True:
            fail(
                f"{name}_audit_structural_gate",
                audit.get("structural_gate"),
                True,
            )

    if comparison.get("passes_structural_gate") is not True:
        fail(
            "comparison_structural_gate",
            comparison.get("passes_structural_gate"),
            True,
        )
    if comparison_audit.get("passes_structural_gate") is not True:
        fail(
            "comparison_audit_structural_gate",
            comparison_audit.get("passes_structural_gate"),
            True,
        )
    if comparison.get("global_entity_concentration_consumed") is not False:
        fail(
            "global_entity_concentration_consumed",
            comparison.get("global_entity_concentration_consumed"),
            False,
        )
    if comparison.get("missing_context_is_not_quarantine") is not True:
        fail(
            "missing_context_is_not_quarantine",
            comparison.get("missing_context_is_not_quarantine"),
            True,
        )

    # Exact sidecar binding.
    if metric.get("measurement_result_identity_id") != ids[
        "measurement_result_identity"
    ]:
        fail(
            "metric_identity_binding",
            metric.get("measurement_result_identity_id"),
            ids["measurement_result_identity"],
        )
    if comparison.get("measurement_result_identity_id") != ids[
        "measurement_result_identity"
    ]:
        fail(
            "comparison_identity_binding",
            comparison.get("measurement_result_identity_id"),
            ids["measurement_result_identity"],
        )
    if trend.get("measurement_result_identity_id") != ids[
        "measurement_result_identity"
    ]:
        fail(
            "trend_identity_binding",
            trend.get("measurement_result_identity_id"),
            ids["measurement_result_identity"],
        )
    if trend.get("comparison_id") != ids["comparison"]:
        fail(
            "trend_comparison_binding",
            trend.get("comparison_id"),
            ids["comparison"],
        )

    # Trend-specific non-leakage and no-policy-reuse guarantees.
    if trend.get("cross_paper_numeric_series_built") is not False:
        fail(
            "cross_paper_numeric_series_built",
            trend.get("cross_paper_numeric_series_built"),
            False,
        )
    if trend.get("numeric_ranking_reused_as_trend_policy") is not False:
        fail(
            "numeric_ranking_reused_at_trend",
            trend.get("numeric_ranking_reused_as_trend_policy"),
            False,
        )
    if trend_audit.get("issues") not in ([], None):
        fail("trend_audit_issues", trend_audit.get("issues"), [])

    if context.get("paper_global_context_fallback_used") is not False:
        fail(
            "paper_global_context_fallback_used",
            context.get("paper_global_context_fallback_used"),
            False,
        )
    if int(context.get("paper_global_leakage_count", 0)) != 0:
        fail(
            "paper_global_leakage_count",
            context.get("paper_global_leakage_count"),
            0,
        )
    if context.get("numeric_ranking_reused_as_trend_policy") is not False:
        fail(
            "numeric_ranking_reused_at_context",
            context.get("numeric_ranking_reused_as_trend_policy"),
            False,
        )
    if context.get("pairwise_contrasts_built") is not False:
        fail(
            "context_source_pairwise_purity",
            context.get("pairwise_contrasts_built"),
            False,
        )
    if context.get("cross_context_assessments_built") is not False:
        fail(
            "context_source_assessment_purity",
            context.get("cross_context_assessments_built"),
            False,
        )

    for key, expected in (
        ("majority_vote_used", False),
        ("same_paper_pairs_allowed", False),
        ("numeric_ranking_reused_as_trend_policy", False),
        ("context_reprojected", False),
        ("causal_status_promoted", False),
    ):
        if assessment.get(key) is not expected:
            fail(
                f"assessment_policy:{key}",
                assessment.get(key),
                expected,
            )

    # Independent row-level guards.
    holdout_set = set(map(str, protocol["holdout_papers"]))
    outside_trends = sorted({
        str(row.get("paper_id", ""))
        for row in trend_rows
        if str(row.get("paper_id", "")) not in holdout_set
    })
    if outside_trends:
        fail("trend_outside_holdout", outside_trends, [])

    causal_correlations = [
        row.get("trend_id")
        for row in trend_rows
        if row.get("evidence_basis") == "reported_correlation"
        and row.get("causal_status") != "not_asserted"
    ]
    if causal_correlations:
        fail(
            "correlation_promoted_to_causation",
            causal_correlations,
            [],
        )

    same_paper_pairs = [
        row.get("contrast_id")
        for row in contrasts
        if row.get("left_paper_id") == row.get("right_paper_id")
    ]
    if same_paper_pairs:
        fail("same_paper_pairing", same_paper_pairs, [])

    bad_reversal = [
        row.get("assessment_id")
        for row in assessments
        if row.get("reversal_pair_ids")
        and row.get("status") != "reversed"
    ]
    if bad_reversal:
        fail("reversal_majority_vote", bad_reversal, [])

    bad_single_paper = [
        row.get("assessment_id")
        for row in assessments
        if len(row.get("paper_ids", [])) < 2
        and row.get("status") != "insufficient"
    ]
    if bad_single_paper:
        fail(
            "same_paper_support_counted_as_replication",
            bad_single_paper,
            [],
        )

    if (
        assessment.get("expected_cross_paper_pair_count")
        != assessment.get("pairwise_contrast_count")
        or assessment.get("pairwise_contrast_count") != len(contrasts)
    ):
        fail(
            "incomplete_pair_generation",
            {
                "expected": assessment.get(
                    "expected_cross_paper_pair_count"
                ),
                "summary": assessment.get("pairwise_contrast_count"),
                "rows": len(contrasts),
            },
            "all equal",
        )

    if (
        assessment.get("relation_count")
        != assessment.get("assessment_count")
        or assessment.get("assessment_count") != len(assessments)
    ):
        fail(
            "one_assessment_per_relation",
            {
                "relations": assessment.get("relation_count"),
                "summary": assessment.get("assessment_count"),
                "rows": len(assessments),
            },
            "all equal",
        )

    status_counts = Counter(
        str(row.get("status", ""))
        for row in assessments
    )
    evidence_basis_counts = Counter(
        str(row.get("evidence_basis", ""))
        for row in trend_rows
    )
    direction_counts = Counter(
        str(row.get("direction", ""))
        for row in trend_rows
    )

    distribution = {
        "holdout_paper_count": len(expected_papers),
        "trend_evidence_count": len(trend_rows),
        "trend_evidence_basis_counts":
            dict(sorted(evidence_basis_counts.items())),
        "trend_direction_counts":
            dict(sorted(direction_counts.items())),
        "paper_local_result_count": len(local_results),
        "context_profile_count": len(profiles),
        "relation_count": assessment.get("relation_count"),
        "cross_paper_pair_count": len(contrasts),
        "assessment_count": len(assessments),
        "status_counts": dict(sorted(status_counts.items())),
        "identity_source_mention_count":
            identity.get("source_mention_count"),
        "identity_scientific_result_count":
            identity.get("scientific_result_count"),
        "metric_definition_context_count":
            metric.get("context_count"),
        "comparison_context_count":
            comparison.get("context_count"),
    }

    return {
        "phase": "alpha4c.4b",
        "holdout_epoch": protocol["holdout_epoch"],
        "verdict": "pass" if not violations else "fail",
        "passes_frozen_trend_holdout_invariants": not violations,
        "count_thresholds_used_for_acceptance": False,
        "holdout_distribution_is_observational_only": True,
        "llm_calls_performed_by_runner": False,
        "bridge_used": False,
        "projection_mode": protocol["mode"],
        "holdout_papers": expected_papers,
        "distribution_observations": distribution,
        "violations": violations,
        "interpretation": (
            "Zero TrendEvidence, zero cross-paper overlap, zero repeated/"
            "reversed/context-specific assessments, or all-insufficient "
            "assessments are valid holdout outcomes. This verdict is based "
            "only on frozen semantic/implementation bindings and structural, "
            "provenance, context-leakage, causality, pairing, and "
            "non-majoritarian safety invariants."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the alpha4c.4b frozen unseen SERS Trend holdout. "
            "The paper set is read only from the frozen alpha4c.4a "
            "protocol; arbitrary paper overrides are intentionally absent."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--protocol-only",
        action="store_true",
        help=(
            "Verify frozen split, semantic IDs, and implementation blobs "
            "without reading holdout canonical graphs."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Verify protocol and snapshot all frozen holdout canonical "
            "inputs without generating projections or Trend outputs."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "After full input preflight, record/print the deterministic "
            "pipeline commands but do not execute them."
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

    protocol, protocol_audit = verify_protocol(protocol_path)
    print("alpha4c.4b frozen protocol: PASS")
    print(
        "Holdout:",
        ", ".join(map(str, protocol["holdout_papers"])),
    )
    print(
        "Split SHA256:",
        protocol_audit["source_split"]["split_sha256"],
    )
    print("Frozen implementation files:", len(
        protocol_audit["implementation_blobs"]
    ))
    if args.protocol_only:
        return 0

    canonical_snapshot = snapshot_canonical_inputs(protocol)
    print("Canonical holdout input readiness: PASS")
    print("Canonical papers:", len(canonical_snapshot))
    if args.preflight_only:
        print(
            "No projection/corpus/Trend output was generated in "
            "--preflight-only mode."
        )
        return 0

    evaluation_root = PROJECT_ROOT / str(protocol["evaluation_root"])
    manifest_path = evaluation_root / "manifest.json"
    report_path = evaluation_root / "holdout_report.json"
    evaluation_root.mkdir(parents=True, exist_ok=True)

    protocol_sha = _sha256(protocol_path)
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        _require_equal(
            "campaign protocol SHA256",
            manifest.get("protocol_sha256"),
            protocol_sha,
        )
        _require_equal(
            "campaign source split SHA256",
            manifest.get("source_split_sha256"),
            protocol_audit["source_split"]["split_sha256"],
        )
        _require_equal(
            "campaign frozen implementation",
            manifest.get("frozen_implementation_blobs"),
            protocol_audit["implementation_blobs"],
        )
        _require_equal(
            "campaign semantics",
            manifest.get("runtime_semantics"),
            protocol_audit["semantics"],
        )
        _require_equal(
            "campaign canonical snapshot",
            manifest.get("canonical_snapshot"),
            canonical_snapshot,
        )
        _require_equal(
            "campaign dry-run mode",
            bool(manifest.get("dry_run")),
            bool(args.dry_run),
        )
    else:
        manifest = {
            "phase": "alpha4c.4b",
            "holdout_epoch": protocol["holdout_epoch"],
            "status": "dry_run" if args.dry_run else "running",
            "created_at_utc": _now(),
            "updated_at_utc": _now(),
            "protocol_path": str(
                protocol_path.relative_to(PROJECT_ROOT)
            ),
            "protocol_sha256": protocol_sha,
            "source_split_sha256":
                protocol_audit["source_split"]["split_sha256"],
            "holdout_papers": list(protocol["holdout_papers"]),
            "future_reserve_count": len(
                protocol["future_reserve_papers"]
            ),
            "runtime_semantics": protocol_audit["semantics"],
            "frozen_implementation_blobs":
                protocol_audit["implementation_blobs"],
            "canonical_snapshot": canonical_snapshot,
            "llm_calls_performed_by_runner": False,
            "bridge_used": False,
            "projection_mode": protocol["mode"],
            "dry_run": bool(args.dry_run),
            "stages": {},
        }
        _atomic_json(manifest_path, manifest)

    _verify_canonical_snapshot_unchanged(canonical_snapshot)
    _run_pipeline(
        protocol=protocol,
        manifest=manifest,
        manifest_path=manifest_path,
        dry_run=bool(args.dry_run),
    )

    manifest["updated_at_utc"] = _now()
    if args.dry_run:
        manifest["status"] = "dry_run_complete"
        _atomic_json(manifest_path, manifest)
        print("alpha4c.4b dry-run: COMPLETE")
        print("Manifest:", manifest_path)
        return 0

    # Recheck freeze after builders finish.
    verify_protocol(protocol_path)
    _verify_canonical_snapshot_unchanged(canonical_snapshot)

    report = _build_report(protocol)
    _atomic_json(report_path, report)
    manifest["holdout_report"] = str(
        report_path.relative_to(PROJECT_ROOT)
    )
    manifest["status"] = (
        "complete"
        if report["passes_frozen_trend_holdout_invariants"]
        else "failed"
    )
    manifest["finished_at_utc"] = _now()
    manifest["updated_at_utc"] = _now()
    _atomic_json(manifest_path, manifest)

    print()
    print("alpha4c.4b frozen Trend holdout verdict:", report["verdict"])
    print(
        "Count thresholds used for acceptance:",
        report["count_thresholds_used_for_acceptance"],
    )
    print(
        "LLM calls performed by runner:",
        report["llm_calls_performed_by_runner"],
    )
    print(
        "Distribution:",
        json.dumps(
            report["distribution_observations"],
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    print("Report:", report_path)
    print("Manifest:", manifest_path)

    if report["violations"]:
        print("Invariant violations:")
        for row in report["violations"]:
            print(
                " -",
                row["code"],
                "observed=",
                row["observed"],
                "expected=",
                row["expected"],
            )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
