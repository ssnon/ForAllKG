from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from campaigns.sers_alpha4_epoch.alpha4.alpha4c5i_dev_compatibility import (
    load_exact_dev_paper_ids,
    verify_frozen_postmortem,
    verify_h1_dev_summary,
)
from campaigns.sers_alpha4_epoch.alpha4.alpha4c5i_dev_input_builder import (
    DEFAULT_GROUNDING,
    load_dev_grounding,
    path_is_closed_reserve,
)
from dac_her.explorer_contracts import (
    ExplorationReport,
    GraphExplorerPacket,
)
from dac_her.hypothesis_contracts import HypothesisContext
from dac_her.hypothesis_trend_input import (
    TrendAwareHypothesisInput,
    validate_hypothesis_context_sha,
    verify_trend_aware_input_sources,
)


ALPHA4C5I_DEV_EXPLORER_SEMANTICS_ID = (
    "sers_alpha4c5i_dev_explorer_context_materialization_v1"
)

SOURCE_DATA_ROOT = Path(
    "evaluation/sers_alpha4c5g/dev_v1/work_data_sers"
)
SOURCE_CORPUS_ID = "sers_alpha4c5g_dev_v1_corpus"
SOURCE_MODE = "evidence"

DEFAULT_OUTPUT_ROOT = Path(
    "evaluation/sers_alpha4c5i/dev_explorer_v1"
)
DEFAULT_INPUT_OUTPUT_ROOT = Path(
    "evaluation/sers_alpha4c5i/dev_compat_v1/input"
)

# Development Explorer settings copied from the pre-execution frozen
# SERS evaluation configuration. They are operational/retrieval settings,
# not learned from Reserve A/B scientific outputs.
TRAVERSAL_ALGORITHM = "top_n"
TRAVERSAL_SOURCE_QUERY = "nanostructure design"
TRAVERSAL_TARGET_QUERY = "SERS performance"
TRAVERSAL_NODE_MAP_K = 20
TRAVERSAL_ENDPOINT_PAIR_K = 12
TRAVERSAL_TOP_K = 8
TRAVERSAL_MAX_DEPTH = 8
TRAVERSAL_REVERSE_PENALTY = 0.6
NODE_INDEX_MODEL = "nomic-ai/nomic-embed-text-v1.5"

EXPLORER_QUESTION = (
    "How do Au/Ag nanostructure design variables and local "
    "experimental context relate to SERS performance?"
)
EXPLORER_OBJECTIVE = "map_evidence"
EXPLORER_MODEL = "openai/gpt-5.6-luna"
EXPLORER_BASE_URL = "https://openrouter.ai/api/v1"
EXPLORER_INSTRUCTOR_MODE = "JSON"
EXPLORER_TEMPERATURE = 0.0
EXPLORER_PARSE_RETRIES = 1
EXPLORER_MAX_REPAIRS = 1

EXPECTED_DEV_COUNT = 53


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def repo_relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def source_corpus_root(root: Path) -> Path:
    return (
        root
        / SOURCE_DATA_ROOT
        / "corpus"
        / SOURCE_CORPUS_ID
        / SOURCE_MODE
    )


def verify_packet_exact_dev(
    packet: GraphExplorerPacket,
    exact_dev: list[str],
) -> list[str]:
    issues: list[str] = []
    if packet.domain_profile_id != "sers_au_ag":
        issues.append(
            "packet domain_profile_id is not sers_au_ag"
        )
    if packet.corpus.corpus_id != SOURCE_CORPUS_ID:
        issues.append(
            "packet corpus_id does not match DEV53 corpus"
        )
    if packet.corpus.projection_mode != SOURCE_MODE:
        issues.append(
            "packet projection_mode is not evidence"
        )
    paper_ids = sorted(
        str(row.paper_id) for row in packet.corpus.papers
    )
    if paper_ids != exact_dev:
        issues.append(
            "packet paper scope is not the exact 53-paper DEV split"
        )
    return issues


def verify_report_lineage(
    packet: GraphExplorerPacket,
    report: ExplorationReport,
) -> list[str]:
    issues: list[str] = []
    if report.source_packet_sha256 != packet.packet_sha256:
        issues.append(
            "report source_packet_sha256 does not match packet"
        )
    if report.task_id != packet.task.task_id:
        issues.append("report task_id does not match packet")
    return issues


def verify_context_lineage(
    packet: GraphExplorerPacket,
    report: ExplorationReport,
    context: HypothesisContext,
) -> list[str]:
    issues: list[str] = []
    try:
        validate_hypothesis_context_sha(context)
    except Exception as exc:
        issues.append(f"context SHA validation failed: {exc}")

    if context.source_packet_id != packet.packet_id:
        issues.append("context source_packet_id mismatch")
    if context.source_packet_sha256 != packet.packet_sha256:
        issues.append("context source_packet_sha256 mismatch")
    if context.source_report_id != report.report_id:
        issues.append("context source_report_id mismatch")
    if context.task_id != packet.task.task_id:
        issues.append("context task_id mismatch")
    if context.corpus_id != SOURCE_CORPUS_ID:
        issues.append("context corpus_id mismatch")
    if context.domain_profile_id != "sers_au_ag":
        issues.append("context domain_profile_id mismatch")
    return issues


def verify_built_trend_input(
    value: TrendAwareHypothesisInput,
    exact_dev: list[str],
    context: HypothesisContext,
) -> list[str]:
    issues: list[str] = []
    try:
        verify_trend_aware_input_sources(value)
    except Exception as exc:
        issues.append(
            f"TrendAwareHypothesisInput source verification failed: {exc}"
        )

    if value.trend_corpus_binding.paper_ids != exact_dev:
        issues.append(
            "TrendAwareHypothesisInput is not exact DEV53"
        )
    if value.grounded_context.context_id != context.context_id:
        issues.append(
            "TrendAwareHypothesisInput context_id mismatch"
        )
    if (
        value.grounded_context.context_sha256
        != context.context_sha256
    ):
        issues.append(
            "TrendAwareHypothesisInput context SHA mismatch"
        )
    return issues


def preflight_issues(
    *,
    root: Path,
    output_root: Path,
    input_output_root: Path,
) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    issues.extend(verify_frozen_postmortem(root))
    issues.extend(verify_h1_dev_summary(root))

    try:
        exact_dev = load_exact_dev_paper_ids(root)
    except Exception as exc:
        return [f"DEV split binding failed: {exc}"], []

    if len(exact_dev) != EXPECTED_DEV_COUNT:
        issues.append(
            f"expected 53 DEV papers; observed {len(exact_dev)}"
        )

    try:
        grounding = load_dev_grounding(
            root=root,
            path=DEFAULT_GROUNDING,
        )
        if grounding is None:
            issues.append("DEV grounding unexpectedly resolved to None")
    except Exception as exc:
        issues.append(f"DEV grounding verification failed: {exc}")

    corpus_root = source_corpus_root(root)
    required_source = (
        corpus_root / "graph.graphml",
        corpus_root / "node_text.jsonl",
    )
    for path in required_source:
        if not path.is_file():
            issues.append(
                f"DEV source corpus artifact missing: {path}"
            )

    if path_is_closed_reserve(corpus_root):
        issues.append(
            "DEV source corpus path unexpectedly matches Reserve guard"
        )

    if output_root.exists():
        issues.append(
            f"DEV Explorer output already exists: {output_root}"
        )
    if input_output_root.exists():
        issues.append(
            f"DEV 5b input output already exists: {input_output_root}"
        )

    required_scripts = (
        "scripts/build_navigation_graph.py",
        "scripts/build_node_index.py",
        "scripts/run_graph_traversal.py",
        "scripts/build_explorer_packet.py",
        "scripts/run_graph_explorer.py",
        "scripts/build_hypothesis_context.py",
        "scripts/build_sers_alpha4c5i_dev_trend_input.py",
    )
    for rel in required_scripts:
        if not (root / rel).is_file():
            issues.append(f"required script missing: {rel}")

    return sorted(set(issues)), exact_dev


def build_execution_manifest(
    *,
    root: Path,
    output_root: Path,
    input_output_root: Path,
    packet: GraphExplorerPacket,
    report: ExplorationReport,
    context: HypothesisContext,
    trend_input: TrendAwareHypothesisInput,
    model: str,
    base_url: str | None,
    instructor_mode: str = EXPLORER_INSTRUCTOR_MODE,
    temperature: float = EXPLORER_TEMPERATURE,
    parse_retries: int = EXPLORER_PARSE_RETRIES,
    max_repairs: int = EXPLORER_MAX_REPAIRS,
) -> dict[str, Any]:
    paths = {
        "navigation_graph":
            output_root / "navigation" / "graph.graphml",
        "navigation_summary":
            output_root / "navigation" / "summary.json",
        "node_index_manifest":
            output_root / "node_index" / "manifest.json",
        "traversal":
            output_root / "explorer" / "traversal.json",
        "packet":
            output_root / "explorer" / "packet.json",
        "explorer_run":
            output_root / "explorer" / "explorer.run.json",
        "explorer_validation":
            output_root / "explorer" / "explorer.validation.json",
        "explorer_report":
            output_root / "explorer" / "explorer.report.json",
        "hypothesis_context":
            output_root / "hypothesis" / "hypothesis_context.json",
        "trend_input":
            input_output_root / "trend_aware_hypothesis_input.json",
        "trend_input_build_manifest":
            input_output_root / "build_manifest.json",
    }

    bindings: dict[str, Any] = {}
    for name, path in paths.items():
        if path.is_file():
            bindings[name] = {
                "path": repo_relative(root, path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }

    payload: dict[str, Any] = {
        "schema_version":
            "sers-alpha4c5i-dev-explorer-context-materialization-v1",
        "semantics_id": ALPHA4C5I_DEV_EXPLORER_SEMANTICS_ID,
        "development_only": True,
        "dev_paper_count": EXPECTED_DEV_COUNT,
        "source_data_root": str(SOURCE_DATA_ROOT),
        "source_corpus_id": SOURCE_CORPUS_ID,
        "source_mode": SOURCE_MODE,
        "traversal": {
            "algorithm": TRAVERSAL_ALGORITHM,
            "source_query": TRAVERSAL_SOURCE_QUERY,
            "target_query": TRAVERSAL_TARGET_QUERY,
            "node_map_k": TRAVERSAL_NODE_MAP_K,
            "endpoint_pair_k": TRAVERSAL_ENDPOINT_PAIR_K,
            "top_k": TRAVERSAL_TOP_K,
            "max_depth": TRAVERSAL_MAX_DEPTH,
            "reverse_penalty": TRAVERSAL_REVERSE_PENALTY,
            "node_index_model": NODE_INDEX_MODEL,
        },
        "explorer": {
            "question": EXPLORER_QUESTION,
            "objective": EXPLORER_OBJECTIVE,
            "model": model,
            "base_url": base_url,
            "instructor_mode": instructor_mode,
            "temperature": temperature,
            "parse_retries": parse_retries,
            "max_repairs": max_repairs,
        },
        "packet_id": packet.packet_id,
        "packet_sha256": packet.packet_sha256,
        "report_id": report.report_id,
        "context_id": context.context_id,
        "context_sha256": context.context_sha256,
        "trend_input_id": trend_input.input_id,
        "trend_input_sha256": trend_input.input_sha256,
        "artifact_bindings": bindings,
        "reserve_a_scientific_read": False,
        "reserve_b_scientific_read": False,
        "reserve_b_rerun": False,
        "new_extraction": False,
        "trend_semantics_modified": False,
        "precision_semantics_modified": False,
        "cross_context_semantics_modified": False,
        "count_thresholds_used_for_acceptance": False,
        "explorer_llm_development_only": True,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(
        canonical
    ).hexdigest()
    return payload
