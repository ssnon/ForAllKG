from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dac_her.alpha4c5i_dev_explorer_context import (
    ALPHA4C5I_DEV_EXPLORER_SEMANTICS_ID,
    DEFAULT_INPUT_OUTPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    EXPLORER_BASE_URL,
    EXPLORER_INSTRUCTOR_MODE,
    EXPLORER_MAX_REPAIRS,
    EXPLORER_MODEL,
    EXPLORER_OBJECTIVE,
    EXPLORER_PARSE_RETRIES,
    EXPLORER_QUESTION,
    EXPLORER_TEMPERATURE,
    NODE_INDEX_MODEL,
    SOURCE_CORPUS_ID,
    SOURCE_DATA_ROOT,
    SOURCE_MODE,
    TRAVERSAL_ALGORITHM,
    TRAVERSAL_ENDPOINT_PAIR_K,
    TRAVERSAL_MAX_DEPTH,
    TRAVERSAL_NODE_MAP_K,
    TRAVERSAL_REVERSE_PENALTY,
    TRAVERSAL_SOURCE_QUERY,
    TRAVERSAL_TARGET_QUERY,
    TRAVERSAL_TOP_K,
    atomic_json,
    build_execution_manifest,
    preflight_issues,
    repo_relative,
    source_corpus_root,
    verify_built_trend_input,
    verify_context_lineage,
    verify_packet_exact_dev,
    verify_report_lineage,
)
from dac_her.explorer_contracts import (
    ExplorationReport,
    GraphExplorerPacket,
)
from dac_her.hypothesis_contracts import HypothesisContext
from dac_her.hypothesis_trend_input import TrendAwareHypothesisInput


ROOT = Path.cwd()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the missing alpha4c.5i DEV53 Explorer context "
            "using the existing GraphAgents Explorer infrastructure, "
            "then feed the accepted HypothesisContext into the existing "
            "DEV-only 5b TrendAwareHypothesisInput builder."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--input-output-root",
        type=Path,
        default=DEFAULT_INPUT_OUTPUT_ROOT,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")

    parser.add_argument(
        "--confirm-development-only",
        action="store_true",
    )
    parser.add_argument(
        "--confirm-explorer-llm",
        action="store_true",
    )

    parser.add_argument("--model", default=EXPLORER_MODEL)
    parser.add_argument("--base-url", default=EXPLORER_BASE_URL)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--instructor-mode",
        default=EXPLORER_INSTRUCTOR_MODE,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=EXPLORER_TEMPERATURE,
    )
    parser.add_argument(
        "--parse-retries",
        type=int,
        default=EXPLORER_PARSE_RETRIES,
    )
    parser.add_argument(
        "--max-repairs",
        type=int,
        choices=(0, 1),
        default=EXPLORER_MAX_REPAIRS,
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


class CommandRunner:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path

    def run(
        self,
        stage: str,
        module: str,
        *args: str,
    ) -> None:
        command = [sys.executable, "-m", module, *args]
        print(
            f"[alpha4c.5i-dev-explorer:{stage}]",
            " ".join(command),
        )
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
        )
        row = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "stage": stage,
            "module": module,
            "argv": command,
            "returncode": result.returncode,
        }
        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with self.log_path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"{module} failed with exit code "
                f"{result.returncode}"
            )


def _write_failure(
    *,
    output_root: Path,
    error: Exception,
) -> None:
    atomic_json(
        output_root / "RUN_FAILED.json",
        {
            "schema_version":
                "sers-alpha4c5i-dev-explorer-failure-v1",
            "semantics_id":
                ALPHA4C5I_DEV_EXPLORER_SEMANTICS_ID,
            "failed_at":
                datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "development_only": True,
            "reserve_a_scientific_read": False,
            "reserve_b_scientific_read": False,
            "reserve_b_rerun": False,
            "new_extraction": False,
            "automatic_output_rollback": False,
        },
    )


def main() -> int:
    args = parse_args()
    output_root = _resolve(args.output_root)
    input_output_root = _resolve(args.input_output_root)

    issues, exact_dev = preflight_issues(
        root=ROOT,
        output_root=output_root,
        input_output_root=input_output_root,
    )

    api_key_present = bool(os.getenv(args.api_key_env))

    print("alpha4c.5i DEV Explorer Context Materialization")
    print("Semantics:", ALPHA4C5I_DEV_EXPLORER_SEMANTICS_ID)
    print("DEV papers:", len(exact_dev))
    print("Source corpus:", SOURCE_CORPUS_ID)
    print("Mode:", SOURCE_MODE)
    print(
        "Traversal:",
        TRAVERSAL_ALGORITHM,
        TRAVERSAL_SOURCE_QUERY,
        "->",
        TRAVERSAL_TARGET_QUERY,
    )
    print("Explorer model:", args.model)
    print("Explorer objective:", EXPLORER_OBJECTIVE)
    print("API key env present:", api_key_present)
    print("Reserve A scientific read: False")
    print("Reserve B scientific read: False")
    print("Reserve B rerun: False")
    print("New extraction: False")

    if not api_key_present:
        issues.append(
            f"API key environment variable is missing: "
            f"{args.api_key_env}"
        )

    if args.preflight:
        if issues:
            print("Preflight: FAIL")
            for issue in sorted(set(issues)):
                print(" -", issue)
            print("Write performed: False")
            print("Explorer LLM calls: 0")
            return 2
        print("Exact DEV53 binding: PASS")
        print("Frozen DEV grounding: PASS")
        print("Source corpus artifacts: PASS")
        print("Output roots absent: PASS")
        print("Preflight: PASS")
        print("Write performed: False")
        print("Explorer LLM calls: 0")
        return 0

    if issues:
        print("Execution readiness: FAIL")
        for issue in sorted(set(issues)):
            print(" -", issue)
        return 2

    if not args.confirm_development_only:
        raise SystemExit(
            "--confirm-development-only is required."
        )
    if not args.confirm_explorer_llm:
        raise SystemExit(
            "--confirm-explorer-llm is required because this DEV "
            "materialization performs a Graph Explorer model call."
        )

    output_root.mkdir(parents=True, exist_ok=False)
    command_log = output_root / "command_log.jsonl"
    runner = CommandRunner(command_log)

    navigation_root = output_root / "navigation"
    node_index_root = output_root / "node_index"
    explorer_root = output_root / "explorer"
    hypothesis_root = output_root / "hypothesis"

    corpus_root = source_corpus_root(ROOT)
    navigation_graph = navigation_root / "graph.graphml"
    traversal_path = explorer_root / "traversal.json"
    packet_path = explorer_root / "packet.json"
    explorer_prefix = explorer_root / "explorer"
    report_path = Path(
        str(explorer_prefix) + ".report.json"
    )
    context_path = (
        hypothesis_root / "hypothesis_context.json"
    )

    try:
        runner.run(
            "navigation_graph",
            "scripts.build_navigation_graph",
            "--corpus-id",
            SOURCE_CORPUS_ID,
            "--data-root",
            str(ROOT / SOURCE_DATA_ROOT),
            "--mode",
            SOURCE_MODE,
            "--corpus-graphml",
            str(corpus_root / "graph.graphml"),
            "--output-dir",
            str(navigation_root),
            "--reverse-penalty",
            str(TRAVERSAL_REVERSE_PENALTY),
        )

        index_args = [
            "--corpus-id",
            SOURCE_CORPUS_ID,
            "--data-root",
            str(ROOT / SOURCE_DATA_ROOT),
            "--mode",
            SOURCE_MODE,
            "--model",
            NODE_INDEX_MODEL,
            "--navigation-graphml",
            str(navigation_graph),
            "--node-text",
            str(corpus_root / "node_text.jsonl"),
            "--output-dir",
            str(node_index_root),
        ]
        if args.device:
            index_args.extend(
                ["--device", str(args.device)]
            )
        runner.run(
            "node_index",
            "scripts.build_node_index",
            *index_args,
        )

        runner.run(
            "graph_traversal",
            "scripts.run_graph_traversal",
            "--corpus-id",
            SOURCE_CORPUS_ID,
            "--domain-profile",
            "sers_au_ag",
            "--data-root",
            str(ROOT / SOURCE_DATA_ROOT),
            "--mode",
            SOURCE_MODE,
            "--algorithm",
            TRAVERSAL_ALGORITHM,
            "--source",
            TRAVERSAL_SOURCE_QUERY,
            "--target",
            TRAVERSAL_TARGET_QUERY,
            "--node-map-k",
            str(TRAVERSAL_NODE_MAP_K),
            "--endpoint-pair-k",
            str(TRAVERSAL_ENDPOINT_PAIR_K),
            "--top-k",
            str(TRAVERSAL_TOP_K),
            "--max-depth",
            str(TRAVERSAL_MAX_DEPTH),
            "--navigation-graphml",
            str(navigation_graph),
            "--index-dir",
            str(node_index_root),
            "--output",
            str(traversal_path),
        )

        runner.run(
            "explorer_packet",
            "scripts.build_explorer_packet",
            "--traversal-result",
            str(traversal_path),
            "--domain-profile",
            "sers_au_ag",
            "--data-root",
            str(ROOT / SOURCE_DATA_ROOT),
            "--corpus-dir",
            str(corpus_root),
            "--question",
            EXPLORER_QUESTION,
            "--objective",
            EXPLORER_OBJECTIVE,
            "--output",
            str(packet_path),
        )

        packet = GraphExplorerPacket.model_validate_json(
            packet_path.read_text(encoding="utf-8")
        )
        packet_issues = verify_packet_exact_dev(
            packet,
            exact_dev,
        )
        if packet_issues:
            raise RuntimeError(
                "Packet DEV53 gate failed before LLM call:\n- "
                + "\n- ".join(packet_issues)
            )
        print(
            "[alpha4c.5i-dev-explorer] packet exact DEV53: PASS"
        )
        print(
            "[alpha4c.5i-dev-explorer] Explorer LLM calls so far: 0"
        )

        explorer_args = [
            "--packet",
            str(packet_path),
            "--model",
            str(args.model),
            "--api-key-env",
            args.api_key_env,
            "--instructor-mode",
            args.instructor_mode,
            "--temperature",
            str(args.temperature),
            "--parse-retries",
            str(args.parse_retries),
            "--max-repairs",
            str(args.max_repairs),
            "--timeout",
            str(args.timeout),
            "--output-prefix",
            str(explorer_prefix),
            "--save-prompt",
        ]
        if args.base_url:
            explorer_args.extend(
                ["--base-url", str(args.base_url)]
            )

        runner.run(
            "graph_explorer_llm",
            "scripts.run_graph_explorer",
            *explorer_args,
        )

        if not report_path.is_file():
            raise RuntimeError(
                "Graph Explorer completed without an accepted "
                "explorer.report.json."
            )
        report = ExplorationReport.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )
        report_issues = verify_report_lineage(
            packet,
            report,
        )
        if report_issues:
            raise RuntimeError(
                "Explorer report lineage failed:\n- "
                + "\n- ".join(report_issues)
            )

        hypothesis_root.mkdir(
            parents=True,
            exist_ok=True,
        )
        runner.run(
            "hypothesis_context",
            "scripts.build_hypothesis_context",
            "--packet",
            str(packet_path),
            "--report",
            str(report_path),
            "--output",
            str(context_path),
            "--path-lineage-output",
            str(
                hypothesis_root
                / "statement_path_lineage.json"
            ),
            "--path-lineage-propagation-output",
            str(
                hypothesis_root
                / "path_lineage_propagation.json"
            ),
        )

        context = HypothesisContext.model_validate_json(
            context_path.read_text(encoding="utf-8")
        )
        context_issues = verify_context_lineage(
            packet,
            report,
            context,
        )
        if context_issues:
            raise RuntimeError(
                "HypothesisContext lineage failed:\n- "
                + "\n- ".join(context_issues)
            )

        runner.run(
            "dev_5b_trend_input",
            "scripts.build_sers_alpha4c5i_dev_trend_input",
            "--build",
            "--confirm-development-only",
            "--context",
            str(context_path),
            "--output-root",
            str(input_output_root),
        )

        trend_input_path = (
            input_output_root
            / "trend_aware_hypothesis_input.json"
        )
        trend_input = (
            TrendAwareHypothesisInput.model_validate_json(
                trend_input_path.read_text(
                    encoding="utf-8"
                )
            )
        )
        input_issues = verify_built_trend_input(
            trend_input,
            exact_dev,
            context,
        )
        if input_issues:
            raise RuntimeError(
                "Built DEV TrendAwareHypothesisInput failed:\n- "
                + "\n- ".join(input_issues)
            )

        manifest = build_execution_manifest(
            root=ROOT,
            output_root=output_root,
            input_output_root=input_output_root,
            packet=packet,
            report=report,
            context=context,
            trend_input=trend_input,
            model=str(args.model),
            base_url=args.base_url,
            instructor_mode=args.instructor_mode,
            temperature=args.temperature,
            parse_retries=args.parse_retries,
            max_repairs=args.max_repairs,
        )
        atomic_json(
            output_root / "execution_manifest.json",
            manifest,
        )
        atomic_json(
            output_root / "RUN_PASS.json",
            {
                "schema_version":
                    "sers-alpha4c5i-dev-explorer-pass-v1",
                "semantics_id":
                    ALPHA4C5I_DEV_EXPLORER_SEMANTICS_ID,
                "passed_at":
                    datetime.now(timezone.utc).isoformat(),
                "development_only": True,
                "packet_id": packet.packet_id,
                "report_id": report.report_id,
                "context_id": context.context_id,
                "trend_input_id": trend_input.input_id,
                "reserve_a_scientific_read": False,
                "reserve_b_scientific_read": False,
                "reserve_b_rerun": False,
                "new_extraction": False,
            },
        )

    except Exception as exc:
        _write_failure(
            output_root=output_root,
            error=exc,
        )
        print(
            "alpha4c.5i DEV Explorer materialization: FAIL",
            file=sys.stderr,
        )
        print("Reason:", exc, file=sys.stderr)
        print(
            "Partial DEV artifacts preserved for diagnosis.",
            file=sys.stderr,
        )
        print("Reserve A/B rerun: False", file=sys.stderr)
        raise

    print("alpha4c.5i DEV Explorer materialization: PASS")
    print("Packet ID:", packet.packet_id)
    print("Report ID:", report.report_id)
    print("Context ID:", context.context_id)
    print("Context SHA256:", context.context_sha256)
    print("Trend input ID:", trend_input.input_id)
    print("Trend input SHA256:", trend_input.input_sha256)
    print("DEV papers:", len(exact_dev))
    print("Reserve A scientific read: False")
    print("Reserve B scientific read: False")
    print("Reserve B rerun: False")
    print("New extraction: False")
    print(
        "Explorer LLM development call completed: True"
    )
    print(
        "Execution manifest:",
        output_root / "execution_manifest.json",
    )
    print(
        "Trend-aware input:",
        input_output_root
        / "trend_aware_hypothesis_input.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
