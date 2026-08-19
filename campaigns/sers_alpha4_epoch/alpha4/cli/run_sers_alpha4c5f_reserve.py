from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from campaigns.sers_alpha4_epoch.alpha4.alpha4c5f_reserve import (
    DEFAULT_5F_PROTOCOL_PATH,
    Alpha4c5fProtocol,
    load_5f_protocol,
    sha256_file,
    verify_5f_protocol,
)


ROOT = Path.cwd()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "alpha4c.5f single-shot frozen v3 reserve orchestrator. "
            "Preflight does not consume the reserve. Execute creates "
            "the consumption marker before any scientific transformation."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_5F_PROTOCOL_PATH,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--execute-reserve", action="store_true")
    parser.add_argument(
        "--confirm-consume-reserve",
        action="store_true",
    )
    return parser.parse_args()


class CampaignFailure(RuntimeError):
    pass


class Runner:
    def __init__(self, protocol: Alpha4c5fProtocol) -> None:
        self.protocol = protocol
        self.eval_root = ROOT / protocol.evaluation_root
        self.work_data = ROOT / protocol.data_root
        self.command_log = self.eval_root / "command_log.jsonl"
        self.marker = (
            self.eval_root / "consumption_started.json"
        )
        self.pass_marker = self.eval_root / "CAMPAIGN_PASS.json"
        self.fail_marker = self.eval_root / "CAMPAIGN_FAIL.json"
        ids = protocol.artifact_ids
        self.corpus_root = (
            self.work_data
            / "corpus"
            / ids.corpus
            / "evidence"
        )
        self.trend_root = (
            self.corpus_root
            / "trend"
            / ids.trend
        )
        self.precision_root = (
            self.trend_root
            / "precision"
            / ids.precision
        )
        self.context_root = (
            self.precision_root
            / "cross_context"
            / ids.context
        )
        self.assessment_root = (
            self.context_root
            / "assessment"
            / ids.assessment
        )
        self.explorer_root = self.eval_root / "explorer"
        self.hypothesis_root = (
            self.eval_root / "hypothesis"
        )

    def verify(self) -> list[str]:
        issues = verify_5f_protocol(
            ROOT,
            self.protocol,
            check_canonical_presence=True,
        )
        # Preflight is metadata/readiness only. Do not parse or semantically
        # inspect reserve graphs here.
        if self.protocol.data_root != (
            "evaluation/sers_alpha4c5f/reserve_v1/"
            "work_data_sers"
        ):
            issues.append("unexpected campaign data_root")
        if self.protocol.evaluation_root != (
            "evaluation/sers_alpha4c5f/reserve_v1"
        ):
            issues.append("unexpected evaluation_root")
        return issues

    def status(self) -> int:
        print("alpha4c.5f reserve status")
        print("Protocol ID:", self.protocol.protocol_id)
        print(
            "Protocol SHA256:",
            self.protocol.protocol_sha256,
        )
        print("Campaign:", self.protocol.campaign_id)
        print("Reserve papers:", len(self.protocol.reserve_paper_ids))
        print("Consumption marker:", self.marker.exists())
        print("PASS marker:", self.pass_marker.exists())
        print("FAIL marker:", self.fail_marker.exists())
        if self.marker.exists():
            print(
                "Consumption:",
                self.marker.read_text(encoding="utf-8").strip(),
            )
        return 0

    def preflight(self) -> int:
        if self.marker.exists():
            raise CampaignFailure(
                "Reserve is already consumed; preflight cannot "
                "re-open the campaign."
            )
        issues = self.verify()
        print("alpha4c.5f frozen reserve preflight")
        print("Protocol ID:", self.protocol.protocol_id)
        print(
            "Protocol SHA256:",
            self.protocol.protocol_sha256,
        )
        print(
            "5e protocol SHA256:",
            self.protocol.evaluation_protocol_sha256,
        )
        print(
            "Reserve manifest SHA256:",
            self.protocol.reserve_manifest_sha256,
        )
        print(
            "Source split SHA256:",
            self.protocol.source_split_sha256,
        )
        print("Exact reserve paper count:", len(
            self.protocol.reserve_paper_ids
        ))
        print(
            "Papers:",
            ", ".join(self.protocol.reserve_paper_ids),
        )
        print("Paper override allowed: False")
        print("Scientific mode: evidence")
        print("Bridge required: False")
        print("New extraction LLM allowed: False")
        print("Explorer question:", self.protocol.explorer.question)
        print(
            "Traversal:",
            self.protocol.traversal.source_query,
            "->",
            self.protocol.traversal.target_query,
            self.protocol.traversal.algorithm,
        )
        print("Explorer model:", self.protocol.explorer.model)
        print("Maker model:", self.protocol.maker.model)
        print("Maker max hypotheses:", 1)
        print("Maker max repairs:", 1)
        print("Temperature:", 0)
        print("Count thresholds used: False")
        print("Reserve consumed: False")
        print("LLM calls: 0")
        if issues:
            print("Preflight: FAIL")
            for issue in issues:
                print(" -", issue)
            return 2
        print("Preflight: PASS")
        return 0

    def _cmd(
        self,
        stage: str,
        args: list[str],
    ) -> None:
        started = time.time()
        event = {
            "stage": stage,
            "started_at": now_iso(),
            "command": args,
        }
        print(f"\n[alpha4c.5f] {stage}")
        print("[alpha4c.5f] command:", " ".join(args))
        result = subprocess.run(
            args,
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        event.update(
            {
                "finished_at": now_iso(),
                "elapsed_seconds": time.time() - started,
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-8000:],
                "stderr_tail": result.stderr[-8000:],
            }
        )
        append_jsonl(self.command_log, event)
        if result.returncode != 0:
            raise CampaignFailure(
                f"{stage} failed with exit code "
                f"{result.returncode}"
            )

    def _python(
        self,
        stage: str,
        module: str,
        *args: str,
    ) -> None:
        self._cmd(
            stage,
            [sys.executable, "-m", module, *args],
        )

    def _freeze_canonical_sources(self) -> None:
        rows = []
        for paper_id in self.protocol.reserve_paper_ids:
            source = (
                ROOT
                / "data_sers"
                / "extracted"
                / paper_id
                / f"{paper_id}.graphml"
            )
            if not source.exists():
                raise CampaignFailure(
                    f"canonical source disappeared: {source}"
                )
            source_sha = sha256_file(source)

            # Scientific parsing begins only after the consumption marker.
            graph = nx.read_graphml(
                source,
                force_multigraph=True,
            )
            domain = str(
                graph.graph.get("domain_profile_id", "")
            )
            if domain and domain != "sers_au_ag":
                raise CampaignFailure(
                    f"canonical graph domain mismatch "
                    f"for {paper_id}: {domain!r}"
                )

            dest = (
                self.work_data
                / "extracted"
                / paper_id
                / f"{paper_id}.graphml"
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            dest_sha = sha256_file(dest)
            if dest_sha != source_sha:
                raise CampaignFailure(
                    f"canonical copy hash mismatch: {paper_id}"
                )
            rows.append(
                {
                    "paper_id": paper_id,
                    "source_path": str(source),
                    "source_sha256": source_sha,
                    "campaign_path": str(dest),
                    "campaign_sha256": dest_sha,
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "domain_profile_id": domain or "sers_au_ag",
                }
            )
        write_json(
            self.eval_root / "canonical_source_lock.json",
            {
                "campaign_id": self.protocol.campaign_id,
                "paper_ids": self.protocol.reserve_paper_ids,
                "source_rows": rows,
                "scientific_parse_after_consumption": True,
            },
        )

    def _build_evidence_substrate(self) -> None:
        ids = self.protocol.artifact_ids
        for paper_id in self.protocol.reserve_paper_ids:
            self._python(
                f"projection:{paper_id}",
                "scripts.build_graphagents_projection",
                "--paper-id", paper_id,
                "--domain-profile", "sers_au_ag",
                "--data-root", str(self.work_data),
                "--mode", "evidence",
            )

        self._python(
            "corpus",
            "scripts.build_corpus_graph",
            "--corpus-id", ids.corpus,
            "--domain-profile", "sers_au_ag",
            "--data-root", str(self.work_data),
            "--paper-ids",
            *self.protocol.reserve_paper_ids,
            "--mode", "evidence",
            "--allow-critical-partial",
        )

        self._python(
            "measurement_result_identity",
            "scripts.build_measurement_result_identities",
            "--domain-profile", "sers_au_ag",
            "--data-root", str(self.work_data),
            "--corpus-id", ids.corpus,
            "--mode", "evidence",
            "--measurement-result-identity-id",
            ids.measurement_result_identity,
        )

        self._python(
            "metric_definition",
            "scripts.build_metric_definition_contexts",
            "--domain-profile", "sers_au_ag",
            "--data-root", str(self.work_data),
            "--corpus-id", ids.corpus,
            "--mode", "evidence",
            "--metric-definition-id", ids.metric_definition,
            "--measurement-result-identity-id",
            ids.measurement_result_identity,
        )

        self._python(
            "comparison",
            "scripts.build_comparison_contexts",
            "--domain-profile", "sers_au_ag",
            "--data-root", str(self.work_data),
            "--corpus-id", ids.corpus,
            "--mode", "evidence",
            "--comparison-id", ids.comparison,
            "--metric-definition-id", ids.metric_definition,
            "--measurement-result-identity-id",
            ids.measurement_result_identity,
        )

        self._python(
            "trend",
            "scripts.build_trend_evidence",
            "--domain-profile", "sers_au_ag",
            "--data-root", str(self.work_data),
            "--corpus-id", ids.corpus,
            "--mode", "evidence",
            "--trend-id", ids.trend,
            "--measurement-result-identity-id",
            ids.measurement_result_identity,
            "--comparison-id", ids.comparison,
        )

        self._python(
            "trend_precision",
            "scripts.build_trend_precision",
            "--domain-profile", "sers_au_ag",
            "--data-root", str(self.work_data),
            "--corpus-id", ids.corpus,
            "--mode", "evidence",
            "--trend-id", ids.trend,
            "--precision-id", ids.precision,
        )

        local_results = self.precision_root / "local_results.jsonl"
        if not local_results.exists():
            raise CampaignFailure(
                f"precision local_results missing: {local_results}"
            )
        local_count = sum(
            1
            for line in local_results.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )
        write_json(
            self.eval_root / "trend_yield.json",
            {
                "local_result_count": local_count,
                "count_thresholds_used_for_acceptance": False,
                "zero_is_valid": True,
            },
        )

        if local_count > 0:
            self._python(
                "cross_context_profiles",
                "scripts.build_cross_context_profiles",
                "--domain-profile", "sers_au_ag",
                "--data-root", str(self.work_data),
                "--corpus-id", ids.corpus,
                "--mode", "evidence",
                "--trend-id", ids.trend,
                "--precision-id", ids.precision,
                "--context-id", ids.context,
            )
            self._python(
                "cross_context_assessments",
                "scripts.build_cross_context_assessments",
                "--domain-profile", "sers_au_ag",
                "--data-root", str(self.work_data),
                "--corpus-id", ids.corpus,
                "--mode", "evidence",
                "--trend-id", ids.trend,
                "--precision-id", ids.precision,
                "--context-id", ids.context,
                "--assessment-id", ids.assessment,
            )

    def _build_explorer_context(self) -> Path:
        ids = self.protocol.artifact_ids
        traversal = self.protocol.traversal
        explorer = self.protocol.explorer

        self._python(
            "navigation_graph",
            "scripts.build_navigation_graph",
            "--corpus-id", ids.corpus,
            "--data-root", str(self.work_data),
            "--mode", "evidence",
            "--reverse-penalty",
            str(traversal.reverse_penalty),
        )
        self._python(
            "node_index",
            "scripts.build_node_index",
            "--corpus-id", ids.corpus,
            "--data-root", str(self.work_data),
            "--mode", "evidence",
            "--model", traversal.node_index_model,
        )

        self.explorer_root.mkdir(
            parents=True,
            exist_ok=True,
        )
        traversal_path = (
            self.explorer_root / "traversal.json"
        )
        self._python(
            "graph_traversal",
            "scripts.run_graph_traversal",
            "--corpus-id", ids.corpus,
            "--domain-profile", "sers_au_ag",
            "--data-root", str(self.work_data),
            "--mode", "evidence",
            "--algorithm", traversal.algorithm,
            "--source", traversal.source_query,
            "--target", traversal.target_query,
            "--node-map-k", str(traversal.node_map_k),
            "--endpoint-pair-k",
            str(traversal.endpoint_pair_k),
            "--top-k", str(traversal.top_k),
            "--max-depth", str(traversal.max_depth),
            "--output", str(traversal_path),
        )

        packet_path = (
            self.explorer_root / "packet.json"
        )
        self._python(
            "explorer_packet",
            "scripts.build_explorer_packet",
            "--traversal-result", str(traversal_path),
            "--domain-profile", "sers_au_ag",
            "--data-root", str(self.work_data),
            "--question", explorer.question,
            "--objective", explorer.objective,
            "--output", str(packet_path),
        )

        explorer_prefix = (
            self.explorer_root / "explorer"
        )
        command = [
            "--packet", str(packet_path),
            "--model", explorer.model,
            "--api-key-env", explorer.api_key_env,
            "--instructor-mode", explorer.instructor_mode,
            "--temperature", str(explorer.temperature),
            "--parse-retries", str(explorer.parse_retries),
            "--max-repairs", str(explorer.max_repairs),
            "--output-prefix", str(explorer_prefix),
            "--save-prompt",
        ]
        if explorer.base_url is not None:
            command.extend(
                ["--base-url", explorer.base_url]
            )
        self._python(
            "graph_explorer_llm",
            "scripts.run_graph_explorer",
            *command,
        )
        report = Path(
            str(explorer_prefix) + ".report.json"
        )
        if not report.exists():
            raise CampaignFailure(
                "Graph Explorer did not produce an accepted report."
            )

        context_path = (
            self.hypothesis_root / "hypothesis_context.json"
        )
        self.hypothesis_root.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._python(
            "hypothesis_context",
            "scripts.build_hypothesis_context",
            "--packet", str(packet_path),
            "--report", str(report),
            "--output", str(context_path),
        )
        return context_path

    def _build_trend_input(
        self,
        hypothesis_context_path: Path,
    ) -> Path:
        ids = self.protocol.artifact_ids
        grounding_path = (
            self.hypothesis_root
            / "trend_hypothesis_grounding.json"
        )
        args = [
            "--trend-dir", str(self.trend_root),
            "--precision-dir", str(self.precision_root),
            "--domain-profile", "sers_au_ag",
            "--output", str(grounding_path),
        ]
        local_results = (
            self.precision_root / "local_results.jsonl"
        )
        local_count = sum(
            1
            for line in local_results.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )
        if local_count > 0:
            args.extend(
                [
                    "--context-dir", str(self.context_root),
                    "--assessment-dir",
                    str(self.assessment_root),
                ]
            )
        self._python(
            "trend_hypothesis_grounding",
            "scripts.build_hypothesis_trend_grounding",
            *args,
        )

        input_path = (
            self.hypothesis_root
            / "trend_aware_hypothesis_input.json"
        )
        self._python(
            "trend_aware_hypothesis_input",
            "scripts.build_hypothesis_trend_input",
            "--context", str(hypothesis_context_path),
            "--trend-grounding", str(grounding_path),
            "--input-semantics-id",
            "sers_au_ag_hypothesis_trend_input_v1_alpha4c5b",
            "--output", str(input_path),
        )
        return input_path

    def _run_maker_and_evaluate(
        self,
        input_path: Path,
    ) -> Path:
        maker = self.protocol.maker
        maker_prefix = (
            self.hypothesis_root / "reserve_maker"
        )
        maker_args = [
            "--input", str(input_path),
            "--model", maker.model,
            "--api-key-env", maker.api_key_env,
            "--instructor-mode", maker.instructor_mode,
            "--temperature", str(maker.temperature),
            "--parse-retries", str(maker.parse_retries),
            "--max-repairs", str(maker.max_repairs),
            "--max-hypotheses",
            str(maker.max_hypotheses),
            "--save-prompt",
            "--telemetry-path",
            str(
                self.hypothesis_root
                / "reserve_maker.telemetry.jsonl"
            ),
            "--output-prefix", str(maker_prefix),
        ]
        if maker.base_url is not None:
            maker_args.extend(
                ["--base-url", maker.base_url]
            )
        self._python(
            "direction_aware_hypothesis_maker_llm",
            "scripts.run_direction_aware_trend_hypothesis_maker",
            *maker_args,
        )

        run_path = Path(str(maker_prefix) + ".run.json")
        portfolio_path = Path(
            str(maker_prefix) + ".portfolio.json"
        )
        if not run_path.exists() or not portfolio_path.exists():
            raise CampaignFailure(
                "Maker did not produce accepted run/portfolio."
            )
        run = json.loads(
            run_path.read_text(encoding="utf-8")
        )
        repair_attempts = int(
            run.get("repair_attempts", -1)
        )
        if repair_attempts == 0:
            final_draft = Path(
                str(maker_prefix) + ".draft.json"
            )
        elif repair_attempts == 1:
            final_draft = Path(
                str(maker_prefix) + ".repair1.draft.json"
            )
        else:
            raise CampaignFailure(
                "Maker repair count violates frozen 5f policy."
            )
        if not final_draft.exists():
            raise CampaignFailure(
                f"final Maker draft missing: {final_draft}"
            )

        evaluation_path = (
            self.hypothesis_root
            / "reserve_evaluation.json"
        )
        self._python(
            "alpha4c5e_reserve_evaluation",
            "scripts.evaluate_direction_aware_trend_hypothesis_run",
            "--mode", "reserve",
            "--protocol",
            self.protocol.evaluation_protocol_path,
            "--reserve-manifest",
            self.protocol.reserve_manifest_path,
            "--input", str(input_path),
            "--run", str(run_path),
            "--final-draft", str(final_draft),
            "--portfolio", str(portfolio_path),
            "--output", str(evaluation_path),
        )
        evaluation = json.loads(
            evaluation_path.read_text(encoding="utf-8")
        )
        if evaluation.get("accepted") is not True:
            raise CampaignFailure(
                "5e reserve evaluation did not accept campaign."
            )
        return evaluation_path

    def execute(self) -> int:
        if not self.protocol.execution_policy.rerun_after_consumption_allowed:
            if self.marker.exists():
                raise CampaignFailure(
                    "Reserve already consumed. 5f is single-shot and "
                    "refuses re-execution."
                )

        issues = self.verify()
        if issues:
            raise CampaignFailure(
                "Pre-execution frozen verification failed:\n- "
                + "\n- ".join(issues)
            )

        if self.work_data.exists():
            raise CampaignFailure(
                "Campaign work_data already exists before consumption. "
                "Refusing ambiguous restart."
            )
        self.eval_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        # The reserve is consumed HERE, before any graph is parsed/copied,
        # projection is built, traversal is run, or LLM is called.
        marker_payload = {
            "campaign_id": self.protocol.campaign_id,
            "protocol_id": self.protocol.protocol_id,
            "protocol_sha256": self.protocol.protocol_sha256,
            "evaluation_protocol_sha256":
                self.protocol.evaluation_protocol_sha256,
            "reserve_manifest_sha256":
                self.protocol.reserve_manifest_sha256,
            "paper_ids": self.protocol.reserve_paper_ids,
            "started_at": now_iso(),
            "reserve_consumed": True,
            "reason": (
                "single-shot alpha4c.5f scientific execution began"
            ),
        }
        write_json(self.marker, marker_payload)
        print(
            "[alpha4c.5f] RESERVE CONSUMED:",
            self.marker,
        )

        try:
            write_json(
                self.eval_root / "campaign_manifest.json",
                {
                    "campaign_id": self.protocol.campaign_id,
                    "protocol_id": self.protocol.protocol_id,
                    "protocol_sha256":
                        self.protocol.protocol_sha256,
                    "reserve_manifest_id":
                        self.protocol.reserve_manifest_id,
                    "reserve_manifest_sha256":
                        self.protocol.reserve_manifest_sha256,
                    "paper_ids":
                        self.protocol.reserve_paper_ids,
                    "state": "running",
                    "started_at": marker_payload["started_at"],
                    "count_thresholds_used_for_acceptance":
                        False,
                },
            )
            self._freeze_canonical_sources()
            self._build_evidence_substrate()
            context_path = self._build_explorer_context()
            input_path = self._build_trend_input(
                context_path
            )
            evaluation_path = (
                self._run_maker_and_evaluate(input_path)
            )
            evaluation = json.loads(
                evaluation_path.read_text(encoding="utf-8")
            )
            success = {
                "campaign_id": self.protocol.campaign_id,
                "protocol_id": self.protocol.protocol_id,
                "protocol_sha256":
                    self.protocol.protocol_sha256,
                "reserve_manifest_sha256":
                    self.protocol.reserve_manifest_sha256,
                "completed_at": now_iso(),
                "reserve_consumed": True,
                "accepted": True,
                "fatal_issue_count":
                    evaluation.get("fatal_issue_count"),
                "hypothesis_count":
                    evaluation.get("hypothesis_count"),
                "abstained": evaluation.get("abstained"),
                "count_thresholds_used_for_acceptance":
                    False,
                "evaluation_path": str(evaluation_path),
            }
            write_json(self.pass_marker, success)
            write_json(
                self.eval_root / "campaign_manifest.json",
                {
                    **success,
                    "state": "pass",
                },
            )
            print("\nalpha4c.5f reserve campaign PASS")
            print("Campaign:", self.protocol.campaign_id)
            print("Reserve consumed: True")
            print(
                "5e accepted:",
                evaluation.get("accepted"),
            )
            print(
                "Fatal issues:",
                evaluation.get("fatal_issue_count"),
            )
            print(
                "Hypotheses:",
                evaluation.get("hypothesis_count"),
            )
            print(
                "Abstained:",
                evaluation.get("abstained"),
            )
            print(
                "Count thresholds used:",
                False,
            )
            print("PASS marker:", self.pass_marker)
            return 0
        except Exception as exc:
            failure = {
                "campaign_id": self.protocol.campaign_id,
                "protocol_id": self.protocol.protocol_id,
                "protocol_sha256":
                    self.protocol.protocol_sha256,
                "failed_at": now_iso(),
                "reserve_consumed": True,
                "accepted": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "rerun_allowed": False,
                "automatic_scientific_output_rollback":
                    False,
                "count_thresholds_used_for_acceptance":
                    False,
            }
            write_json(self.fail_marker, failure)
            write_json(
                self.eval_root / "campaign_manifest.json",
                {
                    **failure,
                    "state": "fail",
                },
            )
            print(
                "\n[alpha4c.5f] CAMPAIGN FAIL; reserve remains "
                "consumed and may not be rerun under this epoch.",
                file=sys.stderr,
            )
            print(
                "[alpha4c.5f] failure marker:",
                self.fail_marker,
                file=sys.stderr,
            )
            raise


def main() -> int:
    args = parse_args()
    protocol = load_5f_protocol(ROOT / args.protocol)
    runner = Runner(protocol)

    if args.status:
        return runner.status()
    if args.preflight:
        return runner.preflight()
    if args.execute_reserve:
        if not args.confirm_consume_reserve:
            raise SystemExit(
                "--confirm-consume-reserve is required for real "
                "single-shot reserve execution."
            )
        return runner.execute()
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
