from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from pipeline_core.discovery.hypothesis_llm import InstructorOpenAICompatibleHypothesisBackend
from pipeline_core.discovery.prospective_identification_materialization_shadow import (
    run_prospective_identification_shadow,
)
from pipeline_core.discovery.research_idea_closed_loop import (
    RealizationLifecycleReport,
    run_realization_lifecycle,
)
from pipeline_core.discovery.research_idea_epistemic_archive import (
    EvidenceRecoveryPlan,
    build_epistemic_decomposition_report,
    build_evidence_recovery_command_plan,
    build_evidence_recovery_plan,
    build_multi_realization_archive,
    build_recovery_relevance_gate,
)
from pipeline_core.discovery.research_idea_offspring_execution import OffspringExecutionReport
from pipeline_core.discovery.research_idea_search_contracts import GenerationalIdeaSearchShadowReport
from pipeline_core.discovery.research_idea_verified_active_search import IdeaEvolutionRequestReport
from pipeline_core.literature.acquisition.contracts import (
    AcquisitionProfile,
    CorpusSelectionReport,
    SelectedCorpusWork,
)
from pipeline_core.literature.catalog_contracts import LiteratureCatalogPacket


def _write(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load(path: Path, model):
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSONL objects: {path}")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if hasattr(row, "model_dump"):
                row = row.model_dump(mode="json")
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _case(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--case must use LABEL=/path/to/case-root")
    label, raw = value.split("=", 1)
    if not label.strip() or not raw.strip():
        raise argparse.ArgumentTypeError("--case must use LABEL=/path/to/case-root")
    return label.strip(), Path(raw).expanduser().resolve()


def _header(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--header must use KEY=VALUE")
    key, raw = value.split("=", 1)
    return key.strip(), raw


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run SIS-v2.8 Epistemic Realization Archive & Evidence Recovery. "
            "Strict HypothesisCard grounding remains unchanged. The runner decomposes "
            "ResearchIdea realizations into grounded/hypothetical/missing-evidence parts, "
            "builds a non-destructive multi-realization archive, and routes true evidence "
            "coverage gaps into the existing literature/KG recovery path."
        )
    )
    p.add_argument("--case", action="append", type=_case, required=True)
    p.add_argument("--results-per-query", type=int, default=30)
    p.add_argument(
        "--evidence-target-total",
        type=int,
        default=12,
        help=(
            "Maximum number of deduplicated evidence-recovery targets/axes sent to "
            "literature discovery; also caps M2 selected works."
        ),
    )
    p.add_argument("--providers", default="semantic_scholar,crossref")
    p.add_argument("--execute-literature-discovery", action="store_true")

    # Optional retry after the trusted acquisition/extraction/publication path has rebuilt context.
    p.add_argument("--recovered-context", action="append", type=_case, default=[])
    p.add_argument(
        "--model",
        default=(os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL") or os.getenv("OPENROUTER_AGENT_MODEL") or ""),
    )
    p.add_argument(
        "--prospective-model",
        default=(os.getenv("GRAPHAGENTS_HYPOTHESIS_CRITIC_MODEL") or os.getenv("OPENROUTER_CRITIC_MODEL") or os.getenv("OPENROUTER_AGENT_MODEL") or ""),
    )
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--instructor-mode", default="JSON")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--parse-retries", type=int, default=2)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--header", action="append", type=_header, default=[])
    p.add_argument("--max-retry-realizations-per-idea", type=int, default=2)
    p.add_argument("--max-retry-ideas-per-generation", type=int, default=8)
    p.add_argument("--max-retry-prospective-audits", type=int, default=12)
    p.add_argument("--skip-retry-prospective", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    return p


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError("None of the expected artifact paths exist: " + ", ".join(map(str, paths)))


def _latest_feedback(work_root: Path, generation: int) -> Path:
    root = work_root / f"g{generation}"
    rows = list(root.glob("round_*.feedback.json"))
    if not rows:
        raise FileNotFoundError(f"No v2.7 feedback reports found under {root}")

    def round_index(path: Path) -> int:
        stem = path.name.split(".", 1)[0]
        try:
            return int(stem.rsplit("_", 1)[-1])
        except ValueError:
            return -1

    return max(rows, key=lambda path: (round_index(path), str(path)))


def _paths(root: Path) -> dict[str, Path]:
    out = root / "scientific_portfolio_shadow"
    work = out / "sis_v2_7_verified_active_search"
    return {
        "out": out,
        "context": _first_existing(root / "hypothesis.context.json", out / "hypothesis.context.json"),
        "source_generational": _first_existing(
            out / "sis_v2_2.source_v2_1_1.generational_shadow.json",
            out / "sis_v2_1_1.feedback_aware.generational_shadow.json",
            out / "sis_v2_1.generational_shadow.json",
        ),
        "g2_execution": out / "sis_v2_3.offspring_execution.json",
        "g3_execution": out / "sis_v2_4.g3_offspring_execution.json",
        "g2_lifecycle": out / "sis_v2_7.g2_final_lifecycle.json",
        "g2_portfolio": out / "sis_v2_7.g2_final.portfolio.json",
        "g3_lifecycle": out / "sis_v2_7.g3_final_lifecycle.json",
        "g3_portfolio": out / "sis_v2_7.g3_final.portfolio.json",
        "evolution_requests": out / "sis_v2_7.idea_evolution_requests.json",
        "v2_7_work": work,
        "v2_8_root": out / "sis_v2_8_epistemic_archive",
        "decomposition": out / "sis_v2_8.epistemic_decomposition.json",
        "archive": out / "sis_v2_8.multi_realization_archive.json",
        "recovery": out / "sis_v2_8.evidence_recovery_plan.json",
        "profile": out / "sis_v2_8.evidence_recovery_profile.yaml",
        "commands": out / "sis_v2_8.evidence_recovery_commands.json",
        "summary": out / "sis_v2_8.case_summary.json",
    }


def _require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} artifact is missing: {path}")


def _research_ideas(
    source: GenerationalIdeaSearchShadowReport,
    g2: OffspringExecutionReport,
    g3: OffspringExecutionReport,
) -> list[Any]:
    by_id = {
        row.idea_id: row
        for row in [*source.research_ideas, *g2.offspring_nodes, *g3.offspring_nodes]
    }
    return list(by_id.values())


def _feedback_from_lifecycle(lifecycle: RealizationLifecycleReport) -> dict[str, Any]:
    records = []
    for row in lifecycle.observations:
        if row.hypothesis_id is None:
            continue
        records.append(
            {
                "hypothesis_id": row.hypothesis_id,
                "idea_id": row.idea_id,
                "prospective_evaluation_state": (
                    "EVALUATED" if row.prospective_status == "COMPLETE" else "NOT_EVALUATED"
                ),
                "residual_evaluation_state": "NOT_EVALUATED",
                "current_evidence_status": row.current_evidence_status,
                "prospective_identifiability": row.prospective_identifiability,
                "directionality_mode": row.directionality_mode,
                "measurement_compatibility_mode": row.measurement_compatibility_mode,
                "residual_epistemic_state": row.residual_epistemic_state,
                "residual_state_reason": row.residual_state_reason,
            }
        )
    return {"records": records}


def _filter_execution(execution: OffspringExecutionReport, idea_ids: set[str]) -> OffspringExecutionReport:
    nodes = [row for row in execution.offspring_nodes if row.idea_id in idea_ids]
    semantic = [row for row in execution.semantic_records if row.idea_id in idea_ids]
    allowed_ids = {row.idea_id for row in nodes}
    semantic = [row for row in semantic if row.idea_id in allowed_ids]
    identity = Counter(row.transition.identity_relation for row in semantic)
    disposition = Counter(row.disposition for row in semantic)
    generated_channel = Counter(row.channel for row in semantic)
    generated_operator = Counter(row.chosen_operator_id for row in semantic)
    accepted = sum(row.accepted_for_realization for row in semantic)
    digest = hashlib.sha256(
        json.dumps(
            {
                "source_report_id": execution.report_id,
                "retry_idea_ids": sorted(allowed_ids),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return execution.model_copy(
        update={
            "report_id": f"g{execution.generation_index}_evidence_recovery_subset:{digest[:20]}",
            "report_sha256": digest,
            "offspring_nodes": nodes,
            "semantic_records": semantic,
            "raw_offspring_count": len(nodes),
            "accepted_for_realization_count": accepted,
            "identity_relation_counts": dict(sorted(identity.items())),
            "disposition_counts": dict(sorted(disposition.items())),
            "generated_count_by_channel": dict(sorted(generated_channel.items())),
            "generated_count_by_operator": dict(sorted(generated_operator.items())),
            "semantic_noop_count": sum(row.disposition == "SEMANTIC_NOOP" for row in semantic),
            "mutation_attempt_count": len(semantic),
            "distinct_child_count": sum(row.transition.identity_relation == "DIFFERENT_IDEA" for row in semantic),
            "mutation_semantic_yield_fraction": (
                sum(row.transition.identity_relation == "DIFFERENT_IDEA" for row in semantic) / len(semantic)
                if semantic else 0.0
            ),
            "indeterminate_probe_count": sum(row.transition.identity_relation == "INDETERMINATE" for row in semantic),
            "channel_drift_child_count": sum(row.disposition == "ACCEPTED_CHANNEL_DRIFT_CHILD" for row in semantic),
            "exact_kernel_duplicate_suppressed_count": 0,
        }
    )


def _run_retry(
    *,
    args: argparse.Namespace,
    label: str,
    recovered_context_path: Path,
    recovery: EvidenceRecoveryPlan,
    g2_execution: OffspringExecutionReport,
    g3_execution: OffspringExecutionReport,
    output_root: Path,
) -> dict[str, Any]:
    if not args.model:
        raise SystemExit("--model (or GRAPHAGENTS_HYPOTHESIS_MODEL) is required with --recovered-context")
    context = _load(recovered_context_path, HypothesisContext)
    retry_ids = set(recovery.retry_candidate_idea_ids)
    headers = dict(args.header)
    backend = InstructorOpenAICompatibleHypothesisBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        temperature=args.temperature,
        parse_retries=args.parse_retries,
        timeout=args.timeout,
        extra_headers=headers,
        telemetry_path=output_root / "retry.telemetry.jsonl",
        telemetry_context={"pipeline": "sis_v2_8_evidence_recovery_retry", "case": label},
    )

    prospective_runner = None
    if not args.skip_retry_prospective:
        prospective_model = args.prospective_model or args.model

        def prospective_runner(*, context, candidate, source_stage, output_prefix):
            return run_prospective_identification_shadow(
                context=context,
                candidate=candidate,
                source_stage=source_stage,
                model=prospective_model,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                parse_retries=args.parse_retries,
                output_prefix=output_prefix,
            )

    payload: dict[str, Any] = {
        "schema_version": "sis-v2-8-evidence-recovery-retry-summary-v1",
        "recovered_context_id": context.context_id,
        "recovered_context_sha256": context.context_sha256,
        "requested_retry_idea_count": len(retry_ids),
        "generations": {},
        "same_research_idea_retry_only": True,
        "automatic_idea_mutation_executed": False,
        "strict_hypothesis_compiler_preserved": True,
        "production_selection_authority": False,
    }
    for generation, execution in ((2, g2_execution), (3, g3_execution)):
        eligible = {row.idea_id for row in execution.offspring_nodes} & retry_ids
        if not eligible:
            payload["generations"][f"G{generation}"] = {
                "retry_idea_count": 0,
                "materialized_hypothesis_count": 0,
                "status": "NO_ELIGIBLE_RETRY_IDEAS",
            }
            continue
        filtered = _filter_execution(execution, eligible)
        lifecycle, portfolio = run_realization_lifecycle(
            execution=filtered,
            context=context,
            backend=backend,
            output_dir=output_root / f"g{generation}",
            max_ideas=min(args.max_retry_ideas_per_generation, len(eligible)),
            max_realizations_per_idea=args.max_retry_realizations_per_idea,
            max_per_parent=max(1, args.max_retry_ideas_per_generation),
            max_repair_attempts=1,
            prospective_runner=prospective_runner,
            max_prospective_audits=args.max_retry_prospective_audits,
        )
        _write(output_root / f"g{generation}.retry_lifecycle.json", lifecycle)
        _write(output_root / f"g{generation}.retry_portfolio.json", portfolio)
        payload["generations"][f"G{generation}"] = {
            "retry_idea_count": len(eligible),
            "target_idea_ids": sorted(eligible),
            "materialized_hypothesis_count": lifecycle.materialized_hypothesis_count,
            "usable_grounded_realization_count": lifecycle.usable_grounded_realization_count,
            "not_operationalizable_count": lifecycle.prospective_not_operationalizable_count,
            "status": "COMPLETE",
            "lifecycle_report_id": lifecycle.report_id,
            "portfolio_id": portfolio.portfolio_id,
        }
    _write(output_root / "retry.summary.json", payload)
    return payload


def _run_command(label: str, cmd: list[str], out: Path) -> int:
    print()
    print("-" * 88)
    print(label)
    print("-" * 88)
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    out.mkdir(parents=True, exist_ok=True)
    safe = "_".join(label.lower().split())
    (out / f"{safe}.stdout.txt").write_text(result.stdout or "", encoding="utf-8")
    (out / f"{safe}.stderr.txt").write_text(result.stderr or "", encoding="utf-8")
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    return result.returncode



def _build_relevance_selection_artifacts(
    *,
    recovery: EvidenceRecoveryPlan,
    evidence_root: Path,
    max_strict_total: int,
) -> dict[str, Any]:
    profile_payload = recovery.targeted_acquisition_profile
    if profile_payload is None:
        return {}
    catalog_path = evidence_root / "m1_catalog" / "catalog.json"
    assessments_path = evidence_root / "m2_selection" / "assessments.jsonl"
    _require(catalog_path, "M1 catalog")
    _require(assessments_path, "M2 assessments")

    catalog_packet = _load(catalog_path, LiteratureCatalogPacket)
    assessment_rows = _load_jsonl(assessments_path)
    gate = build_recovery_relevance_gate(
        acquisition_profile=profile_payload,
        catalog=catalog_packet.model_dump(mode="json"),
        assessments=assessment_rows,
        max_strict_total=max_strict_total,
        recovery_plan=recovery,
    )

    gate_root = evidence_root / "m2_5_relevance"
    gate_root.mkdir(parents=True, exist_ok=True)
    _write(gate_root / "relevance_gate_report.json", gate)

    work_by_id = {row.work_id: row for row in catalog_packet.works}
    record_by_id = {row.work_id: row for row in gate.records}
    strict_rows: list[SelectedCorpusWork] = []
    for work_id in gate.strict_selected_work_ids:
        work = work_by_id[work_id]
        relevance = record_by_id[work_id]
        strict_rows.append(
            SelectedCorpusWork(
                work_id=work.work_id,
                title=work.title,
                doi=work.doi,
                year=work.year,
                venue=work.venue,
                open_access_url=work.open_access_url,
                matched_axes=list(relevance.matched_axes),
                primary_quota_axis=None,
                total_score=float(relevance.original_total_score),
            )
        )
    _write_jsonl(gate_root / "strict_selected_works.jsonl", strict_rows)

    adjacent_rows = [
        row
        for row in gate.records
        if row.relevance_class == "ADJACENT_METHOD_OR_MECHANISM"
    ]
    _write_jsonl(gate_root / "adjacent_candidates.jsonl", adjacent_rows)

    strict_axis_counts = Counter(
        axis
        for row in gate.records
        if row.relevance_class == "STRICT_DOMAIN_RELEVANT"
        for axis in row.matched_axes
    )
    axis_ids = [str(row.get("axis_id")) for row in profile_payload.get("axes") or []]
    strict_selection_report = CorpusSelectionReport(
        profile_id=str(profile_payload["profile_id"]),
        source_catalog_id=catalog_packet.catalog_id,
        candidate_count=len(catalog_packet.works),
        eligible_count=gate.strict_candidate_count,
        manual_review_count=gate.adjacent_candidate_count,
        excluded_count=max(
            0,
            len(catalog_packet.works)
            - gate.strict_candidate_count
            - gate.adjacent_candidate_count,
        ),
        selected_count=len(strict_rows),
        target_total=max_strict_total,
        axis_candidate_counts={axis_id: strict_axis_counts.get(axis_id, 0) for axis_id in axis_ids},
        axis_quota_targets={axis_id: 0 for axis_id in axis_ids},
        axis_primary_selected_counts={axis_id: 0 for axis_id in axis_ids},
        unfilled_axis_quotas={},
        selected_work_ids=[row.work_id for row in strict_rows],
        policy_notes=[
            "SIS-v2.8 post-M2 recovery relevance gate.",
            "target_total is a maximum acquisition budget, not a fill target.",
            "STRICT_DOMAIN_RELEVANT candidates alone may proceed to M3.",
            "ADJACENT_METHOD_OR_MECHANISM candidates are archived and are not positive premises.",
            "OFF_DOMAIN_REJECT candidates are not acquired.",
        ],
        positive_evidence_promotion_performed=False,
    )
    _write(gate_root / "strict_selection_report.json", strict_selection_report)

    print()
    print("-" * 88)
    print("SIS-v2.8 recovery relevance gate")
    print("-" * 88)
    print("shared domain anchors:", gate.derived_domain_anchors[:20])
    print("request-conditioned relevance:", gate.request_conditioning_applied)
    print(
        "strict / adjacent / off-domain:",
        gate.strict_candidate_count,
        "/",
        gate.adjacent_candidate_count,
        "/",
        gate.off_domain_reject_count,
    )
    print(
        "strict selected / max budget:",
        gate.strict_selected_count,
        "/",
        gate.max_strict_acquisition_total,
    )
    if gate.strict_selected_count < gate.max_strict_acquisition_total:
        print("strict selection intentionally under-filled; max budget is not a quota")

    return {
        "report": gate,
        "strict_selection_report": strict_selection_report,
        "relevance_gate_report_path": str(gate_root / "relevance_gate_report.json"),
        "strict_selected_works_path": str(gate_root / "strict_selected_works.jsonl"),
        "strict_selection_report_path": str(gate_root / "strict_selection_report.json"),
        "adjacent_candidate_archive_path": str(gate_root / "adjacent_candidates.jsonl"),
    }


def main() -> int:
    args = parser().parse_args()
    if args.results_per_query < 1 or args.results_per_query > 100:
        raise SystemExit("--results-per-query must be in [1,100]")
    recovered_by_label = {label: path.expanduser().resolve() for label, path in args.recovered_context}
    output = args.output.expanduser().resolve()
    cases: dict[str, Any] = {}
    aggregate = Counter()

    for label, root in args.case:
        paths = _paths(root)
        for key in (
            "context", "source_generational", "g2_execution", "g3_execution",
            "g2_lifecycle", "g2_portfolio", "g3_lifecycle", "g3_portfolio",
            "evolution_requests",
        ):
            _require(paths[key], key)
        g2_feedback_path = _latest_feedback(paths["v2_7_work"], 2)
        g3_feedback_path = _latest_feedback(paths["v2_7_work"], 3)

        context = _load(paths["context"], HypothesisContext)
        source = _load(paths["source_generational"], GenerationalIdeaSearchShadowReport)
        g2_execution = _load(paths["g2_execution"], OffspringExecutionReport)
        g3_execution = _load(paths["g3_execution"], OffspringExecutionReport)
        g2_lifecycle = _load(paths["g2_lifecycle"], RealizationLifecycleReport)
        g3_lifecycle = _load(paths["g3_lifecycle"], RealizationLifecycleReport)
        g2_portfolio = _load(paths["g2_portfolio"], HypothesisPortfolio)
        g3_portfolio = _load(paths["g3_portfolio"], HypothesisPortfolio)
        evolution = _load(paths["evolution_requests"], IdeaEvolutionRequestReport)
        feedbacks = [_load_json(g2_feedback_path), _load_json(g3_feedback_path)]
        ideas = _research_ideas(source, g2_execution, g3_execution)

        decomposition = build_epistemic_decomposition_report(
            research_ideas=ideas,
            lifecycles=[g2_lifecycle, g3_lifecycle],
            portfolios=[g2_portfolio, g3_portfolio],
            feedback_reports=feedbacks,
            evolution_request_report=evolution,
        )
        archive = build_multi_realization_archive(decomposition)
        recovery = build_evidence_recovery_plan(
            decomposition,
            domain_profile_id=context.domain_profile_id,
            results_per_query=args.results_per_query,
            target_total=args.evidence_target_total,
        )
        _write(paths["decomposition"], decomposition)
        _write(paths["archive"], archive)
        _write(paths["recovery"], recovery)

        command_plan = None
        discovery_codes: dict[str, int] = {}
        relevance_artifacts: dict[str, Any] = {}
        if recovery.targeted_acquisition_profile is not None:
            # Validate against the repository's existing acquisition contract before writing.
            AcquisitionProfile.model_validate(recovery.targeted_acquisition_profile)
            paths["profile"].write_text(
                yaml.safe_dump(
                    recovery.targeted_acquisition_profile,
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
            command_plan = build_evidence_recovery_command_plan(
                profile_path=str(paths["profile"]),
                output_root=str(paths["v2_8_root"] / "evidence_recovery"),
                providers=args.providers,
                results_per_query=args.results_per_query,
                python_executable=sys.executable,
            )
            _write(paths["commands"], command_plan)
            if args.execute_literature_discovery:
                discovery_codes["M1_DISCOVERY"] = _run_command(
                    f"{label} SIS-v2.8 targeted literature discovery",
                    command_plan.discovery_command,
                    paths["v2_8_root"] / "evidence_recovery",
                )
                if discovery_codes["M1_DISCOVERY"] == 0:
                    discovery_codes["M2_SELECTION"] = _run_command(
                        f"{label} SIS-v2.8 targeted literature selection",
                        command_plan.selection_command,
                        paths["v2_8_root"] / "evidence_recovery",
                    )
                if discovery_codes.get("M2_SELECTION") == 0:
                    relevance_artifacts = _build_relevance_selection_artifacts(
                        recovery=recovery,
                        evidence_root=paths["v2_8_root"] / "evidence_recovery",
                        max_strict_total=args.evidence_target_total,
                    )
                    discovery_codes["M2_5_RELEVANCE_GATE"] = 0

        retry_summary = None
        if label in recovered_by_label:
            recovered = recovered_by_label[label]
            _require(recovered, f"{label} recovered context")
            retry_summary = _run_retry(
                args=args,
                label=label,
                recovered_context_path=recovered,
                recovery=recovery,
                g2_execution=g2_execution,
                g3_execution=g3_execution,
                output_root=paths["v2_8_root"] / "retry_after_evidence_recovery",
            )

        summary = {
            "schema_version": "sis-v2-8-epistemic-archive-case-summary-v1",
            "case": label,
            "idea_count": decomposition.idea_count,
            "epistemic_realization_count": decomposition.record_count,
            "maturity_counts": decomposition.maturity_counts,
            "missing_evidence_requirement_kind_counts": decomposition.requirement_kind_counts,
            "literature_recovery_requirement_count": decomposition.literature_recovery_requirement_count,
            "kg_retraversal_requirement_count": decomposition.kg_retraversal_requirement_count,
            "archive_entry_count": archive.realization_entry_count,
            "archive_retained_entry_count": archive.retained_entry_count,
            "multi_realization_idea_count": archive.multi_realization_idea_count,
            "epistemic_diversity_retention_count": archive.epistemic_diversity_retention_count,
            "evidence_recovery_route_counts": recovery.route_counts,
            "literature_recovery_request_count_raw": recovery.literature_request_count,
            "literature_recovery_cluster_count": recovery.literature_cluster_count,
            "literature_requests_collapsed_by_clustering_count": recovery.literature_requests_collapsed_by_clustering_count,
            "targeted_literature_recovery_axis_count": recovery.targeted_literature_axis_count,
            "literature_clusters_suppressed_by_budget_count": recovery.literature_clusters_suppressed_by_budget_count,
            "recovery_target_budget": recovery.recovery_target_budget,
            "retry_candidate_idea_count": len(recovery.retry_candidate_idea_ids),
            "targeted_acquisition_profile_created": recovery.targeted_acquisition_profile is not None,
            "literature_discovery_stage_return_codes": discovery_codes,
            "recovery_relevance_gate_executed": bool(relevance_artifacts),
            "recovery_relevance_class_counts": (
                {
                    "STRICT_DOMAIN_RELEVANT": relevance_artifacts["report"].strict_candidate_count,
                    "ADJACENT_METHOD_OR_MECHANISM": relevance_artifacts["report"].adjacent_candidate_count,
                    "OFF_DOMAIN_REJECT": relevance_artifacts["report"].off_domain_reject_count,
                }
                if relevance_artifacts else {}
            ),
            "strict_recovery_selected_count": (
                relevance_artifacts["report"].strict_selected_count if relevance_artifacts else 0
            ),
            "strict_recovery_selection_max_budget": args.evidence_target_total,
            "adjacent_candidate_archive_count": (
                relevance_artifacts["report"].adjacent_candidate_count if relevance_artifacts else 0
            ),
            "relevance_gate_artifact_paths": (
                {key: value for key, value in relevance_artifacts.items() if key.endswith("_path")}
                if relevance_artifacts else {}
            ),
            "retry_after_recovered_context": retry_summary,
            "strict_hypothesis_card_contract_preserved": True,
            "ungrounded_idea_deletion_authority": False,
            "external_literature_direct_premise_injection": False,
            "destructive_realization_replacement_applied": False,
            "automatic_canonical_graph_mutation_executed": False,
            "production_selection_authority": False,
        }
        _write(paths["summary"], summary)
        cases[label] = summary
        aggregate["case_count"] += 1
        aggregate["idea_count"] += decomposition.idea_count
        aggregate["epistemic_realization_count"] += decomposition.record_count
        aggregate["literature_recovery_requirement_count"] += decomposition.literature_recovery_requirement_count
        aggregate["literature_recovery_request_count_raw"] += recovery.literature_request_count
        aggregate["literature_recovery_cluster_count"] += recovery.literature_cluster_count
        aggregate["literature_requests_collapsed_by_clustering_count"] += recovery.literature_requests_collapsed_by_clustering_count
        aggregate["targeted_literature_recovery_axis_count"] += recovery.targeted_literature_axis_count
        aggregate["literature_clusters_suppressed_by_budget_count"] += recovery.literature_clusters_suppressed_by_budget_count
        if relevance_artifacts:
            aggregate["strict_recovery_candidate_count"] += relevance_artifacts["report"].strict_candidate_count
            aggregate["adjacent_recovery_candidate_count"] += relevance_artifacts["report"].adjacent_candidate_count
            aggregate["off_domain_recovery_reject_count"] += relevance_artifacts["report"].off_domain_reject_count
            aggregate["strict_recovery_selected_count"] += relevance_artifacts["report"].strict_selected_count
        aggregate["kg_retraversal_requirement_count"] += decomposition.kg_retraversal_requirement_count
        aggregate["archive_entry_count"] += archive.realization_entry_count
        aggregate["archive_retained_entry_count"] += archive.retained_entry_count
        aggregate["retry_candidate_idea_count"] += len(recovery.retry_candidate_idea_ids)

        print()
        print("=" * 88)
        print(label)
        print("=" * 88)
        print("ideas / epistemic realizations:", decomposition.idea_count, "/", decomposition.record_count)
        print("maturity:", decomposition.maturity_counts)
        print("missing evidence:", decomposition.requirement_kind_counts)
        print("recovery routes:", recovery.route_counts)
        print(
            "literature recovery raw / clusters / targeted / budget:",
            recovery.literature_request_count,
            "/",
            recovery.literature_cluster_count,
            "/",
            recovery.targeted_literature_axis_count,
            "/",
            recovery.recovery_target_budget,
        )
        if relevance_artifacts:
            gate = relevance_artifacts["report"]
            print(
                "recovery relevance strict / adjacent / off-domain:",
                gate.strict_candidate_count,
                "/",
                gate.adjacent_candidate_count,
                "/",
                gate.off_domain_reject_count,
            )
            print(
                "strict recovery selected / max budget:",
                gate.strict_selected_count,
                "/",
                gate.max_strict_acquisition_total,
            )
        print("archive retained / all:", archive.retained_entry_count, "/", archive.realization_entry_count)
        print("multi-realization ideas:", archive.multi_realization_idea_count)
        print("retry candidates:", len(recovery.retry_candidate_idea_ids))
        print("STRICT_HYPOTHESIS_CARD_CONTRACT_PRESERVED=True")
        print("UNGROUNDED_IDEA_DELETION_AUTHORITY=False")
        print("EXTERNAL_LITERATURE_DIRECT_PREMISE_INJECTION=False")
        print("DESTRUCTIVE_REALIZATION_REPLACEMENT=False")

    payload = {
        "schema_version": "sis-v2-8-epistemic-realization-archive-evidence-recovery-cohort-v1",
        "cases": cases,
        "aggregate": dict(aggregate),
        "research_idea_can_survive_without_current_grounded_hypothesis": True,
        "strict_hypothesis_card_contract_preserved": True,
        "multi_realization_archive_is_non_destructive": True,
        "missing_evidence_routes_before_idea_mutation": True,
        "external_literature_requires_positive_evidence_promotion_before_retry": True,
        "automatic_canonical_graph_mutation_executed": False,
        "production_selection_authority": False,
        "scientific_truth_authority": False,
        "novelty_authority": False,
    }
    _write(output, payload)
    print()
    print("SIS-v2.8 Epistemic Realization Archive & Evidence Recovery complete")
    print("aggregate:", dict(aggregate))
    print("output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
