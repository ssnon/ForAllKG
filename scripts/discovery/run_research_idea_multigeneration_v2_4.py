from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.research_idea_multigeneration import (
    InstructorOpenAICompatibleConceptualDeltaBackend,
    build_calibrated_g3_search,
    build_multigeneration_case_summary,
    run_conceptual_delta_audit,
)
from pipeline_core.discovery.research_idea_offspring_execution import (
    InstructorOpenAICompatibleOffspringBackend,
    OffspringExecutionReport,
    OffspringRealizationReport,
    build_offspring_generation_plan,
    execute_offspring_plan,
)
from pipeline_core.discovery.research_idea_search_contracts import (
    GenerationalIdeaSearchShadowReport,
)


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


def _case(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--case must use LABEL=/path/to/case-root")
    label, raw = value.split("=", 1)
    label = label.strip()
    raw = raw.strip()
    if not label or not raw:
        raise argparse.ArgumentTypeError("--case must use LABEL=/path/to/case-root")
    return label, Path(raw).expanduser().resolve()


def _header(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--header values must use KEY=VALUE")
    key, item = value.split("=", 1)
    if not key.strip():
        raise argparse.ArgumentTypeError("header key may not be empty")
    return key.strip(), item


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run SIS-v2.4 independent conceptual-delta calibration over G2, compile "
            "calibrated G2 credit into a bounded G3 parent allocation, execute actual "
            "G3 ResearchIdea offspring, calibrate G3, and emit G0-to-G3 cohort metrics."
        )
    )
    p.add_argument("--case", action="append", type=_case, required=True)
    p.add_argument("--max-g3-parents", type=int, default=8)
    p.add_argument("--max-g3-raw-offspring", type=int, default=12)
    p.add_argument("--g3-semantic-retries", type=int, default=1)
    p.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
            or ""
        ),
    )
    p.add_argument(
        "--audit-model",
        default=(
            os.getenv("GRAPHAGENTS_CRITIC_MODEL")
            or os.getenv("OPENROUTER_CRITIC_MODEL")
            or None
        ),
    )
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--instructor-mode", default="JSON")
    p.add_argument("--offspring-temperature", type=float, default=0.2)
    p.add_argument("--audit-temperature", type=float, default=0.0)
    p.add_argument("--parse-retries", type=int, default=2)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--header", action="append", default=[], type=_header)
    p.add_argument("--output", required=True, type=Path)
    return p


def _case_paths(root: Path) -> dict[str, Path]:
    out = root / "scientific_portfolio_shadow"
    return {
        "source_v211": out / "sis_v2_2.source_v2_1_1.generational_shadow.json",
        "g2_execution": out / "sis_v2_3.offspring_execution.json",
        "g2_realization": out / "sis_v2_3.realization.report.json",
        "g2_audit": out / "sis_v2_4.g2_conceptual_delta.json",
        "g2_population": out / "sis_v2_4.g2_calibrated_population.json",
        "g3_search": out / "sis_v2_4.g3_credit_search.json",
        "g3_plan": out / "sis_v2_4.g3_offspring_plan.json",
        "g3_execution": out / "sis_v2_4.g3_offspring_execution.json",
        "g3_audit": out / "sis_v2_4.g3_conceptual_delta.json",
        "summary": out / "sis_v2_4.multigeneration.summary.json",
        "g2_audit_telemetry": out / "sis_v2_4.g2_conceptual_delta.telemetry.jsonl",
        "g3_offspring_telemetry": out / "sis_v2_4.g3_offspring.telemetry.jsonl",
        "g3_audit_telemetry": out / "sis_v2_4.g3_conceptual_delta.telemetry.jsonl",
    }


def _find_context(root: Path, *, context_id: str, context_sha256: str) -> HypothesisContext:
    matches: list[tuple[Path, HypothesisContext]] = []
    for path in root.rglob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                prefix = handle.read(16384)
            if '"hypothesis-context-v1"' not in prefix:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("schema_version") != "hypothesis-context-v1":
            continue
        if str(payload.get("context_id") or "") != context_id:
            continue
        if str(payload.get("context_sha256") or "") != context_sha256:
            continue
        try:
            context = HypothesisContext.model_validate(payload)
        except Exception:
            continue
        matches.append((path.resolve(), context))
    if not matches:
        raise FileNotFoundError(
            f"No exact-lineage HypothesisContext found under {root}: "
            f"context_id={context_id}"
        )
    matches.sort(key=lambda item: (len(str(item[0])), str(item[0])))
    return matches[0][1]


def main() -> int:
    args = parser().parse_args()
    if args.max_g3_parents < 1 or args.max_g3_raw_offspring < 1:
        raise ValueError("G3 parent/offspring bounds must be >= 1")
    if args.g3_semantic_retries < 0:
        raise ValueError("--g3-semantic-retries must be >= 0")
    if not args.model:
        raise SystemExit(
            "--model is required unless GRAPHAGENTS_HYPOTHESIS_MODEL or "
            "OPENROUTER_AGENT_MODEL is set"
        )

    audit_model = args.audit_model or args.model
    headers = dict(args.header)
    cohort: dict[str, Any] = {
        "schema_version": "sis-v2-4-calibrated-multigeneration-cohort-v1",
        "case_count": len(args.case),
        "cases": {},
        "calibration_is_not_hard_gate": True,
        "verification_is_not_fertility_authority": True,
        "conceptual_audit_is_not_fertility_authority": True,
        "single_scalar_fitness_used": False,
        "generator_model": args.model,
        "conceptual_auditor_model": audit_model,
        "conceptual_auditor_model_distinct_from_generator": audit_model != args.model,
        "production_generation_authority": False,
        "production_selection_authority": False,
        "scientific_truth_authority": False,
        "novelty_authority": False,
    }

    aggregate = {
        "g2_mutation_attempts": 0,
        "g2_deterministic_distinct": 0,
        "g2_calibrated_strong_distinct": 0,
        "g2_calibration_disagreements": 0,
        "g2_materialized": 0,
        "g3_raw_offspring": 0,
        "g3_mutation_attempts": 0,
        "g3_deterministic_distinct": 0,
        "g3_calibrated_strong_distinct": 0,
        "g3_calibration_disagreements": 0,
        "conceptual_audit_llm_calls": 0,
        "g3_offspring_llm_calls": 0,
    }

    for label, root in args.case:
        paths = _case_paths(root)
        missing = [
            str(paths[name])
            for name in ("source_v211", "g2_execution", "g2_realization")
            if not paths[name].is_file()
        ]
        if missing:
            raise FileNotFoundError(f"{label} missing SIS-v2.3 prerequisites: {missing}")

        source_v211 = _load(paths["source_v211"], GenerationalIdeaSearchShadowReport)
        g2_execution = _load(paths["g2_execution"], OffspringExecutionReport)
        g2_realization = _load(paths["g2_realization"], OffspringRealizationReport)
        if not source_v211.research_ideas:
            raise ValueError(f"{label}: source generational population is empty")
        first = source_v211.research_ideas[0]
        context = _find_context(
            root,
            context_id=first.source_context_id,
            context_sha256=first.source_context_sha256,
        )

        g2_audit_backend = InstructorOpenAICompatibleConceptualDeltaBackend(
            model=audit_model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            instructor_mode=args.instructor_mode,
            temperature=args.audit_temperature,
            parse_retries=args.parse_retries,
            timeout=args.timeout,
            extra_headers=headers,
            telemetry_path=paths["g2_audit_telemetry"],
            telemetry_context={"case": label, "generation": 2},
        )
        g2_audit = run_conceptual_delta_audit(
            execution=g2_execution,
            parent_nodes=source_v211.research_ideas,
            research_question=context.question,
            backend=g2_audit_backend,
        )
        _write(paths["g2_audit"], g2_audit)

        g2_population, g3_search = build_calibrated_g3_search(
            execution=g2_execution,
            realization=g2_realization,
            conceptual_audit=g2_audit,
            max_parent_budget=args.max_g3_parents,
        )
        _write(paths["g2_population"], g2_population)
        _write(paths["g3_search"], g3_search)

        g3_plan = build_offspring_generation_plan(
            evolutionary_report=g3_search,
            generational_report=g2_population,
            max_raw_offspring=args.max_g3_raw_offspring,
            generation_index=3,
        )
        _write(paths["g3_plan"], g3_plan)

        g3_backend = InstructorOpenAICompatibleOffspringBackend(
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            instructor_mode=args.instructor_mode,
            temperature=args.offspring_temperature,
            parse_retries=args.parse_retries,
            timeout=args.timeout,
            extra_headers=headers,
            telemetry_path=paths["g3_offspring_telemetry"],
            telemetry_context={"case": label, "generation": 3},
        )
        g3_execution, _ = execute_offspring_plan(
            plan=g3_plan,
            evolutionary_report=g3_search,
            generational_report=g2_population,
            research_question=context.question,
            backend=g3_backend,
            semantic_retry_limit=args.g3_semantic_retries,
        )
        _write(paths["g3_execution"], g3_execution)

        g3_audit_backend = InstructorOpenAICompatibleConceptualDeltaBackend(
            model=audit_model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            instructor_mode=args.instructor_mode,
            temperature=args.audit_temperature,
            parse_retries=args.parse_retries,
            timeout=args.timeout,
            extra_headers=headers,
            telemetry_path=paths["g3_audit_telemetry"],
            telemetry_context={"case": label, "generation": 3},
        )
        g3_audit = run_conceptual_delta_audit(
            execution=g3_execution,
            parent_nodes=g2_population.research_ideas,
            research_question=context.question,
            backend=g3_audit_backend,
        )
        _write(paths["g3_audit"], g3_audit)

        summary = build_multigeneration_case_summary(
            source_generational_report=source_v211,
            g2_execution=g2_execution,
            g2_audit=g2_audit,
            g2_realization=g2_realization,
            g3_search=g3_search,
            g3_execution=g3_execution,
            g3_audit=g3_audit,
        )
        _write(paths["summary"], summary)
        cohort["cases"][label] = {
            "case_root": str(root),
            "g2_conceptual_audit": str(paths["g2_audit"]),
            "g3_credit_search": str(paths["g3_search"]),
            "g3_offspring_execution": str(paths["g3_execution"]),
            "g3_conceptual_audit": str(paths["g3_audit"]),
            "summary": summary.model_dump(mode="json"),
        }

        aggregate["g2_mutation_attempts"] += g2_audit.mutation_attempt_count
        aggregate["g2_deterministic_distinct"] += g2_execution.distinct_child_count
        aggregate["g2_calibrated_strong_distinct"] += sum(
            row.strong_distinct_child
            and next(
                s.channel for s in g2_execution.semantic_records if s.idea_id == row.child_idea_id
            ) != "EXPLOIT"
            for row in g2_audit.records
        )
        aggregate["g2_calibration_disagreements"] += g2_audit.disagreement_count
        aggregate["g2_materialized"] += g2_realization.materialized_hypothesis_count
        aggregate["g3_raw_offspring"] += g3_execution.raw_offspring_count
        aggregate["g3_mutation_attempts"] += g3_audit.mutation_attempt_count
        aggregate["g3_deterministic_distinct"] += g3_execution.distinct_child_count
        aggregate["g3_calibrated_strong_distinct"] += sum(
            row.strong_distinct_child
            and next(
                s.channel for s in g3_execution.semantic_records if s.idea_id == row.child_idea_id
            ) != "EXPLOIT"
            for row in g3_audit.records
        )
        aggregate["g3_calibration_disagreements"] += g3_audit.disagreement_count
        aggregate["conceptual_audit_llm_calls"] += 2
        aggregate["g3_offspring_llm_calls"] += g3_execution.llm_call_count

        print()
        print("=" * 88)
        print(label)
        print("=" * 88)
        print("G2 deterministic identity:", g2_execution.identity_relation_counts)
        print("G2 independent conceptual audit:", g2_audit.category_counts)
        print("G2 calibration disagreements:", g2_audit.disagreement_count)
        print(
            "G2 calibrated mutation yield:",
            f"{g2_audit.calibrated_mutation_yield_fraction:.3f}",
        )
        print("G2 materialized hypotheses:", g2_realization.materialized_hypothesis_count)
        print("G3 parent channels:", g3_search.selected_parent_channel_counts)
        print("G3 disagreement parents retained:", g3_search.disagreement_parent_count)
        print("G3 low-realization TRANSFORM parents:", g3_search.low_realization_transform_parent_count)
        print("G3 raw offspring:", g3_execution.raw_offspring_count)
        print("G3 deterministic identity:", g3_execution.identity_relation_counts)
        print("G3 independent conceptual audit:", g3_audit.category_counts)
        print("G3 calibration disagreements:", g3_audit.disagreement_count)
        print(
            "G3 calibrated mutation yield:",
            f"{g3_audit.calibrated_mutation_yield_fraction:.3f}",
        )
        print("G3 OFFSPRING_LLM_CALLS:", g3_execution.llm_call_count)
        print("CALIBRATION_IS_HARD_GATE=False")
        print("PRODUCTION_GENERATION_AUTHORITY=False")
        print("PRODUCTION_SELECTION_AUTHORITY=False")

    aggregate["g2_calibrated_mutation_yield_fraction"] = (
        aggregate["g2_calibrated_strong_distinct"] / aggregate["g2_mutation_attempts"]
        if aggregate["g2_mutation_attempts"]
        else 0.0
    )
    aggregate["g3_calibrated_mutation_yield_fraction"] = (
        aggregate["g3_calibrated_strong_distinct"] / aggregate["g3_mutation_attempts"]
        if aggregate["g3_mutation_attempts"]
        else 0.0
    )
    cohort["aggregate"] = aggregate
    cohort["new_llm_calls"] = True
    _write(args.output.expanduser().resolve(), cohort)

    print()
    print("SIS-v2.4 calibrated multi-generation cohort complete")
    print("cases:", len(args.case))
    print("generator model:", args.model)
    print("conceptual auditor model:", audit_model)
    print("auditor model distinct from generator:", audit_model != args.model)
    print("aggregate:", aggregate)
    print("CALIBRATION_IS_HARD_GATE=False")
    print("PRODUCTION_GENERATION_AUTHORITY=False")
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print("output:", args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
