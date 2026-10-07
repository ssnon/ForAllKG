from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline_core.discovery.research_idea_epistemic_archive import (
    EpistemicDecompositionReport,
    MultiRealizationArchiveReport,
)
from pipeline_core.discovery.research_idea_parallel_partial_search import (
    ParallelPartialSearchReport,
    build_parallel_partial_search_report,
)


def _case(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--case must use LABEL=/path/to/case-root")
    label, raw = value.split("=", 1)
    if not label.strip() or not raw.strip():
        raise argparse.ArgumentTypeError("--case must use LABEL=/path/to/case-root")
    return label.strip(), Path(raw).expanduser().resolve()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run SIS-v2.8b Parallel Partial-Realization Search. This deterministic "
            "shadow keeps retained partial/speculative/evidence-seeking realizations "
            "alive in the search population, converts evidence requirements into "
            "non-blocking epistemic debt, and emits child-idea evolution handoffs. "
            "No literature acquisition, LLM generation, or graph mutation is executed."
        )
    )
    p.add_argument("--case", action="append", type=_case, required=True)
    p.add_argument(
        "--acquisition-persistence-threshold",
        type=int,
        default=2,
        help=(
            "Repeated equivalent evidence debt occurrences required before acquisition "
            "becomes an optional escalation. This never makes acquisition blocking."
        ),
    )
    p.add_argument("--output", type=Path, required=True)
    return p


def _load(path: Path, model):
    if not path.is_file():
        raise FileNotFoundError(f"Required SIS artifact is missing: {path}")
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _paths(root: Path) -> dict[str, Path]:
    out = root / "scientific_portfolio_shadow"
    v28b = out / "sis_v2_8b_parallel_partial_search"
    return {
        "decomposition": out / "sis_v2_8.epistemic_decomposition.json",
        "archive": out / "sis_v2_8.multi_realization_archive.json",
        "root": v28b,
        "report": out / "sis_v2_8b.parallel_partial_search.json",
        "debts": v28b / "epistemic_debt_tickets.json",
        "handoffs": v28b / "idea_evolution_handoffs.json",
        "members": v28b / "search_members.json",
        "states": v28b / "idea_search_states.json",
        "summary": out / "sis_v2_8b.case_summary.json",
    }


def _summary(report: ParallelPartialSearchReport) -> dict[str, Any]:
    return {
        "schema_version": "sis-v2-8b-parallel-partial-case-summary-v1",
        "report_id": report.report_id,
        "idea_count": report.idea_count,
        "active_idea_count": report.active_idea_count,
        "member_count": report.member_count,
        "retained_member_count": report.retained_member_count,
        "active_member_count": report.active_member_count,
        "partially_grounded_active_count": report.partially_grounded_active_count,
        "speculative_active_count": report.speculative_active_count,
        "evidence_seeking_active_count": report.evidence_seeking_active_count,
        "child_parent_eligible_member_count": report.child_parent_eligible_member_count,
        "same_idea_search_eligible_member_count": report.same_idea_search_eligible_member_count,
        "epistemic_debt_count": report.epistemic_debt_count,
        "acquisition_escalation_eligible_debt_count": (
            report.acquisition_escalation_eligible_debt_count
        ),
        "evolution_handoff_count": report.evolution_handoff_count,
        "maturity_counts": report.maturity_counts,
        "lane_counts": report.lane_counts,
        "evidence_recovery_is_blocking_inner_loop": False,
        "partial_realizations_remain_search_reproductive": True,
        "speculative_falsifiable_realizations_remain_search_reproductive": True,
        "evidence_acquisition_is_optional_escalation": True,
        "strict_hypothesis_card_contract_preserved": True,
        "new_retrieval_calls": False,
        "new_llm_calls": False,
        "offspring_generation_executed": False,
        "canonical_graph_mutated": False,
    }


def main() -> int:
    args = parser().parse_args()
    if args.acquisition_persistence_threshold < 1:
        raise SystemExit("--acquisition-persistence-threshold must be >= 1")

    cohort: dict[str, Any] = {}
    aggregate = Counter()
    for label, root in args.case:
        paths = _paths(root)
        decomposition = _load(paths["decomposition"], EpistemicDecompositionReport)
        archive = _load(paths["archive"], MultiRealizationArchiveReport)
        report = build_parallel_partial_search_report(
            decomposition=decomposition,
            archive=archive,
            acquisition_persistence_threshold=args.acquisition_persistence_threshold,
        )
        _write(paths["report"], report)
        _write(paths["members"], [row.model_dump(mode="json") for row in report.members])
        _write(paths["states"], [row.model_dump(mode="json") for row in report.idea_states])
        _write(paths["debts"], [row.model_dump(mode="json") for row in report.epistemic_debts])
        _write(paths["handoffs"], [row.model_dump(mode="json") for row in report.evolution_handoffs])
        summary = _summary(report)
        _write(paths["summary"], summary)
        cohort[label] = summary

        aggregate["case_count"] += 1
        aggregate["idea_count"] += report.idea_count
        aggregate["active_idea_count"] += report.active_idea_count
        aggregate["member_count"] += report.member_count
        aggregate["active_member_count"] += report.active_member_count
        aggregate["speculative_active_count"] += report.speculative_active_count
        aggregate["evidence_seeking_active_count"] += report.evidence_seeking_active_count
        aggregate["child_parent_eligible_member_count"] += report.child_parent_eligible_member_count
        aggregate["epistemic_debt_count"] += report.epistemic_debt_count
        aggregate["acquisition_escalation_eligible_debt_count"] += (
            report.acquisition_escalation_eligible_debt_count
        )
        aggregate["evolution_handoff_count"] += report.evolution_handoff_count

        print()
        print("=" * 88)
        print(label)
        print("=" * 88)
        print("ideas active / all:", report.active_idea_count, "/", report.idea_count)
        print("members active / retained / all:", report.active_member_count, "/", report.retained_member_count, "/", report.member_count)
        print("partial / speculative / evidence-seeking active:", report.partially_grounded_active_count, "/", report.speculative_active_count, "/", report.evidence_seeking_active_count)
        print("child-parent eligible members:", report.child_parent_eligible_member_count)
        print("same-idea search eligible members:", report.same_idea_search_eligible_member_count)
        print("epistemic debt / acquisition-escalation eligible:", report.epistemic_debt_count, "/", report.acquisition_escalation_eligible_debt_count)
        print("idea evolution handoffs:", report.evolution_handoff_count)
        print("lanes:", report.lane_counts)
        print("EVIDENCE_RECOVERY_BLOCKING_INNER_LOOP=False")
        print("PARTIAL_REALIZATIONS_REMAIN_SEARCH_REPRODUCTIVE=True")
        print("SPECULATIVE_REALIZATIONS_REMAIN_SEARCH_REPRODUCTIVE=True")
        print("EVIDENCE_ACQUISITION_OPTIONAL_ESCALATION=True")
        print("OFFSPRING_GENERATION_EXECUTED=False")
        print("CANONICAL_GRAPH_MUTATED=False")

    payload = {
        "schema_version": "sis-v2-8b-parallel-partial-realization-search-cohort-v1",
        "cases": cohort,
        "aggregate": dict(aggregate),
        "evidence_recovery_is_blocking_inner_loop": False,
        "partial_realizations_remain_search_reproductive": True,
        "speculative_falsifiable_realizations_remain_search_reproductive": True,
        "evidence_acquisition_is_optional_escalation": True,
        "strict_hypothesis_card_contract_preserved": True,
        "new_retrieval_calls": False,
        "new_llm_calls": False,
        "offspring_generation_executed": False,
        "canonical_graph_mutated": False,
        "production_generation_authority": False,
        "production_selection_authority": False,
    }
    output = args.output.expanduser().resolve()
    _write(output, payload)
    print()
    print("SIS-v2.8b Parallel Partial-Realization Search complete")
    print("aggregate:", dict(aggregate))
    print("output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
