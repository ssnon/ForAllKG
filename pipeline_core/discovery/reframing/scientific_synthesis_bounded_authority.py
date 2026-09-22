from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from pipeline_core.discovery.reframing.scientific_synthesis_n10_authority import (
    ScientificSynthesisN10FilterReport,
    StrictLegacyN10Resolution,
    resolve_strict_legacy_n10_portfolio,
)
from pipeline_core.discovery.reframing.scientific_synthesis_specification_repair import (
    ScientificSynthesisRepairReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScientificSynthesisBoundedRoundReport(StrictModel):
    schema_version: Literal[
        "scientific-synthesis-bounded-n10-round-report-v1"
    ] = "scientific-synthesis-bounded-n10-round-report-v1"

    report_id: str
    source_initial_portfolio_id: str
    source_repaired_portfolio_id: str
    first_pass_survivor_portfolio_id: str
    repaired_pass_survivor_portfolio_id: str
    output_portfolio_id: str

    initial_candidate_count: int = Field(ge=0)
    first_pass_survivor_count: int = Field(ge=0)
    repair_eligible_count: int = Field(ge=0)
    repaired_candidate_count: int = Field(ge=0)
    repaired_pass_survivor_count: int = Field(ge=0)
    final_scientific_survivor_count: int = Field(ge=0)

    first_pass_positive_authority_required: Literal[True] = True
    repaired_pass_positive_authority_required: Literal[True] = True
    conditional_is_positive: Literal[False] = False
    repair_depth_limit: Literal[1] = 1
    further_repair_allowed: Literal[False] = False
    every_repaired_survivor_received_fresh_n10: Literal[True] = True
    legacy_production_selection_mutated: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class ScientificMergedBoundedN10Report(StrictModel):
    schema_version: Literal[
        "scientific-merged-bounded-n10-authority-report-v1"
    ] = "scientific-merged-bounded-n10-authority-report-v1"

    report_id: str
    source_context_id: str
    legacy_resolution: StrictLegacyN10Resolution
    source_scientific_bounded_report_id: str
    source_scientific_survivor_portfolio_id: str
    output_portfolio_id: str

    legacy_survivor_count: int = Field(ge=0)
    scientific_synthesis_survivor_count: int = Field(ge=0)
    merged_survivor_count: int = Field(ge=0)

    every_scientific_survivor_has_role_aware_positive_n10_authority: Literal[
        True
    ] = True
    conditional_is_positive: Literal[False] = False
    scientific_repair_depth_limit: Literal[1] = 1
    further_scientific_repair_allowed: Literal[False] = False
    legacy_production_selection_mutated: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def _validate_context_lineage(
    *,
    context: HypothesisContext,
    label: str,
    portfolio: HypothesisPortfolio,
) -> None:
    if portfolio.source_context_id != context.context_id:
        raise ValueError(f"{label} context ID mismatch")
    if portfolio.source_context_sha256 != context.context_sha256:
        raise ValueError(f"{label} context SHA mismatch")
    if portfolio.domain_profile_id != context.domain_profile_id:
        raise ValueError(f"{label} domain mismatch")
    if portfolio.source_report_id != context.source_report_id:
        raise ValueError(f"{label} source report mismatch")
    if portfolio.source_report_sha256 != context.source_report_sha256:
        raise ValueError(f"{label} source report SHA mismatch")


def merge_scientific_synthesis_n10_rounds(
    *,
    context: HypothesisContext,
    initial_portfolio: HypothesisPortfolio,
    first_pass_survivors: HypothesisPortfolio,
    first_pass_filter: ScientificSynthesisN10FilterReport,
    repair_report: ScientificSynthesisRepairReport,
    repaired_portfolio: HypothesisPortfolio,
    repaired_pass_survivors: HypothesisPortfolio,
    repaired_pass_filter: ScientificSynthesisN10FilterReport,
) -> tuple[HypothesisPortfolio, ScientificSynthesisBoundedRoundReport]:
    for label, portfolio in (
        ("initial scientific synthesis", initial_portfolio),
        ("first-pass scientific synthesis survivors", first_pass_survivors),
        ("repaired scientific synthesis", repaired_portfolio),
        ("repaired-pass scientific synthesis survivors", repaired_pass_survivors),
    ):
        _validate_context_lineage(
            context=context,
            label=label,
            portfolio=portfolio,
        )

    if first_pass_filter.source_portfolio_id != initial_portfolio.portfolio_id:
        raise ValueError("first-pass N10 filter/source portfolio mismatch")
    if first_pass_filter.output_portfolio_id != first_pass_survivors.portfolio_id:
        raise ValueError("first-pass N10 filter/survivor portfolio mismatch")
    if repair_report.source_portfolio_id != initial_portfolio.portfolio_id:
        raise ValueError("repair report/initial portfolio mismatch")
    if repair_report.output_portfolio_id != repaired_portfolio.portfolio_id:
        raise ValueError("repair report/repaired portfolio mismatch")
    if repaired_pass_filter.source_portfolio_id != repaired_portfolio.portfolio_id:
        raise ValueError("repaired-pass N10 filter/source portfolio mismatch")
    if repaired_pass_filter.output_portfolio_id != repaired_pass_survivors.portfolio_id:
        raise ValueError("repaired-pass N10 filter/survivor portfolio mismatch")

    initial_ids = {row.hypothesis_id for row in initial_portfolio.hypotheses}
    first_ids = {row.hypothesis_id for row in first_pass_survivors.hypotheses}
    repaired_ids = {row.hypothesis_id for row in repaired_portfolio.hypotheses}
    repaired_survivor_ids = {
        row.hypothesis_id for row in repaired_pass_survivors.hypotheses
    }

    if not first_ids <= initial_ids:
        raise ValueError("first-pass survivor absent from initial synthesis portfolio")
    if not repaired_survivor_ids <= repaired_ids:
        raise ValueError("repaired-pass survivor absent from repaired portfolio")

    source_by_repaired = {
        row.repaired_hypothesis_id: row.source_hypothesis_id
        for row in repair_report.entries
        if row.status == "repaired" and row.repaired_hypothesis_id is not None
    }
    if set(source_by_repaired) != repaired_ids:
        raise ValueError("repair report membership does not match repaired portfolio")
    repaired_sources = set(source_by_repaired.values())
    if not repaired_sources <= initial_ids:
        raise ValueError("repair source is absent from initial synthesis portfolio")
    if repaired_sources & first_ids:
        raise ValueError(
            "first-pass positive survivor must not receive bounded specification repair"
        )

    first_positive_ids = {
        row.hypothesis_id for row in first_pass_filter.decisions if row.kept
    }
    repaired_positive_ids = {
        row.hypothesis_id for row in repaired_pass_filter.decisions if row.kept
    }
    if first_positive_ids != first_ids:
        raise ValueError("first-pass survivor membership lacks positive N10 decision")
    if repaired_positive_ids != repaired_survivor_ids:
        raise ValueError("repaired survivor membership lacks fresh positive N10 decision")

    overlap = first_ids & repaired_survivor_ids
    if overlap:
        raise ValueError("first-pass/repaired-pass survivor ID collision")

    cards = [
        *first_pass_survivors.hypotheses,
        *repaired_pass_survivors.hypotheses,
    ]
    abstention = None
    if not cards:
        abstention = "no_scientific_synthesis_hypotheses_survived_bounded_strict_n10"

    output_id = _stable_id(
        "hypothesis_portfolio",
        "scientific_synthesis_bounded_n10_survivors",
        initial_portfolio.portfolio_id,
        repaired_portfolio.portfolio_id,
        *[row.hypothesis_id for row in cards],
        abstention or "",
    )
    output = HypothesisPortfolio(
        portfolio_id=output_id,
        domain_profile_id=context.domain_profile_id,
        source_context_id=context.context_id,
        source_context_sha256=context.context_sha256,
        source_report_id=context.source_report_id,
        source_report_sha256=context.source_report_sha256,
        hypotheses=cards,
        abstention_reason=abstention,
    )
    report = ScientificSynthesisBoundedRoundReport(
        report_id=_stable_id(
            "scientific_synthesis_bounded_n10_round",
            output.portfolio_id,
            initial_portfolio.portfolio_id,
            repaired_portfolio.portfolio_id,
        ),
        source_initial_portfolio_id=initial_portfolio.portfolio_id,
        source_repaired_portfolio_id=repaired_portfolio.portfolio_id,
        first_pass_survivor_portfolio_id=first_pass_survivors.portfolio_id,
        repaired_pass_survivor_portfolio_id=repaired_pass_survivors.portfolio_id,
        output_portfolio_id=output.portfolio_id,
        initial_candidate_count=len(initial_portfolio.hypotheses),
        first_pass_survivor_count=len(first_pass_survivors.hypotheses),
        repair_eligible_count=repair_report.eligible_repair_count,
        repaired_candidate_count=len(repaired_portfolio.hypotheses),
        repaired_pass_survivor_count=len(repaired_pass_survivors.hypotheses),
        final_scientific_survivor_count=len(cards),
    )
    return output, report


def merge_legacy_with_bounded_scientific_survivors(
    *,
    run_dir: str | Path,
    context: HypothesisContext,
    scientific_survivors: HypothesisPortfolio,
    scientific_report: ScientificSynthesisBoundedRoundReport,
) -> tuple[HypothesisPortfolio, ScientificMergedBoundedN10Report]:
    _validate_context_lineage(
        context=context,
        label="bounded scientific synthesis survivors",
        portfolio=scientific_survivors,
    )
    if scientific_report.output_portfolio_id != scientific_survivors.portfolio_id:
        raise ValueError("bounded scientific report/survivor portfolio mismatch")

    legacy_resolution = resolve_strict_legacy_n10_portfolio(run_dir=run_dir)
    legacy = HypothesisPortfolio.model_validate_json(
        Path(legacy_resolution.portfolio_path).read_text(encoding="utf-8")
    )
    _validate_context_lineage(
        context=context,
        label="legacy strict-N10 survivors",
        portfolio=legacy,
    )

    legacy_ids = {row.hypothesis_id for row in legacy.hypotheses}
    scientific_ids = {row.hypothesis_id for row in scientific_survivors.hypotheses}
    overlap = sorted(legacy_ids & scientific_ids)
    if overlap:
        raise ValueError(
            "legacy/scientific bounded survivor ID collision: "
            + ", ".join(overlap)
        )

    cards = [*legacy.hypotheses, *scientific_survivors.hypotheses]
    abstention = None
    if not cards:
        abstention = (
            "no_legacy_or_scientific_hypotheses_survived_bounded_strict_n10"
        )

    output_id = _stable_id(
        "hypothesis_portfolio",
        "legacy_plus_scientific_bounded_n10_survivors",
        legacy.portfolio_id,
        scientific_survivors.portfolio_id,
        *[row.hypothesis_id for row in cards],
        abstention or "",
    )
    output = HypothesisPortfolio(
        portfolio_id=output_id,
        domain_profile_id=context.domain_profile_id,
        source_context_id=context.context_id,
        source_context_sha256=context.context_sha256,
        source_report_id=context.source_report_id,
        source_report_sha256=context.source_report_sha256,
        hypotheses=cards,
        abstention_reason=abstention,
    )
    report = ScientificMergedBoundedN10Report(
        report_id=_stable_id(
            "scientific_merged_bounded_n10_authority",
            output.portfolio_id,
            scientific_report.report_id,
            legacy.portfolio_id,
        ),
        source_context_id=context.context_id,
        legacy_resolution=legacy_resolution,
        source_scientific_bounded_report_id=scientific_report.report_id,
        source_scientific_survivor_portfolio_id=scientific_survivors.portfolio_id,
        output_portfolio_id=output.portfolio_id,
        legacy_survivor_count=len(legacy.hypotheses),
        scientific_synthesis_survivor_count=len(scientific_survivors.hypotheses),
        merged_survivor_count=len(cards),
    )
    return output, report


__all__ = [
    "ScientificMergedBoundedN10Report",
    "ScientificSynthesisBoundedRoundReport",
    "merge_legacy_with_bounded_scientific_survivors",
    "merge_scientific_synthesis_n10_rounds",
]
