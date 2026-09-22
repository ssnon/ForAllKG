from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from pipeline_core.discovery.nonobviousness_production_gate_v2 import (
    build_nonobviousness_production_gate_v2,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SYNTHESIS_AUTHORITY_SCOPE = "scientific_cross_lane_synthesis_candidate"
ROLE_AWARE_AUTHORITY_SOURCE = "n10_role_aware_nonobviousness_v2"
ROLE_AWARE_POSITIVE_REQUIREMENT = (
    "ELIGIBLE_AND_ROLE_AWARE_POSITIVE_NONOBVIOUSNESS"
)

SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODE_HARD_FILTER = "hard_filter"
SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODE_CERTIFICATION_ONLY = "certification_only"
SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODES = (
    SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODE_HARD_FILTER,
    SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODE_CERTIFICATION_ONLY,
)


class StrictLegacyN10Resolution(StrictModel):
    schema_version: Literal[
        "strict-legacy-n10-resolution-v1"
    ] = "strict-legacy-n10-resolution-v1"
    run_dir: str
    portfolio_path: str
    authority_kind: Literal[
        "bounded_n10",
        "hard_filter_n10",
        "certified_novelty_subset",
    ]
    hypothesis_count: int = Field(ge=0)
    positive_n10_authority_required: Literal[True] = True


class ScientificSynthesisN10Decision(StrictModel):
    hypothesis_id: str
    selection_class: Literal["ELIGIBLE", "CONDITIONAL", "INELIGIBLE"]
    positive_nonobviousness_authority: bool
    fallback_allowed: bool
    kept: bool
    action: str | None = None
    reason_codes: list[str] = Field(default_factory=list)



ScientificSynthesisNoveltyStatus = Literal[
    "NOVELTY_CERTIFIED",
    "NOVELTY_UNRESOLVED",
    "NOVELTY_REJECTED",
]


class ScientificSynthesisNoveltyCertificationDecision(StrictModel):
    hypothesis_id: str
    selection_class: Literal["ELIGIBLE", "CONDITIONAL", "INELIGIBLE"]
    positive_nonobviousness_authority: bool
    fallback_allowed: bool
    certification_status: ScientificSynthesisNoveltyStatus
    novelty_certified: bool
    candidate_retained: Literal[True] = True
    action: str | None = None
    unresolved_dimensions: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class ScientificSynthesisNoveltyCertificationReport(StrictModel):
    schema_version: Literal[
        "scientific-synthesis-novelty-certification-report-v1"
    ] = "scientific-synthesis-novelty-certification-report-v1"

    report_id: str
    source_portfolio_id: str
    candidate_portfolio_id: str
    certified_portfolio_id: str
    candidate_count: int = Field(ge=0)
    certified_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    decisions: list[ScientificSynthesisNoveltyCertificationDecision]

    authority_mode: Literal["certification_only"] = "certification_only"
    candidate_portfolio_preserved: Literal[True] = True
    candidate_retention_is_not_novelty_authority: Literal[True] = True
    conditional_is_positive: Literal[False] = False
    rejected_candidate_deleted: Literal[False] = False
    certified_requires_positive_n10_authority: Literal[True] = True
    semantic_or_feasibility_can_upgrade_novelty: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class ScientificCertificationMergedReport(StrictModel):
    schema_version: Literal[
        "scientific-certification-merged-authority-report-v1"
    ] = "scientific-certification-merged-authority-report-v1"

    report_id: str
    source_context_id: str
    legacy_resolution: StrictLegacyN10Resolution
    scientific_source_portfolio_id: str
    scientific_certification_report_id: str
    scientific_certified_portfolio_id: str
    merged_candidate_portfolio_id: str
    merged_certified_portfolio_id: str

    legacy_strict_authority_count: int = Field(ge=0)
    scientific_candidate_count: int = Field(ge=0)
    scientific_certified_count: int = Field(ge=0)
    scientific_unresolved_count: int = Field(ge=0)
    scientific_rejected_count: int = Field(ge=0)
    merged_candidate_count: int = Field(ge=0)
    merged_certified_count: int = Field(ge=0)

    scientific_candidates_preserved_independent_of_novelty_status: Literal[
        True
    ] = True
    certified_view_contains_only_positive_n10_authority: Literal[True] = True
    candidate_retention_is_not_novelty_authority: Literal[True] = True
    downstream_semantic_feasibility_cannot_upgrade_novelty: Literal[
        True
    ] = True
    legacy_production_selection_mutated: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class ScientificSynthesisN10FilterReport(StrictModel):
    schema_version: Literal[
        "scientific-synthesis-n10-filter-report-v1"
    ] = "scientific-synthesis-n10-filter-report-v1"
    report_id: str
    source_portfolio_id: str
    source_gate_scope: Literal[
        "scientific_cross_lane_synthesis_candidate"
    ] = "scientific_cross_lane_synthesis_candidate"
    output_portfolio_id: str
    candidate_count: int = Field(ge=0)
    survivor_count: int = Field(ge=0)
    conditional_rejected_count: int = Field(ge=0)
    ineligible_rejected_count: int = Field(ge=0)
    decisions: list[ScientificSynthesisN10Decision]

    role_aware_positive_authority_required: Literal[True] = True
    conditional_is_positive: Literal[False] = False
    absence_is_novelty: Literal[False] = False
    bounded_continuation_offered_to_synthesis: Literal[False] = False
    repair_opportunity_parity_with_legacy: Literal[False] = False
    first_pass_positive_bar_preserved: Literal[True] = True
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class ScientificMergedN10Report(StrictModel):
    schema_version: Literal[
        "scientific-merged-n10-authority-report-v1"
    ] = "scientific-merged-n10-authority-report-v1"
    report_id: str
    source_context_id: str
    legacy_resolution: StrictLegacyN10Resolution
    scientific_synthesis_source_portfolio_id: str
    scientific_synthesis_survivor_portfolio_id: str
    output_portfolio_id: str
    legacy_survivor_count: int = Field(ge=0)
    scientific_synthesis_survivor_count: int = Field(ge=0)
    merged_survivor_count: int = Field(ge=0)

    every_scientific_synthesis_survivor_has_positive_n10_authority: Literal[
        True
    ] = True
    legacy_and_scientific_survivors_share_role_aware_positive_bar: Literal[
        True
    ] = True
    synthesis_bounded_repair_not_performed: Literal[True] = True
    legacy_production_selection_mutated: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


def _stable_id(prefix: str, *parts: object) -> str:
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require_file(path: Path, *, label: str) -> Path:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path.resolve()


def promote_scientific_synthesis_n10_gate(
    *,
    candidate_gate: dict[str, Any],
) -> dict[str, Any]:
    promoted = build_nonobviousness_production_gate_v2(
        candidate_gate=candidate_gate,
    )

    if promoted.get("schema_version") != "scientific-novelty-fallback-gate-v2":
        raise ValueError("unexpected role-aware v2 production schema")
    if promoted.get("production_authority") is not True:
        raise ValueError("role-aware v2 gate lacks production authority")
    if promoted.get("authority_scope") != "alpha6_original_fallback":
        raise ValueError("unexpected source authority scope")
    if promoted.get("authority_source") != ROLE_AWARE_AUTHORITY_SOURCE:
        raise ValueError("unexpected role-aware authority source")
    if promoted.get("positive_authority_requires") != ROLE_AWARE_POSITIVE_REQUIREMENT:
        raise ValueError("unexpected role-aware positive authority contract")
    if promoted.get("conditional_is_positive") is not False:
        raise ValueError("CONDITIONAL must remain non-positive")
    if promoted.get("absence_is_novelty") is not False:
        raise ValueError("search-bounded absence must not become novelty")
    if promoted.get("candidate_semantics_preserved") is not True:
        raise ValueError("candidate semantics must remain preserved")

    result = deepcopy(promoted)
    result["authority_scope"] = SYNTHESIS_AUTHORITY_SCOPE
    result["candidate_origin"] = "scientific_cross_lane_synthesis"
    result["bounded_continuation_offered"] = False
    result["repair_opportunity_parity_with_legacy"] = False
    result["first_pass_positive_bar_preserved"] = True
    return result


def filter_scientific_synthesis_portfolio_by_n10(
    *,
    portfolio: HypothesisPortfolio,
    production_gate: dict[str, Any],
) -> tuple[HypothesisPortfolio, ScientificSynthesisN10FilterReport]:
    if production_gate.get("schema_version") != "scientific-novelty-fallback-gate-v2":
        raise ValueError("unexpected scientific synthesis N10 gate schema")
    if production_gate.get("production_authority") is not True:
        raise ValueError("scientific synthesis gate lacks production authority")
    if production_gate.get("authority_scope") != SYNTHESIS_AUTHORITY_SCOPE:
        raise ValueError("scientific synthesis gate has wrong authority scope")
    if production_gate.get("authority_source") != ROLE_AWARE_AUTHORITY_SOURCE:
        raise ValueError("scientific synthesis gate has wrong authority source")
    if production_gate.get("conditional_is_positive") is not False:
        raise ValueError("CONDITIONAL must remain non-positive")
    if production_gate.get("absence_is_novelty") is not False:
        raise ValueError("search-bounded absence must not become novelty")

    rows = production_gate.get("gates")
    if not isinstance(rows, list):
        raise ValueError("scientific synthesis gate rows must be a list")

    row_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("scientific synthesis gate row must be an object")
        hypothesis_id = str(row.get("hypothesis_id") or "").strip()
        if not hypothesis_id:
            raise ValueError("scientific synthesis gate row missing hypothesis_id")
        if hypothesis_id in row_by_id:
            raise ValueError("duplicate scientific synthesis gate hypothesis_id")
        row_by_id[hypothesis_id] = row

    expected_ids = {card.hypothesis_id for card in portfolio.hypotheses}
    if set(row_by_id) != expected_ids:
        raise ValueError(
            "scientific synthesis N10 gate membership does not match portfolio"
        )

    survivors = []
    decisions: list[ScientificSynthesisN10Decision] = []
    conditional = 0
    ineligible = 0

    for card in portfolio.hypotheses:
        row = row_by_id[card.hypothesis_id]
        selection_class = str(row.get("selection_class") or "")
        positive = row.get("positive_nonobviousness_authority")
        fallback = row.get("fallback_allowed")

        if selection_class not in {"ELIGIBLE", "CONDITIONAL", "INELIGIBLE"}:
            raise ValueError(
                f"unsupported scientific synthesis selection class: {selection_class}"
            )
        if not isinstance(positive, bool) or not isinstance(fallback, bool):
            raise ValueError("scientific synthesis authority booleans are malformed")

        expected_keep = (
            selection_class == "ELIGIBLE"
            and positive is True
            and fallback is True
        )
        if selection_class == "ELIGIBLE" and not expected_keep:
            raise ValueError(
                "ELIGIBLE scientific synthesis candidate lacks positive authority"
            )
        if selection_class != "ELIGIBLE" and positive:
            raise ValueError(
                "non-ELIGIBLE scientific synthesis candidate claims positive authority"
            )
        if selection_class == "CONDITIONAL":
            conditional += 1
        elif selection_class == "INELIGIBLE":
            ineligible += 1

        if expected_keep:
            survivors.append(card)

        decisions.append(
            ScientificSynthesisN10Decision(
                hypothesis_id=card.hypothesis_id,
                selection_class=selection_class,
                positive_nonobviousness_authority=positive,
                fallback_allowed=fallback,
                kept=expected_keep,
                action=row.get("action"),
                reason_codes=list(row.get("reason_codes") or []),
            )
        )

    abstention = None
    if not survivors:
        abstention = (
            "no_scientific_synthesis_hypotheses_received_positive_n10_authority"
        )

    output_id = _stable_id(
        "hypothesis_portfolio",
        "scientific_synthesis_n10_survivors",
        portfolio.portfolio_id,
        *[card.hypothesis_id for card in survivors],
        abstention or "",
    )
    output = HypothesisPortfolio(
        portfolio_id=output_id,
        domain_profile_id=portfolio.domain_profile_id,
        source_context_id=portfolio.source_context_id,
        source_context_sha256=portfolio.source_context_sha256,
        source_report_id=portfolio.source_report_id,
        source_report_sha256=portfolio.source_report_sha256,
        hypotheses=survivors,
        abstention_reason=abstention,
    )
    report = ScientificSynthesisN10FilterReport(
        report_id=_stable_id(
            "scientific_synthesis_n10_filter",
            portfolio.portfolio_id,
            output.portfolio_id,
        ),
        source_portfolio_id=portfolio.portfolio_id,
        output_portfolio_id=output.portfolio_id,
        candidate_count=len(portfolio.hypotheses),
        survivor_count=len(survivors),
        conditional_rejected_count=conditional,
        ineligible_rejected_count=ineligible,
        decisions=decisions,
    )
    return output, report



def _scientific_gate_rows(
    *,
    portfolio: HypothesisPortfolio,
    production_gate: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if production_gate.get("schema_version") != "scientific-novelty-fallback-gate-v2":
        raise ValueError("unexpected scientific synthesis N10 gate schema")
    if production_gate.get("production_authority") is not True:
        raise ValueError("scientific synthesis gate lacks production authority")
    if production_gate.get("authority_scope") != SYNTHESIS_AUTHORITY_SCOPE:
        raise ValueError("scientific synthesis gate has wrong authority scope")
    if production_gate.get("authority_source") != ROLE_AWARE_AUTHORITY_SOURCE:
        raise ValueError("scientific synthesis gate has wrong authority source")
    if production_gate.get("conditional_is_positive") is not False:
        raise ValueError("CONDITIONAL must remain non-positive")
    if production_gate.get("absence_is_novelty") is not False:
        raise ValueError("search-bounded absence must not become novelty")

    rows = production_gate.get("gates")
    if not isinstance(rows, list):
        raise ValueError("scientific synthesis gate rows must be a list")

    row_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("scientific synthesis gate row must be an object")
        hypothesis_id = str(row.get("hypothesis_id") or "").strip()
        if not hypothesis_id:
            raise ValueError("scientific synthesis gate row missing hypothesis_id")
        if hypothesis_id in row_by_id:
            raise ValueError("duplicate scientific synthesis gate hypothesis_id")
        row_by_id[hypothesis_id] = row

    expected_ids = {
        card.hypothesis_id
        for card in portfolio.hypotheses
    }
    if set(row_by_id) != expected_ids:
        raise ValueError(
            "scientific synthesis N10 gate membership does not match portfolio"
        )
    return row_by_id


def _unresolved_dimensions_from_gate(
    row: dict[str, Any],
) -> list[str]:
    action = str(row.get("action") or "")
    reasons = {
        str(value)
        for value in (row.get("reason_codes") or [])
    }
    dimensions: set[str] = set()

    if (
        "SPECIFICATION" in action
        or "atomic_specification_incomplete" in reasons
    ):
        dimensions.add("SPECIFICATION")

    if (
        action == "RESOLVE_NOVELTY_BEARING_EVIDENCE"
        or "candidate_not_ready_for_adjudication" in reasons
        or "role_aware_v2_conditional_fail_closed" in reasons
    ):
        dimensions.add("EVIDENCE_CLOSURE")

    if (
        action == "RESOLVE_NOVELTY_BEARING_PRIOR_ART_RELATION"
        or "partial_prior_art_requires_resolution" in reasons
    ):
        dimensions.add("PRIOR_ART_RELATION")

    return sorted(dimensions)


def split_scientific_synthesis_portfolio_by_n10_certification(
    *,
    portfolio: HypothesisPortfolio,
    production_gate: dict[str, Any],
) -> tuple[
    HypothesisPortfolio,
    HypothesisPortfolio,
    ScientificSynthesisNoveltyCertificationReport,
]:
    """Preserve all scientific candidates while separating N10 certification.

    Candidate membership and novelty authority are deliberately independent:

      ELIGIBLE + positive authority -> NOVELTY_CERTIFIED
      CONDITIONAL                  -> NOVELTY_UNRESOLVED
      INELIGIBLE                   -> NOVELTY_REJECTED

    None of the three states removes a card from the scientific candidate
    portfolio. Only NOVELTY_CERTIFIED cards enter the certified subset.
    """

    row_by_id = _scientific_gate_rows(
        portfolio=portfolio,
        production_gate=production_gate,
    )

    certified_cards = []
    decisions: list[ScientificSynthesisNoveltyCertificationDecision] = []
    certified_count = 0
    unresolved_count = 0
    rejected_count = 0

    for card in portfolio.hypotheses:
        row = row_by_id[card.hypothesis_id]
        selection_class = str(row.get("selection_class") or "")
        positive = row.get("positive_nonobviousness_authority")
        fallback = row.get("fallback_allowed")

        if selection_class not in {"ELIGIBLE", "CONDITIONAL", "INELIGIBLE"}:
            raise ValueError(
                "unsupported scientific synthesis selection class: "
                + selection_class
            )
        if not isinstance(positive, bool) or not isinstance(fallback, bool):
            raise ValueError(
                "scientific synthesis authority booleans are malformed"
            )

        if selection_class == "ELIGIBLE":
            if not positive or not fallback:
                raise ValueError(
                    "ELIGIBLE scientific synthesis candidate lacks "
                    "positive N10 authority"
                )
            status: ScientificSynthesisNoveltyStatus = "NOVELTY_CERTIFIED"
            novelty_certified = True
            certified_cards.append(card)
            certified_count += 1

        elif selection_class == "CONDITIONAL":
            if positive:
                raise ValueError(
                    "CONDITIONAL scientific synthesis candidate unexpectedly "
                    "claims positive authority"
                )
            status = "NOVELTY_UNRESOLVED"
            novelty_certified = False
            unresolved_count += 1

        else:
            if positive:
                raise ValueError(
                    "INELIGIBLE scientific synthesis candidate unexpectedly "
                    "claims positive authority"
                )
            status = "NOVELTY_REJECTED"
            novelty_certified = False
            rejected_count += 1

        decisions.append(
            ScientificSynthesisNoveltyCertificationDecision(
                hypothesis_id=card.hypothesis_id,
                selection_class=selection_class,
                positive_nonobviousness_authority=positive,
                fallback_allowed=fallback,
                certification_status=status,
                novelty_certified=novelty_certified,
                action=row.get("action"),
                unresolved_dimensions=(
                    _unresolved_dimensions_from_gate(row)
                    if status == "NOVELTY_UNRESOLVED"
                    else []
                ),
                reason_codes=list(row.get("reason_codes") or []),
            )
        )

    certified_abstention = None
    if not certified_cards:
        certified_abstention = (
            "no_scientific_synthesis_hypotheses_received_positive_n10_authority"
        )

    certified_id = _stable_id(
        "hypothesis_portfolio",
        "scientific_synthesis_novelty_certified",
        portfolio.portfolio_id,
        *[card.hypothesis_id for card in certified_cards],
        certified_abstention or "",
    )
    certified_portfolio = HypothesisPortfolio(
        portfolio_id=certified_id,
        domain_profile_id=portfolio.domain_profile_id,
        source_context_id=portfolio.source_context_id,
        source_context_sha256=portfolio.source_context_sha256,
        source_report_id=portfolio.source_report_id,
        source_report_sha256=portfolio.source_report_sha256,
        hypotheses=certified_cards,
        abstention_reason=certified_abstention,
    )

    # Candidate view is intentionally the exact scientific synthesis portfolio.
    candidate_portfolio = portfolio.model_copy(deep=True)

    report = ScientificSynthesisNoveltyCertificationReport(
        report_id=_stable_id(
            "scientific_synthesis_novelty_certification",
            portfolio.portfolio_id,
            certified_portfolio.portfolio_id,
            *[
                (
                    decision.hypothesis_id,
                    decision.certification_status,
                )
                for decision in decisions
            ],
        ),
        source_portfolio_id=portfolio.portfolio_id,
        candidate_portfolio_id=candidate_portfolio.portfolio_id,
        certified_portfolio_id=certified_portfolio.portfolio_id,
        candidate_count=len(portfolio.hypotheses),
        certified_count=certified_count,
        unresolved_count=unresolved_count,
        rejected_count=rejected_count,
        decisions=decisions,
    )
    return candidate_portfolio, certified_portfolio, report


def merge_legacy_strict_with_scientific_certification(
    *,
    context: HypothesisContext,
    legacy_resolution: StrictLegacyN10Resolution,
    scientific_candidates: HypothesisPortfolio,
    scientific_certified: HypothesisPortfolio,
    certification_report: ScientificSynthesisNoveltyCertificationReport,
) -> tuple[
    HypothesisPortfolio,
    HypothesisPortfolio,
    ScientificCertificationMergedReport,
]:
    """Build discovery-candidate and novelty-certified downstream views.

    The candidate view contains strict legacy survivors plus every scientific
    synthesis candidate. The certified view contains strict legacy survivors
    plus only scientific candidates with positive N10 novelty authority.
    """

    legacy = HypothesisPortfolio.model_validate_json(
        Path(legacy_resolution.portfolio_path).read_text(encoding="utf-8")
    )

    for label, portfolio in (
        ("legacy", legacy),
        ("scientific candidates", scientific_candidates),
        ("scientific certified", scientific_certified),
    ):
        if portfolio.source_context_id != context.context_id:
            raise ValueError(f"{label} portfolio/context ID mismatch")
        if portfolio.source_context_sha256 != context.context_sha256:
            raise ValueError(f"{label} portfolio/context SHA mismatch")
        if portfolio.domain_profile_id != context.domain_profile_id:
            raise ValueError(f"{label} portfolio/context domain mismatch")
        if portfolio.source_report_id != context.source_report_id:
            raise ValueError(f"{label} portfolio/context source report mismatch")
        if portfolio.source_report_sha256 != context.source_report_sha256:
            raise ValueError(
                f"{label} portfolio/context source report SHA mismatch"
            )

    if certification_report.source_portfolio_id != scientific_candidates.portfolio_id:
        raise ValueError("scientific certification source portfolio mismatch")
    if certification_report.candidate_portfolio_id != scientific_candidates.portfolio_id:
        raise ValueError("scientific certification candidate portfolio mismatch")
    if certification_report.certified_portfolio_id != scientific_certified.portfolio_id:
        raise ValueError("scientific certification subset portfolio mismatch")

    candidate_ids = {
        card.hypothesis_id
        for card in scientific_candidates.hypotheses
    }
    certified_ids = {
        card.hypothesis_id
        for card in scientific_certified.hypotheses
    }
    if not certified_ids <= candidate_ids:
        raise ValueError(
            "scientific certified subset contains card absent from candidates"
        )

    legacy_ids = {
        card.hypothesis_id
        for card in legacy.hypotheses
    }
    overlap = sorted(legacy_ids & candidate_ids)
    if overlap:
        raise ValueError(
            "legacy/scientific candidate hypothesis ID collision: "
            + ", ".join(overlap)
        )

    merged_candidate_cards = [
        *legacy.hypotheses,
        *scientific_candidates.hypotheses,
    ]
    merged_certified_cards = [
        *legacy.hypotheses,
        *scientific_certified.hypotheses,
    ]

    candidate_abstention = None
    if not merged_candidate_cards:
        candidate_abstention = "no_legacy_or_scientific_discovery_candidates"

    certified_abstention = None
    if not merged_certified_cards:
        certified_abstention = (
            "no_legacy_or_scientific_hypotheses_have_positive_n10_authority"
        )

    merged_candidate_id = _stable_id(
        "hypothesis_portfolio",
        "legacy_plus_scientific_discovery_candidates",
        legacy.portfolio_id,
        scientific_candidates.portfolio_id,
        *[card.hypothesis_id for card in merged_candidate_cards],
        candidate_abstention or "",
    )
    merged_candidate_portfolio = HypothesisPortfolio(
        portfolio_id=merged_candidate_id,
        domain_profile_id=context.domain_profile_id,
        source_context_id=context.context_id,
        source_context_sha256=context.context_sha256,
        source_report_id=context.source_report_id,
        source_report_sha256=context.source_report_sha256,
        hypotheses=merged_candidate_cards,
        abstention_reason=candidate_abstention,
    )

    merged_certified_id = _stable_id(
        "hypothesis_portfolio",
        "legacy_plus_scientific_novelty_certified",
        legacy.portfolio_id,
        scientific_certified.portfolio_id,
        *[card.hypothesis_id for card in merged_certified_cards],
        certified_abstention or "",
    )
    merged_certified_portfolio = HypothesisPortfolio(
        portfolio_id=merged_certified_id,
        domain_profile_id=context.domain_profile_id,
        source_context_id=context.context_id,
        source_context_sha256=context.context_sha256,
        source_report_id=context.source_report_id,
        source_report_sha256=context.source_report_sha256,
        hypotheses=merged_certified_cards,
        abstention_reason=certified_abstention,
    )

    report = ScientificCertificationMergedReport(
        report_id=_stable_id(
            "scientific_certification_merged_authority",
            merged_candidate_portfolio.portfolio_id,
            merged_certified_portfolio.portfolio_id,
            certification_report.report_id,
        ),
        source_context_id=context.context_id,
        legacy_resolution=legacy_resolution,
        scientific_source_portfolio_id=scientific_candidates.portfolio_id,
        scientific_certification_report_id=certification_report.report_id,
        scientific_certified_portfolio_id=scientific_certified.portfolio_id,
        merged_candidate_portfolio_id=merged_candidate_portfolio.portfolio_id,
        merged_certified_portfolio_id=merged_certified_portfolio.portfolio_id,
        legacy_strict_authority_count=len(legacy.hypotheses),
        scientific_candidate_count=len(scientific_candidates.hypotheses),
        scientific_certified_count=certification_report.certified_count,
        scientific_unresolved_count=certification_report.unresolved_count,
        scientific_rejected_count=certification_report.rejected_count,
        merged_candidate_count=len(merged_candidate_cards),
        merged_certified_count=len(merged_certified_cards),
    )
    return (
        merged_candidate_portfolio,
        merged_certified_portfolio,
        report,
    )


def resolve_strict_legacy_n10_portfolio(
    *,
    run_dir: str | Path,
) -> StrictLegacyN10Resolution:
    run = Path(run_dir).expanduser().resolve()
    manifest_path = _require_file(
        run / "e2e_runner.manifest.json",
        label="legacy E2E manifest",
    )
    manifest = _load_object(manifest_path)

    bounded = manifest.get("n10_bounded_continuation")
    if isinstance(bounded, dict) and bounded.get("enabled") is True:
        path_text = str(bounded.get("final_portfolio") or "").strip()
        if not path_text:
            raise ValueError("bounded N10 manifest lacks final_portfolio")
        path = _require_file(Path(path_text), label="bounded N10 final portfolio")
        portfolio = HypothesisPortfolio.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        return StrictLegacyN10Resolution(
            run_dir=str(run),
            portfolio_path=str(path),
            authority_kind="bounded_n10",
            hypothesis_count=len(portfolio.hypotheses),
        )

    post = manifest.get("post_generation_n10_authority")
    if not isinstance(post, dict):
        raise ValueError(
            "legacy run has no strict post-generation N10 authority envelope"
        )
    mode = str(post.get("mode") or "")
    if mode == "hard_filter":
        path_text = str(post.get("downstream_portfolio") or "").strip()
        if not path_text:
            raise ValueError("hard-filter N10 manifest lacks downstream_portfolio")
        kind = "hard_filter_n10"
    elif mode == "certification_only":
        path_text = str(post.get("certified_novelty_portfolio") or "").strip()
        if not path_text:
            raise ValueError(
                "certification-only N10 manifest lacks certified_novelty_portfolio"
            )
        kind = "certified_novelty_subset"
    else:
        raise ValueError(f"unsupported strict legacy N10 authority mode: {mode!r}")

    path = _require_file(Path(path_text), label="strict legacy N10 portfolio")
    portfolio = HypothesisPortfolio.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    return StrictLegacyN10Resolution(
        run_dir=str(run),
        portfolio_path=str(path),
        authority_kind=kind,
        hypothesis_count=len(portfolio.hypotheses),
    )


def merge_legacy_and_scientific_n10_survivors(
    *,
    context: HypothesisContext,
    legacy_resolution: StrictLegacyN10Resolution,
    scientific_source_portfolio: HypothesisPortfolio,
    scientific_survivors: HypothesisPortfolio,
) -> tuple[HypothesisPortfolio, ScientificMergedN10Report]:
    legacy = HypothesisPortfolio.model_validate_json(
        Path(legacy_resolution.portfolio_path).read_text(encoding="utf-8")
    )

    for label, portfolio in (
        ("legacy", legacy),
        ("scientific source", scientific_source_portfolio),
        ("scientific survivors", scientific_survivors),
    ):
        if portfolio.source_context_id != context.context_id:
            raise ValueError(f"{label} portfolio/context ID mismatch")
        if portfolio.source_context_sha256 != context.context_sha256:
            raise ValueError(f"{label} portfolio/context SHA mismatch")
        if portfolio.domain_profile_id != context.domain_profile_id:
            raise ValueError(f"{label} portfolio/context domain mismatch")
        if portfolio.source_report_id != context.source_report_id:
            raise ValueError(f"{label} portfolio/context source report mismatch")
        if portfolio.source_report_sha256 != context.source_report_sha256:
            raise ValueError(f"{label} portfolio/context source report SHA mismatch")

    scientific_source_ids = {
        card.hypothesis_id for card in scientific_source_portfolio.hypotheses
    }
    scientific_survivor_ids = {
        card.hypothesis_id for card in scientific_survivors.hypotheses
    }
    if not scientific_survivor_ids <= scientific_source_ids:
        raise ValueError("scientific survivor is absent from source synthesis portfolio")

    legacy_ids = {card.hypothesis_id for card in legacy.hypotheses}
    overlap = sorted(legacy_ids & scientific_survivor_ids)
    if overlap:
        raise ValueError(
            "legacy/scientific N10 survivor hypothesis ID collision: "
            + ", ".join(overlap)
        )

    merged_cards = [
        *legacy.hypotheses,
        *scientific_survivors.hypotheses,
    ]
    abstention = None
    if not merged_cards:
        abstention = "no_legacy_or_scientific_hypotheses_survived_strict_n10"

    output_id = _stable_id(
        "hypothesis_portfolio",
        "legacy_plus_scientific_n10_survivors",
        legacy.portfolio_id,
        scientific_survivors.portfolio_id,
        *[card.hypothesis_id for card in merged_cards],
        abstention or "",
    )
    merged = HypothesisPortfolio(
        portfolio_id=output_id,
        domain_profile_id=context.domain_profile_id,
        source_context_id=context.context_id,
        source_context_sha256=context.context_sha256,
        source_report_id=context.source_report_id,
        source_report_sha256=context.source_report_sha256,
        hypotheses=merged_cards,
        abstention_reason=abstention,
    )
    report = ScientificMergedN10Report(
        report_id=_stable_id(
            "scientific_merged_n10_authority",
            output_id,
            legacy.portfolio_id,
            scientific_survivors.portfolio_id,
        ),
        source_context_id=context.context_id,
        legacy_resolution=legacy_resolution,
        scientific_synthesis_source_portfolio_id=scientific_source_portfolio.portfolio_id,
        scientific_synthesis_survivor_portfolio_id=scientific_survivors.portfolio_id,
        output_portfolio_id=merged.portfolio_id,
        legacy_survivor_count=len(legacy.hypotheses),
        scientific_synthesis_survivor_count=len(scientific_survivors.hypotheses),
        merged_survivor_count=len(merged_cards),
    )
    return merged, report


__all__ = [
    "ROLE_AWARE_AUTHORITY_SOURCE",
    "ROLE_AWARE_POSITIVE_REQUIREMENT",
    "SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODE_CERTIFICATION_ONLY",
    "SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODE_HARD_FILTER",
    "SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODES",
    "SYNTHESIS_AUTHORITY_SCOPE",
    "ScientificCertificationMergedReport",
    "ScientificMergedN10Report",
    "ScientificSynthesisN10FilterReport",
    "ScientificSynthesisNoveltyCertificationDecision",
    "ScientificSynthesisNoveltyCertificationReport",
    "StrictLegacyN10Resolution",
    "filter_scientific_synthesis_portfolio_by_n10",
    "merge_legacy_and_scientific_n10_survivors",
    "merge_legacy_strict_with_scientific_certification",
    "promote_scientific_synthesis_n10_gate",
    "resolve_strict_legacy_n10_portfolio",
    "split_scientific_synthesis_portfolio_by_n10_certification",
]
