from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
    LiteratureQueryPlan,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.pre_n10_regeneration_reentry_v2 import (
    PreN10RegenerationReentryReportV2,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v2 import (
    PreN10ScientificContractReportV2,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pretty_json_bytes(value: object) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def write_exact_or_validate(path: Path, value: object) -> None:
    expected = _pretty_json_bytes(value)
    resolved = path.expanduser().resolve()
    if resolved.exists():
        if resolved.read_bytes() != expected:
            raise ValueError(
                "existing write-once exploratory novelty artifact differs: "
                + str(resolved)
            )
        return
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("xb") as handle:
        handle.write(expected)


def _slug(value: str) -> str:
    result = str(value).replace(":", "_").replace("/", "_")
    if not result or result in {".", ".."}:
        raise ValueError("invalid exploratory lineage ID")
    return result


def select_ready_novelty_claim_ids(
    hypothesis_contract: object,
) -> list[str]:
    """Select only already-strict-ready novelty-bearing claims.

    This is intentionally NOT a relaxed scientific contract. Claims with
    incomplete source identity, unsupported atomic kind, or incomplete binding
    specification are never promoted by this function.

    The only relaxation is at the HYPOTHESIS level: supporting/testing claims
    may remain incomplete while already-ready NOVELTY_BEARING claims are
    allowed to receive bounded prior-art search.
    """

    selected: list[str] = []
    for claim in getattr(hypothesis_contract, "claims", []):
        if (
            getattr(claim, "contract_status", None)
            == "READY_FOR_N10_CONTRACT"
            and getattr(claim, "novelty_selection_role", None)
            == "NOVELTY_BEARING"
        ):
            selected.append(str(getattr(claim, "claim_id")))
    return selected


def build_exploratory_query_plan(
    *,
    source_plan: LiteratureQueryPlan,
    hypothesis_id: str,
    selected_claim_ids: list[str],
) -> LiteratureQueryPlan:
    selected = set(selected_claim_ids)
    if not selected:
        raise ValueError(
            "exploratory query plan requires at least one selected claim"
        )

    groups = [
        row
        for row in source_plan.claims
        if row.hypothesis_id == hypothesis_id
    ]
    if len(groups) != 1:
        raise ValueError(
            "exploratory source plan must resolve exactly one claim group"
        )

    group = groups[0]
    source_claim_ids = {row.claim_id for row in group.claims}
    unknown = sorted(selected - source_claim_ids)
    if unknown:
        raise ValueError(
            "exploratory selected claim IDs absent from source query plan: "
            + repr(unknown)
        )

    kept_claims = [
        row.model_dump(mode="json")
        for row in group.claims
        if row.claim_id in selected
    ]
    if len(kept_claims) != len(selected):
        raise ValueError(
            "exploratory selected claim cardinality mismatch"
        )

    kept_queries = [
        row.model_dump(mode="json")
        for row in source_plan.queries
        if row.hypothesis_id == hypothesis_id
        and row.claim_id in selected
    ]
    query_claim_ids = {
        str(row["claim_id"])
        for row in kept_queries
        if row.get("claim_id")
    }
    claims_without_queries = sorted(selected - query_claim_ids)
    if claims_without_queries:
        raise ValueError(
            "exploratory selected claim lacks frozen retrieval query: "
            + repr(claims_without_queries)
        )

    payload = source_plan.model_dump(mode="json")
    payload["claims"] = [
        {
            **group.model_dump(mode="json"),
            "claims": kept_claims,
        }
    ]
    payload["queries"] = kept_queries
    payload.pop("plan_id", None)
    payload.pop("plan_sha256", None)

    digest = _sha256_json(payload)
    return LiteratureQueryPlan(
        **payload,
        plan_id=(
            "literature_query_plan:pre_n10_exploratory:"
            + digest[:20]
        ),
        plan_sha256=digest,
    )


class PreN10ExploratoryNoveltyLineagePlanV1(StrictModel):
    source_hypothesis_id: str
    regenerated_hypothesis_id: str

    portfolio_path: str
    portfolio_id: str
    portfolio_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_query_plan_path: str
    source_query_plan_id: str
    source_query_plan_file_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    source_contract_v2_path: str
    source_contract_v2_id: str
    source_contract_v2_file_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    exploratory_query_plan_path: str
    exploratory_query_plan_id: str
    exploratory_query_plan_file_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    domain_profile_id: str
    selected_claim_ids: list[str] = Field(min_length=1)
    excluded_claim_ids: list[str] = Field(default_factory=list)

    source_hypothesis_contract_status: Literal[
        "REQUIRES_PRE_N10_INTERVENTION"
    ]
    source_semantic_disposition: Literal["PASS"]

    eligibility_basis: Literal[
        "STRICT_READY_NOVELTY_BEARING_CLAIM_SUBSET"
    ] = "STRICT_READY_NOVELTY_BEARING_CLAIM_SUBSET"

    strict_pre_n10_ready: Literal[False] = False
    exploratory_novelty_search_eligible: Literal[True] = True

    source_representation_mutated: Literal[False] = False
    incomplete_claim_promoted: Literal[False] = False
    certification_authority_created: Literal[False] = False
    n9_authority_created: Literal[False] = False
    n10_authority_created: Literal[False] = False


class PreN10ExploratoryNoveltyPlanV1(StrictModel):
    schema_version: Literal[
        "pre-n10-exploratory-novelty-plan-v1"
    ] = "pre-n10-exploratory-novelty-plan-v1"

    plan_id: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_reentry_report_id: str
    source_reentry_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_reentry_report_path: str
    source_reentry_report_file_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    lineages: list[PreN10ExploratoryNoveltyLineagePlanV1]
    lineage_count: int = Field(ge=0)
    selected_claim_count: int = Field(ge=0)

    hypothesis_level_recall_relaxation_only: Literal[True] = True
    claim_level_strict_readiness_preserved: Literal[True] = True
    only_ready_novelty_bearing_claims_searched: Literal[True] = True
    incomplete_claims_excluded_from_search: Literal[True] = True

    certification_contract_unchanged: Literal[True] = True
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    vpost_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_plan(self) -> "PreN10ExploratoryNoveltyPlanV1":
        if self.lineage_count != len(self.lineages):
            raise ValueError("exploratory lineage_count mismatch")
        if self.selected_claim_count != sum(
            len(row.selected_claim_ids)
            for row in self.lineages
        ):
            raise ValueError("exploratory selected_claim_count mismatch")
        body = self.model_dump(mode="json")
        observed_id = body.pop("plan_id")
        observed_sha = body.pop("plan_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("exploratory novelty plan SHA mismatch")
        if observed_id != (
            "pre_n10_exploratory_novelty_plan_v1:"
            + expected_sha[:20]
        ):
            raise ValueError("exploratory novelty plan ID mismatch")
        return self


class PreN10ExploratoryNoveltyLineageResultV1(StrictModel):
    source_hypothesis_id: str
    regenerated_hypothesis_id: str
    selected_claim_ids: list[str] = Field(min_length=1)

    external_report_path: str
    external_report_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    external_report_id: str
    external_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    external_status: str
    claim_statuses: dict[str, str]

    exploratory_only: Literal[True] = True
    novelty_certification_authority: Literal[False] = False
    candidate_survival_authority: Literal[False] = False
    incomplete_claim_promoted: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    vpost_performed: Literal[False] = False


class PreN10ExploratoryNoveltyReportV1(StrictModel):
    schema_version: Literal[
        "pre-n10-exploratory-novelty-report-v1"
    ] = "pre-n10-exploratory-novelty-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_plan_id: str
    source_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    lineages: list[PreN10ExploratoryNoveltyLineageResultV1]
    lineage_count: int = Field(ge=0)
    selected_claim_count: int = Field(ge=0)
    external_status_counts: dict[str, int]

    exploratory_only: Literal[True] = True
    search_bounded_prior_art_only: Literal[True] = True
    absence_is_not_novelty_proof: Literal[True] = True

    strict_certification_contract_unchanged: Literal[True] = True
    novelty_certification_authority: Literal[False] = False
    candidate_survival_authority: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    vpost_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "PreN10ExploratoryNoveltyReportV1":
        if self.lineage_count != len(self.lineages):
            raise ValueError("exploratory report lineage_count mismatch")
        if self.selected_claim_count != sum(
            len(row.selected_claim_ids)
            for row in self.lineages
        ):
            raise ValueError("exploratory report selected_claim_count mismatch")
        expected = Counter(row.external_status for row in self.lineages)
        if dict(sorted(expected.items())) != dict(
            sorted(self.external_status_counts.items())
        ):
            raise ValueError("exploratory external status counts mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("exploratory novelty report SHA mismatch")
        if observed_id != (
            "pre_n10_exploratory_novelty_report_v1:"
            + expected_sha[:20]
        ):
            raise ValueError("exploratory novelty report ID mismatch")
        return self


def build_pre_n10_exploratory_novelty_plan_v1(
    *,
    reentry_report_path: Path,
    output_root: Path,
) -> PreN10ExploratoryNoveltyPlanV1:
    reentry_file = reentry_report_path.expanduser().resolve()
    if not reentry_file.is_file():
        raise ValueError(
            "missing regeneration re-entry report: " + str(reentry_file)
        )

    reentry = PreN10RegenerationReentryReportV2.model_validate_json(
        reentry_file.read_text(encoding="utf-8")
    )
    root = output_root.expanduser().resolve()
    rows: list[PreN10ExploratoryNoveltyLineagePlanV1] = []

    for lineage in reentry.lineages:
        if (
            lineage.final_status != "PRE_N10_INTERVENTION_REQUIRED"
            or not lineage.semantic_admissible_for_pre_n10
            or lineage.semantic_disposition != "PASS"
        ):
            continue

        required = {
            "portfolio": lineage.regenerated_portfolio_path,
            "query_plan": lineage.query_plan_path,
            "contract_v2": lineage.contract_v2_report_path,
        }
        if any(not value for value in required.values()):
            continue

        portfolio_path = Path(str(required["portfolio"])).expanduser().resolve()
        query_plan_path = Path(str(required["query_plan"])).expanduser().resolve()
        contract_v2_path = Path(
            str(required["contract_v2"])
        ).expanduser().resolve()

        if not (
            portfolio_path.is_file()
            and query_plan_path.is_file()
            and contract_v2_path.is_file()
        ):
            raise ValueError(
                "exploratory eligible re-entry lineage references "
                "missing canonical artifacts: "
                + lineage.source_hypothesis_id
            )

        portfolio = HypothesisPortfolio.model_validate_json(
            portfolio_path.read_text(encoding="utf-8")
        )
        source_plan = LiteratureQueryPlan.model_validate_json(
            query_plan_path.read_text(encoding="utf-8")
        )
        contract = PreN10ScientificContractReportV2.model_validate_json(
            contract_v2_path.read_text(encoding="utf-8")
        )

        if len(portfolio.hypotheses) != 1:
            raise ValueError(
                "exploratory lane requires one regenerated hypothesis "
                "per re-entry lineage"
            )
        hypothesis = portfolio.hypotheses[0]

        contracts = [
            row
            for row in contract.hypotheses
            if row.hypothesis_id == hypothesis.hypothesis_id
        ]
        if len(contracts) != 1:
            raise ValueError(
                "exploratory V2 contract hypothesis must resolve exactly once"
            )
        h_contract = contracts[0]
        if h_contract.contract_status != "REQUIRES_PRE_N10_INTERVENTION":
            continue

        selected = select_ready_novelty_claim_ids(h_contract)
        if not selected:
            continue

        all_claim_ids = [row.claim_id for row in h_contract.claims]
        excluded = [
            claim_id
            for claim_id in all_claim_ids
            if claim_id not in set(selected)
        ]

        exploratory_plan = build_exploratory_query_plan(
            source_plan=source_plan,
            hypothesis_id=hypothesis.hypothesis_id,
            selected_claim_ids=selected,
        )

        lineage_root = (
            root
            / "lineage"
            / _slug(lineage.source_hypothesis_id)
        )
        exploratory_plan_path = (
            lineage_root / "exploratory.claims_queries.json"
        )
        write_exact_or_validate(
            exploratory_plan_path,
            exploratory_plan,
        )

        rows.append(
            PreN10ExploratoryNoveltyLineagePlanV1(
                source_hypothesis_id=lineage.source_hypothesis_id,
                regenerated_hypothesis_id=hypothesis.hypothesis_id,
                portfolio_path=str(portfolio_path),
                portfolio_id=portfolio.portfolio_id,
                portfolio_file_sha256=sha256_file(portfolio_path),
                source_query_plan_path=str(query_plan_path),
                source_query_plan_id=source_plan.plan_id,
                source_query_plan_file_sha256=sha256_file(query_plan_path),
                source_contract_v2_path=str(contract_v2_path),
                source_contract_v2_id=contract.report_id,
                source_contract_v2_file_sha256=sha256_file(contract_v2_path),
                exploratory_query_plan_path=str(
                    exploratory_plan_path.resolve()
                ),
                exploratory_query_plan_id=exploratory_plan.plan_id,
                exploratory_query_plan_file_sha256=sha256_file(
                    exploratory_plan_path
                ),
                domain_profile_id=portfolio.domain_profile_id,
                selected_claim_ids=selected,
                excluded_claim_ids=excluded,
                source_hypothesis_contract_status=h_contract.contract_status,
                source_semantic_disposition="PASS",
            )
        )

    body = {
        "schema_version": "pre-n10-exploratory-novelty-plan-v1",
        "source_reentry_report_id": reentry.report_id,
        "source_reentry_report_sha256": reentry.report_sha256,
        "source_reentry_report_path": str(reentry_file),
        "source_reentry_report_file_sha256": sha256_file(reentry_file),
        "lineages": [row.model_dump(mode="json") for row in rows],
        "lineage_count": len(rows),
        "selected_claim_count": sum(
            len(row.selected_claim_ids) for row in rows
        ),
        "hypothesis_level_recall_relaxation_only": True,
        "claim_level_strict_readiness_preserved": True,
        "only_ready_novelty_bearing_claims_searched": True,
        "incomplete_claims_excluded_from_search": True,
        "certification_contract_unchanged": True,
        "n9_performed": False,
        "n10_performed": False,
        "vpost_performed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return PreN10ExploratoryNoveltyPlanV1(
        **body,
        plan_id=(
            "pre_n10_exploratory_novelty_plan_v1:"
            + digest[:20]
        ),
        plan_sha256=digest,
    )


def build_pre_n10_exploratory_novelty_report_v1(
    *,
    plan: PreN10ExploratoryNoveltyPlanV1,
    lineage_reports: list[tuple[PreN10ExploratoryNoveltyLineagePlanV1, ExternalNoveltyReport, Path]],
) -> PreN10ExploratoryNoveltyReportV1:
    by_source = {
        row.source_hypothesis_id: row
        for row in plan.lineages
    }
    if len(by_source) != len(plan.lineages):
        raise ValueError("duplicate exploratory plan source lineage")

    results: list[PreN10ExploratoryNoveltyLineageResultV1] = []
    for lineage, report, report_path in lineage_reports:
        expected = by_source.get(lineage.source_hypothesis_id)
        if expected is None or expected != lineage:
            raise ValueError("exploratory report lineage not present in plan")
        if report.source_portfolio_id != lineage.portfolio_id:
            raise ValueError(
                "exploratory external report/source portfolio mismatch"
            )
        cards = [
            card
            for card in report.cards
            if card.hypothesis_id == lineage.regenerated_hypothesis_id
        ]
        if len(cards) != 1:
            raise ValueError(
                "exploratory external report must contain exactly one "
                "regenerated hypothesis card"
            )
        card = cards[0]
        claim_statuses = {
            row.claim_id: row.status
            for row in card.claim_reviews
            if row.claim_id in set(lineage.selected_claim_ids)
        }
        if set(claim_statuses) != set(lineage.selected_claim_ids):
            raise ValueError(
                "exploratory external report did not review exactly the "
                "selected ready novelty-bearing claims"
            )

        results.append(
            PreN10ExploratoryNoveltyLineageResultV1(
                source_hypothesis_id=lineage.source_hypothesis_id,
                regenerated_hypothesis_id=lineage.regenerated_hypothesis_id,
                selected_claim_ids=list(lineage.selected_claim_ids),
                external_report_path=str(report_path.resolve()),
                external_report_file_sha256=sha256_file(report_path),
                external_report_id=report.report_id,
                external_report_sha256=report.report_sha256,
                external_status=card.status,
                claim_statuses=claim_statuses,
            )
        )

    counts = Counter(row.external_status for row in results)
    body = {
        "schema_version": "pre-n10-exploratory-novelty-report-v1",
        "source_plan_id": plan.plan_id,
        "source_plan_sha256": plan.plan_sha256,
        "lineages": [row.model_dump(mode="json") for row in results],
        "lineage_count": len(results),
        "selected_claim_count": sum(
            len(row.selected_claim_ids) for row in results
        ),
        "external_status_counts": dict(sorted(counts.items())),
        "exploratory_only": True,
        "search_bounded_prior_art_only": True,
        "absence_is_not_novelty_proof": True,
        "strict_certification_contract_unchanged": True,
        "novelty_certification_authority": False,
        "candidate_survival_authority": False,
        "n9_performed": False,
        "n10_performed": False,
        "vpost_performed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return PreN10ExploratoryNoveltyReportV1(
        **body,
        report_id=(
            "pre_n10_exploratory_novelty_report_v1:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


__all__ = [
    "PreN10ExploratoryNoveltyLineagePlanV1",
    "PreN10ExploratoryNoveltyLineageResultV1",
    "PreN10ExploratoryNoveltyPlanV1",
    "PreN10ExploratoryNoveltyReportV1",
    "build_exploratory_query_plan",
    "build_pre_n10_exploratory_novelty_plan_v1",
    "build_pre_n10_exploratory_novelty_report_v1",
    "select_ready_novelty_claim_ids",
    "sha256_file",
    "write_exact_or_validate",
]
