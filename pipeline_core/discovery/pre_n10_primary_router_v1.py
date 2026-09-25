from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import LiteratureQueryPlan
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.pre_n10_decomposition_primary_v1 import (
    execute_pre_n10_decomposition_primary_v1,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v1 import (
    PreN10ScientificContractReportV1,
    build_pre_n10_scientific_contract_v1,
)
from pipeline_core.discovery.preverifier_contract_gate_v2 import RouterHint
from pipeline_core.discovery.pre_n10_source_alignment_primary_v1 import (
    execute_pre_n10_source_alignment_primary_v1,
)
from pipeline_core.discovery.pre_n10_specification_repair_primary_v1 import (
    SpecificationRepairBackend,
    execute_pre_n10_specification_repair_primary_v1,
)
from pipeline_core.discovery.prospective_routed_primary_materializer_v2 import (
    SourceAlignmentAuditBackend,
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
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pretty_json_bytes(value: object) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_exact_or_validate(path: Path, payload: object) -> str:
    expected = _pretty_json_bytes(payload)
    path = path.expanduser().resolve()
    if path.exists():
        observed = path.read_bytes()
        if observed != expected:
            raise ValueError(
                "existing write-once pre-N10 primary-router artifact differs: "
                + str(path)
            )
        return hashlib.sha256(observed).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(expected)
    return hashlib.sha256(expected).hexdigest()


def _slug(value: str) -> str:
    slug = str(value).replace(":", "_").replace("/", "_")
    if not slug or slug in {".", ".."}:
        raise ValueError("invalid hypothesis ID")
    return slug


_HINT_PRECEDENCE: dict[RouterHint, int] = {
    "PROCEED_TO_LITERAL_ENDPOINT_BINDING": 0,
    "SPECIFICATION_REPAIR_REVIEW": 1,
    "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW": 2,
    "DECOMPOSE_OR_REGENERATE_REVIEW": 3,
}


def _dominant_hint(hints: list[RouterHint]) -> RouterHint:
    if not hints:
        return "PROCEED_TO_LITERAL_ENDPOINT_BINDING"
    unknown = [hint for hint in hints if hint not in _HINT_PRECEDENCE]
    if unknown:
        raise ValueError("unknown pre-N10 router hint: " + repr(sorted(set(unknown))))
    return max(hints, key=lambda hint: _HINT_PRECEDENCE[hint])


PrimaryRouteV1 = Literal[
    "PASSTHROUGH_READY",
    "SPECIFICATION_REPAIR",
    "SOURCE_ALIGNMENT",
    "ATOMIC_DECOMPOSITION",
    "MIXED_ROUTE_REGENERATION_REQUIRED",
]


class PreN10PrimaryRouterHypothesisResultV1(StrictModel):
    hypothesis_id: str
    source_contract_status: str
    observed_router_hints: list[RouterHint]
    dominant_router_hint: RouterHint
    route: PrimaryRouteV1
    homogeneous_intervention_route: bool

    child_primary_report_type: str | None = None
    child_primary_report_path: str | None = None
    child_primary_report_id: str | None = None
    child_primary_report_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    post_contract_status: str
    recovered_for_n10: bool
    regeneration_fallback_required: bool

    primary_intervention_performed: bool
    multiple_primary_interventions_performed: Literal[False] = False
    retrieval_performed: Literal[False] = False
    novelty_assessment_performed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    verifier_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_state(self) -> "PreN10PrimaryRouterHypothesisResultV1":
        ready = self.post_contract_status == "READY_FOR_N10"
        if self.recovered_for_n10 != ready:
            raise ValueError("recovered_for_n10 mismatch")
        if self.regeneration_fallback_required == ready:
            raise ValueError("regeneration fallback must complement readiness")

        child_fields = (
            self.child_primary_report_type,
            self.child_primary_report_path,
            self.child_primary_report_id,
            self.child_primary_report_sha256,
        )
        if self.primary_intervention_performed:
            if any(value is None for value in child_fields):
                raise ValueError("performed primary intervention requires child report")
            if not self.homogeneous_intervention_route:
                raise ValueError("primary intervention requires homogeneous route")
        else:
            if any(value is not None for value in child_fields):
                raise ValueError("non-intervention route cannot carry child report")
        if self.route == "MIXED_ROUTE_REGENERATION_REQUIRED":
            if self.homogeneous_intervention_route:
                raise ValueError("mixed route cannot be homogeneous")
            if not self.regeneration_fallback_required:
                raise ValueError("mixed route must require regeneration")
        return self


class PreN10PrimaryRouterReportV1(StrictModel):
    schema_version: Literal[
        "pre-n10-primary-router-report-v1"
    ] = "pre-n10-primary-router-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_contract_report_id: str
    source_contract_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_portfolio_id: str
    source_portfolio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_query_plan_id: str
    source_query_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    post_primary_query_plan_id: str
    post_primary_query_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    post_contract_report_id: str
    post_contract_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    hypotheses: list[PreN10PrimaryRouterHypothesisResultV1]
    hypothesis_count: int = Field(ge=0)
    route_counts: dict[str, int]
    primary_intervention_count: int = Field(ge=0)
    recovered_for_n10_count: int = Field(ge=0)
    regeneration_fallback_required_count: int = Field(ge=0)
    mixed_route_fallback_count: int = Field(ge=0)

    one_primary_intervention_max_per_hypothesis: Literal[True] = True
    mixed_routes_regenerate_without_chained_primary_repairs: Literal[True] = True
    route_precedence_frozen: Literal[True] = True

    retrieval_performed: Literal[False] = False
    novelty_assessment_performed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    regeneration_performed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    verifier_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "PreN10PrimaryRouterReportV1":
        if self.hypothesis_count != len(self.hypotheses):
            raise ValueError("hypothesis_count mismatch")
        counts = Counter(row.route for row in self.hypotheses)
        if dict(sorted(counts.items())) != dict(sorted(self.route_counts.items())):
            raise ValueError("route_counts mismatch")
        if self.primary_intervention_count != sum(
            row.primary_intervention_performed for row in self.hypotheses
        ):
            raise ValueError("primary_intervention_count mismatch")
        if self.recovered_for_n10_count != sum(
            row.recovered_for_n10 for row in self.hypotheses
        ):
            raise ValueError("recovered_for_n10_count mismatch")
        if self.regeneration_fallback_required_count != sum(
            row.regeneration_fallback_required for row in self.hypotheses
        ):
            raise ValueError("regeneration fallback count mismatch")
        if self.mixed_route_fallback_count != sum(
            row.route == "MIXED_ROUTE_REGENERATION_REQUIRED"
            for row in self.hypotheses
        ):
            raise ValueError("mixed-route fallback count mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("pre-N10 primary router SHA mismatch")
        if observed_id != "pre_n10_primary_router_v1:" + expected_sha[:20]:
            raise ValueError("pre-N10 primary router ID mismatch")
        return self


SpecificationRepairBackendFactory = Callable[
    [str, Path], SpecificationRepairBackend
]
SourceAlignmentAuditBackendFactory = Callable[
    [str, Path], SourceAlignmentAuditBackend
]


def _subset_portfolio(
    source: HypothesisPortfolio,
    *,
    hypothesis_id: str,
) -> HypothesisPortfolio:
    cards = [row for row in source.hypotheses if row.hypothesis_id == hypothesis_id]
    if len(cards) != 1:
        raise ValueError("subset hypothesis must resolve exactly one card")
    payload = source.model_dump(mode="json")
    payload["hypotheses"] = [cards[0].model_dump(mode="json")]
    payload["abstention_reason"] = None
    payload["portfolio_id"] = (
        "hypothesis_portfolio:pre_n10_primary_subset:"
        + _sha256_json(
            {
                "source_portfolio_id": source.portfolio_id,
                "hypothesis_id": hypothesis_id,
            }
        )[:20]
    )
    return HypothesisPortfolio.model_validate(payload)


def _subset_query_plan(
    source: LiteratureQueryPlan,
    *,
    hypothesis_id: str,
    subset_portfolio_id: str,
) -> LiteratureQueryPlan:
    groups = [row for row in source.claims if row.hypothesis_id == hypothesis_id]
    if len(groups) != 1:
        raise ValueError("subset hypothesis must resolve exactly one claim group")
    payload = source.model_dump(mode="json")
    payload["source_portfolio_id"] = subset_portfolio_id
    payload["claims"] = [groups[0].model_dump(mode="json")]
    payload["queries"] = [
        row.model_dump(mode="json")
        for row in source.queries
        if row.hypothesis_id == hypothesis_id
    ]
    payload.pop("plan_id")
    payload.pop("plan_sha256")
    digest = _sha256_json(payload)
    return LiteratureQueryPlan(
        **payload,
        plan_id="literature_query_plan:pre_n10_primary_subset:" + digest[:20],
        plan_sha256=digest,
    )


def _merge_hypothesis_plan(
    source: LiteratureQueryPlan,
    *,
    hypothesis_id: str,
    child: LiteratureQueryPlan,
) -> LiteratureQueryPlan:
    child_groups = [row for row in child.claims if row.hypothesis_id == hypothesis_id]
    if len(child_groups) != 1:
        raise ValueError("child plan must contain exactly one routed claim group")

    payload = source.model_dump(mode="json")
    groups = []
    replaced = 0
    for group in payload["claims"]:
        if group["hypothesis_id"] == hypothesis_id:
            groups.append(child_groups[0].model_dump(mode="json"))
            replaced += 1
        else:
            groups.append(group)
    if replaced != 1:
        raise ValueError("routed hypothesis claim group did not merge exactly once")
    payload["claims"] = groups

    payload["queries"] = [
        row
        for row in payload["queries"]
        if row["hypothesis_id"] != hypothesis_id
    ] + [
        row.model_dump(mode="json")
        for row in child.queries
        if row.hypothesis_id == hypothesis_id
    ]

    payload.pop("plan_id")
    payload.pop("plan_sha256")
    digest = _sha256_json(payload)
    return LiteratureQueryPlan(
        **payload,
        plan_id="literature_query_plan:pre_n10_primary_routed:" + digest[:20],
        plan_sha256=digest,
    )


def execute_pre_n10_primary_router_v1(
    *,
    portfolio_path: Path,
    query_plan_path: Path,
    contract_report: PreN10ScientificContractReportV1,
    output_root: Path,
    specification_repair_backend_factory: SpecificationRepairBackendFactory | None = None,
    source_alignment_audit_backend_factory: SourceAlignmentAuditBackendFactory | None = None,
) -> tuple[
    LiteratureQueryPlan,
    PreN10ScientificContractReportV1,
    PreN10PrimaryRouterReportV1,
]:
    portfolio_file = portfolio_path.expanduser().resolve()
    query_file = query_plan_path.expanduser().resolve()
    root = output_root.expanduser().resolve()

    if _sha256_file(portfolio_file) != contract_report.source_portfolio_sha256:
        raise ValueError("source portfolio changed after V_pre freeze")
    if _sha256_file(query_file) != contract_report.source_query_plan_sha256:
        raise ValueError("source query plan changed after V_pre freeze")

    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_file.read_text(encoding="utf-8")
    )
    source_plan = LiteratureQueryPlan.model_validate_json(
        query_file.read_text(encoding="utf-8")
    )
    if portfolio.portfolio_id != contract_report.source_portfolio_id:
        raise ValueError("portfolio/contract ID mismatch")
    if source_plan.plan_id != contract_report.source_query_plan_id:
        raise ValueError("query-plan/contract ID mismatch")

    contract_by_hypothesis = {
        row.hypothesis_id: row for row in contract_report.hypotheses
    }
    portfolio_ids = [row.hypothesis_id for row in portfolio.hypotheses]
    if set(contract_by_hypothesis) != set(portfolio_ids):
        raise ValueError("portfolio/contract hypothesis populations differ")

    working_plan = source_plan
    provisional: dict[str, dict[str, object]] = {}

    for hypothesis_id in portfolio_ids:
        contract_h = contract_by_hypothesis[hypothesis_id]
        not_ready_hints: list[RouterHint] = [
            claim.router_hint
            for claim in contract_h.claims
            if claim.contract_status == "NOT_READY_FOR_N10_CONTRACT"
        ]
        observed = list(dict.fromkeys(not_ready_hints))
        dominant = _dominant_hint(not_ready_hints)

        if contract_h.contract_status == "READY_FOR_N10":
            provisional[hypothesis_id] = {
                "source_contract_status": contract_h.contract_status,
                "observed_router_hints": [],
                "dominant_router_hint": dominant,
                "route": "PASSTHROUGH_READY",
                "homogeneous": True,
                "primary_performed": False,
                "child": None,
            }
            continue

        if not not_ready_hints:
            raise ValueError(
                "intervention-required hypothesis lacks not-ready claim: "
                + hypothesis_id
            )

        if len(set(not_ready_hints)) != 1:
            provisional[hypothesis_id] = {
                "source_contract_status": contract_h.contract_status,
                "observed_router_hints": observed,
                "dominant_router_hint": dominant,
                "route": "MIXED_ROUTE_REGENERATION_REQUIRED",
                "homogeneous": False,
                "primary_performed": False,
                "child": None,
            }
            continue

        route_dir = root / "primary" / _slug(hypothesis_id)
        subset_portfolio = _subset_portfolio(
            portfolio,
            hypothesis_id=hypothesis_id,
        )
        subset_plan = _subset_query_plan(
            working_plan,
            hypothesis_id=hypothesis_id,
            subset_portfolio_id=subset_portfolio.portfolio_id,
        )
        subset_portfolio_path = route_dir / "source.portfolio.json"
        subset_plan_path = route_dir / "source.claims_queries.json"
        _write_exact_or_validate(subset_portfolio_path, subset_portfolio)
        _write_exact_or_validate(subset_plan_path, subset_plan)
        subset_contract = build_pre_n10_scientific_contract_v1(
            portfolio_path=subset_portfolio_path,
            query_plan_path=subset_plan_path,
            claim_decomposition_request_count=0,
        )
        _write_exact_or_validate(route_dir / "contract.before_primary.json", subset_contract)

        if dominant == "SPECIFICATION_REPAIR_REVIEW":
            if specification_repair_backend_factory is None:
                raise ValueError("specification repair route lacks backend factory")
            child_plan, _child_contract, child_report = (
                execute_pre_n10_specification_repair_primary_v1(
                    portfolio_path=subset_portfolio_path,
                    query_plan_path=subset_plan_path,
                    contract_report=subset_contract,
                    repair_backend=specification_repair_backend_factory(
                        hypothesis_id,
                        route_dir,
                    ),
                    output_root=route_dir / "specification_repair",
                )
            )
            route: PrimaryRouteV1 = "SPECIFICATION_REPAIR"
            child_type = "pre-n10-specification-repair-primary-report-v1"
            child_path = route_dir / "specification_repair" / "primary.report.json"
        elif dominant == "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW":
            if source_alignment_audit_backend_factory is None:
                raise ValueError("source alignment route lacks audit backend factory")
            child_plan, _child_contract, child_report = (
                execute_pre_n10_source_alignment_primary_v1(
                    portfolio_path=subset_portfolio_path,
                    query_plan_path=subset_plan_path,
                    contract_report=subset_contract,
                    audit_backend=source_alignment_audit_backend_factory(
                        hypothesis_id,
                        route_dir,
                    ),
                    output_root=route_dir / "source_alignment",
                )
            )
            route = "SOURCE_ALIGNMENT"
            child_type = "pre-n10-source-alignment-primary-report-v1"
            child_path = route_dir / "source_alignment" / "primary.report.json"
        elif dominant == "DECOMPOSE_OR_REGENERATE_REVIEW":
            child_plan, _child_contract, child_report = (
                execute_pre_n10_decomposition_primary_v1(
                    portfolio_path=subset_portfolio_path,
                    query_plan_path=subset_plan_path,
                    contract_report=subset_contract,
                    output_root=route_dir / "decomposition",
                )
            )
            route = "ATOMIC_DECOMPOSITION"
            child_type = "pre-n10-decomposition-primary-report-v1"
            child_path = route_dir / "decomposition" / "primary.report.json"
        else:
            raise ValueError(
                "intervention-required hypothesis unexpectedly routed to proceed"
            )

        if not child_path.is_file():
            raise ValueError("child primary report artifact missing: " + str(child_path))
        working_plan = _merge_hypothesis_plan(
            working_plan,
            hypothesis_id=hypothesis_id,
            child=child_plan,
        )
        provisional[hypothesis_id] = {
            "source_contract_status": contract_h.contract_status,
            "observed_router_hints": observed,
            "dominant_router_hint": dominant,
            "route": route,
            "homogeneous": True,
            "primary_performed": True,
            "child": (
                child_type,
                str(child_path),
                child_report.report_id,
                child_report.report_sha256,
            ),
        }

    post_plan_path = root / "post_primary.claims_queries.json"
    _write_exact_or_validate(post_plan_path, working_plan)
    post_contract = build_pre_n10_scientific_contract_v1(
        portfolio_path=portfolio_file,
        query_plan_path=post_plan_path,
        claim_decomposition_request_count=0,
    )
    post_contract_path = root / "contract.after_primary_router.json"
    _write_exact_or_validate(post_contract_path, post_contract)
    post_by_hypothesis = {
        row.hypothesis_id: row for row in post_contract.hypotheses
    }

    rows: list[PreN10PrimaryRouterHypothesisResultV1] = []
    for hypothesis_id in portfolio_ids:
        state = provisional[hypothesis_id]
        post_h = post_by_hypothesis[hypothesis_id]
        route = str(state["route"])
        ready = post_h.contract_status == "READY_FOR_N10"
        if route == "MIXED_ROUTE_REGENERATION_REQUIRED" and ready:
            raise ValueError("unchanged mixed route unexpectedly became ready")
        child = state["child"]
        child_type = child_path = child_id = child_sha = None
        if child is not None:
            child_type, child_path, child_id, child_sha = child
        rows.append(
            PreN10PrimaryRouterHypothesisResultV1(
                hypothesis_id=hypothesis_id,
                source_contract_status=str(state["source_contract_status"]),
                observed_router_hints=list(state["observed_router_hints"]),
                dominant_router_hint=state["dominant_router_hint"],
                route=route,
                homogeneous_intervention_route=bool(state["homogeneous"]),
                child_primary_report_type=child_type,
                child_primary_report_path=child_path,
                child_primary_report_id=child_id,
                child_primary_report_sha256=child_sha,
                post_contract_status=post_h.contract_status,
                recovered_for_n10=ready,
                regeneration_fallback_required=not ready,
                primary_intervention_performed=bool(state["primary_performed"]),
            )
        )

    route_counts = Counter(row.route for row in rows)
    body = {
        "schema_version": "pre-n10-primary-router-report-v1",
        "source_contract_report_id": contract_report.report_id,
        "source_contract_report_sha256": contract_report.report_sha256,
        "source_portfolio_id": portfolio.portfolio_id,
        "source_portfolio_sha256": _sha256_file(portfolio_file),
        "source_query_plan_id": source_plan.plan_id,
        "source_query_plan_sha256": source_plan.plan_sha256,
        "post_primary_query_plan_id": working_plan.plan_id,
        "post_primary_query_plan_sha256": working_plan.plan_sha256,
        "post_contract_report_id": post_contract.report_id,
        "post_contract_report_sha256": post_contract.report_sha256,
        "hypotheses": [row.model_dump(mode="json") for row in rows],
        "hypothesis_count": len(rows),
        "route_counts": dict(sorted(route_counts.items())),
        "primary_intervention_count": sum(
            row.primary_intervention_performed for row in rows
        ),
        "recovered_for_n10_count": sum(row.recovered_for_n10 for row in rows),
        "regeneration_fallback_required_count": sum(
            row.regeneration_fallback_required for row in rows
        ),
        "mixed_route_fallback_count": sum(
            row.route == "MIXED_ROUTE_REGENERATION_REQUIRED" for row in rows
        ),
        "one_primary_intervention_max_per_hypothesis": True,
        "mixed_routes_regenerate_without_chained_primary_repairs": True,
        "route_precedence_frozen": True,
        "retrieval_performed": False,
        "novelty_assessment_performed": False,
        "n9_performed": False,
        "n10_performed": False,
        "regeneration_performed": False,
        "endpoint_binding_performed": False,
        "verifier_performed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    report = PreN10PrimaryRouterReportV1(
        **body,
        report_id="pre_n10_primary_router_v1:" + digest[:20],
        report_sha256=digest,
    )
    _write_exact_or_validate(root / "primary_router.report.json", report)
    return working_plan, post_contract, report


__all__ = [
    "PreN10PrimaryRouterHypothesisResultV1",
    "PreN10PrimaryRouterReportV1",
    "PrimaryRouteV1",
    "execute_pre_n10_primary_router_v1",
]
