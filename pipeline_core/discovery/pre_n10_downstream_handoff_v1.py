from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import LiteratureQueryPlan
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.hypothesis_semantic_disposition import (
    HypothesisSemanticDispositionV1,
)
from pipeline_core.discovery.pre_n10_initial_semantic_gate_v1 import (
    PreN10InitialSemanticGateReportV1,
)
from pipeline_core.discovery.pre_n10_primary_router_v1 import (
    PreN10PrimaryRouterReportV1,
)
from pipeline_core.discovery.pre_n10_regeneration_reentry_v2 import (
    PreN10RegenerationReentryReportV2,
)
from pipeline_core.discovery.pre_n10_regeneration_v1 import (
    PreN10RegenerationExecutionReportV1,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v1 import (
    PreN10ScientificContractReportV1,
    build_pre_n10_scientific_contract_v1,
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
                "existing write-once pre-N10 downstream handoff artifact differs: "
                + str(path)
            )
        return hashlib.sha256(observed).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(expected)
    return hashlib.sha256(expected).hexdigest()


def _slug(value: str) -> str:
    result = str(value).replace(":", "_").replace("/", "_")
    if not result or result in {".", ".."}:
        raise ValueError("invalid hypothesis ID")
    return result


DownstreamOriginV1 = Literal[
    "INITIAL_PRIMARY_READY",
    "REGENERATED_REENTRY_READY",
]


class PreN10DownstreamHandoffLineageV1(StrictModel):
    lineage_id: str
    origin: DownstreamOriginV1
    source_hypothesis_id: str
    downstream_hypothesis_id: str

    portfolio_path: str
    portfolio_id: str
    portfolio_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_plan_path: str
    query_plan_id: str
    query_plan_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_report_path: str
    contract_report_id: str
    contract_report_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    semantic_disposition_path: str
    semantic_disposition_id: str
    semantic_disposition_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_disposition: Literal["PASS"] = "PASS"

    pre_n10_status: Literal["READY_FOR_N10"] = "READY_FOR_N10"
    eligible_for_external_novelty: Literal[True] = True

    scientific_text_mutated_by_handoff: Literal[False] = False
    new_scientific_content_added_by_handoff: Literal[False] = False
    external_novelty_performed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    verifier_performed: Literal[False] = False


class PreN10BlockedRegenerationLineageV1(StrictModel):
    source_hypothesis_id: str
    final_status: str
    semantic_status: str
    pre_n10_disposition: str | None = None
    ready_for_n10: Literal[False] = False
    eligible_for_external_novelty: Literal[False] = False
    second_regeneration_allowed: Literal[False] = False


class PreN10DownstreamHandoffReportV1(StrictModel):
    schema_version: Literal[
        "pre-n10-downstream-handoff-report-v1"
    ] = "pre-n10-downstream-handoff-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_initial_semantic_gate_report_id: str
    source_initial_semantic_gate_report_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    source_primary_router_report_id: str
    source_primary_router_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_regeneration_report_id: str | None = None
    source_regeneration_report_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    source_regeneration_reentry_report_id: str | None = None
    source_regeneration_reentry_report_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

    lineages: list[PreN10DownstreamHandoffLineageV1]
    blocked_regeneration_lineages: list[PreN10BlockedRegenerationLineageV1]

    source_hypothesis_count: int = Field(ge=0)
    initial_ready_lineage_count: int = Field(ge=0)
    regeneration_fallback_source_count: int = Field(ge=0)
    regenerated_ready_lineage_count: int = Field(ge=0)
    blocked_after_regeneration_count: int = Field(ge=0)
    eligible_for_external_novelty_count: int = Field(ge=0)

    initial_semantic_disposition_required: Literal[True] = True
    regenerated_semantic_disposition_required: Literal[True] = True
    pre_n10_ready_required_for_external_novelty: Literal[True] = True
    regeneration_is_one_shot: Literal[True] = True
    second_regeneration_performed: Literal[False] = False

    external_novelty_performed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    verifier_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "PreN10DownstreamHandoffReportV1":
        if self.initial_ready_lineage_count != sum(
            row.origin == "INITIAL_PRIMARY_READY" for row in self.lineages
        ):
            raise ValueError("initial-ready lineage count mismatch")
        if self.regenerated_ready_lineage_count != sum(
            row.origin == "REGENERATED_REENTRY_READY" for row in self.lineages
        ):
            raise ValueError("regenerated-ready lineage count mismatch")
        if self.blocked_after_regeneration_count != len(
            self.blocked_regeneration_lineages
        ):
            raise ValueError("blocked regeneration count mismatch")
        if self.eligible_for_external_novelty_count != len(self.lineages):
            raise ValueError("external-novelty eligibility count mismatch")
        if self.regeneration_fallback_source_count != (
            self.regenerated_ready_lineage_count
            + self.blocked_after_regeneration_count
        ):
            raise ValueError("regeneration fallback accounting mismatch")
        if self.source_hypothesis_count != (
            self.initial_ready_lineage_count
            + self.regeneration_fallback_source_count
        ):
            raise ValueError("source hypothesis accounting mismatch")

        regen_fields = (
            self.source_regeneration_report_id,
            self.source_regeneration_report_sha256,
            self.source_regeneration_reentry_report_id,
            self.source_regeneration_reentry_report_sha256,
        )
        if self.regeneration_fallback_source_count:
            if any(value is None for value in regen_fields):
                raise ValueError("fallback handoff requires regeneration lineage reports")
        elif any(value is not None for value in regen_fields):
            raise ValueError("no-fallback handoff cannot carry regeneration reports")

        source_ids = [row.source_hypothesis_id for row in self.lineages]
        source_ids.extend(
            row.source_hypothesis_id for row in self.blocked_regeneration_lineages
        )
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source hypothesis appears in multiple handoff dispositions")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("pre-N10 downstream handoff SHA mismatch")
        if observed_id != "pre_n10_downstream_handoff_v1:" + expected_sha[:20]:
            raise ValueError("pre-N10 downstream handoff ID mismatch")
        return self


def _subset_portfolio(
    source: HypothesisPortfolio,
    *,
    hypothesis_id: str,
    lineage_key: str,
) -> HypothesisPortfolio:
    cards = [row for row in source.hypotheses if row.hypothesis_id == hypothesis_id]
    if len(cards) != 1:
        raise ValueError("downstream subset must resolve exactly one hypothesis")
    payload = source.model_dump(mode="json")
    payload["hypotheses"] = [cards[0].model_dump(mode="json")]
    payload["abstention_reason"] = None
    payload["portfolio_id"] = (
        "hypothesis_portfolio:pre_n10_downstream:"
        + _sha256_json(
            {
                "source_portfolio_id": source.portfolio_id,
                "hypothesis_id": hypothesis_id,
                "lineage_key": lineage_key,
            }
        )[:20]
    )
    return HypothesisPortfolio.model_validate(payload)


def _subset_query_plan(
    source: LiteratureQueryPlan,
    *,
    hypothesis_id: str,
    subset_portfolio_id: str,
    lineage_key: str,
) -> LiteratureQueryPlan:
    groups = [row for row in source.claims if row.hypothesis_id == hypothesis_id]
    if len(groups) != 1:
        raise ValueError("downstream subset must resolve exactly one claim group")
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
        plan_id="literature_query_plan:pre_n10_downstream:" + digest[:20],
        plan_sha256=digest,
    )


def _materialize_ready_lineage(
    *,
    source_portfolio: HypothesisPortfolio,
    source_plan: LiteratureQueryPlan,
    hypothesis_id: str,
    source_hypothesis_id: str,
    origin: DownstreamOriginV1,
    semantic_disposition_path: Path,
    semantic_disposition_id: str,
    output_root: Path,
    lineage_key: str,
) -> PreN10DownstreamHandoffLineageV1:
    disposition_file = semantic_disposition_path.expanduser().resolve()
    if not disposition_file.is_file():
        raise ValueError("missing semantic disposition artifact: " + str(disposition_file))
    disposition = HypothesisSemanticDispositionV1.model_validate_json(
        disposition_file.read_text(encoding="utf-8")
    )
    if disposition.disposition_id != semantic_disposition_id:
        raise ValueError("semantic disposition ID mismatch")
    if disposition.source_portfolio_id != source_portfolio.portfolio_id:
        raise ValueError("semantic disposition/source portfolio ID mismatch")
    if disposition.source_portfolio_sha256 != _sha256_json(source_portfolio):
        raise ValueError("semantic disposition/source portfolio SHA mismatch")
    if (
        disposition.disposition != "PASS"
        or not disposition.semantic_admissible_for_pre_n10
    ):
        raise ValueError("downstream handoff requires semantic PASS")

    subset_portfolio = _subset_portfolio(
        source_portfolio,
        hypothesis_id=hypothesis_id,
        lineage_key=lineage_key,
    )
    subset_plan = _subset_query_plan(
        source_plan,
        hypothesis_id=hypothesis_id,
        subset_portfolio_id=subset_portfolio.portfolio_id,
        lineage_key=lineage_key,
    )
    root = output_root / "lineage" / _slug(lineage_key)
    portfolio_path = root / "portfolio.json"
    query_plan_path = root / "claims_queries.json"
    contract_path = root / "contract.json"
    _write_exact_or_validate(portfolio_path, subset_portfolio)
    _write_exact_or_validate(query_plan_path, subset_plan)
    contract = build_pre_n10_scientific_contract_v1(
        portfolio_path=portfolio_path,
        query_plan_path=query_plan_path,
        claim_decomposition_request_count=0,
    )
    if contract.disposition != "READY_FOR_N10":
        raise ValueError(
            "handoff normalization lost pre-N10 readiness: " + hypothesis_id
        )
    _write_exact_or_validate(contract_path, contract)

    return PreN10DownstreamHandoffLineageV1(
        lineage_id=(
            "pre_n10_downstream_lineage_v1:"
            + _sha256_json(
                {
                    "origin": origin,
                    "source_hypothesis_id": source_hypothesis_id,
                    "downstream_hypothesis_id": hypothesis_id,
                    "portfolio_id": subset_portfolio.portfolio_id,
                    "query_plan_id": subset_plan.plan_id,
                    "contract_report_id": contract.report_id,
                    "semantic_disposition_id": disposition.disposition_id,
                }
            )[:20]
        ),
        origin=origin,
        source_hypothesis_id=source_hypothesis_id,
        downstream_hypothesis_id=hypothesis_id,
        portfolio_path=str(portfolio_path.resolve()),
        portfolio_id=subset_portfolio.portfolio_id,
        portfolio_file_sha256=_sha256_file(portfolio_path),
        query_plan_path=str(query_plan_path.resolve()),
        query_plan_id=subset_plan.plan_id,
        query_plan_file_sha256=_sha256_file(query_plan_path),
        contract_report_path=str(contract_path.resolve()),
        contract_report_id=contract.report_id,
        contract_report_file_sha256=_sha256_file(contract_path),
        semantic_disposition_path=str(disposition_file),
        semantic_disposition_id=disposition.disposition_id,
        semantic_disposition_file_sha256=_sha256_file(disposition_file),
    )


def build_pre_n10_downstream_handoff_v1(
    *,
    initial_semantic_gate: PreN10InitialSemanticGateReportV1,
    initial_portfolio_path: Path,
    post_primary_query_plan_path: Path,
    primary_router_report: PreN10PrimaryRouterReportV1,
    output_root: Path,
    regeneration_report: PreN10RegenerationExecutionReportV1 | None = None,
    regeneration_reentry_report: PreN10RegenerationReentryReportV2 | None = None,
) -> PreN10DownstreamHandoffReportV1:
    if not initial_semantic_gate.pre_n10_entry_authorized:
        raise ValueError("initial semantic gate did not authorize pre-N10 entry")
    if initial_semantic_gate.semantic_disposition != "PASS":
        raise ValueError("initial semantic gate lacks PASS disposition")
    if (
        initial_semantic_gate.semantic_disposition_path is None
        or initial_semantic_gate.semantic_disposition_id is None
    ):
        raise ValueError("initial semantic gate lacks disposition artifact")

    portfolio_file = initial_portfolio_path.expanduser().resolve()
    plan_file = post_primary_query_plan_path.expanduser().resolve()
    root = output_root.expanduser().resolve()
    if not portfolio_file.is_file() or not plan_file.is_file():
        raise ValueError("initial handoff source artifacts are missing")

    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_file.read_text(encoding="utf-8")
    )
    plan = LiteratureQueryPlan.model_validate_json(
        plan_file.read_text(encoding="utf-8")
    )
    if portfolio.portfolio_id != initial_semantic_gate.source_portfolio_id:
        raise ValueError("initial semantic gate/portfolio ID mismatch")
    if _sha256_file(portfolio_file) != initial_semantic_gate.source_portfolio_file_sha256:
        raise ValueError("initial semantic gate/portfolio file SHA mismatch")
    if primary_router_report.source_portfolio_id != portfolio.portfolio_id:
        raise ValueError("primary router/portfolio ID mismatch")
    if primary_router_report.source_portfolio_sha256 != _sha256_file(portfolio_file):
        raise ValueError("primary router/portfolio SHA mismatch")
    if plan.plan_id != primary_router_report.post_primary_query_plan_id:
        raise ValueError("primary router/post-plan ID mismatch")
    if plan.plan_sha256 != primary_router_report.post_primary_query_plan_sha256:
        raise ValueError("primary router/post-plan SHA mismatch")

    portfolio_ids = {row.hypothesis_id for row in portfolio.hypotheses}
    router_ids = {row.hypothesis_id for row in primary_router_report.hypotheses}
    if portfolio_ids != router_ids:
        raise ValueError("primary router/portfolio hypothesis populations differ")

    lineages: list[PreN10DownstreamHandoffLineageV1] = []
    fallback_ids: set[str] = set()
    disposition_path = Path(initial_semantic_gate.semantic_disposition_path)
    for row in primary_router_report.hypotheses:
        if row.recovered_for_n10:
            lineages.append(
                _materialize_ready_lineage(
                    source_portfolio=portfolio,
                    source_plan=plan,
                    hypothesis_id=row.hypothesis_id,
                    source_hypothesis_id=row.hypothesis_id,
                    origin="INITIAL_PRIMARY_READY",
                    semantic_disposition_path=disposition_path,
                    semantic_disposition_id=str(
                        initial_semantic_gate.semantic_disposition_id
                    ),
                    output_root=root,
                    lineage_key="initial:" + row.hypothesis_id,
                )
            )
        else:
            if not row.regeneration_fallback_required:
                raise ValueError("non-ready primary row lacks regeneration fallback")
            fallback_ids.add(row.hypothesis_id)

    blocked: list[PreN10BlockedRegenerationLineageV1] = []
    if fallback_ids:
        if regeneration_report is None or regeneration_reentry_report is None:
            raise ValueError("primary fallback population requires regeneration and re-entry reports")
        if regeneration_report.source_primary_report_id != primary_router_report.report_id:
            raise ValueError("regeneration/primary-router ID mismatch")
        if regeneration_report.source_primary_report_sha256 != primary_router_report.report_sha256:
            raise ValueError("regeneration/primary-router SHA mismatch")
        regen_ids = {row.source_hypothesis_id for row in regeneration_report.lineages}
        if regen_ids != fallback_ids:
            raise ValueError("regeneration population differs from primary fallback population")
        if (
            regeneration_reentry_report.source_regeneration_report_id
            != regeneration_report.report_id
        ):
            raise ValueError("re-entry/regeneration report ID mismatch")
        if (
            regeneration_reentry_report.source_regeneration_report_sha256
            != regeneration_report.report_sha256
        ):
            raise ValueError("re-entry/regeneration report SHA mismatch")
        reentry_ids = {
            row.source_hypothesis_id for row in regeneration_reentry_report.lineages
        }
        if reentry_ids != fallback_ids:
            raise ValueError("re-entry population differs from primary fallback population")

        for row in regeneration_reentry_report.lineages:
            if not row.ready_for_n10:
                blocked.append(
                    PreN10BlockedRegenerationLineageV1(
                        source_hypothesis_id=row.source_hypothesis_id,
                        final_status=row.final_status,
                        semantic_status=row.semantic_status,
                        pre_n10_disposition=row.pre_n10_disposition,
                    )
                )
                continue
            if row.final_status != "PRE_N10_READY":
                raise ValueError("ready regeneration lineage has incompatible final status")
            if row.regenerated_portfolio_path is None or row.query_plan_path is None:
                raise ValueError("ready regeneration lineage lacks downstream artifacts")
            if row.semantic_disposition_path is None or row.semantic_disposition_id is None:
                raise ValueError("ready regeneration lineage lacks semantic disposition")

            regen_portfolio_path = Path(row.regenerated_portfolio_path).expanduser().resolve()
            regen_plan_path = Path(row.query_plan_path).expanduser().resolve()
            regen_contract_path = Path(str(row.contract_report_path)).expanduser().resolve()
            for path in (regen_portfolio_path, regen_plan_path, regen_contract_path):
                if not path.is_file():
                    raise ValueError("ready regeneration source artifact missing: " + str(path))
            regen_portfolio = HypothesisPortfolio.model_validate_json(
                regen_portfolio_path.read_text(encoding="utf-8")
            )
            regen_plan = LiteratureQueryPlan.model_validate_json(
                regen_plan_path.read_text(encoding="utf-8")
            )
            regen_contract = PreN10ScientificContractReportV1.model_validate_json(
                regen_contract_path.read_text(encoding="utf-8")
            )
            if len(regen_portfolio.hypotheses) != 1:
                raise ValueError(
                    "v1 downstream handoff requires one regenerated hypothesis per source"
                )
            hypothesis = regen_portfolio.hypotheses[0]
            if regen_portfolio.portfolio_id != row.regenerated_portfolio_id:
                raise ValueError("re-entry/regenerated portfolio ID mismatch")
            if regen_plan.plan_id != row.query_plan_id:
                raise ValueError("re-entry/query-plan ID mismatch")
            if regen_contract.report_id != row.contract_report_id:
                raise ValueError("re-entry/contract report ID mismatch")
            if regen_contract.disposition != "READY_FOR_N10":
                raise ValueError("ready re-entry lineage has non-ready contract")
            lineages.append(
                _materialize_ready_lineage(
                    source_portfolio=regen_portfolio,
                    source_plan=regen_plan,
                    hypothesis_id=hypothesis.hypothesis_id,
                    source_hypothesis_id=row.source_hypothesis_id,
                    origin="REGENERATED_REENTRY_READY",
                    semantic_disposition_path=Path(row.semantic_disposition_path),
                    semantic_disposition_id=row.semantic_disposition_id,
                    output_root=root,
                    lineage_key=(
                        "regenerated:"
                        + row.source_hypothesis_id
                        + ":"
                        + hypothesis.hypothesis_id
                    ),
                )
            )
    else:
        if regeneration_report is not None or regeneration_reentry_report is not None:
            raise ValueError("no primary fallback exists but regeneration reports were supplied")

    initial_count = sum(row.origin == "INITIAL_PRIMARY_READY" for row in lineages)
    regenerated_count = sum(
        row.origin == "REGENERATED_REENTRY_READY" for row in lineages
    )
    body = {
        "schema_version": "pre-n10-downstream-handoff-report-v1",
        "source_initial_semantic_gate_report_id": initial_semantic_gate.report_id,
        "source_initial_semantic_gate_report_sha256": initial_semantic_gate.report_sha256,
        "source_primary_router_report_id": primary_router_report.report_id,
        "source_primary_router_report_sha256": primary_router_report.report_sha256,
        "source_regeneration_report_id": (
            regeneration_report.report_id if regeneration_report is not None else None
        ),
        "source_regeneration_report_sha256": (
            regeneration_report.report_sha256 if regeneration_report is not None else None
        ),
        "source_regeneration_reentry_report_id": (
            regeneration_reentry_report.report_id
            if regeneration_reentry_report is not None
            else None
        ),
        "source_regeneration_reentry_report_sha256": (
            regeneration_reentry_report.report_sha256
            if regeneration_reentry_report is not None
            else None
        ),
        "lineages": [row.model_dump(mode="json") for row in lineages],
        "blocked_regeneration_lineages": [
            row.model_dump(mode="json") for row in blocked
        ],
        "source_hypothesis_count": len(primary_router_report.hypotheses),
        "initial_ready_lineage_count": initial_count,
        "regeneration_fallback_source_count": len(fallback_ids),
        "regenerated_ready_lineage_count": regenerated_count,
        "blocked_after_regeneration_count": len(blocked),
        "eligible_for_external_novelty_count": len(lineages),
        "initial_semantic_disposition_required": True,
        "regenerated_semantic_disposition_required": True,
        "pre_n10_ready_required_for_external_novelty": True,
        "regeneration_is_one_shot": True,
        "second_regeneration_performed": False,
        "external_novelty_performed": False,
        "n9_performed": False,
        "n10_performed": False,
        "endpoint_binding_performed": False,
        "verifier_performed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    report = PreN10DownstreamHandoffReportV1(
        **body,
        report_id="pre_n10_downstream_handoff_v1:" + digest[:20],
        report_sha256=digest,
    )
    _write_exact_or_validate(root / "downstream_handoff.report.json", report)
    return report


__all__ = [
    "DownstreamOriginV1",
    "PreN10BlockedRegenerationLineageV1",
    "PreN10DownstreamHandoffLineageV1",
    "PreN10DownstreamHandoffReportV1",
    "build_pre_n10_downstream_handoff_v1",
]
