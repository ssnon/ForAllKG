from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import LiteratureQueryPlan
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.preverifier_contract_gate_v2 import (
    PreVerifierContractGateV2Report,
)
from pipeline_core.discovery.prospective_regeneration_downstream_n10_v2 import (
    ProspectiveRegenerationExternalN10ReportV2,
    RegenerationNoveltyCertificationReportV2,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    HypothesisBindingStatus,
    RelationalAtomicBindingHypothesisPlan,
    RelationalAtomicBindingPlan,
    assess_claim_binding_readiness,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


class RegenerationBindingCompatibilityManifestV2(StrictModel):
    schema_version: Literal["regeneration-binding-compatibility-manifest-v2"] = (
        "regeneration-binding-compatibility-manifest-v2"
    )
    manifest_id: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_external_n10_report_id: str
    source_external_n10_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_portfolio_id: str
    source_final_hypothesis_ids: list[str]
    regenerated_hypothesis_ids: list[str]
    source_certification_reports: list[str]
    diagnostic_compatibility_only: Literal[True] = True
    per_hypothesis_source_portfolio_authoritative: Literal[True] = True
    per_hypothesis_query_plan_authoritative: Literal[True] = True
    alpha6_provenance_asserted: Literal[False] = False
    scientific_content_mutated: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_manifest(self):
        n = len(self.regenerated_hypothesis_ids)
        if len(self.source_final_hypothesis_ids) != n:
            raise ValueError("source/regenerated lineage count mismatch")
        if len(self.source_certification_reports) != n:
            raise ValueError("certification lineage count mismatch")
        body = self.model_dump(mode="json")
        observed_id = body.pop("manifest_id")
        observed_sha = body.pop("manifest_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("compatibility manifest SHA mismatch")
        if observed_id != "regeneration_binding_compatibility_manifest_v2:" + expected_sha[:20]:
            raise ValueError("compatibility manifest ID mismatch")
        return self


class RegenerationReachabilityLineageV2(StrictModel):
    source_final_hypothesis_id: str
    regenerated_hypothesis_id: str
    n10_certification_status: str
    n10_selection_class: str
    gate_ready: bool
    novelty_bearing_gate_ready_claim_ids: list[str] = Field(default_factory=list)


class RegenerationReachabilityComparisonV2(StrictModel):
    schema_version: Literal["regeneration-preverifier-reachability-comparison-v2"] = (
        "regeneration-preverifier-reachability-comparison-v2"
    )
    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_id: Literal["P16", "P17", "P18", "P19", "P20"]
    source_external_n10_report_id: str
    source_binding_plan_id: str
    source_gate_report_id: str
    lineage_count: int = Field(ge=0)
    initial_gate_ready_hypothesis_count: int = Field(ge=0)
    primary_gate_ready_hypothesis_count: int = Field(ge=0)
    regenerated_gate_ready_hypothesis_count: int = Field(ge=0)
    n10_certified_count: int = Field(ge=0)
    n10_unresolved_count: int = Field(ge=0)
    n10_rejected_count: int = Field(ge=0)
    lineages: list[RegenerationReachabilityLineageV2]
    endpoint_binding_performed: Literal[False] = False
    verifier_performed: Literal[False] = False
    second_regeneration_performed: Literal[False] = False
    scientific_content_mutated: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self):
        if self.lineage_count != len(self.lineages):
            raise ValueError("lineage_count mismatch")
        if self.regenerated_gate_ready_hypothesis_count != sum(x.gate_ready for x in self.lineages):
            raise ValueError("regenerated gate-ready count mismatch")
        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("reachability comparison SHA mismatch")
        if observed_id != "regeneration_preverifier_reachability_comparison_v2:" + expected_sha[:20]:
            raise ValueError("reachability comparison ID mismatch")
        return self


def gate_ready_hypothesis_ids(report: PreVerifierContractGateV2Report) -> set[str]:
    return {
        row.final_hypothesis_id
        for row in report.rows
        if row.gate_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
        and row.novelty_selection_role == "NOVELTY_BEARING"
    }


def build_compatibility_portfolio_v2(*, portfolios: list[HypothesisPortfolio]) -> HypothesisPortfolio:
    if not portfolios:
        raise ValueError("compatibility portfolio requires source portfolios")
    first = portfolios[0]
    for row in portfolios[1:]:
        for field in (
            "domain_profile_id", "source_context_id", "source_context_sha256",
            "source_report_id", "source_report_sha256",
        ):
            if getattr(row, field) != getattr(first, field):
                raise ValueError("regenerated portfolio metadata mismatch: " + field)
    cards = [card for portfolio in portfolios for card in portfolio.hypotheses]
    ids = [card.hypothesis_id for card in cards]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate regenerated hypothesis IDs")
    return HypothesisPortfolio(
        portfolio_id=_stable_id(
            "hypothesis_portfolio", "regeneration_binding_compatibility_v2",
            first.source_context_sha256, *ids,
        ),
        domain_profile_id=first.domain_profile_id,
        source_context_id=first.source_context_id,
        source_context_sha256=first.source_context_sha256,
        source_report_id=first.source_report_id,
        source_report_sha256=first.source_report_sha256,
        hypotheses=cards,
        abstention_reason=None,
    )


def build_compatibility_manifest_v2(
    *,
    external_n10: ProspectiveRegenerationExternalN10ReportV2,
    compatibility_portfolio: HypothesisPortfolio,
    certification_paths: list[Path],
) -> RegenerationBindingCompatibilityManifestV2:
    body = {
        "schema_version": "regeneration-binding-compatibility-manifest-v2",
        "source_external_n10_report_id": external_n10.report_id,
        "source_external_n10_report_sha256": external_n10.report_sha256,
        "compatibility_portfolio_id": compatibility_portfolio.portfolio_id,
        "source_final_hypothesis_ids": [x.source_final_hypothesis_id for x in external_n10.lineages],
        "regenerated_hypothesis_ids": [x.regenerated_hypothesis_id for x in external_n10.lineages],
        "source_certification_reports": [str(x) for x in certification_paths],
        "diagnostic_compatibility_only": True,
        "per_hypothesis_source_portfolio_authoritative": True,
        "per_hypothesis_query_plan_authoritative": True,
        "alpha6_provenance_asserted": False,
        "scientific_content_mutated": False,
        "production_selection_changed": False,
    }
    digest = _sha256_json(body)
    return RegenerationBindingCompatibilityManifestV2(
        **body,
        manifest_id="regeneration_binding_compatibility_manifest_v2:" + digest[:20],
        manifest_sha256=digest,
    )


def _binding_status(*, ready_count: int, novelty_ready_count: int) -> HypothesisBindingStatus:
    if ready_count == 0:
        return "NO_BINDABLE_CLAIMS"
    if novelty_ready_count == 0:
        return "NO_NOVELTY_BEARING_BINDABLE_CLAIM"
    return "READY_FOR_LITERAL_ENDPOINT_BINDING"


def build_regeneration_relational_atomic_binding_plan_v2(
    *,
    run_dir: Path,
    external_n10: ProspectiveRegenerationExternalN10ReportV2,
    certification_paths: list[Path],
    compatibility_portfolio_path: Path,
    compatibility_manifest_path: Path,
) -> RelationalAtomicBindingPlan:
    if len(certification_paths) != external_n10.lineage_count:
        raise ValueError("certification path population mismatch")
    external_by_source = {x.source_final_hypothesis_id: x for x in external_n10.lineages}
    if len(external_by_source) != len(external_n10.lineages):
        raise ValueError("duplicate source final hypothesis IDs")

    hypothesis_plans = []
    seen_sources = set()
    for certification_path in certification_paths:
        certification = RegenerationNoveltyCertificationReportV2.model_validate_json(
            certification_path.read_text(encoding="utf-8")
        )
        source_id = certification.source_final_hypothesis_id
        if source_id in seen_sources:
            raise ValueError("duplicate certification source lineage")
        seen_sources.add(source_id)
        external_row = external_by_source.get(source_id)
        if external_row is None:
            raise ValueError("certification lineage absent from external N10 report")
        if len(certification.decisions) != 1 or len(certification.candidate_artifacts) != 1:
            raise ValueError("regeneration certification requires one decision/artifact")

        decision = certification.decisions[0]
        artifact = certification.candidate_artifacts[0]
        if decision.regenerated_hypothesis_id != external_row.regenerated_hypothesis_id:
            raise ValueError("regenerated hypothesis lineage mismatch")
        if decision.certification_status != external_row.certification_status:
            raise ValueError("certification status mismatch")
        if decision.n10_selection_class != external_row.n10_selection_class:
            raise ValueError("N10 selection class mismatch")

        candidate_id = decision.regenerated_hypothesis_id
        source_portfolio_path = Path(artifact.source_portfolio).expanduser().resolve()
        query_plan_path = Path(artifact.query_plan).expanduser().resolve()
        source_portfolio = HypothesisPortfolio.model_validate_json(
            source_portfolio_path.read_text(encoding="utf-8")
        )
        if source_portfolio.portfolio_id != certification.source_regenerated_portfolio_id:
            raise ValueError("source portfolio/certification ID mismatch")
        cards = [x for x in source_portfolio.hypotheses if x.hypothesis_id == candidate_id]
        if len(cards) != 1:
            raise ValueError("regenerated source portfolio must resolve one candidate card")

        query_plan = LiteratureQueryPlan.model_validate_json(
            query_plan_path.read_text(encoding="utf-8")
        )
        if query_plan.source_portfolio_id != source_portfolio.portfolio_id:
            raise ValueError("regenerated query-plan/source-portfolio provenance mismatch")
        claim_groups = [x for x in query_plan.claims if x.hypothesis_id == candidate_id]
        if len(claim_groups) != 1:
            raise ValueError("regenerated query plan must resolve one candidate claim group")

        claim_plans = [
            assess_claim_binding_readiness(
                claim=claim,
                candidate_hypothesis_id=candidate_id,
                final_hypothesis_id=candidate_id,
            )
            for claim in claim_groups[0].claims
        ]
        ready_count = sum(x.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING" for x in claim_plans)
        novelty_ready_count = sum(
            x.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
            and x.novelty_selection_role == "NOVELTY_BEARING"
            for x in claim_plans
        )
        hypothesis_plans.append(
            RelationalAtomicBindingHypothesisPlan(
                original_hypothesis_id=source_id,
                candidate_hypothesis_id=candidate_id,
                final_hypothesis_id=candidate_id,
                alpha6_decision="regeneration_v2",
                certification_status=decision.certification_status,
                n10_selection_class=decision.n10_selection_class,
                source_candidate_portfolio=str(source_portfolio_path),
                source_candidate_portfolio_sha256=_sha256_file(source_portfolio_path),
                source_query_plan=str(query_plan_path),
                source_query_plan_sha256=_sha256_file(query_plan_path),
                claim_count=len(claim_plans),
                binding_ready_claim_count=ready_count,
                novelty_bearing_binding_ready_claim_count=novelty_ready_count,
                binding_status=_binding_status(
                    ready_count=ready_count,
                    novelty_ready_count=novelty_ready_count,
                ),
                claims=claim_plans,
            )
        )

    if seen_sources != set(external_by_source):
        raise ValueError("binding-plan certification population differs from external N10")

    h_counts = Counter(x.binding_status for x in hypothesis_plans)
    c_counts = Counter(c.binding_status for x in hypothesis_plans for c in x.claims)
    body = {
        "schema_version": "relational-atomic-binding-plan-v1",
        "run_dir": str(run_dir.expanduser().resolve()),
        "source_alpha6_candidate_portfolio": str(compatibility_portfolio_path.expanduser().resolve()),
        "source_alpha6_candidate_portfolio_sha256": _sha256_file(compatibility_portfolio_path.expanduser().resolve()),
        "source_certification_report": str(compatibility_manifest_path.expanduser().resolve()),
        "source_certification_report_sha256": _sha256_file(compatibility_manifest_path.expanduser().resolve()),
        "hypotheses": [x.model_dump(mode="json") for x in hypothesis_plans],
        "hypothesis_count": len(hypothesis_plans),
        "ready_hypothesis_count": sum(x.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING" for x in hypothesis_plans),
        "not_ready_hypothesis_count": sum(x.binding_status != "READY_FOR_LITERAL_ENDPOINT_BINDING" for x in hypothesis_plans),
        "claim_count": sum(x.claim_count for x in hypothesis_plans),
        "binding_ready_claim_count": sum(x.binding_ready_claim_count for x in hypothesis_plans),
        "novelty_bearing_binding_ready_claim_count": sum(x.novelty_bearing_binding_ready_claim_count for x in hypothesis_plans),
        "hypothesis_status_counts": dict(sorted(h_counts.items())),
        "claim_status_counts": dict(sorted(c_counts.items())),
        "certification_only_source_required": True,
        "candidate_final_authority_equivalence_required": True,
        "endpoint_binding_performed": False,
        "verifier_result_observed": False,
        "production_selection_authority": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return RelationalAtomicBindingPlan(
        **body,
        plan_id="relational_atomic_binding_plan:" + digest[:20],
        plan_sha256=digest,
    )


def build_reachability_comparison_v2(
    *,
    case_id: str,
    external_n10: ProspectiveRegenerationExternalN10ReportV2,
    binding_plan: RelationalAtomicBindingPlan,
    initial_gate: PreVerifierContractGateV2Report,
    primary_gate: PreVerifierContractGateV2Report,
    regenerated_gate: PreVerifierContractGateV2Report,
) -> RegenerationReachabilityComparisonV2:
    if case_id != external_n10.case_id:
        raise ValueError("case/external-N10 mismatch")
    regenerated_ready = gate_ready_hypothesis_ids(regenerated_gate)
    rows = []
    for lineage in external_n10.lineages:
        claim_ids = [
            x.claim_id
            for x in regenerated_gate.rows
            if x.final_hypothesis_id == lineage.regenerated_hypothesis_id
            and x.gate_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
            and x.novelty_selection_role == "NOVELTY_BEARING"
        ]
        rows.append(
            RegenerationReachabilityLineageV2(
                source_final_hypothesis_id=lineage.source_final_hypothesis_id,
                regenerated_hypothesis_id=lineage.regenerated_hypothesis_id,
                n10_certification_status=lineage.certification_status,
                n10_selection_class=lineage.n10_selection_class,
                gate_ready=lineage.regenerated_hypothesis_id in regenerated_ready,
                novelty_bearing_gate_ready_claim_ids=claim_ids,
            )
        )
    body = {
        "schema_version": "regeneration-preverifier-reachability-comparison-v2",
        "case_id": case_id,
        "source_external_n10_report_id": external_n10.report_id,
        "source_binding_plan_id": binding_plan.plan_id,
        "source_gate_report_id": regenerated_gate.report_id,
        "lineage_count": len(rows),
        "initial_gate_ready_hypothesis_count": len(gate_ready_hypothesis_ids(initial_gate)),
        "primary_gate_ready_hypothesis_count": len(gate_ready_hypothesis_ids(primary_gate)),
        "regenerated_gate_ready_hypothesis_count": len(regenerated_ready),
        "n10_certified_count": external_n10.certified_count,
        "n10_unresolved_count": external_n10.unresolved_count,
        "n10_rejected_count": external_n10.rejected_count,
        "lineages": [x.model_dump(mode="json") for x in rows],
        "endpoint_binding_performed": False,
        "verifier_performed": False,
        "second_regeneration_performed": False,
        "scientific_content_mutated": False,
        "production_selection_changed": False,
    }
    digest = _sha256_json(body)
    return RegenerationReachabilityComparisonV2(
        **body,
        report_id="regeneration_preverifier_reachability_comparison_v2:" + digest[:20],
        report_sha256=digest,
    )
