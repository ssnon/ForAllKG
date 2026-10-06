from __future__ import annotations

import hashlib
import json

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio

from domains.sers.validation_evidence_intake_contracts import (
    SERSRouteEvidenceSubmissionBundle,
)
from domains.sers.validation_orchestration_contracts import (
    SERSRouteEvidenceBundle,
    SERSRouteEvidenceRecord,
)
from domains.sers.validation_routing_contracts import (
    SERSHypothesisValidationPlanBundle,
    SERSValidationRoute,
)


_INTAKE_VERSION = "sers-route-evidence-intake-v0"


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(_canonical(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:20]}"


def _resolve_route(
    *,
    hypothesis_id: str,
    route_kind: str,
    route_id: str | None,
    plans: SERSHypothesisValidationPlanBundle,
) -> SERSValidationRoute:
    plan = next((row for row in plans.plans if row.hypothesis_id == hypothesis_id), None)
    if plan is None:
        raise ValueError(f"evidence submission references missing hypothesis {hypothesis_id}")

    if route_id is not None:
        route = next((row for row in plan.routes if row.route_id == route_id), None)
        if route is None:
            raise ValueError(f"evidence submission references missing route {route_id}")
        if route.route_kind != route_kind:
            raise ValueError("evidence submission route_kind does not match route_id")
        return route

    candidates = [row for row in plan.routes if row.route_kind == route_kind]
    if len(candidates) != 1:
        raise ValueError(
            "evidence submission requires explicit route_id when route kind does not "
            f"resolve uniquely: hypothesis={hypothesis_id}, route_kind={route_kind}, "
            f"matches={len(candidates)}"
        )
    return candidates[0]


class SERSRouteEvidenceIntakeCompiler:
    compiler_version = _INTAKE_VERSION

    def compile(
        self,
        portfolio: HypothesisPortfolio,
        plans: SERSHypothesisValidationPlanBundle,
        submissions: SERSRouteEvidenceSubmissionBundle,
    ) -> SERSRouteEvidenceBundle:
        if portfolio.domain_profile_id != "sers_au_ag":
            raise ValueError("SERS route-evidence intake requires sers_au_ag portfolio")
        if plans.source_portfolio_id != portfolio.portfolio_id:
            raise ValueError("validation-plan bundle/portfolio lineage mismatch")

        cards = {row.hypothesis_id: row for row in portfolio.hypotheses}
        records: list[SERSRouteEvidenceRecord] = []

        for submission in submissions.submissions:
            card = cards.get(submission.hypothesis_id)
            if card is None:
                raise ValueError(
                    f"evidence submission references missing hypothesis {submission.hypothesis_id}"
                )
            route = _resolve_route(
                hypothesis_id=submission.hypothesis_id,
                route_kind=submission.route_kind,
                route_id=submission.route_id,
                plans=plans,
            )

            generation_overlap = bool(
                set(card.source_paper_ids) & set(submission.source_paper_ids)
            )
            if generation_overlap:
                if submission.source_overlap_with_generation is False:
                    raise ValueError(
                        "submission declares independent source but source_paper_ids overlap "
                        "hypothesis-generation papers"
                    )
                overlap_flag: bool | None = True
            else:
                overlap_flag = submission.source_overlap_with_generation

            records.append(SERSRouteEvidenceRecord(
                evidence_id=_stable_id(
                    "sers_route_evidence",
                    submission.submission_id,
                    route.route_id,
                    self.compiler_version,
                ),
                hypothesis_id=submission.hypothesis_id,
                route_id=route.route_id,
                route_kind=route.route_kind,
                evidence_role=submission.evidence_role,
                review_readiness=submission.review_readiness,
                relation_to_claim=submission.relation_to_claim,
                target_observables=list(submission.target_observables),
                source_ids=list(submission.source_ids),
                source_paper_ids=list(submission.source_paper_ids),
                source_overlap_with_generation=overlap_flag,
                limitations=list(submission.limitations),
                provenance_notes=[
                    *submission.provenance_notes,
                    f"compiled_from_submission:{submission.submission_id}",
                ],
            ))

        return SERSRouteEvidenceBundle(
            bundle_id=_stable_id(
                "sers_route_evidence_bundle",
                portfolio.portfolio_id,
                plans.bundle_id,
                submissions.bundle_id,
                *(row.evidence_id for row in records),
                self.compiler_version,
            ),
            records=records,
            record_count=len(records),
        )
