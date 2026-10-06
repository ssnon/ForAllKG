from __future__ import annotations

import hashlib
import json
import re

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisPortfolio,
)

from domains.sers.classical_em_handoff_contracts import (
    SERSClassicalEMHandoff,
    SERSClassicalEMHandoffBundle,
)
from domains.sers.fdtd_applicability import SERSFDTDApplicabilityAnalyzer
from domains.sers.fdtd_applicability_contracts import SERSFDTDApplicabilityBundle
from domains.sers.simulation_compiler import SERSHypothesisSimulationCompiler
from domains.sers.simulation_contracts import (
    SERSSimulationCompilationBundle,
    SERSSimulationValidationBundle,
)
from domains.sers.simulation_validator import SERSDeterministicSimulationValidator
from domains.sers.validation_routing_contracts import (
    SERSHypothesisValidationPlanBundle,
    SERSValidationRoute,
)


_HANDOFF_VERSION = "sers-classical-em-handoff-v0"
_PREDICTION_FIELD = re.compile(
    r"^predicted_observations\[(?P<index>\d+)\]\.(?P<part>observable|rationale)$"
)
_ACTIVE_TOP_LEVEL_FIELDS = {"hypothesis_statement", "inferential_bridge"}


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(_canonical(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:20]}"


def _classical_em_route(plan) -> SERSValidationRoute | None:
    rows = [row for row in plan.routes if row.route_kind == "classical_em"]
    if len(rows) > 1:
        raise ValueError(
            f"multiple classical_em routes for hypothesis {plan.hypothesis_id}"
        )
    return rows[0] if rows else None


def _validate_route_source_fields(route: SERSValidationRoute, card: HypothesisCard) -> None:
    if not route.source_fields:
        raise ValueError("classical_em route requires source_fields")
    for field in route.source_fields:
        if field in _ACTIVE_TOP_LEVEL_FIELDS:
            continue
        match = _PREDICTION_FIELD.fullmatch(field)
        if match and int(match.group("index")) < len(card.predicted_observations):
            continue
        raise ValueError(
            "classical_em route contains a non-claim or unknown source field: "
            + field
        )


def _route_scoped_card(
    card: HypothesisCard,
    route: SERSValidationRoute,
) -> HypothesisCard:
    """Remove context-only text before existing FDTD compilation.

    The canonical hypothesis remains unchanged. This transient card exists only
    to ensure that the FDTD compiler cannot obtain geometry or numerical input
    from assumptions/title that did not create the active classical-EM route.
    """
    _validate_route_source_fields(route, card)
    fields = set(route.source_fields)

    predictions = []
    for index, row in enumerate(card.predicted_observations):
        observable_field = f"predicted_observations[{index}].observable"
        rationale_field = f"predicted_observations[{index}].rationale"
        predictions.append(row.model_copy(update={
            "observable": row.observable if observable_field in fields else "",
            "rationale": row.rationale if rationale_field in fields else "",
        }))

    return card.model_copy(update={
        # v0.1 routing deliberately treats title/assumptions as context only.
        "title": "",
        "hypothesis_statement": (
            card.hypothesis_statement if "hypothesis_statement" in fields else ""
        ),
        "inferential_bridge": (
            card.inferential_bridge if "inferential_bridge" in fields else ""
        ),
        "predicted_observations": predictions,
        "assumptions": [],
    })


def _handoff_status(disposition: str) -> str:
    mapping = {
        "runnable_shadow": "compiled_runnable_shadow",
        "requires_concretization": "compiled_requires_concretization",
        "unsupported": "compiled_unsupported",
        "not_applicable": "compiled_not_applicable",
        "requires_interpretation": "compiled_requires_interpretation",
        "invalid": "compiled_invalid",
    }
    try:
        return mapping[disposition]
    except KeyError as exc:
        raise ValueError(f"unknown simulation disposition: {disposition}") from exc


class SERSClassicalEMHandoffBuilder:
    """Connect routed classical-EM claims to the existing FDTD spec stack.

    This layer does not execute Meep and does not create physics authority. It
    proves a narrower invariant first: only text fields that created the active
    classical-EM route are allowed to feed the existing FDTD compiler.
    """

    builder_version = _HANDOFF_VERSION

    def build(
        self,
        portfolio: HypothesisPortfolio,
        validation_plans: SERSHypothesisValidationPlanBundle,
        *,
        fdtd_applicability: SERSFDTDApplicabilityBundle | None = None,
    ) -> tuple[
        SERSClassicalEMHandoffBundle,
        SERSSimulationCompilationBundle,
        SERSSimulationValidationBundle,
        SERSFDTDApplicabilityBundle,
    ]:
        if portfolio.domain_profile_id != "sers_au_ag":
            raise ValueError("SERS classical-EM handoff requires domain_profile_id='sers_au_ag'")
        if validation_plans.source_portfolio_id != portfolio.portfolio_id:
            raise ValueError("validation-plan bundle/source portfolio mismatch")
        if validation_plans.domain_profile_id != portfolio.domain_profile_id:
            raise ValueError("validation-plan bundle/domain mismatch")

        plan_ids = [row.hypothesis_id for row in validation_plans.plans]
        if len(plan_ids) != len(set(plan_ids)):
            raise ValueError("duplicate validation plan for hypothesis")
        cards_by_id = {row.hypothesis_id: row for row in portfolio.hypotheses}
        missing_cards = sorted(set(plan_ids) - set(cards_by_id))
        if missing_cards:
            raise ValueError(
                "validation plan references hypothesis absent from portfolio: "
                + ", ".join(missing_cards)
            )

        selected_ids = set(plan_ids)
        if fdtd_applicability is None:
            fdtd_applicability = SERSFDTDApplicabilityAnalyzer().analyze_portfolio(
                portfolio,
                hypothesis_ids=selected_ids,
            )
        if fdtd_applicability.source_portfolio_id != portfolio.portfolio_id:
            raise ValueError("FDTD applicability/source portfolio mismatch")
        if fdtd_applicability.domain_profile_id != portfolio.domain_profile_id:
            raise ValueError("FDTD applicability/domain mismatch")
        if (
            validation_plans.source_fdtd_applicability_bundle_id
            != fdtd_applicability.bundle_id
        ):
            raise ValueError(
                "validation-plan/FDTD-applicability lineage mismatch; rerun routing "
                "and handoff from the same deterministic applicability artifact"
            )

        applicability_by_id = {
            row.hypothesis_id: row for row in fdtd_applicability.reports
        }
        if len(applicability_by_id) != len(fdtd_applicability.reports):
            raise ValueError("duplicate FDTD applicability report for hypothesis")
        if selected_ids != set(applicability_by_id):
            raise ValueError(
                "FDTD applicability must cover exactly the hypotheses in the validation plan"
            )

        routed: list[tuple[object, SERSValidationRoute, HypothesisCard]] = []
        not_routed: list[str] = []
        for plan in validation_plans.plans:
            route = _classical_em_route(plan)
            if route is None:
                not_routed.append(plan.hypothesis_id)
                continue
            if route.readiness != "ready":
                raise ValueError(
                    f"classical_em route is not ready for {plan.hypothesis_id}: "
                    f"{route.readiness}"
                )
            if not route.source_fdtd_subclaim_ids:
                raise ValueError("classical_em route must reference FDTD subclaim ids")

            applicability = applicability_by_id[plan.hypothesis_id]
            available_subclaims = {row.subclaim_id for row in applicability.fdtd_subclaims}
            route_subclaims = set(route.source_fdtd_subclaim_ids)
            if not route_subclaims <= available_subclaims:
                raise ValueError(
                    "classical_em route references FDTD subclaim absent from its "
                    f"applicability report: {plan.hypothesis_id}"
                )

            card = cards_by_id[plan.hypothesis_id]
            routed.append((plan, route, _route_scoped_card(card, route)))

        routed_ids = {plan.hypothesis_id for plan, _, _ in routed}
        scoped_portfolio = portfolio.model_copy(update={
            "hypotheses": [card for _, _, card in routed],
            "abstention_reason": None if routed else "no classical-EM route selected",
        })

        compilation = SERSHypothesisSimulationCompiler().compile_portfolio(
            scoped_portfolio,
            hypothesis_ids=routed_ids,
            applicability_bundle=fdtd_applicability,
        )
        validation = SERSDeterministicSimulationValidator().validate_bundle(compilation)

        spec_by_id = {row.hypothesis_id: row for row in compilation.specs}
        report_by_id = {row.hypothesis_id: row for row in validation.reports}
        if set(spec_by_id) != routed_ids or set(report_by_id) != routed_ids:
            raise ValueError("routed FDTD compilation/validation lost hypothesis lineage")

        handoffs: list[SERSClassicalEMHandoff] = []
        for plan, route, _ in routed:
            applicability = applicability_by_id[plan.hypothesis_id]
            spec = spec_by_id[plan.hypothesis_id]
            report = report_by_id[plan.hypothesis_id]

            if spec.source_applicability_report_id != applicability.report_id:
                raise ValueError("simulation spec/applicability report lineage mismatch")
            if set(spec.fdtd_subclaim_ids) != set(route.source_fdtd_subclaim_ids):
                raise ValueError(
                    "simulation spec contains FDTD subclaims outside the routed "
                    f"classical-EM claim for {plan.hypothesis_id}"
                )

            allowed_provenance_fields = set(route.source_fields) | {
                "applicability.baseline_wavelength_nm"
            }
            leaked = sorted({
                row.source_field
                for row in spec.parameter_provenance
                if row.source_field not in allowed_provenance_fields
            })
            if leaked:
                raise ValueError(
                    "route-scoped FDTD compilation consumed context-only source fields: "
                    + ", ".join(leaked)
                )

            handoffs.append(SERSClassicalEMHandoff(
                handoff_id=_stable_id(
                    "sers_classical_em_handoff",
                    plan.plan_id,
                    route.route_id,
                    spec.spec_id,
                    report.report_id,
                    self.builder_version,
                ),
                hypothesis_id=plan.hypothesis_id,
                source_validation_plan_id=plan.plan_id,
                source_validation_route_id=route.route_id,
                source_fdtd_applicability_report_id=applicability.report_id,
                source_fdtd_subclaim_ids=route.source_fdtd_subclaim_ids,
                route_scoped_source_fields=route.source_fields,
                source_simulation_spec_id=spec.spec_id,
                source_simulation_validation_report_id=report.report_id,
                simulation_disposition=report.disposition,
                handoff_status=_handoff_status(report.disposition),
                required_for_outcome_assessment=route.required_for_outcome_assessment,
                required_for_mechanism_assessment=route.required_for_mechanism_assessment,
            ))

        bundle = SERSClassicalEMHandoffBundle(
            bundle_id=_stable_id(
                "sers_classical_em_handoff_bundle",
                portfolio.portfolio_id,
                validation_plans.bundle_id,
                fdtd_applicability.bundle_id,
                compilation.bundle_id,
                validation.bundle_id,
                *(row.handoff_id for row in handoffs),
                self.builder_version,
            ),
            source_portfolio_id=portfolio.portfolio_id,
            source_validation_plan_bundle_id=validation_plans.bundle_id,
            source_fdtd_applicability_bundle_id=fdtd_applicability.bundle_id,
            source_simulation_compilation_bundle_id=compilation.bundle_id,
            source_simulation_validation_bundle_id=validation.bundle_id,
            handoffs=handoffs,
            routed_hypothesis_count=len(handoffs),
            not_routed_hypothesis_ids=sorted(not_routed),
        )
        return bundle, compilation, validation, fdtd_applicability
