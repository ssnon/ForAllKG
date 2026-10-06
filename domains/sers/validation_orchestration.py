from __future__ import annotations

import hashlib
import json
import re

from pipeline_core.discovery.hypothesis_contracts import HypothesisCard, HypothesisPortfolio

from domains.sers.classical_em_readiness_contracts import (
    SERSClassicalEMReadinessAssessment,
    SERSClassicalEMReadinessBundle,
)
from domains.sers.validation_orchestration_contracts import (
    SERSHypothesisValidationState,
    SERSIntegratedExperimentRequirement,
    SERSRouteEvidenceBundle,
    SERSRouteEvidenceRecord,
    SERSRouteValidationState,
    SERSValidationOrchestrationBundle,
)
from domains.sers.validation_routing_contracts import (
    SERSHypothesisValidationPlan,
    SERSHypothesisValidationPlanBundle,
    SERSValidationRoute,
)


_ORCHESTRATOR_VERSION = "sers-validation-orchestrator-v0"


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(_canonical(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:20]}"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _prediction_indexes(route: SERSValidationRoute) -> list[int]:
    indexes: list[int] = []
    for field in route.source_fields:
        match = re.match(r"predicted_observations\[(\d+)\]\.", field)
        if match:
            indexes.append(int(match.group(1)))
    return sorted(set(indexes))


def _experiment_requirement(
    card: HypothesisCard,
    plan: SERSHypothesisValidationPlan,
    route: SERSValidationRoute,
) -> SERSIntegratedExperimentRequirement:
    indexes = _prediction_indexes(route)
    if indexes:
        predictions = [
            card.predicted_observations[index]
            for index in indexes
            if index < len(card.predicted_observations)
        ]
    else:
        # The integrated route is the whole measured outcome branch.  If route
        # lineage does not name a specific prediction index, retain all source
        # predictions rather than inventing a narrower experiment target.
        predictions = list(card.predicted_observations)

    target_norms = {_normalized(row) for row in route.target_observables if row.strip()}
    falsifiers = [
        row
        for row in card.falsification_criteria
        if not target_norms
        or any(
            target in _normalized(row.observable)
            or _normalized(row.observable) in target
            for target in target_norms
        )
    ]
    if not falsifiers:
        # Fail safe: preserving the source hypothesis's falsification criteria is
        # preferable to silently dropping them from experiment requirements.
        falsifiers = list(card.falsification_criteria)

    return SERSIntegratedExperimentRequirement(
        requirement_id=_stable_id(
            "sers_integrated_experiment_requirement",
            card.hypothesis_id,
            plan.plan_id,
            route.route_id,
            _ORCHESTRATOR_VERSION,
        ),
        hypothesis_id=card.hypothesis_id,
        source_validation_plan_id=plan.plan_id,
        source_validation_route_id=route.route_id,
        route_target_observables=list(route.target_observables),
        source_prediction_ids=[row.observation_id for row in predictions],
        source_falsification_criterion_ids=[
            row.criterion_id for row in falsifiers
        ],
        comparison_requirements=[
            "define an explicit baseline/control for the claimed integrated SERS contrast",
            "preserve the hypothesis-stated focal comparison while holding non-focal conditions comparable",
        ],
        control_principles=[
            "record substrate/sample identity and measurement provenance",
            "hold non-focal acquisition and analyte conditions comparable across the focal contrast",
        ],
        unresolved_protocol_dimensions=[
            "fabrication_recipe",
            "sample_count_and_replication",
            "instrument_configuration",
            "measurement_order_and_randomization",
            "statistical_acceptance_criterion",
        ],
    )


def _external_evidence_is_independent(
    record: SERSRouteEvidenceRecord,
    card: HypothesisCard,
) -> tuple[bool, list[str]]:
    blockers: list[str] = []

    overlap_from_ids = bool(set(record.source_paper_ids) & set(card.source_paper_ids))
    explicit_overlap = record.source_overlap_with_generation is True
    overlap = overlap_from_ids or explicit_overlap

    if overlap:
        blockers.append("source_overlap_with_generation")

    if record.evidence_role == "generation_evidence":
        blockers.append("generation_evidence_not_independent_validation")
        return False, blockers

    if overlap:
        return False, blockers

    if record.evidence_role == "independent_validation":
        if record.source_overlap_with_generation is False:
            return True, blockers
        if record.source_paper_ids:
            # If concrete source paper IDs are available and none overlap with the
            # hypothesis-generation papers, the independence check is explicit
            # enough for routing-state purposes.
            return True, blockers
        blockers.append("independence_not_proven")
        return False, blockers

    if record.evidence_role == "experimental_result":
        # A new experimental result may be validation evidence even when it has
        # no literature paper ID.  Explicit source overlap still blocks it above.
        return True, blockers

    blockers.append("operator_observation_not_independent_validation")
    return False, blockers


def _non_em_route_state(
    route: SERSValidationRoute,
    card: HypothesisCard,
    records: list[SERSRouteEvidenceRecord],
    *,
    experiment_requirement_id: str | None,
) -> SERSRouteValidationState:
    if not records:
        next_action = {
            "molecular_spectroscopy": "collect_independent_molecular_spectroscopy_evidence",
            "surface_chemistry": "collect_independent_surface_chemistry_evidence",
            "fabrication_process": "collect_fabrication_process_evidence",
            "integrated_sers_experiment": "review_integrated_sers_experiment_requirement_for_execution",
        }.get(route.route_kind, "collect_route_evidence")
        return SERSRouteValidationState(
            route_id=route.route_id,
            route_kind=route.route_kind,
            required_for_outcome_assessment=route.required_for_outcome_assessment,
            required_for_mechanism_assessment=route.required_for_mechanism_assessment,
            state="no_evidence",
            blockers=["route_evidence_missing"],
            next_actions=[next_action],
            experiment_requirement_id=experiment_requirement_id,
        )

    blockers: list[str] = []
    qualified: list[SERSRouteEvidenceRecord] = []
    for record in records:
        independent, independence_blockers = _external_evidence_is_independent(
            record, card
        )
        if record.review_readiness != "review_ready":
            blockers.append("evidence_not_review_ready")
        blockers.extend(independence_blockers)
        if record.review_readiness == "review_ready" and independent:
            qualified.append(record)

    if qualified:
        state = "review_ready"
        next_actions = ["perform_component_scientific_review"]
        blockers = []
    else:
        state = "evidence_available_not_reviewable"
        next_actions = [
            {
                "molecular_spectroscopy": "strengthen_independent_molecular_spectroscopy_evidence",
                "surface_chemistry": "strengthen_independent_surface_chemistry_evidence",
                "fabrication_process": "strengthen_fabrication_process_evidence",
                "integrated_sers_experiment": "strengthen_integrated_sers_experiment_evidence",
            }.get(route.route_kind, "strengthen_route_evidence")
        ]

    return SERSRouteValidationState(
        route_id=route.route_id,
        route_kind=route.route_kind,
        required_for_outcome_assessment=route.required_for_outcome_assessment,
        required_for_mechanism_assessment=route.required_for_mechanism_assessment,
        state=state,
        evidence_ids=[row.evidence_id for row in records],
        blockers=_unique(blockers),
        next_actions=_unique(next_actions),
        experiment_requirement_id=experiment_requirement_id,
    )


def _classical_em_route_state(
    route: SERSValidationRoute,
    assessment: SERSClassicalEMReadinessAssessment | None,
) -> SERSRouteValidationState:
    if assessment is None:
        return SERSRouteValidationState(
            route_id=route.route_id,
            route_kind=route.route_kind,
            required_for_outcome_assessment=route.required_for_outcome_assessment,
            required_for_mechanism_assessment=route.required_for_mechanism_assessment,
            state="no_evidence",
            blockers=["classical_em_readiness_missing"],
            next_actions=["run_classical_em_validation_pipeline"],
        )

    blockers = _unique(
        list(assessment.readiness_blockers) + list(assessment.numerical_blockers)
    )
    deferred = bool(assessment.deferred_high_cost_actions)

    # Current readiness-v0 intentionally never grants component review.  Keep
    # this branch forward-compatible with a later readiness contract that may.
    review_permitted = bool(
        getattr(assessment, "component_review_permitted", False)
    )
    if review_permitted:
        state = "review_ready"
        next_actions = ["perform_component_scientific_review"]
        blockers = []
    else:
        state = "evidence_available_not_reviewable"
        next_actions = list(assessment.next_research_actions)
        if deferred:
            next_actions.append("defer_classical_em_high_cost_escalation")

    return SERSRouteValidationState(
        route_id=route.route_id,
        route_kind=route.route_kind,
        required_for_outcome_assessment=route.required_for_outcome_assessment,
        required_for_mechanism_assessment=route.required_for_mechanism_assessment,
        state=state,
        evidence_ids=[assessment.source_evidence_id],
        blockers=blockers,
        next_actions=_unique(next_actions),
        high_cost_escalation_deferred=deferred,
    )


def _recommend_next_action(
    route_states: list[SERSRouteValidationState],
) -> tuple[str | None, str | None, str, str]:
    unresolved = [row for row in route_states if row.state == "requires_interpretation"]
    if unresolved:
        row = unresolved[0]
        return (
            row.route_kind,
            row.route_id,
            "resolve_route_interpretation",
            "At least one routed claim is semantically unresolved; scientific validator work should not be launched until its meaning is clarified.",
        )

    required_mechanism = [
        row
        for row in route_states
        if row.required_for_mechanism_assessment
        and row.route_kind != "integrated_sers_experiment"
        and row.state != "review_ready"
    ]
    if required_mechanism:
        # Prefer filling a completely empty non-EM evidence branch before paying
        # for additional work on an already-observed, explicitly deferred EM
        # branch. This is the intended anti-tunnel-vision behavior of S4.0.
        def priority(row: SERSRouteValidationState) -> tuple[int, int, str]:
            non_em = row.route_kind != "classical_em"
            if row.state == "no_evidence" and non_em:
                bucket = 0
            elif row.state == "evidence_available_not_reviewable" and non_em:
                bucket = 1
            elif row.state == "no_evidence":
                bucket = 2
            elif row.high_cost_escalation_deferred:
                bucket = 4
            else:
                bucket = 3
            return (bucket, len(row.blockers), row.route_id)

        row = sorted(required_mechanism, key=priority)[0]
        action = next(
            (
                value
                for value in row.next_actions
                if value != "defer_classical_em_high_cost_escalation"
            ),
            row.next_actions[0] if row.next_actions else "resolve_route_evidence_gap",
        )
        return (
            row.route_kind,
            row.route_id,
            action,
            "A mechanism-required route is not yet review-ready; resolving that missing component takes precedence over whole-hypothesis assessment.",
        )

    required_integrated = [
        row
        for row in route_states
        if row.route_kind == "integrated_sers_experiment"
        and (row.required_for_mechanism_assessment or row.required_for_outcome_assessment)
        and row.state != "review_ready"
    ]
    if required_integrated:
        row = required_integrated[0]
        return (
            row.route_kind,
            row.route_id,
            row.next_actions[0]
            if row.next_actions
            else "review_integrated_sers_experiment_requirement_for_execution",
            "Non-integrated mechanism routes are review-ready; the routed integrated SERS outcome now requires experimental evidence.",
        )

    required_outcome = [
        row
        for row in route_states
        if row.required_for_outcome_assessment and row.state != "review_ready"
    ]
    if required_outcome:
        row = required_outcome[0]
        return (
            row.route_kind,
            row.route_id,
            row.next_actions[0] if row.next_actions else "resolve_outcome_evidence_gap",
            "Outcome-required routed evidence remains incomplete.",
        )

    return (
        None,
        None,
        "perform_multi_route_scientific_review",
        "All required routed evidence packages are review-ready; the next step is scientific component review rather than additional validator execution.",
    )


class SERSValidationOrchestrator:
    """Assemble routed validation progress into a research-state artifact.

    S4.0 is intentionally an orchestration layer, not a truth engine.  It tracks
    what evidence exists, whether it is ready for scientific review, which route
    remains unresolved, and what research action is cheapest/most informative
    next.  It never turns route evidence into hypothesis support, contradiction,
    rejection, experiment promotion, or self-feedback authority.
    """

    orchestrator_version = _ORCHESTRATOR_VERSION

    def build(
        self,
        portfolio: HypothesisPortfolio,
        validation_plans: SERSHypothesisValidationPlanBundle,
        *,
        classical_em_readiness: SERSClassicalEMReadinessBundle | None = None,
        route_evidence: SERSRouteEvidenceBundle | None = None,
    ) -> SERSValidationOrchestrationBundle:
        if portfolio.domain_profile_id != "sers_au_ag":
            raise ValueError("SERS validation orchestration requires sers_au_ag portfolio")
        if validation_plans.source_portfolio_id != portfolio.portfolio_id:
            raise ValueError("validation-plan bundle/portfolio lineage mismatch")

        card_by_id = {row.hypothesis_id: row for row in portfolio.hypotheses}

        readiness_by_route: dict[str, SERSClassicalEMReadinessAssessment] = {}
        if classical_em_readiness is not None:
            for assessment in classical_em_readiness.assessments:
                if assessment.source_validation_route_id in readiness_by_route:
                    raise ValueError(
                        "duplicate classical-EM readiness assessment for route "
                        f"{assessment.source_validation_route_id}"
                    )
                readiness_by_route[assessment.source_validation_route_id] = assessment

        evidence_by_route: dict[str, list[SERSRouteEvidenceRecord]] = {}
        if route_evidence is not None:
            for record in route_evidence.records:
                evidence_by_route.setdefault(record.route_id, []).append(record)

        states: list[SERSHypothesisValidationState] = []
        requirements: list[SERSIntegratedExperimentRequirement] = []

        known_route_ids = {
            route.route_id
            for plan in validation_plans.plans
            for route in plan.routes
        }
        unknown_evidence_routes = set(evidence_by_route) - known_route_ids
        if unknown_evidence_routes:
            raise ValueError(
                "route evidence references unknown validation route(s): "
                + ", ".join(sorted(unknown_evidence_routes))
            )
        unknown_readiness_routes = set(readiness_by_route) - known_route_ids
        if unknown_readiness_routes:
            raise ValueError(
                "classical-EM readiness references unknown validation route(s): "
                + ", ".join(sorted(unknown_readiness_routes))
            )

        for plan in validation_plans.plans:
            card = card_by_id.get(plan.hypothesis_id)
            if card is None:
                raise ValueError(
                    f"validation plan references missing hypothesis {plan.hypothesis_id}"
                )

            route_states: list[SERSRouteValidationState] = []
            requirement_ids: list[str] = []

            for route in plan.routes:
                requirement = None
                if route.route_kind == "integrated_sers_experiment":
                    requirement = _experiment_requirement(card, plan, route)
                    requirements.append(requirement)
                    requirement_ids.append(requirement.requirement_id)

                records = evidence_by_route.get(route.route_id, [])
                for record in records:
                    if record.hypothesis_id != plan.hypothesis_id:
                        raise ValueError("route evidence/hypothesis lineage mismatch")
                    if record.route_kind != route.route_kind:
                        raise ValueError("route evidence/route-kind lineage mismatch")

                if route.route_kind == "unresolved":
                    route_states.append(SERSRouteValidationState(
                        route_id=route.route_id,
                        route_kind=route.route_kind,
                        required_for_outcome_assessment=route.required_for_outcome_assessment,
                        required_for_mechanism_assessment=route.required_for_mechanism_assessment,
                        state="requires_interpretation",
                        blockers=["route_requires_human_interpretation"],
                        next_actions=["resolve_route_interpretation"],
                    ))
                    continue

                if route.route_kind == "classical_em":
                    if records:
                        raise ValueError(
                            "classical_em route evidence must enter through the "
                            "classical-EM readiness lineage, not generic route evidence"
                        )
                    route_states.append(_classical_em_route_state(
                        route, readiness_by_route.get(route.route_id)
                    ))
                    continue

                route_states.append(_non_em_route_state(
                    route,
                    card,
                    records,
                    experiment_requirement_id=(
                        requirement.requirement_id if requirement is not None else None
                    ),
                ))

            mechanism_required = [
                row for row in route_states if row.required_for_mechanism_assessment
            ]
            if not plan.mechanism_claim_present:
                mechanism_state = "not_claimed"
            elif mechanism_required and all(
                row.state == "review_ready" for row in mechanism_required
            ):
                mechanism_state = "review_ready"
            else:
                mechanism_state = "incomplete"

            outcome_required = [
                row for row in route_states if row.required_for_outcome_assessment
            ]
            outcome_state = (
                "review_ready"
                if outcome_required
                and all(row.state == "review_ready" for row in outcome_required)
                else "incomplete"
            )

            if any(row.state == "requires_interpretation" for row in route_states):
                overall = "requires_interpretation"
            elif mechanism_state == "incomplete":
                overall = "mechanism_evidence_incomplete"
            elif outcome_state == "incomplete":
                overall = "outcome_evidence_incomplete"
            else:
                overall = "routed_evidence_review_ready"

            next_kind, next_route_id, next_action, rationale = _recommend_next_action(
                route_states
            )

            states.append(SERSHypothesisValidationState(
                state_id=_stable_id(
                    "sers_hypothesis_validation_state",
                    plan.plan_id,
                    *(row.model_dump(mode="json") for row in route_states),
                    self.orchestrator_version,
                ),
                hypothesis_id=plan.hypothesis_id,
                source_validation_plan_id=plan.plan_id,
                route_states=route_states,
                mechanism_evidence_state=mechanism_state,
                outcome_evidence_state=outcome_state,
                overall_research_state=overall,
                recommended_next_route_kind=next_kind,
                recommended_next_route_id=next_route_id,
                recommended_next_action=next_action,
                rationale=rationale,
                experiment_requirement_ids=requirement_ids,
            ))

        return SERSValidationOrchestrationBundle(
            bundle_id=_stable_id(
                "sers_validation_orchestration_bundle",
                portfolio.portfolio_id,
                validation_plans.bundle_id,
                classical_em_readiness.bundle_id
                if classical_em_readiness is not None
                else None,
                route_evidence.bundle_id if route_evidence is not None else None,
                *(row.state_id for row in states),
                *(row.requirement_id for row in requirements),
                self.orchestrator_version,
            ),
            source_portfolio_id=portfolio.portfolio_id,
            source_validation_plan_bundle_id=validation_plans.bundle_id,
            source_classical_em_readiness_bundle_id=(
                classical_em_readiness.bundle_id
                if classical_em_readiness is not None
                else None
            ),
            source_route_evidence_bundle_id=(
                route_evidence.bundle_id if route_evidence is not None else None
            ),
            states=states,
            state_count=len(states),
            experiment_requirements=requirements,
            experiment_requirement_count=len(requirements),
        )
