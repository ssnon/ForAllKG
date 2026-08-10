from __future__ import annotations

import hashlib

from dac_her.feasibility_contracts import FeasibilityHypothesis, FeasibilityIntake
from dac_her.scope_contracts import ScientificScope
from dac_her.validation_contracts import ValidationSpecification


_DUAL_CHECKS = [
    "structural_validity",
    "pair_stability",
    "aggregation_risk",
    "operating_state_stability",
]
_SINGLE_CHECKS = [
    "structural_validity",
    "isolated_site_stability",
    "aggregation_risk",
    "operating_state_stability",
]
_GENERAL_CHECKS = [
    "structural_validity",
    "aggregation_risk",
    "operating_state_stability",
]


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:length]}"


def _strategy(scope: ScientificScope) -> str:
    if scope.hypothesis_level == "context_extension":
        return "context_comparison"
    if scope.hypothesis_level in {"material_family", "comparative_study"}:
        return "comparative_computational_study"
    if scope.hypothesis_level == "mechanism":
        return "mechanism_validation"
    if scope.hypothesis_level == "candidate_specific":
        return "candidate_specific_computation"
    return "hybrid"


def _base_checks(scope: ScientificScope) -> tuple[list[str], list[str], list[str]]:
    if scope.catalyst_class == "dual_atom":
        return (
            list(_DUAL_CHECKS),
            ["isolated_site_stability"],
            ["isolated_single_atom_synthesis"],
        )
    if scope.catalyst_class == "single_atom":
        return (
            list(_SINGLE_CHECKS),
            ["pair_stability"],
            ["atomic_pair_selective_synthesis"],
        )
    if scope.catalyst_class == "mixed_atomic_site":
        return (
            sorted(set(_DUAL_CHECKS + _SINGLE_CHECKS)),
            [],
            [],
        )
    return (list(_GENERAL_CHECKS), [], [])


def _mechanism_checks(h: FeasibilityHypothesis) -> list[str]:
    text = " ".join([
        h.statement,
        h.inferential_bridge,
        *(p.observable for p in h.predictions),
    ]).lower()
    rows: list[str] = []
    if "hydrogen adsorption" in text or "adsorption thermodynamics" in text:
        rows.append("hydrogen_adsorption")
    if "charge transfer" in text or "electronic" in text:
        rows.append("electronic_structure")
    if "water dissociation" in text or "water activation" in text or "volmer" in text:
        rows.append("water_dissociation")
    if any(term in text for term in ("mechanism", "pathway", "heyrovsky", "tafel")):
        rows.append("reaction_pathway")
    return rows


class ValidationSpecificationCompiler:
    def compile_intake(
        self,
        intake: FeasibilityIntake,
        scopes: list[ScientificScope],
    ) -> list[ValidationSpecification]:
        scope_by_id = {row.hypothesis_id: row for row in scopes}
        return [
            self.compile(h, scope_by_id[h.hypothesis_id])
            for h in intake.hypotheses
        ]

    def compile(
        self,
        hypothesis: FeasibilityHypothesis,
        scope: ScientificScope,
    ) -> ValidationSpecification:
        required, not_applicable, exp_not_applicable = _base_checks(scope)
        required = sorted(set(required + _mechanism_checks(hypothesis)))

        controlled: list[str] = []
        varied = list(scope.independent_variables)
        statement = hypothesis.statement.lower()

        if "at a given nitrogen coordination number" in statement:
            controlled.append("nitrogen_coordination_number")
            varied = [row for row in varied if row != "nitrogen_coordination_number"]
        if scope.hypothesis_level == "context_extension" and len(scope.environments) >= 2:
            if "reaction_environment" not in varied:
                varied.append("reaction_environment")
            controlled.extend(["catalyst_identity", "coordination_motif"])

        if scope.hypothesis_level in {"material_family", "comparative_study"}:
            controlled.extend(["metal_identity_or_pair", "support_model"])
        controlled = sorted(set(controlled))
        varied = sorted(set(varied))

        concretization: list[str] = []
        if scope.requires_candidate_concretization:
            if not scope.metals and scope.catalyst_class == "dual_atom":
                concretization.append("select explicit metal pair(s) for the comparison set")
            elif not scope.metals and scope.catalyst_class == "single_atom":
                concretization.append("select explicit isolated metal site(s) for the comparison set")
            elif not scope.metals and scope.catalyst_class == "mixed_atomic_site":
                concretization.append("select explicit SAC and DAC comparator identities")
            concretization.append("define concrete support/coordination structures")
            if scope.reaction == "HER":
                concretization.append("define electrolyte/pH and electrochemical comparison conditions")

        comparisons: list[str] = []
        if scope.hypothesis_level == "material_family":
            comparisons.append("compare multiple structures spanning the stated coordination/design variables")
        elif scope.hypothesis_level == "comparative_study":
            comparisons.append("hold the stated controlled variables comparable while varying the focal geometry/descriptor")
        elif scope.hypothesis_level == "context_extension":
            comparisons.append("evaluate matched catalyst/site models across the stated reaction environments")
        elif scope.hypothesis_level == "mechanism":
            comparisons.append("compare the proposed mechanism against at least one plausible alternative pathway")

        success_patterns = [
            f"{row.observable}: expected_direction={row.expected_direction}"
            for row in hypothesis.predictions
        ]
        falsification_patterns = [
            f"{row.observable}: {row.falsifying_outcome}"
            for row in hypothesis.falsifiers
        ]

        next_actions: list[str] = []
        if concretization:
            next_actions.append("concretize validation systems before launching high-fidelity computation")
        if required:
            next_actions.append("resolve scope-applicable physics checks with evidence/database/computational backends")
        next_actions.append("retrieve direct or analogous synthesis/characterization precedent for the concretized systems")

        return ValidationSpecification(
            specification_id=_stable_id(
                "validation_specification",
                hypothesis.hypothesis_id,
                scope.scope_id,
                _strategy(scope),
                ",".join(required),
                ",".join(varied),
            ),
            hypothesis_id=hypothesis.hypothesis_id,
            source_scope_id=scope.scope_id,
            validation_strategy=_strategy(scope),  # type: ignore[arg-type]
            requires_candidate_concretization=scope.requires_candidate_concretization,
            controlled_variables=controlled,
            varied_variables=varied,
            primary_observables=list(scope.dependent_observables),
            secondary_observables=[
                row for row in ("structural_stability", "operating_state_stability")
                if row not in scope.dependent_observables
            ],
            required_comparisons=comparisons,
            candidate_concretization_requirements=concretization,
            required_physics_checks=required,
            not_applicable_physics_checks=not_applicable,
            not_applicable_experimental_capabilities=exp_not_applicable,
            success_patterns=success_patterns,
            falsification_patterns=falsification_patterns,
            next_actions=next_actions,
        )
