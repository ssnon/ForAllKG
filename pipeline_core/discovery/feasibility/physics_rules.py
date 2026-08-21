from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from pipeline_core.discovery.feasibility.feasibility_contracts import FeasibilityHypothesis
from pipeline_core.discovery.feasibility.physics_contracts import PhysicsCheckRequest
from pipeline_core.discovery.feasibility.scope_contracts import ScientificScope
from pipeline_core.runtime.validation_contracts import ValidationSpecification


_CHECK_TERMS: dict[str, tuple[str, ...]] = {
    "hydrogen_adsorption": (
        "hydrogen adsorption",
        "h adsorption",
        "h*",
        "delta g h",
        "Δg_h",
        "adsorption free energy",
        "adsorption thermodynamics",
    ),
    "water_dissociation": (
        "water dissociation",
        "water activation",
        "h2o dissociation",
        "volmer",
        "alkaline her",
    ),
    "oh_binding": (
        "oh adsorption",
        "oh*",
        "hydroxyl",
    ),
    "electronic_structure": (
        "charge transfer",
        "charge redistribution",
        "bader",
        "d-band",
        "orbital",
        "electronic structure",
    ),
    "reaction_pathway": (
        "mechanism",
        "reaction pathway",
        "volmer",
        "heyrovsky",
        "tafel",
        "barrier",
        "activation energy",
    ),
    "aggregation_risk": (
        "aggregation",
        "sinter",
        "migration",
        "cluster",
        "agglomeration",
    ),
    "thermodynamic_stability": (
        "formation energy",
        "binding energy",
        "thermodynamic",
        "stability",
    ),
    "operating_state_stability": (
        "operando",
        "in situ",
        "reconstruction",
        "surface state",
        "electrochemical stability",
        "acidic",
        "alkaline",
    ),
    "pair_stability": (
        "dual-atom",
        "dual atom",
        "atomic pair",
        "metal pair",
        "tm2@",
    ),
    "isolated_site_stability": (
        "single-atom",
        "single atom",
        "isolated metal atom",
        "isolated atom",
    ),
    "structural_validity": (
        "coordination",
        "geometry",
        "structure",
        "site",
    ),
}


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:length]}"


def _normalized_text(hypothesis: FeasibilityHypothesis) -> str:
    chunks = [
        hypothesis.title,
        hypothesis.statement,
        hypothesis.inferential_bridge,
        *hypothesis.assumptions,
        *(row.observable for row in hypothesis.predictions),
        *(row.rationale for row in hypothesis.predictions),
        *(row.text for row in hypothesis.premises),
    ]
    text = " ".join(chunks).lower().replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text)


def _matched_terms(text: str, terms: Iterable[str]) -> list[str]:
    return sorted({term for term in terms if term.lower() in text})


class PhysicsCheckPlanner:
    """Scope-aware deterministic planner for feasibility v0.2.

    Required checks come from ValidationSpecification, which is compiled from the
    *target hypothesis scope*. Premises may supply evidence for a check, but they no
    longer decide whether a SAC or DAC check applies.
    """

    def plan(
        self,
        hypothesis: FeasibilityHypothesis,
        scope: ScientificScope,
        specification: ValidationSpecification,
    ) -> list[PhysicsCheckRequest]:
        text = _normalized_text(hypothesis)
        requests: list[PhysicsCheckRequest] = []
        for check_type in specification.required_physics_checks:
            terms = _CHECK_TERMS.get(check_type, ())
            matched = _matched_terms(text, terms)
            requests.append(
                PhysicsCheckRequest(
                    request_id=_stable_id(
                        "physics_request",
                        hypothesis.hypothesis_id,
                        scope.scope_id,
                        specification.specification_id,
                        check_type,
                    ),
                    hypothesis_id=hypothesis.hypothesis_id,
                    source_scope_id=scope.scope_id,
                    source_validation_specification_id=specification.specification_id,
                    check_type=check_type,
                    scientific_domain=scope.scientific_domain,
                    reason=(
                        f"Scope-aware validation requires {check_type} for "
                        f"catalyst_class={scope.catalyst_class}, "
                        f"hypothesis_level={scope.hypothesis_level}."
                    ),
                    relevant_terms=matched,
                )
            )
        return requests
