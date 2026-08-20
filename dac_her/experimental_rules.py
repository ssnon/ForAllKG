from __future__ import annotations

import hashlib
import re

from pipeline_core.experimental_contracts import ExperimentalRequirement
from pipeline_core.feasibility_contracts import FeasibilityHypothesis
from pipeline_core.scope_contracts import ScientificScope
from pipeline_core.validation_contracts import ValidationSpecification


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:length]}"


def _text(hypothesis: FeasibilityHypothesis) -> str:
    chunks = [
        hypothesis.title,
        hypothesis.statement,
        hypothesis.inferential_bridge,
        *hypothesis.assumptions,
        *(row.observable for row in hypothesis.predictions),
        *(row.rationale for row in hypothesis.predictions),
    ]
    return re.sub(
        r"\s+",
        " ",
        " ".join(chunks).lower().replace("–", "-").replace("—", "-"),
    )


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


_HER_TERMS = (
    "hydrogen evolution",
    " her ",
    "her activity",
    "overpotential",
    "tafel",
    "exchange current",
    "current density",
)

_MECHANISM_TERMS = (
    "water dissociation",
    "water activation",
    "charge transfer",
    "charge redistribution",
    "hydrogen spillover",
    "reaction pathway",
    "mechanism",
    "active site",
    "volmer",
    "heyrovsky",
    "tafel",
)

_OPERANDO_TERMS = (
    "operando",
    "in situ",
    "working state",
    "surface reconstruction",
    "active site",
)


class GenericExperimentalRequirementPlanner:
    """Scope-aware laboratory-agnostic requirement planner for v0.2."""

    def plan(
        self,
        hypothesis: FeasibilityHypothesis,
        scope: ScientificScope,
        specification: ValidationSpecification,
    ) -> list[ExperimentalRequirement]:
        text = f" {_text(hypothesis)} "
        rows: list[ExperimentalRequirement] = []

        def add(category: str, capability: str, necessity: str, rationale: str) -> None:
            if capability in specification.not_applicable_experimental_capabilities:
                return
            rows.append(
                ExperimentalRequirement(
                    requirement_id=_stable_id(
                        "experimental_requirement",
                        hypothesis.hypothesis_id,
                        scope.scope_id,
                        category,
                        capability,
                    ),
                    category=category,
                    capability=capability,
                    necessity=necessity,  # type: ignore[arg-type]
                    rationale=rationale,
                    scientific_domain=scope.scientific_domain,
                )
            )

        if scope.catalyst_class == "dual_atom":
            add(
                "synthesis",
                "atomic_pair_selective_synthesis",
                "required",
                "The target hypothesis is explicitly DAC/atomic-pair scoped, so validation requires a route that suppresses isolated-single-atom, cluster, and nanoparticle alternatives.",
            )
            add(
                "characterization",
                "atomic_resolution_microscopy",
                "required",
                "A DAC structural assignment generally needs direct or near-direct evidence of paired atomic sites.",
            )
            add(
                "characterization",
                "xray_absorption_coordination_analysis",
                "required",
                "Ensemble coordination information should complement microscopy when assigning a dual-atom local structure.",
            )
        elif scope.catalyst_class == "single_atom":
            add(
                "synthesis",
                "isolated_single_atom_synthesis",
                "required",
                "The target hypothesis is explicitly single-atom scoped, so validation requires stabilization of isolated sites while suppressing pair/cluster/nanoparticle formation.",
            )
            add(
                "characterization",
                "atomic_resolution_microscopy",
                "required",
                "Evidence for atomically isolated sites is needed to support a single-atom structural assignment.",
            )
            add(
                "characterization",
                "xray_absorption_coordination_analysis",
                "required",
                "Coordination-sensitive spectroscopy should complement microscopy and support the isolated-site local environment.",
            )
        elif scope.catalyst_class == "mixed_atomic_site":
            add(
                "synthesis",
                "atomic_site_state_control",
                "required",
                "The hypothesis compares or combines SAC and DAC states, requiring explicit control of site nuclearity and alternative structures.",
            )
            add(
                "characterization",
                "atomic_resolution_microscopy",
                "required",
                "Site-nuclearity comparison requires direct structural evidence for the relevant atomic-site states.",
            )
            add(
                "characterization",
                "xray_absorption_coordination_analysis",
                "required",
                "Coordination-sensitive evidence is needed to distinguish local environments across site nuclearities.",
            )
        elif scope.catalyst_class == "general_atomic_site":
            add(
                "synthesis",
                "atomically_dispersed_site_synthesis",
                "recommended",
                "The hypothesis concerns atomically dispersed sites but does not resolve SAC versus DAC; generic site-isolation control is therefore appropriate.",
            )

        if _contains_any(text, _HER_TERMS) or scope.reaction == "HER":
            add(
                "electrochemistry",
                "her_polarization_and_kinetic_testing",
                "required",
                "The stated HER response should be evaluated with standard polarization/kinetic measurements under explicit conditions.",
            )
            add(
                "electrochemistry",
                "electrochemical_stability_testing",
                "required",
                "Catalyst performance claims require stability/durability testing in the relevant electrolyte and potential regime.",
            )

        if _contains_any(text, _MECHANISM_TERMS):
            add(
                "characterization",
                "mechanism_sensitive_characterization",
                "recommended",
                "Mechanistic attribution should be supported by measurements sensitive to the proposed intermediate, electronic change, or reaction step.",
            )

        if _contains_any(text, _OPERANDO_TERMS) or len(scope.environments) >= 2:
            add(
                "characterization",
                "operando_or_in_situ_validation",
                "recommended",
                "Working-state or environment-dependent claims benefit from operando/in-situ evidence rather than ex-situ structure alone.",
            )

        seen: set[tuple[str, str]] = set()
        deduped: list[ExperimentalRequirement] = []
        for row in rows:
            key = (row.category, row.capability)
            if key not in seen:
                seen.add(key)
                deduped.append(row)
        return deduped
