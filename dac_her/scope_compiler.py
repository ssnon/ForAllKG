from __future__ import annotations

import hashlib
import re

from dac_her.feasibility_contracts import FeasibilityHypothesis, FeasibilityIntake
from dac_her.scope_contracts import ScientificScope


_DUAL_TERMS = (
    "dual-atom",
    "dual atom",
    "dual-site",
    "dual site",
    "atomic pair",
    "metal pair",
    "bimetallic pair",
    "dimer",
    "tm2@",
)

_SINGLE_TERMS = (
    "single-atom",
    "single atom",
    "isolated metal atom",
    "isolated atom",
    "sac",
)

_ENVIRONMENT_TERMS = {
    "acidic": ("acidic", "acid electrolyte"),
    "alkaline": ("alkaline", "basic electrolyte", "koh"),
    "neutral": ("neutral electrolyte", "neutral ph", "near-neutral"),
    "pH-universal": ("ph-universal", "ph universal", "universal ph"),
}

_VARIABLE_TERMS = {
    "nitrogen_coordination_number": (
        "nitrogen coordination number",
        "n coordination number",
        "coordination number",
    ),
    "local_coordination_geometry": (
        "local nitrogen coordination geometry",
        "local coordination geometry",
        "local geometry",
        "coordination geometry",
    ),
    "metal_identity": ("metal identity", "metal species", "metal pair"),
    "support_environment": ("support", "substrate", "graphene", "carbon support"),
    "reaction_environment": ("acidic", "alkaline", "reaction environment", "electrolyte"),
}

_OBSERVABLE_TERMS = {
    "HER_activity": ("her activity", "hydrogen evolution activity", "overpotential", "tafel"),
    "hydrogen_adsorption_thermodynamics": (
        "hydrogen adsorption thermodynamics",
        "hydrogen adsorption",
        "adsorption free energy",
        "delta g h",
        "Δg_h",
    ),
    "charge_transfer": ("charge transfer", "charge redistribution", "bader"),
    "water_dissociation": ("water dissociation", "water activation", "volmer"),
    "structural_stability": ("structural stability", "pair stability", "stability"),
}


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:length]}"


def _normalize(text: str) -> str:
    text = text.lower().replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def _claim_surface(h: FeasibilityHypothesis) -> str:
    # Deliberately excludes premises and the inferential bridge from catalyst-class
    # classification. Those may mention a different source system than the proposed
    # hypothesis (e.g., a SAC extension supported partly by DAC evidence).
    chunks = [
        h.title,
        h.statement,
        *(row.observable for row in h.predictions),
        *(row.rationale for row in h.predictions),
        *(row.observable for row in h.falsifiers),
        *(row.falsifying_outcome for row in h.falsifiers),
    ]
    return _normalize(" ".join(chunks))


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _extract_environments(text: str) -> list[str]:
    rows: list[str] = []
    for name, terms in _ENVIRONMENT_TERMS.items():
        if _contains_any(text, terms):
            rows.append(name)
    return rows


def _extract_variables(text: str) -> list[str]:
    rows: list[str] = []
    for name, terms in _VARIABLE_TERMS.items():
        if _contains_any(text, terms):
            rows.append(name)
    return rows


def _extract_observables(text: str) -> list[str]:
    rows: list[str] = []
    for name, terms in _OBSERVABLE_TERMS.items():
        if _contains_any(text, terms):
            rows.append(name)
    return rows


def _extract_metals(text: str) -> list[str]:
    # Conservative extraction: only hyphenated element-symbol pairs and explicit
    # single symbols followed by site/atom are accepted. Generic TM2 is excluded.
    symbols = re.findall(r"\b([A-Z][a-z]?)[-–]([A-Z][a-z]?)\b", text)
    rows = {symbol for pair in symbols for symbol in pair}
    return sorted(rows)


class HypothesisScopeCompiler:
    """Compile hypothesis claim surface into an explicit scientific scope.

    This deterministic layer prevents support premises from changing the target
    catalyst class. It is intentionally conservative and emits warnings rather than
    silently guessing when the target system is ambiguous.
    """

    def compile_intake(self, intake: FeasibilityIntake) -> list[ScientificScope]:
        return [self.compile(h) for h in intake.hypotheses]

    def compile(self, hypothesis: FeasibilityHypothesis) -> ScientificScope:
        text = _claim_surface(hypothesis)
        has_dual = _contains_any(text, _DUAL_TERMS)
        has_single = _contains_any(text, _SINGLE_TERMS)
        warnings: list[str] = []

        if has_dual and has_single:
            catalyst_class = "mixed_atomic_site"
            catalyst_reason = "The hypothesis claim surface explicitly references both dual-atom and single-atom systems."
            confidence = "medium"
        elif has_dual:
            catalyst_class = "dual_atom"
            catalyst_reason = "The hypothesis claim surface explicitly identifies a dual-atom/atomic-pair system."
            confidence = "high"
        elif has_single:
            catalyst_class = "single_atom"
            catalyst_reason = "The hypothesis claim surface explicitly identifies a single-atom/isolated-site system."
            confidence = "high"
        elif any(term in text for term in ("atomic site", "atomically dispersed", "coordination site")):
            catalyst_class = "general_atomic_site"
            catalyst_reason = "The hypothesis concerns atomically dispersed/coordination sites but does not resolve SAC versus DAC."
            confidence = "medium"
        else:
            catalyst_class = "unknown"
            catalyst_reason = "The target catalyst class is not explicit on the hypothesis claim surface."
            confidence = "low"
            warnings.append("catalyst_class_not_explicit")

        htype = hypothesis.hypothesis_type.lower()
        if htype == "context_dependency" or any(
            phrase in text
            for phrase in ("extend to", "may extend", "between acidic and alkaline", "across acidic and alkaline")
        ):
            level = "context_extension"
            level_reason = "The hypothesis transfers a relationship across catalyst or reaction contexts."
        elif htype == "mechanistic_extension" and "within " in text and any(
            phrase in text
            for phrase in ("catalysts", "coordination number", "local geometry", "vary non-monotonically", "as a function")
        ):
            level = "material_family"
            level_reason = "The hypothesis concerns a trend across a catalyst family rather than one fully specified candidate."
        elif htype == "descriptor_mediation" or any(
            phrase in text
            for phrase in ("at a given", "with comparable", "different local geometries", "compared coordination")
        ):
            level = "comparative_study"
            level_reason = "The hypothesis requires controlled comparison between related structures or descriptor states."
        elif "within " in text and any(
            phrase in text
            for phrase in ("catalysts", "coordination number", "local geometry", "vary non-monotonically", "as a function")
        ):
            level = "material_family"
            level_reason = "The hypothesis concerns a trend across a catalyst family rather than one fully specified candidate."
        elif htype in {"mechanistic_extension", "mechanism"}:
            level = "mechanism"
            level_reason = "The hypothesis proposes a mechanistic extension without a fully specified comparison matrix."
        else:
            level = "unknown"
            level_reason = "The hypothesis level could not be determined conservatively from the claim surface."
            warnings.append("hypothesis_level_not_explicit")

        environments = _extract_environments(text)
        variables = _extract_variables(text)
        observables = _extract_observables(text)
        metals = _extract_metals(hypothesis.statement)

        # Distinguish controlled versus varied variables downstream; scope records
        # the variables present without over-interpreting the design.
        independent = list(variables)

        reaction = "HER" if _contains_any(
            f" {text} ",
            (" her ", "her activity", "hydrogen evolution"),
        ) else "unknown"

        requires_concretization = level != "candidate_specific"
        if not metals and catalyst_class in {"dual_atom", "single_atom", "mixed_atomic_site"}:
            requires_concretization = True
            warnings.append("metal_identity_not_concrete")

        return ScientificScope(
            scope_id=_stable_id(
                "scientific_scope",
                hypothesis.hypothesis_id,
                catalyst_class,
                level,
                ",".join(environments),
                ",".join(variables),
            ),
            hypothesis_id=hypothesis.hypothesis_id,
            catalyst_class=catalyst_class,  # type: ignore[arg-type]
            hypothesis_level=level,  # type: ignore[arg-type]
            reaction=reaction,  # type: ignore[arg-type]
            environments=environments,
            metals=metals,
            coordination_variables=[
                row for row in variables
                if row in {"nitrogen_coordination_number", "local_coordination_geometry"}
            ],
            independent_variables=independent,
            dependent_observables=observables,
            requires_candidate_concretization=requires_concretization,
            scope_confidence=confidence,  # type: ignore[arg-type]
            scope_warnings=sorted(set(warnings)),
            catalyst_class_rationale=catalyst_reason,
            hypothesis_level_rationale=level_reason,
        )
