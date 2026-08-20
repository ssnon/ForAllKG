from __future__ import annotations

import hashlib
import re

from pipeline_core.feasibility_contracts import FeasibilityHypothesis, FeasibilityIntake
from pipeline_core.scope_contracts import ScientificScope


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

# Explicit element symbols/names accepted as concrete metal identities.
# C/N/O/H are intentionally absent so coordination/support formulas such as N4
# do not become false "metal" hits. Generic TM/TM2 is also intentionally absent.
_METAL_SYMBOLS = (
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
)

_METAL_NAMES = {
    "scandium": "Sc",
    "titanium": "Ti",
    "vanadium": "V",
    "chromium": "Cr",
    "manganese": "Mn",
    "iron": "Fe",
    "cobalt": "Co",
    "nickel": "Ni",
    "copper": "Cu",
    "zinc": "Zn",
    "yttrium": "Y",
    "zirconium": "Zr",
    "niobium": "Nb",
    "molybdenum": "Mo",
    "technetium": "Tc",
    "ruthenium": "Ru",
    "rhodium": "Rh",
    "palladium": "Pd",
    "silver": "Ag",
    "cadmium": "Cd",
    "hafnium": "Hf",
    "tantalum": "Ta",
    "tungsten": "W",
    "rhenium": "Re",
    "osmium": "Os",
    "iridium": "Ir",
    "platinum": "Pt",
    "gold": "Au",
    "mercury": "Hg",
}

_CANDIDATE_FAMILY_TERMS = (
    "as a function",
    "series",
    "across",
    "multiple structures",
    "coordination number",
    "different local geometries",
    "vary non-monotonically",
    "trend",
)

_CANDIDATE_CONTEXT_TERMS = (
    "extend to",
    "may extend",
    "between acidic and alkaline",
    "across acidic and alkaline",
    "reaction environments",
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
    """Extract explicit metal identities without treating N/C/O/H as metals.

    v2.7.0 only recovered hyphenated symbol pairs (e.g. Fe-Ru), which made
    candidate-specific SAC targets effectively invisible. v2.7.1 accepts
    standalone transition-metal symbols and common element names while
    deliberately excluding generic TM/TM2 placeholders.
    """
    symbol_pattern = "|".join(sorted(_METAL_SYMBOLS, key=len, reverse=True))
    rows = set(re.findall(rf"\b(?:{symbol_pattern})\b", text))

    lowered = text.lower()
    for name, symbol in _METAL_NAMES.items():
        if re.search(rf"\b{re.escape(name)}\b", lowered):
            rows.add(symbol)

    return sorted(rows)


def _candidate_identity_is_concrete(
    *,
    catalyst_class: str,
    metals: list[str],
) -> bool:
    if catalyst_class == "dual_atom":
        return len(metals) >= 2
    if catalyst_class == "single_atom":
        return len(metals) >= 1
    if catalyst_class == "mixed_atomic_site":
        return len(metals) >= 2
    return False


def _candidate_specific_from_claim(
    *,
    text: str,
    catalyst_class: str,
    metals: list[str],
) -> bool:
    """Conservatively identify a single explicit target rather than a family.

    This is intentionally narrower than merely seeing a metal name. A claim that
    explicitly compares environments/geometries or describes a family/trend stays
    comparative/material-family even when concrete metals are mentioned.
    """
    if not _candidate_identity_is_concrete(
        catalyst_class=catalyst_class,
        metals=metals,
    ):
        return False
    if _contains_any(text, _CANDIDATE_CONTEXT_TERMS):
        return False
    if _contains_any(text, _CANDIDATE_FAMILY_TERMS):
        return False
    return True


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

        metals = _extract_metals(hypothesis.statement)
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
        elif _candidate_specific_from_claim(
            text=text,
            catalyst_class=catalyst_class,
            metals=metals,
        ):
            level = "candidate_specific"
            level_reason = (
                "The hypothesis targets an explicit metal/site identity without "
                "requiring a family, controlled-comparison, or context-extension design."
            )
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
            catalyst_class=catalyst_class,  # legacy compatibility
            system_class=catalyst_class,
            scientific_domain="electrocatalysis",
            hypothesis_level=level,
            reaction=reaction,  # legacy compatibility
            process=reaction,
            environments=environments,
            metals=metals,
            components=metals,
            coordination_variables=[
                row for row in variables
                if row in {"nitrogen_coordination_number", "local_coordination_geometry"}
            ],
            structural_variables=[
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
