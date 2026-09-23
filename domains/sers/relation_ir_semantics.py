from __future__ import annotations

import re
from dataclasses import dataclass


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, str(text or ""), re.I) is not None


_HOTSPOT = r"\bhot\s*spots?\b|\bhotspots?\b"
_EM_CONTEXT = (
    r"\bplasmon(?:ic|ics)?\b|"
    r"\bsers\b|"
    r"surface[- ]enhanced\s+raman|"
    r"\braman\b|"
    r"near[- ]?field|"
    r"electromagnetic|"
    r"localized\s+surface\s+plasmon"
)
_ENERGETIC_CONTEXT = (
    r"\bhmx\b|"
    r"\brdx\b|"
    r"\btatb\b|"
    r"\bexplosive\b|"
    r"\bdetonation\b|"
    r"\bshock(?:ed|ing)?\b|"
    r"\benergetic\s+material"
)
_RIS = r"\bris\b"
_REFRACTIVE_INDEX = r"refractive\s+index"
_REFRACTIVE_INDEX_SENSITIVITY = (
    r"refractive\s+index\s+sensitivity"
)

# General relation-IR concept families used by typed retrieval. These rules
# identify what a source phrase denotes; they do not establish scientific
# truth, novelty, evidence strength, or production authority.
#
# Enhancement factor is materially ambiguous outside Raman/SERS usage, so it
# requires explicit SERS/Raman/plasmonic context. The remaining phrases carry
# their scientific role directly in the source text and do not inherit a type
# merely because the active domain profile is SERS.
_SERS_ENHANCEMENT_FACTOR = (
    r"\bsers\s+enhancement[- ]factor\b|"
    r"\bsers\s+(?:ef|aef)\b|"
    r"\benhancement[- ]factor\b"
)
_MEASUREMENT_REPRODUCIBILITY = (
    r"\bmeasurement\s+(?:reproducib\w*|repeatab\w*)\b|"
    r"\b(?:reproducib\w*|repeatab\w*)\b|"
    r"\brelative\s+standard\s+deviation\b|"
    r"\brsd\b"
)
_NANOSTRUCTURE_DESIGN_VARIABLE = (
    r"\bnano[- ]?structure[- ]+design[- ]+modification\b|"
    r"\bnano[- ]?structure[- ]+"
    r"(?:design|geometry|morphology|size|spacing|separation|"
    r"architecture|shape)\b"
)
_MATERIAL_COMPOSITION_VARIABLE = (
    r"\b(?:substrate|surface|material)\s+composition\b|"
    r"\bcomposition\s+of\s+(?:the\s+)?"
    r"(?:substrate|surface|material)\b"
)


@dataclass(frozen=True)
class SERSRelationTypingAdapter:
    """Conservative typed-concept rules for SERS prior-art diagnostics.

    The rules intentionally require contextual support. A bare "hotspot" does
    not become an electromagnetic hotspot merely because the active domain is
    SERS, and bare "RIS" remains ambiguous unless refractive-index context is
    explicit.
    """

    adapter_id: str = "sers_relation_typing_v1"

    def type_labels(
        self,
        *,
        surface_text: str,
        relation_context: str,
    ) -> tuple[str, ...]:
        surface = str(surface_text or "")
        context = str(relation_context or "")
        joined = f"{surface} {context}"

        labels: list[str] = []

        if _has(_HOTSPOT, joined) and _has(_EM_CONTEXT, joined):
            labels.append("electromagnetic_plasmonic_hotspot")

        if _has(_HOTSPOT, joined) and _has(_ENERGETIC_CONTEXT, joined):
            labels.append("energetic_material_hotspot")

        if (
            _has(_REFRACTIVE_INDEX_SENSITIVITY, joined)
            or (
                _has(_RIS, joined)
                and _has(_REFRACTIVE_INDEX, joined)
            )
        ):
            labels.append("refractive_index_sensitivity")

        # General concept-family typing is surface-local. Context may
        # disambiguate an enhancement-factor phrase, but it must not make an
        # unrelated neighboring concept inherit the same type.
        if (
            _has(_SERS_ENHANCEMENT_FACTOR, surface)
            and _has(_EM_CONTEXT, joined)
        ):
            labels.append("sers_enhancement_metric")

        if _has(_MEASUREMENT_REPRODUCIBILITY, surface):
            labels.append("measurement_reproducibility_metric")

        if _has(_NANOSTRUCTURE_DESIGN_VARIABLE, surface):
            labels.append("nanostructure_design_variable")

        if _has(_MATERIAL_COMPOSITION_VARIABLE, surface):
            labels.append("material_composition_variable")

        return tuple(dict.fromkeys(labels))

    def ambiguity_labels(
        self,
        *,
        surface_text: str,
        relation_context: str,
        type_labels: tuple[str, ...],
    ) -> tuple[str, ...]:
        surface = str(surface_text or "")
        context = str(relation_context or "")
        joined = f"{surface} {context}"
        labels: list[str] = []

        if _has(_RIS, surface) and (
            "refractive_index_sensitivity"
            not in type_labels
        ):
            labels.append("bare_RIS_without_refractive_index_identity")

        if _has(_HOTSPOT, surface) and not {
            "electromagnetic_plasmonic_hotspot",
            "energetic_material_hotspot",
        }.intersection(type_labels):
            labels.append("hotspot_without_typed_physical_context")

        # A single text carrying both hotspot meanings is itself ambiguous.
        if {
            "electromagnetic_plasmonic_hotspot",
            "energetic_material_hotspot",
        }.issubset(type_labels):
            labels.append("hotspot_cross_domain_identity_collision")

        # If the surface is only an acronym, do not silently inherit an
        # unrelated expansion from the active domain.
        if (
            _has(_RIS, surface)
            and not _has(_REFRACTIVE_INDEX, joined)
        ):
            labels.append("RIS_expansion_not_source_explicit")

        return tuple(dict.fromkeys(labels))

    def conflicting_type_pairs(
        self,
        *,
        relation_types: tuple[str, ...],
        document_types: tuple[str, ...],
    ) -> tuple[tuple[str, str], ...]:
        conflicts: list[tuple[str, str]] = []

        if (
            "electromagnetic_plasmonic_hotspot"
            in relation_types
            and "energetic_material_hotspot"
            in document_types
        ):
            conflicts.append(
                (
                    "electromagnetic_plasmonic_hotspot",
                    "energetic_material_hotspot",
                )
            )

        if (
            "energetic_material_hotspot"
            in relation_types
            and "electromagnetic_plasmonic_hotspot"
            in document_types
        ):
            conflicts.append(
                (
                    "energetic_material_hotspot",
                    "electromagnetic_plasmonic_hotspot",
                )
            )

        return tuple(conflicts)


SERS_RELATION_TYPING_ADAPTER = SERSRelationTypingAdapter()


__all__ = [
    "SERSRelationTypingAdapter",
    "SERS_RELATION_TYPING_ADAPTER",
]
