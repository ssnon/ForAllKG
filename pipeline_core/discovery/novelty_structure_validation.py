from __future__ import annotations

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimScientificStructure,
    NoveltyStructureBasis,
)


_DEFAULTS = {
    "inferential_distance": "LOCAL_REPHRASE",
    "mechanistic_necessity": "NO_NEW_MECHANISM",
    "regime_specificity": "NONE",
    "counterintuitiveness": "EXPECTED",
    "testable_distinctiveness": "GENERIC",
}


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _extractive(
    candidate: str,
    source_texts: list[str],
) -> bool:
    needle = _norm(candidate)
    if not needle:
        return False
    return any(
        needle in _norm(source)
        for source in source_texts
        if _norm(source)
    )


def _mentions_identity(
    text: str,
    identity_terms: list[str],
) -> bool:
    if not identity_terms:
        return True

    haystack = _norm(text)
    return any(
        _norm(term)
        and _norm(term) in haystack
        for term in identity_terms
    )


def compile_claim_scientific_structure(
    draft: NoveltyClaimScientificStructure,
    *,
    identity_terms: list[str],
    source_texts: list[str],
) -> tuple[
    NoveltyClaimScientificStructure,
    tuple[str, ...],
]:
    """Fail-closed compilation of LLM-declared atomic structure.

    Strong structural categories are accepted only when the supplied
    basis is:
      1. extractive from the source hypothesis specification; and
      2. branch-specific when an atomic prior-art identity exists.

    This compiler validates provenance and categorical coherence.
    It does not infer scientific structure from keywords or free text.
    """

    reasons: list[str] = []
    valid_basis: list[NoveltyStructureBasis] = []
    seen: set[tuple[str, str]] = set()

    for row in draft.basis:
        cleaned = " ".join(row.source_text.split())
        key = (row.feature, cleaned)

        if key in seen:
            continue
        seen.add(key)

        if not _extractive(cleaned, source_texts):
            reasons.append(
                f"structure_basis_not_extractive:{row.feature}"
            )
            continue

        if not _mentions_identity(
            cleaned,
            identity_terms,
        ):
            reasons.append(
                f"structure_basis_branch_identity_missing:{row.feature}"
            )
            continue

        valid_basis.append(
            NoveltyStructureBasis(
                feature=row.feature,
                source_text=cleaned,
            )
        )

    features = {
        row.feature
        for row in valid_basis
    }

    def supported(*allowed: str) -> bool:
        return bool(
            features.intersection(allowed)
        )

    update = {
        "basis": valid_basis,
    }

    flag_support = {
        "introduces_new_mechanism": "new_mechanism",
        "introduces_threshold": "threshold",
        "introduces_regime_change": "regime_change",
        "introduces_reversal": "reversal",
        "introduces_mechanism_switch": "mechanism_switch",
    }

    for attr, feature in flag_support.items():
        value = bool(getattr(draft, attr))
        if value and not supported(feature):
            update[attr] = False
            reasons.append(
                f"unsupported_structure_flag:{feature}"
            )

    inferential = draft.inferential_distance
    if (
        inferential != "LOCAL_REPHRASE"
        and not supported(
            "inferential_distance",
            "new_mechanism",
            "threshold",
            "regime_change",
            "reversal",
            "mechanism_switch",
        )
    ):
        update["inferential_distance"] = (
            _DEFAULTS["inferential_distance"]
        )
        reasons.append(
            "unsupported_structure_category:inferential_distance"
        )

    mechanistic = draft.mechanistic_necessity
    if (
        mechanistic != "NO_NEW_MECHANISM"
        and not supported(
            "mechanistic_necessity",
            "new_mechanism",
            "mechanism_switch",
        )
    ):
        update["mechanistic_necessity"] = (
            _DEFAULTS["mechanistic_necessity"]
        )
        reasons.append(
            "unsupported_structure_category:mechanistic_necessity"
        )

    regime = draft.regime_specificity
    if (
        regime != "NONE"
        and not supported(
            "regime_specificity",
            "threshold",
            "regime_change",
            "reversal",
            "mechanism_switch",
        )
    ):
        update["regime_specificity"] = (
            _DEFAULTS["regime_specificity"]
        )
        reasons.append(
            "unsupported_structure_category:regime_specificity"
        )

    counter = draft.counterintuitiveness
    if (
        counter != "EXPECTED"
        and not supported(
            "counterintuitiveness"
        )
    ):
        update["counterintuitiveness"] = (
            _DEFAULTS["counterintuitiveness"]
        )
        reasons.append(
            "unsupported_structure_category:counterintuitiveness"
        )

    distinct = draft.testable_distinctiveness
    if (
        distinct != "GENERIC"
        and not supported(
            "testable_distinctiveness"
        )
    ):
        update["testable_distinctiveness"] = (
            _DEFAULTS["testable_distinctiveness"]
        )
        reasons.append(
            "unsupported_structure_category:testable_distinctiveness"
        )

    result = draft.model_copy(
        update=update
    )

    # --------------------------------------------------------------
    # Categorical coherence. Fail closed; do not infer missing
    # strong categories from neighboring labels.
    # --------------------------------------------------------------

    coherence_update: dict[str, object] = {}

    if (
        result.introduces_threshold
        and result.regime_specificity != "THRESHOLD"
    ):
        coherence_update[
            "introduces_threshold"
        ] = False
        reasons.append(
            "incoherent_structure:threshold_without_threshold_regime"
        )

    if (
        result.regime_specificity == "THRESHOLD"
        and not result.introduces_threshold
    ):
        coherence_update[
            "regime_specificity"
        ] = "NONE"
        reasons.append(
            "incoherent_structure:threshold_regime_without_threshold_flag"
        )

    if (
        result.introduces_reversal
        and result.regime_specificity != "REVERSAL"
    ):
        coherence_update[
            "introduces_reversal"
        ] = False
        reasons.append(
            "incoherent_structure:reversal_without_reversal_regime"
        )

    if (
        result.regime_specificity == "REVERSAL"
        and not result.introduces_reversal
    ):
        coherence_update[
            "regime_specificity"
        ] = "NONE"
        reasons.append(
            "incoherent_structure:reversal_regime_without_reversal_flag"
        )

    if (
        result.introduces_mechanism_switch
        and (
            result.regime_specificity
            != "MECHANISM_SWITCH"
            or result.mechanistic_necessity
            != "MECHANISM_SWITCH_REQUIRED"
        )
    ):
        coherence_update[
            "introduces_mechanism_switch"
        ] = False
        reasons.append(
            "incoherent_structure:mechanism_switch_contract"
        )

    if (
        result.regime_specificity == "MECHANISM_SWITCH"
        and not result.introduces_mechanism_switch
    ):
        coherence_update[
            "regime_specificity"
        ] = "NONE"
        reasons.append(
            "incoherent_structure:mechanism_switch_regime_without_flag"
        )

    if (
        result.mechanistic_necessity
        == "MECHANISM_SWITCH_REQUIRED"
        and not result.introduces_mechanism_switch
    ):
        coherence_update[
            "mechanistic_necessity"
        ] = "NO_NEW_MECHANISM"
        reasons.append(
            "incoherent_structure:mechanism_switch_requirement_without_flag"
        )

    if (
        result.regime_specificity == "HYSTERESIS"
        and not result.introduces_regime_change
    ):
        coherence_update[
            "regime_specificity"
        ] = "NONE"
        reasons.append(
            "incoherent_structure:hysteresis_without_regime_change"
        )

    if (
        result.introduces_new_mechanism
        and result.mechanistic_necessity
        not in {
            "NEW_BRIDGE_REQUIRED",
            "MECHANISM_SWITCH_REQUIRED",
        }
    ):
        coherence_update[
            "introduces_new_mechanism"
        ] = False
        reasons.append(
            "incoherent_structure:new_mechanism_without_requirement"
        )

    result = result.model_copy(
        update=coherence_update
    )

    # NEW_REGIME_STRUCTURE should correspond to an explicit
    # higher-order regime structure.
    if (
        result.inferential_distance
        == "NEW_REGIME_STRUCTURE"
        and not (
            result.introduces_threshold
            or result.introduces_regime_change
            or result.introduces_reversal
            or result.introduces_mechanism_switch
        )
    ):
        result = result.model_copy(
            update={
                "inferential_distance":
                    "LOCAL_REPHRASE"
            }
        )
        reasons.append(
            "incoherent_structure:new_regime_distance_without_regime_structure"
        )

    return (
        result,
        tuple(dict.fromkeys(reasons)),
    )
