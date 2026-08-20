from __future__ import annotations

import re
from typing import Iterable


_SPECIFIC_DISTANCE_METRICS = {
    "interatomic_distance",
    "exafs_radial_peak_position",
    "fitted_scattering_path_length",
    "dft_optimized_bond_length",
}
_GENERIC_DISTANCE_METRICS = {
    "bond_distance",
    "bond_length",
    "interatomic_distance",
}


def refine_distance_metric_id(
    *,
    entry_id: str | None,
    label: str | None,
    source_texts: Iterable[str | None],
) -> str | None:
    """Refine generic distance metrics by determination method.

    The same angstrom-valued expression can denote a microscopy-derived
    interatomic separation, an uncorrected FT-EXAFS radial peak, a fitted
    EXAFS scattering-path length, or a DFT-optimized bond length. Those are
    not interchangeable and must remain separate metric families.
    """
    raw_id = str(entry_id or "").strip()
    if raw_id in _SPECIFIC_DISTANCE_METRICS and raw_id != "interatomic_distance":
        return raw_id

    raw_label = str(label or "").strip().lower()
    candidate = raw_id.lower()
    if not (
        candidate in _GENERIC_DISTANCE_METRICS
        or "bond distance" in raw_label
        or "bond length" in raw_label
        or "interatomic distance" in raw_label
    ):
        return entry_id

    text = " | ".join(
        str(value).strip().lower()
        for value in source_texts
        if value is not None and str(value).strip()
    )
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")

    fitted_terms = (
        "exafs-fitted",
        "exafs fitted",
        "quantitative exafs fit",
        "exafs fitting",
        "fitted path",
        "fitted distance",
        "scattering path",
        "coordination path",
        "fit-derived",
    )
    radial_terms = (
        "ft-exafs peak",
        "fourier-transform exafs peak",
        "fourier transform exafs peak",
        "radial peak",
        "r-space peak",
        "peak position",
    )
    dft_terms = (
        "dft",
        "density functional",
        "optimized geometry",
        "geometry optimization",
        "optimized bond",
        "calculated bond length",
        "computed bond length",
    )
    microscopy_terms = (
        "haadf",
        "stem",
        "tem",
        "microscopy",
        "atom pairs",
        "atomic pairs",
        "statistical analysis",
        "directly imaged",
    )

    if any(term in text for term in fitted_terms):
        return "fitted_scattering_path_length"
    if any(term in text for term in radial_terms):
        return "exafs_radial_peak_position"
    if any(term in text for term in dft_terms):
        return "dft_optimized_bond_length"
    if any(term in text for term in microscopy_terms):
        return "interatomic_distance"
    return entry_id



def refine_semantic_metric_id(
    *,
    entry_id: str,
    label: str,
    source_texts: Iterable[Any],
) -> str:
    """Correct high-confidence metric/category mismatches before registry lookup."""
    text = " | ".join(
        str(value).strip().lower()
        for value in source_texts
        if value is not None and str(value).strip()
    )
    raw_label = str(label or "").strip().lower()
    combined = f"{raw_label} | {text}"

    if re.search(r"\b(?:average\s+)?oxidation state\b|\bvalence state\b", combined):
        return "oxidation_state"
    if re.search(r"\bepr\b.*\bg\s*(?:=|value|factor)\b|\bg\s*=\s*2\.", combined):
        return "epr_g_factor"
    if "pcohp" in combined and "antibond" in combined and "energy" in combined:
        return "pcohp_antibonding_state_energy"
    if re.search(r"\bcoordination number\b|\bcn\s*(?:=|of)\b", combined):
        return "coordination_number"
    return refine_distance_metric_id(
        entry_id=entry_id,
        label=label,
        source_texts=source_texts,
    )
