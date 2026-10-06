from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisPortfolio,
)

from domains.sers.fdtd_applicability_contracts import (
    SERSFDTDApplicabilityBundle,
    SERSFDTDApplicabilityReport,
    SERSFDTDSubclaim,
)


_ANALYZER_VERSION = "sers-fdtd-applicability-analyzer-v0"


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(_canonical(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:20]}"


def _normalize(text: str) -> str:
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class _TextChunk:
    field: str
    text: str


def _chunks(card: HypothesisCard) -> list[_TextChunk]:
    rows = [
        _TextChunk("title", card.title),
        _TextChunk("hypothesis_statement", card.hypothesis_statement),
        _TextChunk("inferential_bridge", card.inferential_bridge),
    ]
    rows.extend(
        _TextChunk(
            f"predicted_observations[{index}].observable",
            row.observable,
        )
        for index, row in enumerate(card.predicted_observations)
    )
    rows.extend(
        _TextChunk(
            f"predicted_observations[{index}].rationale",
            row.rationale,
        )
        for index, row in enumerate(card.predicted_observations)
    )
    rows.extend(
        _TextChunk(f"assumptions[{index}]", text)
        for index, text in enumerate(card.assumptions)
    )
    return rows


def _surface(chunks: list[_TextChunk]) -> str:
    return _normalize(" ".join(row.text for row in chunks))


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in patterns
    )


def _matching_fields(
    chunks: list[_TextChunk],
    patterns: tuple[str, ...],
) -> list[str]:
    return sorted({
        row.field
        for row in chunks
        if _matches(row.text, patterns)
    })


def _extract_baseline_wavelength_nm(
    chunks: list[_TextChunk],
) -> float | None:
    patterns = (
        r"substrate[- ]only.{0,40}?(?P<value>\d+(?:\.\d+)?)\s*nm",
        r"(?P<value>\d+(?:\.\d+)?)\s*nm.{0,40}?"
        r"(?:resonance[- ]matching|baseline|reference)\s+choice",
        r"(?:baseline|reference).{0,30}?"
        r"(?P<value>\d+(?:\.\d+)?)\s*nm",
    )
    for chunk in chunks:
        text = _normalize(chunk.text)
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return float(match.group("value"))
    return None


def _detect_analytes(surface: str) -> list[str]:
    rows: list[str] = []
    if re.search(r"\bmethylene\s+blue\b", surface, flags=re.IGNORECASE):
        rows.append("methylene blue")
    if re.search(r"\b4[- ]?mercaptobenzoic\s+acid\b", surface, flags=re.IGNORECASE):
        rows.append("4-mercaptobenzoic acid")
    if re.search(r"\brhodamine\s+6g\b", surface, flags=re.IGNORECASE):
        rows.append("rhodamine 6G")
    return rows


def _detect_geometry_terms(surface: str) -> list[str]:
    rows: list[str] = []
    if _matches(surface, (r"\bcore[- ]?shell\b", r"\bau\s*@\s*ag\b", r"\bag\s*@\s*au\b")):
        rows.append("core_shell")
    if re.search(r"\bnanorods?\b", surface, flags=re.IGNORECASE):
        rows.append("nanorod")
    if re.search(r"\bdimer\b", surface, flags=re.IGNORECASE):
        rows.append("dimer")
    if re.search(r"\bnanospheres?\b", surface, flags=re.IGNORECASE):
        rows.append("nanosphere")
    if _matches(surface, (r"\baao\b", r"\bdimples?\b", r"\bnanopores?\b")):
        rows.append("substrate_topography")
    return rows


class SERSFDTDApplicabilityAnalyzer:
    """Deterministically split SERS hypotheses by FDTD applicability.

    This layer decides only whether classical electromagnetic FDTD can address
    all, part, or none of the stated hypothesis. It does not decide scientific
    truth and does not turn non-FDTD subclaims into negative evidence.
    """

    analyzer_version = _ANALYZER_VERSION

    _EM_PATTERNS = (
        r"\belectromagnetic\b",
        r"\bnear[- ]field\b",
        r"\bhot\s*spots?\b",
        r"\blspr\b",
        r"localized\s+surface\s+plasmon",
        r"\bplasmon(?:ic)?\b",
        r"\bsubstrate[- ]resonance\b",
        r"\bresonance[- ]matching\b",
        r"\bfield\s+enhancement\b",
    )

    _NON_FDTD_PATTERNS = (
        r"\banalyte[- ]specific\b",
        r"\bresonance\s+raman\b",
        r"\braman\s+cross[- ]section\b",
        r"\bchemical\s+enhancement\b",
        r"\bcharge\s+transfer\b",
        r"\badsorption\b",
        r"\bmolecular\s+orientation\b",
    )

    _SERS_INTENSITY_PATTERNS = (
        r"\bsers\s+intensity\b",
        r"\braman\s+intensity\b",
        r"\bsers\s+(?:signal|response)\b",
    )

    _WAVELENGTH_AXIS_PATTERNS = (
        r"\bexcitation\s+wavelength\b",
        r"\bwavelength[- ]depend",
        r"\bspectral\s+response\b",
        r"\bintensity\s+maximum\b",
        r"\bmaximum.{0,30}\bwavelength\b",
        r"\bwavelength.{0,30}\bmaxim",
        r"\bshift(?:ed)?\b.{0,40}\b785\s*nm\b",
    )

    _GAP_AXIS_PATTERNS = (
        r"\breduc(?:e|es|ed|ing)\b.{0,30}\bgap\b",
        r"\bincreas(?:e|es|ed|ing)\b.{0,30}\bgap\b",
        r"\bsmaller\s+(?:interparticle\s+)?gap\b",
        r"\blarger\s+(?:interparticle\s+)?gap\b",
        r"\bgap[- ]depend",
    )

    def analyze_portfolio(
        self,
        portfolio: HypothesisPortfolio,
        *,
        hypothesis_ids: set[str] | None = None,
    ) -> SERSFDTDApplicabilityBundle:
        if portfolio.domain_profile_id != "sers_au_ag":
            raise ValueError(
                "SERS FDTD applicability analyzer requires "
                "domain_profile_id='sers_au_ag'"
            )

        cards = [
            card
            for card in portfolio.hypotheses
            if hypothesis_ids is None
            or card.hypothesis_id in hypothesis_ids
        ]

        if hypothesis_ids is not None:
            found = {row.hypothesis_id for row in cards}
            missing = sorted(hypothesis_ids - found)
            if missing:
                raise ValueError(
                    "requested hypothesis_id(s) absent from portfolio: "
                    + ", ".join(missing)
                )

        reports = [
            self.analyze_card(
                card,
                source_portfolio_id=portfolio.portfolio_id,
                domain_profile_id=portfolio.domain_profile_id,
            )
            for card in cards
        ]

        return SERSFDTDApplicabilityBundle(
            bundle_id=_stable_id(
                "sers_fdtd_applicability_bundle",
                portfolio.portfolio_id,
                *(row.report_id for row in reports),
                self.analyzer_version,
            ),
            source_portfolio_id=portfolio.portfolio_id,
            domain_profile_id=portfolio.domain_profile_id,
            reports=reports,
        )

    def analyze_card(
        self,
        card: HypothesisCard,
        *,
        source_portfolio_id: str,
        domain_profile_id: str,
    ) -> SERSFDTDApplicabilityReport:
        if domain_profile_id != "sers_au_ag":
            raise ValueError(
                "SERS FDTD applicability analyzer requires "
                "domain_profile_id='sers_au_ag'"
            )
        if card.domain_profile_id != domain_profile_id:
            raise ValueError(
                "hypothesis/domain mismatch: "
                f"{card.hypothesis_id} has {card.domain_profile_id!r}"
            )

        chunks = _chunks(card)
        surface = _surface(chunks)

        has_em = _matches(surface, self._EM_PATTERNS)
        has_non_fdtd = _matches(surface, self._NON_FDTD_PATTERNS)
        has_sers_intensity = _matches(
            surface,
            self._SERS_INTENSITY_PATTERNS,
        )
        has_wavelength_axis = _matches(
            surface,
            self._WAVELENGTH_AXIS_PATTERNS,
        )
        has_gap_axis = _matches(surface, self._GAP_AXIS_PATTERNS)

        analytes = _detect_analytes(surface)
        geometry_terms = _detect_geometry_terms(surface)
        baseline_wavelength_nm = _extract_baseline_wavelength_nm(chunks)

        study_axes: list[str] = []
        if has_wavelength_axis:
            study_axes.append("wavelength")
        if has_gap_axis:
            study_axes.append("gap")

        fdtd_subclaims: list[SERSFDTDSubclaim] = []
        non_fdtd_subclaims: list[SERSFDTDSubclaim] = []

        if has_em or (has_wavelength_axis and geometry_terms):
            target_observables = [
                "electromagnetic near-field enhancement",
                "substrate electromagnetic spectral response",
            ]
            if has_gap_axis:
                target_observables.append("gap-dependent field localization")
            fdtd_subclaims.append(
                SERSFDTDSubclaim(
                    subclaim_id=_stable_id(
                        "sers_fdtd_subclaim",
                        card.hypothesis_id,
                        "fdtd_computable",
                        tuple(study_axes),
                        tuple(geometry_terms),
                    ),
                    kind="fdtd_computable",
                    statement=(
                        "Evaluate the substrate-side classical electromagnetic "
                        "response over the hypothesis-relevant design or "
                        "excitation axis."
                    ),
                    rationale=(
                        "Classical FDTD can resolve substrate geometry/material "
                        "dependent electromagnetic fields and resonance trends."
                    ),
                    target_observables=target_observables,
                    source_fields=_matching_fields(
                        chunks,
                        self._EM_PATTERNS + self._WAVELENGTH_AXIS_PATTERNS
                        + self._GAP_AXIS_PATTERNS,
                    ),
                )
            )

        # SERS intensity is not automatically identified with the FDTD EM proxy.
        # When the hypothesis attributes a total-SERS shift to analyte-specific,
        # chemical, adsorption, orientation, or Raman-response physics, preserve
        # that portion outside the FDTD subclaim.
        if has_non_fdtd or (has_sers_intensity and bool(analytes)):
            non_fdtd_subclaims.append(
                SERSFDTDSubclaim(
                    subclaim_id=_stable_id(
                        "sers_fdtd_subclaim",
                        card.hypothesis_id,
                        "non_fdtd",
                        tuple(analytes),
                    ),
                    kind="non_fdtd",
                    statement=(
                        "Resolve the analyte/reporting contribution to the "
                        "observed SERS response separately from the substrate "
                        "electromagnetic response."
                    ),
                    rationale=(
                        "Analyte-specific Raman response, chemical enhancement, "
                        "charge transfer, adsorption, and molecular orientation "
                        "are not established by a classical substrate-only FDTD "
                        "calculation."
                    ),
                    target_observables=[
                        "analyte-specific SERS spectral response",
                        "total measured SERS intensity optimum",
                    ],
                    source_fields=_matching_fields(
                        chunks,
                        self._NON_FDTD_PATTERNS
                        + self._SERS_INTENSITY_PATTERNS,
                    ),
                )
            )

        if fdtd_subclaims and non_fdtd_subclaims:
            applicability = "partial"
            rationale = (
                "The hypothesis contains a substrate electromagnetic component "
                "that FDTD can address, but the full stated SERS outcome also "
                "depends on non-FDTD analyte/reporting physics."
            )
        elif fdtd_subclaims:
            applicability = "direct"
            rationale = (
                "The hypothesis target is represented as a classical substrate "
                "electromagnetic response that FDTD can address directly."
            )
        elif non_fdtd_subclaims:
            applicability = "not_applicable"
            rationale = (
                "The hypothesis target is dominated by analyte/chemical/Raman "
                "physics that a classical substrate-only FDTD calculation does "
                "not establish."
            )
        else:
            applicability = "requires_interpretation"
            rationale = (
                "The hypothesis does not expose a sufficiently explicit "
                "electromagnetic or non-FDTD target for deterministic routing."
            )

        return SERSFDTDApplicabilityReport(
            report_id=_stable_id(
                "sers_fdtd_applicability",
                source_portfolio_id,
                card.hypothesis_id,
                applicability,
                tuple(study_axes),
                baseline_wavelength_nm,
                self.analyzer_version,
            ),
            source_portfolio_id=source_portfolio_id,
            hypothesis_id=card.hypothesis_id,
            domain_profile_id=domain_profile_id,
            applicability=applicability,  # type: ignore[arg-type]
            rationale=rationale,
            fdtd_subclaims=fdtd_subclaims,
            non_fdtd_subclaims=non_fdtd_subclaims,
            study_axes=study_axes,  # type: ignore[arg-type]
            baseline_wavelength_nm=baseline_wavelength_nm,
            detected_analytes=analytes,
            detected_geometry_terms=geometry_terms,
        )
