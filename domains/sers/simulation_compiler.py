from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisPortfolio,
)

from domains.sers.fdtd_applicability import (
    SERSFDTDApplicabilityAnalyzer,
)
from domains.sers.fdtd_applicability_contracts import (
    SERSFDTDApplicabilityBundle,
    SERSFDTDApplicabilityReport,
)
from domains.sers.simulation_contracts import (
    SERSParameterProvenance,
    SERSSimulationCompilationBundle,
    SERSSimulationSpec,
    SERSSweepDefinition,
)


_COMPILER_VERSION = "sers-simulation-spec-compiler-v0.1"


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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize(text: str) -> str:
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"\s+", " ", text).strip()


def _number() -> str:
    return r"(?P<value>\d+(?:\.\d+)?)"


@dataclass(frozen=True)
class _TextChunk:
    field: str
    text: str


def _hypothesis_chunks(card: HypothesisCard) -> list[_TextChunk]:
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


def _find_number(
    chunks: list[_TextChunk],
    patterns: tuple[str, ...],
) -> tuple[float, _TextChunk, str] | None:
    for chunk in chunks:
        text = _normalize(chunk.text)
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return float(match.group("value")), chunk, match.group(0)
    return None




def _find_reported_resonance_wavelengths(
    chunks: list[_TextChunk],
) -> list[tuple[float, _TextChunk, str]]:
    """Extract only wavelengths explicitly described as resonance positions.

    This is intentionally narrower than generic wavelength extraction so a
    phrase such as "resonances at 512 and 772 nm ... match 785 nm excitation"
    yields 512/772 as resonance landmarks while leaving 785 as the comparison
    or excitation reference.
    """

    rows: list[tuple[float, _TextChunk, str]] = []
    seen: set[float] = set()
    pair_pattern = re.compile(
        r"\bresonances?\s+(?:at|near|around)\s+"
        r"(?P<a>\d+(?:\.\d+)?)\s*(?:nm)?\s*"
        r"(?:and|,)\s*(?P<b>\d+(?:\.\d+)?)\s*nm\b",
        flags=re.IGNORECASE,
    )
    single_pattern = re.compile(
        r"\bresonance\s+(?:at|near|around)\s+"
        r"(?P<a>\d+(?:\.\d+)?)\s*nm\b",
        flags=re.IGNORECASE,
    )

    for chunk in chunks:
        text = _normalize(chunk.text)
        for match in pair_pattern.finditer(text):
            for key in ("a", "b"):
                value = float(match.group(key))
                if value not in seen:
                    rows.append((value, chunk, match.group(0)))
                    seen.add(value)
        for match in single_pattern.finditer(text):
            value = float(match.group("a"))
            if value not in seen:
                rows.append((value, chunk, match.group(0)))
                seen.add(value)
    return rows

def _provenance(
    *,
    parameter: str,
    source_kind: str,
    chunk: _TextChunk,
    source_text: str,
    derivation: str | None = None,
) -> SERSParameterProvenance:
    return SERSParameterProvenance(
        parameter=parameter,
        source_kind=source_kind,  # type: ignore[arg-type]
        source_field=chunk.field,
        source_text=source_text,
        derivation=derivation,
    )


def _contains(text: str, patterns: tuple[str, ...]) -> bool:
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in patterns
    )


def _first_chunk(
    chunks: list[_TextChunk],
    pattern: str,
) -> _TextChunk:
    return next(
        (
            row
            for row in chunks
            if re.search(pattern, row.text, flags=re.IGNORECASE)
        ),
        chunks[0],
    )


class SERSHypothesisSimulationCompiler:
    """Conservative deterministic compiler from SERS hypothesis to FDTD intent.

    The compiler consumes an explicit FDTD-applicability report. If the caller
    does not provide one, the deterministic applicability analyzer is invoked.
    Missing physical parameters remain missing rather than being invented.
    """

    compiler_version = _COMPILER_VERSION

    _DIMER_PATTERNS = (
        r"\bdimer\b",
        r"\bparticle\s+pair\b",
        r"\bnanoparticle\s+pair\b",
        r"\bpair\s+of\s+(?:\w+\s+){0,3}(?:nano)?particles?\b",
        r"\btwo\s+(?:\w+\s+){0,3}nanospheres?\b",
    )

    _SPHERE_PATTERNS = (
        r"\bnanospheres?\b",
        r"\bspherical\s+(?:nano)?particles?\b",
        r"\bspheres?\b",
    )

    _NANOROD_PATTERNS = (
        r"\bnanorods?\b",
        r"\bnano[- ]?rods?\b",
    )

    _CORE_SHELL_PATTERNS = (
        r"\bcore[- ]?shell\b",
        r"\bau\s*@\s*ag\b",
        r"\bag\s*@\s*au\b",
    )

    _TOPOGRAPHY_PATTERNS = (
        r"\baao\b",
        r"\banodic\s+alum(?:ina|inum\s+oxide)\b",
        r"\bdimples?\b",
        r"\bnanopores?\b",
    )

    _GAP_SWEEP_PATTERNS = (
        r"\breduc(?:e|es|ed|ing)\b.{0,30}\bgap\b",
        r"\bincreas(?:e|es|ed|ing)\b.{0,30}\bgap\b",
        r"\bsmaller\s+(?:interparticle\s+)?gap\b",
        r"\blarger\s+(?:interparticle\s+)?gap\b",
        r"\bvary(?:ing)?\b.{0,20}\bgap\b",
        r"\bgap[- ]depend",
        r"\bas\s+(?:the\s+)?gap\b",
    )

    def compile_portfolio(
        self,
        portfolio: HypothesisPortfolio,
        *,
        hypothesis_ids: set[str] | None = None,
        applicability_bundle: SERSFDTDApplicabilityBundle | None = None,
    ) -> SERSSimulationCompilationBundle:
        if portfolio.domain_profile_id != "sers_au_ag":
            raise ValueError(
                "SERS simulation compiler requires "
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

        if applicability_bundle is None:
            applicability_bundle = (
                SERSFDTDApplicabilityAnalyzer().analyze_portfolio(
                    portfolio,
                    hypothesis_ids=hypothesis_ids,
                )
            )

        if applicability_bundle.source_portfolio_id != portfolio.portfolio_id:
            raise ValueError("applicability bundle/source portfolio mismatch")
        if applicability_bundle.domain_profile_id != portfolio.domain_profile_id:
            raise ValueError("applicability bundle/domain mismatch")

        applicability_by_id = {
            row.hypothesis_id: row
            for row in applicability_bundle.reports
        }

        specs: list[SERSSimulationSpec] = []
        for card in cards:
            applicability = applicability_by_id.get(card.hypothesis_id)
            if applicability is None:
                raise ValueError(
                    "missing FDTD applicability report for "
                    f"{card.hypothesis_id}"
                )
            specs.append(
                self.compile_card(
                    card,
                    applicability=applicability,
                    source_portfolio_id=portfolio.portfolio_id,
                    domain_profile_id=portfolio.domain_profile_id,
                )
            )

        return SERSSimulationCompilationBundle(
            bundle_id=_stable_id(
                "sers_simulation_bundle",
                portfolio.portfolio_id,
                applicability_bundle.bundle_id,
                *(row.spec_id for row in specs),
                self.compiler_version,
            ),
            source_portfolio_id=portfolio.portfolio_id,
            source_applicability_bundle_id=applicability_bundle.bundle_id,
            domain_profile_id=portfolio.domain_profile_id,
            specs=specs,
        )

    def compile_card(
        self,
        card: HypothesisCard,
        *,
        applicability: SERSFDTDApplicabilityReport,
        source_portfolio_id: str,
        domain_profile_id: str,
    ) -> SERSSimulationSpec:
        if domain_profile_id != "sers_au_ag":
            raise ValueError(
                "SERS simulation compiler requires "
                "domain_profile_id='sers_au_ag'"
            )
        if card.domain_profile_id != domain_profile_id:
            raise ValueError(
                "hypothesis/domain mismatch: "
                f"{card.hypothesis_id} has {card.domain_profile_id!r}"
            )
        if applicability.hypothesis_id != card.hypothesis_id:
            raise ValueError("applicability/hypothesis mismatch")

        chunks = _hypothesis_chunks(card)
        surface = _surface(chunks)
        surface_lower = surface.lower()

        provenance: list[SERSParameterProvenance] = []
        unsupported: list[str] = []
        notes: list[str] = []

        has_core_shell = _contains(surface, self._CORE_SHELL_PATTERNS)
        has_nanorod = _contains(surface, self._NANOROD_PATTERNS)
        has_dimer = _contains(surface, self._DIMER_PATTERNS)
        has_sphere = _contains(surface, self._SPHERE_PATTERNS)

        if _contains(surface, self._TOPOGRAPHY_PATTERNS):
            unsupported.append("substrate_topography")

        if has_core_shell and has_nanorod:
            geometry_family = "core_shell_nanorod"
        elif has_dimer and has_sphere and not has_core_shell:
            geometry_family = "sphere_dimer"
        else:
            geometry_family = None

        if has_core_shell and has_sphere and has_dimer:
            unsupported.append("core_shell_sphere_dimer")
        if has_dimer and not has_sphere and not has_nanorod:
            notes.append(
                "dimer intent detected but particle shape is not explicit"
            )
        if has_sphere and not has_dimer and not has_core_shell:
            notes.append(
                "spherical particle intent detected but dimer/pair geometry "
                "is not explicit"
            )

        has_au = bool(
            re.search(r"\b(?:Au|gold)\b", surface, flags=re.IGNORECASE)
        )
        has_ag = bool(
            re.search(r"\b(?:Ag|silver)\b", surface, flags=re.IGNORECASE)
        )

        left_material = None
        right_material = None
        core_material = None
        shell_material = None

        au_ag = bool(
            re.search(r"\bau\s*@\s*ag\b", surface, flags=re.IGNORECASE)
        )
        ag_au = bool(
            re.search(r"\bag\s*@\s*au\b", surface, flags=re.IGNORECASE)
        )

        if geometry_family == "core_shell_nanorod":
            if au_ag:
                core_material = "Au"
                shell_material = "Ag"
            elif ag_au:
                core_material = "Ag"
                shell_material = "Au"
            elif has_au and has_ag:
                notes.append(
                    "core-shell Au/Ag materials detected but core/shell order "
                    "is not explicit"
                )

            if core_material is not None:
                material_chunk = _first_chunk(
                    chunks,
                    r"\b(?:Au\s*@\s*Ag|Ag\s*@\s*Au|core[- ]?shell)\b",
                )
                provenance.append(
                    _provenance(
                        parameter="core_shell_materials",
                        source_kind="hypothesis_text",
                        chunk=material_chunk,
                        source_text=material_chunk.text,
                    )
                )
        elif geometry_family == "sphere_dimer":
            if has_au and not has_ag:
                left_material = right_material = "Au"
            elif has_ag and not has_au:
                left_material = right_material = "Ag"
            elif has_au and has_ag:
                left_material = "Au"
                right_material = "Ag"

            if left_material is not None:
                material_chunk = _first_chunk(
                    chunks,
                    r"\b(?:Au|Ag|gold|silver)\b",
                )
                provenance.append(
                    _provenance(
                        parameter="material_pair",
                        source_kind="hypothesis_text",
                        chunk=material_chunk,
                        source_text=material_chunk.text,
                    )
                )

        gap_match = _find_number(
            chunks,
            (
                rf"\b{_number()}\s*nm(?:\s+wide)?\s+"
                r"(?:interparticle\s+)?(?:nano[- ]?)?gap\b",
                r"\b(?:interparticle\s+)?(?:nano[- ]?)?gap"
                rf"(?:\s+(?:of|=|:))?\s*{_number()}\s*nm\b",
            ),
        )
        gap_nm = None
        if gap_match:
            gap_nm, chunk, excerpt = gap_match
            provenance.append(
                _provenance(
                    parameter="gap_nm",
                    source_kind="hypothesis_text",
                    chunk=chunk,
                    source_text=excerpt,
                )
            )

        radius_match = _find_number(
            chunks,
            (
                rf"\b{_number()}\s*nm[- ]radius\b",
                rf"\b{_number()}\s*nm\s+(?:particle\s+)?radius\b",
                r"\bradius(?:\s+(?:of|=|:))?\s*"
                rf"{_number()}\s*nm\b",
            ),
        )
        diameter_match = _find_number(
            chunks,
            (
                rf"\b{_number()}\s*nm[- ]diameter\b",
                rf"\b{_number()}\s*nm\s+(?:particle\s+)?diameter\b",
                r"\bdiameter(?:\s+(?:of|=|:))?\s*"
                rf"{_number()}\s*nm\b",
            ),
        )

        particle_radius_nm = None
        if geometry_family == "sphere_dimer":
            if radius_match:
                particle_radius_nm, chunk, excerpt = radius_match
                provenance.append(
                    _provenance(
                        parameter="particle_radius_nm",
                        source_kind="hypothesis_text",
                        chunk=chunk,
                        source_text=excerpt,
                    )
                )
            elif diameter_match:
                diameter_nm, chunk, excerpt = diameter_match
                particle_radius_nm = diameter_nm / 2.0
                provenance.append(
                    _provenance(
                        parameter="particle_radius_nm",
                        source_kind="derived_hypothesis_text",
                        chunk=chunk,
                        source_text=excerpt,
                        derivation="particle_radius_nm = diameter_nm / 2",
                    )
                )

        nanorod_length_match = _find_number(
            chunks,
            (
                rf"\b{_number()}\s*nm[- ]long\b",
                r"\b(?:nano)?rod\s+length(?:\s+(?:of|=|:))?\s*"
                rf"{_number()}\s*nm\b",
                r"\blength(?:\s+(?:of|=|:))?\s*"
                rf"{_number()}\s*nm\b",
            ),
        )
        nanorod_diameter_match = _find_number(
            chunks,
            (
                r"\b(?:nano)?rod\s+diameter(?:\s+(?:of|=|:))?\s*"
                rf"{_number()}\s*nm\b",
                rf"\b{_number()}\s*nm[- ]diameter\s+nanorod\b",
            ),
        )
        shell_thickness_match = _find_number(
            chunks,
            (
                r"\bshell\s+thickness(?:\s+(?:of|=|:))?\s*"
                rf"{_number()}\s*nm\b",
                rf"\b{_number()}\s*nm[- ]thick\s+shell\b",
            ),
        )

        nanorod_length_nm = None
        nanorod_diameter_nm = None
        shell_thickness_nm = None
        for parameter, match in (
            ("nanorod_length_nm", nanorod_length_match),
            ("nanorod_diameter_nm", nanorod_diameter_match),
            ("shell_thickness_nm", shell_thickness_match),
        ):
            if match is None or geometry_family != "core_shell_nanorod":
                continue
            value, chunk, excerpt = match
            if parameter == "nanorod_length_nm":
                nanorod_length_nm = value
            elif parameter == "nanorod_diameter_nm":
                nanorod_diameter_nm = value
            else:
                shell_thickness_nm = value
            provenance.append(
                _provenance(
                    parameter=parameter,
                    source_kind="hypothesis_text",
                    chunk=chunk,
                    source_text=excerpt,
                )
            )

        reported_resonance_matches = _find_reported_resonance_wavelengths(chunks)
        reported_resonance_wavelengths_nm = [
            value for value, _, _ in reported_resonance_matches
        ]
        for value, chunk, excerpt in reported_resonance_matches:
            provenance.append(
                _provenance(
                    parameter="reported_resonance_wavelengths_nm",
                    source_kind="hypothesis_text",
                    chunk=chunk,
                    source_text=excerpt,
                    derivation=(
                        "Source-derived spectral calibration landmark: "
                        f"{value:g} nm"
                    ),
                )
            )

        baseline_wavelength_nm = applicability.baseline_wavelength_nm
        if baseline_wavelength_nm is not None:
            provenance.append(
                SERSParameterProvenance(
                    parameter="baseline_wavelength_nm",
                    source_kind="applicability_report",
                    source_field="applicability.baseline_wavelength_nm",
                    source_text=(
                        "baseline/reference wavelength detected from "
                        f"hypothesis text: {baseline_wavelength_nm:g} nm"
                    ),
                    derivation=(
                        "Comparison anchor only; not treated as a fixed "
                        "excitation wavelength for a wavelength sweep."
                    ),
                )
            )

        wavelength_match = _find_number(
            chunks,
            (
                rf"\b{_number()}\s*nm\s+"
                r"(?:excitation|laser|illumination|wavelength)\b",
                r"\b(?:excitation|laser|illumination|wavelength)"
                rf"(?:\s+(?:at|of|=|:))?\s*{_number()}\s*nm\b",
            ),
        )
        excitation_wavelength_nm = None
        is_wavelength_sweep = "wavelength" in applicability.study_axes
        if wavelength_match and not is_wavelength_sweep:
            excitation_wavelength_nm, chunk, excerpt = wavelength_match
            provenance.append(
                _provenance(
                    parameter="excitation_wavelength_nm",
                    source_kind="hypothesis_text",
                    chunk=chunk,
                    source_text=excerpt,
                )
            )

        polarization = None
        polarization_match = None
        if geometry_family == "sphere_dimer":
            if re.search(
                r"\b(?:parallel|longitudinal)\b.{0,30}\b(?:dimer\s+)?axis\b",
                surface,
                flags=re.IGNORECASE,
            ):
                polarization = "parallel_to_dimer_axis"
                polarization_match = "parallel/longitudinal to dimer axis"
            elif re.search(
                r"\b(?:perpendicular|transverse)\b.{0,30}\b(?:dimer\s+)?axis\b",
                surface,
                flags=re.IGNORECASE,
            ):
                polarization = "perpendicular_to_dimer_axis"
                polarization_match = "perpendicular/transverse to dimer axis"
        elif geometry_family == "core_shell_nanorod":
            if re.search(
                r"\b(?:parallel|longitudinal)\b.{0,35}\b(?:nanorod|rod|long)\s+axis\b",
                surface,
                flags=re.IGNORECASE,
            ) or re.search(r"\blongitudinal\s+polarization\b", surface, flags=re.IGNORECASE):
                polarization = "parallel_to_nanorod_long_axis"
                polarization_match = "parallel/longitudinal to nanorod long axis"
            elif re.search(
                r"\b(?:perpendicular|transverse)\b.{0,35}\b(?:nanorod|rod|long)\s+axis\b",
                surface,
                flags=re.IGNORECASE,
            ) or re.search(r"\btransverse\s+polarization\b", surface, flags=re.IGNORECASE):
                polarization = "perpendicular_to_nanorod_long_axis"
                polarization_match = "perpendicular/transverse to nanorod long axis"

        if polarization is not None:
            chunk = _first_chunk(
                chunks,
                r"\b(?:parallel|longitudinal|perpendicular|transverse)\b",
            )
            provenance.append(
                _provenance(
                    parameter="polarization",
                    source_kind="hypothesis_text",
                    chunk=chunk,
                    source_text=polarization_match or chunk.text,
                )
            )

        surrounding_medium = None
        medium_chunk = None
        if re.search(r"\b(?:water|aqueous)\b", surface, flags=re.IGNORECASE):
            surrounding_medium = "water"
            medium_chunk = _first_chunk(chunks, r"\b(?:water|aqueous)\b")
        elif re.search(r"\bair\b", surface, flags=re.IGNORECASE):
            surrounding_medium = "air"
            medium_chunk = _first_chunk(chunks, r"\bair\b")

        if surrounding_medium is not None and medium_chunk is not None:
            provenance.append(
                _provenance(
                    parameter="surrounding_medium",
                    source_kind="hypothesis_text",
                    chunk=medium_chunk,
                    source_text=medium_chunk.text,
                )
            )

        is_gap_sweep = (
            "gap" in applicability.study_axes
            or _contains(surface, self._GAP_SWEEP_PATTERNS)
        )

        if is_wavelength_sweep:
            study_kind = "wavelength_sweep"
            sweep = SERSSweepDefinition(
                parameter="excitation_wavelength_nm",
                values=[],
            )
            notes.append(
                "wavelength dependence detected; v0.1 preserves the "
                "baseline wavelength separately and does not invent sweep bounds"
            )
        elif is_gap_sweep:
            study_kind = "parameter_sweep"
            sweep = SERSSweepDefinition(parameter="gap_nm", values=[])
            notes.append(
                "gap dependence detected; v0.1 does not invent sweep bounds "
                "or sweep values"
            )
        else:
            study_kind = "single_candidate"
            sweep = None

        spec_payload = {
            "source_portfolio_id": source_portfolio_id,
            "source_applicability_report_id": applicability.report_id,
            "hypothesis_id": card.hypothesis_id,
            "domain_profile_id": domain_profile_id,
            "fdtd_applicability": applicability.applicability,
            "study_axes": list(applicability.study_axes),
            "study_kind": study_kind,
            "geometry_family": geometry_family,
            "left_material": left_material,
            "right_material": right_material,
            "core_material": core_material,
            "shell_material": shell_material,
            "particle_radius_nm": particle_radius_nm,
            "gap_nm": gap_nm,
            "nanorod_length_nm": nanorod_length_nm,
            "nanorod_diameter_nm": nanorod_diameter_nm,
            "shell_thickness_nm": shell_thickness_nm,
            "baseline_wavelength_nm": baseline_wavelength_nm,
            "reported_resonance_wavelengths_nm": reported_resonance_wavelengths_nm,
            "excitation_wavelength_nm": excitation_wavelength_nm,
            "polarization": polarization,
            "surrounding_medium": surrounding_medium,
            "sweep": sweep.model_dump(mode="json") if sweep is not None else None,
            "unsupported_features": sorted(set(unsupported)),
        }

        return SERSSimulationSpec(
            spec_id=_stable_id(
                "sers_simulation_spec",
                card.hypothesis_id,
                spec_payload,
                self.compiler_version,
            ),
            source_portfolio_id=source_portfolio_id,
            source_applicability_report_id=applicability.report_id,
            hypothesis_id=card.hypothesis_id,
            domain_profile_id=domain_profile_id,
            source_hypothesis_text_sha256=_sha256_text(surface_lower),
            fdtd_applicability=applicability.applicability,
            fdtd_subclaim_ids=[
                row.subclaim_id for row in applicability.fdtd_subclaims
            ],
            non_fdtd_subclaim_ids=[
                row.subclaim_id for row in applicability.non_fdtd_subclaims
            ],
            study_axes=list(applicability.study_axes),
            study_kind=study_kind,  # type: ignore[arg-type]
            geometry_family=geometry_family,  # type: ignore[arg-type]
            left_material=left_material,  # type: ignore[arg-type]
            right_material=right_material,  # type: ignore[arg-type]
            particle_radius_nm=particle_radius_nm,
            gap_nm=gap_nm,
            core_material=core_material,  # type: ignore[arg-type]
            shell_material=shell_material,  # type: ignore[arg-type]
            nanorod_length_nm=nanorod_length_nm,
            nanorod_diameter_nm=nanorod_diameter_nm,
            shell_thickness_nm=shell_thickness_nm,
            baseline_wavelength_nm=baseline_wavelength_nm,
            reported_resonance_wavelengths_nm=reported_resonance_wavelengths_nm,
            excitation_wavelength_nm=excitation_wavelength_nm,
            polarization=polarization,  # type: ignore[arg-type]
            surrounding_medium=surrounding_medium,  # type: ignore[arg-type]
            sweep=sweep,
            parameter_provenance=provenance,
            unsupported_features=sorted(set(unsupported)),
            compiler_notes=notes,
        )
