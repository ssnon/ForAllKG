from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from pipeline_core.discovery.hypothesis_contracts import HypothesisCard, HypothesisPortfolio

from domains.sers.fdtd_applicability import SERSFDTDApplicabilityAnalyzer
from domains.sers.fdtd_applicability_contracts import (
    SERSFDTDApplicabilityBundle,
    SERSFDTDApplicabilityReport,
)
from domains.sers.validation_routing_contracts import (
    SERSHypothesisValidationPlan,
    SERSHypothesisValidationPlanBundle,
    SERSValidationContextNote,
    SERSValidationRoute,
)


_ROUTER_VERSION = "sers-validation-router-v01"


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def _claim_chunks(card: HypothesisCard) -> list[_TextChunk]:
    """Fields that are allowed to create active validation routes.

    Assumptions and the generated title are intentionally excluded: they may
    describe transferability limits or confounders, but they are not themselves
    scientific claims that should create validator work.
    """
    rows = [
        _TextChunk("hypothesis_statement", card.hypothesis_statement),
        _TextChunk("inferential_bridge", card.inferential_bridge),
    ]
    rows.extend(
        _TextChunk(f"predicted_observations[{index}].observable", row.observable)
        for index, row in enumerate(card.predicted_observations)
    )
    rows.extend(
        _TextChunk(f"predicted_observations[{index}].rationale", row.rationale)
        for index, row in enumerate(card.predicted_observations)
    )
    return rows


def _context_chunks(card: HypothesisCard) -> list[_TextChunk]:
    return [
        _TextChunk(f"assumptions[{index}]", text)
        for index, text in enumerate(card.assumptions)
    ]


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _matching_fields(chunks: list[_TextChunk], patterns: tuple[str, ...]) -> list[str]:
    return sorted({row.field for row in chunks if _matches(row.text, patterns)})


class SERSHypothesisValidationRouter:
    """Route SERS hypothesis components to validator families without judging truth.

    The existing FDTD applicability artifact is preserved as the authoritative
    statement about whether a classical-EM subclaim exists. This router only adds
    the missing higher-level scientific validation plan: EM computation is one
    branch, while molecular/surface/process contributions and total measured SERS
    outcomes remain routed separately.
    """

    router_version = _ROUTER_VERSION

    _MOLECULAR_ANCHOR_PATTERNS = (
        r"\banalytes?\b",
        r"\bmolecules?\b",
        r"\bmolecular\b",
        r"\breporters?\b",
        r"\bdyes?\b",
        r"\bmethylene\s+blue\b",
        r"\brhodamine\s+6g\b",
        r"\b4[- ]?mercaptobenzoic\s+acid\b",
    )

    _MOLECULAR_BEHAVIOR_PATTERNS = (
        r"\banalyte[- ]specific\b",
        r"\bmolecular\s+(?:resonance|absorption|response)\b",
        r"\bresonance\s+raman\b",
        r"\braman\s+cross[- ]section\b",
        r"\bvibrational\s+(?:mode|response)\b",
        r"\bdye\s+absorption\b",
        r"\bwavelength[- ]depend(?:ent|ence)\b",
    )

    _SURFACE_CHEMISTRY_PATTERNS = (
        r"\bchemical\s+enhancement\b",
        r"\bcharge\s+transfer\b",
        r"\badsor(?:b|ption|bed)\b",
        r"\bsurface\s+(?:association|affinity|binding|chemistry)\b",
        r"\bchemical\s+association\b",
        r"\bbinding\s+(?:energy|affinity)\b",
        r"\bmolecular\s+orientation\b",
    )

    _PROCESS_PATTERNS = (
        r"\bsynthes(?:is|ized|ize)\b",
        r"\bfabricat(?:e|ed|ion)\b",
        r"\banneal(?:ing|ed)?\b",
        r"\btemperature\s+treatment\b",
        r"\bdeposition\b",
        r"\breproducib",
        r"\buniformity\b",
        r"\bstability\b",
        r"\baging\b",
        r"\brsd\b",
    )

    _INTEGRATED_SERS_PATTERNS = (
        r"\bsers\s+(?:intensity|signal|response|enhancement\s+factor|ef)\b",
        r"\braman\s+(?:intensity|signal)\b",
        r"\blimit\s+of\s+detection\b",
        r"\blod\b",
        r"\bspot[- ]to[- ]spot\b",
        r"\bbatch[- ]to[- ]batch\b",
    )

    _MECHANISM_CLAIM_PATTERNS = (
        r"\bbecause\b",
        r"\bdue\s+to\b",
        r"\bthrough\b",
        r"\bvia\b",
        r"\bmediated\s+by\b",
        r"\bmodif(?:y|ies|ied|ying)\b",
        r"\bcontrol(?:s|led|ling)?\b",
        r"\bcontribut(?:e|es|ed|ing)\b",
        r"\bcaus(?:e|es|ed|ing)\b",
        r"\bdriv(?:e|es|en|ing)\b",
        r"\byield(?:s|ed|ing)?\b",
        r"\battribut(?:e|ed|ion)\b",
        r"\bmechanis(?:m|tic)\b",
    )

    def plan_portfolio(
        self,
        portfolio: HypothesisPortfolio,
        *,
        fdtd_applicability: SERSFDTDApplicabilityBundle | None = None,
        hypothesis_ids: set[str] | None = None,
    ) -> SERSHypothesisValidationPlanBundle:
        if portfolio.domain_profile_id != "sers_au_ag":
            raise ValueError(
                "SERS validation router requires domain_profile_id='sers_au_ag'"
            )

        if fdtd_applicability is None:
            fdtd_applicability = SERSFDTDApplicabilityAnalyzer().analyze_portfolio(
                portfolio,
                hypothesis_ids=hypothesis_ids,
            )
        self._validate_lineage(portfolio, fdtd_applicability)

        cards = {
            card.hypothesis_id: card
            for card in portfolio.hypotheses
            if hypothesis_ids is None or card.hypothesis_id in hypothesis_ids
        }
        if hypothesis_ids is not None:
            missing = sorted(hypothesis_ids - set(cards))
            if missing:
                raise ValueError(
                    "requested hypothesis_id(s) absent from portfolio: "
                    + ", ".join(missing)
                )

        selected_reports = [
            row
            for row in fdtd_applicability.reports
            if row.hypothesis_id in cards
        ]
        applicability_by_hypothesis = {
            row.hypothesis_id: row for row in selected_reports
        }
        if len(applicability_by_hypothesis) != len(selected_reports):
            raise ValueError("duplicate FDTD applicability report for hypothesis")
        if set(cards) != set(applicability_by_hypothesis):
            raise ValueError(
                "FDTD applicability reports must cover every selected hypothesis"
            )

        plans = [
            self.plan_card(cards[hypothesis_id], applicability_by_hypothesis[hypothesis_id])
            for hypothesis_id in sorted(cards)
        ]
        return SERSHypothesisValidationPlanBundle(
            bundle_id=_stable_id(
                "sers_validation_plan_bundle",
                portfolio.portfolio_id,
                fdtd_applicability.bundle_id,
                *(row.plan_id for row in plans),
                self.router_version,
            ),
            source_portfolio_id=portfolio.portfolio_id,
            source_fdtd_applicability_bundle_id=fdtd_applicability.bundle_id,
            plans=plans,
        )

    def plan_card(
        self,
        card: HypothesisCard,
        applicability: SERSFDTDApplicabilityReport,
    ) -> SERSHypothesisValidationPlan:
        if card.domain_profile_id != "sers_au_ag":
            raise ValueError("SERS validation router received non-SERS hypothesis")
        if applicability.hypothesis_id != card.hypothesis_id:
            raise ValueError("hypothesis/applicability mismatch")

        claim_chunks = _claim_chunks(card)
        context_chunks = _context_chunks(card)
        active_field_names = {row.field for row in claim_chunks}
        routes: list[SERSValidationRoute] = []

        integrated_fields = _matching_fields(
            claim_chunks,
            self._INTEGRATED_SERS_PATTERNS,
        )
        integrated_required = bool(integrated_fields)
        mechanism_surface = _normalize(" ".join(row.text for row in claim_chunks))
        mechanism_claim_present = _matches(
            mechanism_surface,
            self._MECHANISM_CLAIM_PATTERNS,
        )

        active_fdtd_subclaims = []
        active_fdtd_fields: list[str] = []
        for row in applicability.fdtd_subclaims:
            fields = sorted(
                field for field in row.source_fields if field in active_field_names
            )
            if fields:
                active_fdtd_subclaims.append(row)
                active_fdtd_fields.extend(fields)

        if active_fdtd_subclaims:
            routes.append(SERSValidationRoute(
                route_id=_stable_id(
                    "sers_validation_route",
                    card.hypothesis_id,
                    "classical_em",
                    *(row.subclaim_id for row in active_fdtd_subclaims),
                ),
                route_kind="classical_em",
                statement=(
                    "Evaluate the hypothesis's substrate-side classical "
                    "electromagnetic mechanism without treating the result as a "
                    "verdict on total SERS performance."
                ),
                rationale=(
                    "The existing FDTD applicability analysis identified an EM "
                    "subclaim. FDTD is retained as the currently implemented "
                    "backend, while FEM/DDA/BEM remain scientifically compatible "
                    "alternative classical-EM solvers."
                ),
                source_fields=sorted(set(active_fdtd_fields)),
                target_observables=sorted({
                    observable
                    for row in active_fdtd_subclaims
                    for observable in row.target_observables
                }),
                validator_candidates=["fdtd", "fem", "dda", "bem"],
                readiness="ready",
                source_fdtd_subclaim_ids=[
                    row.subclaim_id for row in active_fdtd_subclaims
                ],
                required_for_outcome_assessment=not integrated_required,
                required_for_mechanism_assessment=mechanism_claim_present,
            ))

        molecular_surface = _normalize(" ".join(row.text for row in claim_chunks))
        has_molecular_anchor = _matches(
            molecular_surface,
            self._MOLECULAR_ANCHOR_PATTERNS,
        )
        has_molecular_behavior = _matches(
            molecular_surface,
            self._MOLECULAR_BEHAVIOR_PATTERNS,
        )
        molecular_fields = (
            _matching_fields(
                claim_chunks,
                self._MOLECULAR_ANCHOR_PATTERNS
                + self._MOLECULAR_BEHAVIOR_PATTERNS,
            )
            if has_molecular_anchor and has_molecular_behavior
            else []
        )
        if molecular_fields:
            routes.append(SERSValidationRoute(
                route_id=_stable_id(
                    "sers_validation_route",
                    card.hypothesis_id,
                    "molecular_spectroscopy",
                    tuple(molecular_fields),
                ),
                route_kind="molecular_spectroscopy",
                statement=(
                    "Resolve molecule/analyte-specific spectral behavior separately "
                    "from the substrate electromagnetic response."
                ),
                rationale=(
                    "Molecular resonance, Raman cross-section, and analyte-specific "
                    "wavelength response are not established by substrate-only FDTD."
                ),
                source_fields=molecular_fields,
                target_observables=[
                    "analyte-specific Raman spectral response",
                    "molecular excitation dependence",
                ],
                validator_candidates=[
                    "spectroscopy_experiment",
                    "literature_evidence",
                    "quantum_chemistry_future",
                ],
                readiness="deferred",
                required_for_outcome_assessment=not integrated_required,
                required_for_mechanism_assessment=mechanism_claim_present,
            ))

        chemistry_fields = _matching_fields(
            claim_chunks, self._SURFACE_CHEMISTRY_PATTERNS
        )
        if chemistry_fields:
            routes.append(SERSValidationRoute(
                route_id=_stable_id(
                    "sers_validation_route",
                    card.hypothesis_id,
                    "surface_chemistry",
                    tuple(chemistry_fields),
                ),
                route_kind="surface_chemistry",
                statement=(
                    "Resolve adsorption, association, orientation, charge-transfer, "
                    "or chemical-enhancement claims with chemistry-aware evidence."
                ),
                rationale=(
                    "These mechanisms depend on molecule-surface interactions that "
                    "a classical Maxwell solver does not determine."
                ),
                source_fields=chemistry_fields,
                target_observables=[
                    "molecule-surface interaction",
                    "chemical contribution to SERS",
                ],
                validator_candidates=[
                    "surface_chemistry_experiment",
                    "dft_future",
                    "molecular_simulation_future",
                    "literature_evidence",
                ],
                readiness="deferred",
                required_for_outcome_assessment=not integrated_required,
                required_for_mechanism_assessment=mechanism_claim_present,
            ))

        process_fields = _matching_fields(claim_chunks, self._PROCESS_PATTERNS)
        if process_fields:
            routes.append(SERSValidationRoute(
                route_id=_stable_id(
                    "sers_validation_route",
                    card.hypothesis_id,
                    "fabrication_process",
                    tuple(process_fields),
                ),
                route_kind="fabrication_process",
                statement=(
                    "Evaluate whether fabrication/process conditions and their "
                    "variability realize the proposed substrate behavior."
                ),
                rationale=(
                    "Synthesis, fabrication tolerance, reproducibility, and aging "
                    "are experimental/process properties rather than Maxwell-only "
                    "observables."
                ),
                source_fields=process_fields,
                target_observables=[
                    "fabrication realizability",
                    "substrate reproducibility and stability",
                ],
                validator_candidates=[
                    "fabrication_experiment",
                    "characterization_experiment",
                    "literature_evidence",
                ],
                readiness="deferred",
                required_for_outcome_assessment=not integrated_required,
                required_for_mechanism_assessment=mechanism_claim_present,
            ))

        if integrated_required:
            routes.append(SERSValidationRoute(
                route_id=_stable_id(
                    "sers_validation_route",
                    card.hypothesis_id,
                    "integrated_sers_experiment",
                    tuple(integrated_fields),
                ),
                route_kind="integrated_sers_experiment",
                statement=(
                    "Test the hypothesis-level measured SERS outcome after the "
                    "mechanistic subclaims have been screened."
                ),
                rationale=(
                    "Total measured SERS performance integrates EM enhancement with "
                    "molecular, surface-chemical, fabrication, and measurement "
                    "effects; it cannot be closed by FDTD alone."
                ),
                source_fields=integrated_fields,
                target_observables=[
                    "measured SERS performance under controlled comparison",
                ],
                validator_candidates=["sers_experiment"],
                readiness="deferred",
                required_for_outcome_assessment=True,
                required_for_mechanism_assessment=mechanism_claim_present,
            ))

        context_notes: list[SERSValidationContextNote] = []
        for chunk in context_chunks:
            topics: list[str] = []
            if _matches(chunk.text, self._PROCESS_PATTERNS):
                topics.append("fabrication_process")
            if _matches(chunk.text, self._SURFACE_CHEMISTRY_PATTERNS):
                topics.append("surface_chemistry")
            if (
                _matches(chunk.text, self._MOLECULAR_ANCHOR_PATTERNS)
                and _matches(chunk.text, self._MOLECULAR_BEHAVIOR_PATTERNS)
            ):
                topics.append("molecular_spectroscopy")
            if _matches(chunk.text, self._INTEGRATED_SERS_PATTERNS):
                topics.append("integrated_sers_experiment")
            if any(
                chunk.field in row.source_fields
                for row in applicability.fdtd_subclaims
            ):
                topics.append("classical_em")
            context_notes.append(SERSValidationContextNote(
                context_id=_stable_id(
                    "sers_validation_context",
                    card.hypothesis_id,
                    chunk.field,
                    chunk.text,
                ),
                source_field=chunk.field,
                text=chunk.text,
                matched_topics=sorted(set(topics)),
            ))

        if not routes or applicability.applicability == "requires_interpretation":
            routes.append(SERSValidationRoute(
                route_id=_stable_id(
                    "sers_validation_route",
                    card.hypothesis_id,
                    "unresolved",
                ),
                route_kind="unresolved",
                statement=(
                    "Resolve the hypothesis's validation mechanism before assigning "
                    "a computational or experimental backend."
                ),
                rationale=(
                    "The current deterministic router cannot preserve the hypothesis "
                    "meaning while assigning a more specific validation route."
                ),
                source_fields=[
                    "hypothesis_statement",
                    "inferential_bridge",
                ],
                target_observables=[],
                validator_candidates=["human_review"],
                readiness="requires_interpretation",
            ))

        return SERSHypothesisValidationPlan(
            plan_id=_stable_id(
                "sers_validation_plan",
                card.hypothesis_id,
                applicability.report_id,
                *(row.route_id for row in routes),
                self.router_version,
            ),
            source_portfolio_id=applicability.source_portfolio_id,
            source_fdtd_applicability_report_id=applicability.report_id,
            hypothesis_id=card.hypothesis_id,
            fdtd_applicability=applicability.applicability,
            routes=routes,
            context_notes=context_notes,
            mechanism_claim_present=mechanism_claim_present,
            mechanism_assessment=(
                "deferred_pending_routed_evidence"
                if mechanism_claim_present
                else "not_claimed"
            ),
            integrated_experiment_required=integrated_required,
        )

    @staticmethod
    def _validate_lineage(
        portfolio: HypothesisPortfolio,
        applicability: SERSFDTDApplicabilityBundle,
    ) -> None:
        if applicability.domain_profile_id != portfolio.domain_profile_id:
            raise ValueError("portfolio/FDTD-applicability domain mismatch")
        if applicability.source_portfolio_id != portfolio.portfolio_id:
            raise ValueError("FDTD applicability was produced from a different portfolio")
