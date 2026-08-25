from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


@dataclass(frozen=True)
class CandidateUnitApplicabilityDecision:
    owner_class: str
    gap_class: str
    eligible: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SERSAuAgCandidateUnitApplicability:
    """Domain-owned applicability contract for gap-control discovery.

    This is deliberately narrower than generic semantic relevance.

    The RCF correction chain validated:
    - explicit nanogap / interparticle-gap ownership;
    - structural proxies for interparticle/sidewall spacing or separation;
    - rejection of anchor-only attachment;
    - rejection of unrelated generic distance variables.

    UNKNOWN stop families are not classified by this adapter and therefore
    must not activate the locked RCF correction chain.
    """

    adapter_id = "sers_au_ag_candidate_unit_applicability_v1"
    domain_profile_id = "sers_au_ag"

    _STOP_PATTERNS = (
        re.compile(r"\bnano[\s-]*gaps?\b", re.I),
        re.compile(
            r"\binter[\s-]*particle\s+gaps?\b",
            re.I,
        ),
        re.compile(
            r"\b(?:nanoparticle|particle|interior)\s+gaps?\b",
            re.I,
        ),
    )

    _STRICT_OWNER_PATTERNS = _STOP_PATTERNS

    # RCF-5.7 validated structural proxies.
    # Deliberately excludes arbitrary "... distance" expressions such as
    # adsorbate-surface distance.
    _PROXY_OWNER_PATTERNS = (
        re.compile(
            r"\binter[\s-]*particle\s+spacing\b",
            re.I,
        ),
        re.compile(
            r"\binter[\s-]*particle\s+separation\b",
            re.I,
        ),
        re.compile(
            r"\bsidewall\s+spacing\b",
            re.I,
        ),
        re.compile(
            r"\bsidewall\s+separation\b",
            re.I,
        ),
    )

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(str(text or "").split())

    @classmethod
    def _matches(
        cls,
        text: str,
        patterns: tuple[re.Pattern[str], ...],
    ) -> bool:
        value = cls._normalize(text)
        return any(pattern.search(value) for pattern in patterns)

    @classmethod
    def _unit_text(cls, unit: Any) -> str:
        return " | ".join(
            part
            for part in (
                cls._normalize(getattr(unit, "label", "")),
                cls._normalize(getattr(unit, "proposed_subject", "")),
                cls._normalize(getattr(unit, "proposed_relation", "")),
                cls._normalize(getattr(unit, "proposed_object", "")),
            )
            if part
        )

    @classmethod
    def _anchor_text(cls, unit: Any) -> str:
        anchors = getattr(unit, "anchors", ()) or ()
        return " | ".join(
            cls._normalize(getattr(anchor, "label", ""))
            for anchor in anchors
            if cls._normalize(getattr(anchor, "label", ""))
        )

    def supports_stop(self, stop: str | None) -> bool:
        value = self._normalize(stop or "")
        return bool(value) and self._matches(
            value,
            self._STOP_PATTERNS,
        )

    def relevance_context(self, stop: str) -> str:
        if not self.supports_stop(stop):
            raise ValueError(
                "SERS gap applicability adapter does not support stop "
                f"{stop!r}"
            )

        # Exact RCF intervention semantics:
        #   "... ; nanogap control ; candidate scientific bridge ; ..."
        return f"{self._normalize(stop)} control"

    def classify(
        self,
        unit: Any,
        *,
        stop: str,
    ) -> CandidateUnitApplicabilityDecision:
        if not self.supports_stop(stop):
            return CandidateUnitApplicabilityDecision(
                owner_class="UNKNOWN_STOP_FAMILY",
                gap_class="UNKNOWN",
                eligible=True,
                reason=(
                    "stop family is outside this domain-owned "
                    "applicability contract"
                ),
            )

        unit_text = self._unit_text(unit)

        if self._matches(
            unit_text,
            self._STRICT_OWNER_PATTERNS,
        ):
            return CandidateUnitApplicabilityDecision(
                owner_class="UNIT_OWNED_GAP_CONTROL",
                gap_class="STRICT_GAP_CONTROL",
                eligible=True,
                reason=(
                    "candidate unit itself explicitly owns a "
                    "nanogap/interparticle-gap variable"
                ),
            )

        if self._matches(
            unit_text,
            self._PROXY_OWNER_PATTERNS,
        ):
            return CandidateUnitApplicabilityDecision(
                owner_class="UNIT_OWNED_GAP_CONTROL",
                gap_class="GAP_CONTROL_PROXY",
                eligible=True,
                reason=(
                    "candidate unit itself owns the validated "
                    "structural gap-control proxy"
                ),
            )

        anchor_text = self._anchor_text(unit)

        if (
            self._matches(
                anchor_text,
                self._STRICT_OWNER_PATTERNS,
            )
            or self._matches(
                anchor_text,
                self._PROXY_OWNER_PATTERNS,
            )
        ):
            return CandidateUnitApplicabilityDecision(
                owner_class="ANCHOR_CONTEXT_ONLY",
                gap_class="NONE",
                eligible=False,
                reason=(
                    "gap concept occurs only in grounding-anchor context; "
                    "the candidate unit itself does not own it"
                ),
            )

        return CandidateUnitApplicabilityDecision(
            owner_class="NO_GAP_ATTACHMENT",
            gap_class="NONE",
            eligible=False,
            reason=(
                "candidate unit has no direct or validated-proxy "
                "gap-control attachment"
            ),
        )


SERS_AU_AG_CANDIDATE_UNIT_APPLICABILITY = (
    SERSAuAgCandidateUnitApplicability()
)
