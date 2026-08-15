from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable

from dac_her.hypothesis_trend_directional_contracts import (
    DirectionAwareTrendHypothesisPortfolioDraft,
)
from dac_her.hypothesis_trend_directional_exposure import (
    DirectionalTrendMakerExposure,
    build_directional_trend_maker_exposure,
    verify_directional_trend_maker_exposure,
)
from dac_her.hypothesis_trend_input import TrendAwareHypothesisInput
from dac_her.hypothesis_trend_prompt import (
    TrendAwareHypothesisPromptAssembler,
)


PROMPT_VERSION = (
    "direction-aware-trend-hypothesis-maker-prompt-v1-alpha4c5d1"
)


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DirectionAwareTrendHypothesisPrompt:
    prompt_version: str
    exposure_id: str
    exposure_sha256: str
    system_prompt: str
    user_prompt: str
    prompt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        exposure: DirectionalTrendMakerExposure,
        system_prompt: str,
        user_prompt: str,
    ) -> "DirectionAwareTrendHypothesisPrompt":
        canonical = _compact_json(
            {
                "prompt_version": PROMPT_VERSION,
                "exposure_id": exposure.exposure_id,
                "exposure_sha256": exposure.exposure_sha256,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return cls(
            prompt_version=PROMPT_VERSION,
            exposure_id=exposure.exposure_id,
            exposure_sha256=exposure.exposure_sha256,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_sha256=_sha256(canonical),
        )


DIRECTION_SYSTEM_ADDENDUM = """
ALPHA4C.5D.1 DIRECTIONAL PRECISION ADDENDUM

The alpha4c Trend sign convention is frozen:
- positive: as the independent variable INCREASES, the dependent observable INCREASES.
- negative: as the independent variable INCREASES, the dependent observable DECREASES.
- unchanged: dependent observable remains unchanged under an independent-variable increase.
- non_monotonic: no single monotonic dependent change is licensed.
- mixed/ambiguous/unspecified directions map to unspecified.

For every Trend-grounded directional statement, use the canonical independent-variable INCREASE frame. Do not re-express it using decreasing/smaller/lower/reduced independent-variable language. This intentionally prevents silent sign inversion.

Every selected positive Trend support view MUST appear in at least one predicted_observation.trend_direction_bindings row. Copy the exact view_id, set independent_change="increase", and copy the exact dependent_change printed by the directional exposure. The prediction expected_direction must agree with the binding.

The response model for this runtime is DirectionAwareTrendHypothesisPortfolioDraft. All alpha4c.5d provenance, limitation-companion, causal/universal, numeric, novelty, and protocol restrictions remain in force.
""".strip()


class DirectionAwareTrendHypothesisPromptAssembler:
    def __init__(
        self,
        *,
        statement_text_limit: int = 1100,
        max_hypotheses: int = 3,
    ) -> None:
        if max_hypotheses < 1:
            raise ValueError("max_hypotheses must be >= 1")
        self.statement_text_limit = int(statement_text_limit)
        self.max_hypotheses = int(max_hypotheses)
        self.base_assembler = TrendAwareHypothesisPromptAssembler(
            statement_text_limit=self.statement_text_limit,
            max_hypotheses=self.max_hypotheses,
        )

    def build(
        self,
        source: TrendAwareHypothesisInput,
        *,
        exposure: DirectionalTrendMakerExposure | None = None,
    ) -> DirectionAwareTrendHypothesisPrompt:
        exposure = (
            exposure
            or build_directional_trend_maker_exposure(source)
        )
        verify_directional_trend_maker_exposure(source, exposure)

        base_prompt = self.base_assembler.build(source)
        system_prompt = base_prompt.system_prompt.replace(
            (
                "Return only the structured "
                "TrendAwareHypothesisPortfolioDraft requested by the caller."
            ),
            (
                "Return only the structured "
                "DirectionAwareTrendHypothesisPortfolioDraft requested "
                "by the caller."
            ),
        )
        system_prompt = (
            system_prompt
            + "\n\n"
            + DIRECTION_SYSTEM_ADDENDUM
        )

        user_prompt = base_prompt.user_prompt.replace(
            "Return TrendAwareHypothesisPortfolioDraft only,",
            "Return DirectionAwareTrendHypothesisPortfolioDraft only,",
        )

        lines = [
            "",
            "ALPHA4C.5D.1 CANONICAL DIRECTION BINDINGS",
            "==========================================",
            "- All signs below are interpreted under independent_change=increase.",
            "- A positive Trend view selected as positive support MUST bind dependent_change=increase.",
            "- A negative Trend view selected as positive support MUST bind dependent_change=decrease.",
            "- Do not use decreasing/smaller/lower/reduced wording for the bound independent variable anywhere in the hypothesis statement, inferential bridge, or Trend-bound prediction.",
        ]
        if not exposure.views:
            lines.append("- NONE")
        for row in exposure.views:
            view = row.source_view
            lines.append(
                f"- view_id={view.view_id}; "
                f"allowed_use_role={view.allowed_use_role}; "
                f"lane={view.lane}; "
                f"relation={view.independent_variable_key}->"
                f"{view.dependent_observable_key}; "
                f"directions={','.join(view.directions) or '-'}; "
                f"shapes={','.join(view.shapes) or '-'}"
            )
            lines.append(
                "  canonical_direction: "
                "independent_change=increase; "
                f"expected_dependent_change="
                f"{row.expected_dependent_change}"
            )
            if row.direction_binding_required_if_positive:
                lines.append(
                    "  MANDATORY DIRECTION BINDING IF SELECTED: "
                    f"view_id={view.view_id}; "
                    "independent_change=increase; "
                    "dependent_change="
                    f"{row.expected_dependent_change}"
                )

        lines.extend(
            [
                "",
                "DIRECTIONAL OUTPUT ADDENDUM",
                "===========================",
                "- Every selected positive Trend view must appear in trend_direction_bindings of at least one predicted observation.",
                "- trend_direction_bindings[].view_id must be the exact positive-support Trend view ID.",
                "- trend_direction_bindings[].independent_change must be 'increase'.",
                "- trend_direction_bindings[].dependent_change must exactly match expected_dependent_change printed above.",
                "- A Trend-bound prediction expected_direction must match its binding: increase->increase, decrease->decrease, non_monotonic->non_monotonic, unchanged/unspecified->unspecified.",
                "- Preserve all REQUIRED COMPANIONS from the alpha4c.5d section above.",
                "- The directional addendum does not authorize causality, universality, numbers, protocols, or external novelty.",
            ]
        )
        user_prompt = user_prompt + "\n" + "\n".join(lines)

        return DirectionAwareTrendHypothesisPrompt.create(
            exposure=exposure,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def repair_feedback(
        self,
        *,
        previous_draft: DirectionAwareTrendHypothesisPortfolioDraft,
        issues: Iterable[object],
    ) -> str:
        issue_lines = []
        for issue in issues:
            issue_lines.append(
                "- "
                + str(getattr(issue, "code", "UNKNOWN"))
                + " @ "
                + str(getattr(issue, "location", ""))
                + ": "
                + str(getattr(issue, "message", issue))
            )
        return "\n".join(
            [
                "DIRECTION-AWARE TREND REPAIR REQUEST",
                "====================================",
                "Repair only the exact deterministic contract failures below.",
                "Preserve all Explorer/Trend provenance namespaces and every required limitation companion.",
                "For direction failures, use the canonical independent-variable INCREASE frame and copy expected_dependent_change exactly.",
                "Do not introduce new evidence IDs, values, protocols, novelty claims, causal authority, or universal authority.",
                "If the hypothesis cannot be repaired, remove it or abstain.",
                "",
                "ISSUES",
                *issue_lines,
                "",
                "PREVIOUS DRAFT",
                previous_draft.model_dump_json(indent=2),
            ]
        )
