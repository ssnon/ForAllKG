from __future__ import annotations

from dac_her.hypothesis_trend_directional_contracts import (
    DirectionAwareTrendHypothesisPortfolio,
)
from dac_her.hypothesis_trend_directional_validator import (
    DirectionAwareTrendHypothesisValidator,
)
from dac_her.hypothesis_trend_input import TrendAwareHypothesisInput
from dac_her.hypothesis_trend_validator import (
    TrendHypothesisValidationResult,
)


HYPOTHESIS_TREND_HARDENED_VALIDATOR_SEMANTICS_ID = (
    "hypothesis_trend_contract_hardened_validator_v1_alpha4c5i"
)


class ContractHardenedTrendHypothesisValidator:
    semantics_id = HYPOTHESIS_TREND_HARDENED_VALIDATOR_SEMANTICS_ID

    def __init__(
        self,
        *,
        directional_validator:
            DirectionAwareTrendHypothesisValidator | None = None,
    ) -> None:
        self.directional_validator = (
            directional_validator
            or DirectionAwareTrendHypothesisValidator()
        )

    def validate(
        self,
        source: TrendAwareHypothesisInput,
        portfolio: DirectionAwareTrendHypothesisPortfolio,
    ) -> TrendHypothesisValidationResult:
        base = self.directional_validator.validate(
            source,
            portfolio,
        )
        return TrendHypothesisValidationResult(
            semantics_id=self.semantics_id,
            passes=base.passes,
            errors=base.errors,
            warnings=base.warnings,
            issues=base.issues,
        )
