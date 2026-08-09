from __future__ import annotations

from dataclasses import dataclass

from dac_her.hypothesis_compiler import HypothesisCompiler
from dac_her.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
    HypothesisPortfolioDraft,
)
from dac_her.hypothesis_validation import HypothesisValidationResult, HypothesisValidator


@dataclass(frozen=True)
class HypothesisSubstrateOutcome:
    portfolio: HypothesisPortfolio
    validation: HypothesisValidationResult

    @property
    def accepted(self) -> bool:
        return self.validation.passes


class HypothesisMakerSubstrate:
    """Deterministic v2.6.0 compile/validate boundary.

    No model is called here. v2.6.1 can place an LLM-owned HypothesisPortfolioDraft
    in front of this class without changing the deterministic scientific boundary.
    """

    def __init__(
        self,
        *,
        compiler: HypothesisCompiler | None = None,
        validator: HypothesisValidator | None = None,
    ) -> None:
        self.compiler = compiler or HypothesisCompiler()
        self.validator = validator or HypothesisValidator()

    def run(
        self,
        context: HypothesisContext,
        draft: HypothesisPortfolioDraft,
    ) -> HypothesisSubstrateOutcome:
        portfolio = self.compiler.compile(context, draft)
        validation = self.validator.validate(context, portfolio)
        return HypothesisSubstrateOutcome(portfolio=portfolio, validation=validation)
