from types import SimpleNamespace

from pipeline_core.discovery.discovery_axis_runtime import (
    DiscoveryAxisSynthesisRuntime,
)


class _Compiler:
    def __init__(self, portfolio):
        self.portfolio = portfolio

    def compile(self, context, draft):
        return self.portfolio


class _Validator:
    def validate(self, context, portfolio):
        return SimpleNamespace(
            passes=False,
            issues=[
                SimpleNamespace(
                    severity="error",
                    code="TEST_VALIDATION_ERROR",
                ),
                SimpleNamespace(
                    severity="warning",
                    code="TEST_VALIDATION_WARNING",
                ),
            ],
        )


def test_compile_validate_discards_portfolio_on_validation_failure():
    invalid_portfolio = object()
    runtime = DiscoveryAxisSynthesisRuntime.__new__(
        DiscoveryAxisSynthesisRuntime
    )
    runtime.compiler = _Compiler(invalid_portfolio)
