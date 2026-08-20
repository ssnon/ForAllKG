from __future__ import annotations

from dataclasses import dataclass, field

from dac_her.experimental_runtime import ExperimentalRealizabilityRuntime
from pipeline_core.feasibility_contracts import FeasibilityIntake
from dac_her.physics_runtime import PhysicsFeasibilityRuntime
from dac_her.scope_compiler import HypothesisScopeCompiler
from pipeline_core.scope_contracts import ScientificScope
from pipeline_core.validation_contracts import ValidationSpecification
from dac_her.validation_specification import ValidationSpecificationCompiler


@dataclass
class DacHerFeasibilityAdapter:
    """Compatibility-preserving adapter around the existing DAC-HER v0.2 stack.

    Alpha2 deliberately does not rewrite the validated HER scope/validation/
    physics/experimental rules. It moves ownership behind an explicit adapter
    boundary first, so another domain cannot accidentally inherit HER rules.
    """

    adapter_id: str = "dac_her"
    domain_profile_id: str = "dac_her"
    scope_compiler: HypothesisScopeCompiler = field(
        default_factory=HypothesisScopeCompiler
    )
    specification_compiler: ValidationSpecificationCompiler = field(
        default_factory=ValidationSpecificationCompiler
    )

    def __post_init__(self) -> None:
        self.physics_runtime = PhysicsFeasibilityRuntime(
            scope_compiler=self.scope_compiler,
            specification_compiler=self.specification_compiler,
        )
        self.experimental_runtime = ExperimentalRealizabilityRuntime(
            scope_compiler=self.scope_compiler,
            specification_compiler=self.specification_compiler,
        )

    def compile_scopes(
        self,
        intake: FeasibilityIntake,
    ) -> list[ScientificScope]:
        return self.scope_compiler.compile_intake(intake)

    def compile_specifications(
        self,
        intake: FeasibilityIntake,
        scopes: list[ScientificScope],
    ) -> list[ValidationSpecification]:
        return self.specification_compiler.compile_intake(intake, scopes)

    def run_physics(
        self,
        intake: FeasibilityIntake,
        scopes: list[ScientificScope],
        specifications: list[ValidationSpecification],
    ):
        return self.physics_runtime.run_intake(
            intake,
            scopes,
            specifications,
        )

    def run_experimental(
        self,
        intake: FeasibilityIntake,
        physics_reports,
        scopes: list[ScientificScope],
        specifications: list[ValidationSpecification],
    ):
        return self.experimental_runtime.run_intake(
            intake,
            physics_reports,
            scopes,
            specifications,
        )


DAC_HER_FEASIBILITY_ADAPTER = DacHerFeasibilityAdapter()
