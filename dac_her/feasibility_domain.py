from __future__ import annotations

from typing import Protocol

from dac_her.experimental_contracts import ExperimentalRealizabilityReport
from dac_her.feasibility_contracts import FeasibilityIntake
from dac_her.physics_contracts import PhysicsFeasibilityReport
from dac_her.scope_contracts import ScientificScope
from dac_her.validation_contracts import ValidationSpecification


class FeasibilityDomainAdapter(Protocol):
    """Domain-specific scientific feasibility boundary.

    Core orchestration owns provenance, artifact persistence, stage ordering, and
    candidate decision. The adapter owns domain-specific interpretation needed to
    turn a hypothesis into scientific scope, validation design, physics checks,
    and experimental-realizability reports.
    """

    adapter_id: str
    domain_profile_id: str

    def compile_scopes(
        self,
        intake: FeasibilityIntake,
    ) -> list[ScientificScope]: ...

    def compile_specifications(
        self,
        intake: FeasibilityIntake,
        scopes: list[ScientificScope],
    ) -> list[ValidationSpecification]: ...

    def run_physics(
        self,
        intake: FeasibilityIntake,
        scopes: list[ScientificScope],
        specifications: list[ValidationSpecification],
    ) -> list[PhysicsFeasibilityReport]: ...

    def run_experimental(
        self,
        intake: FeasibilityIntake,
        physics_reports: list[PhysicsFeasibilityReport],
        scopes: list[ScientificScope],
        specifications: list[ValidationSpecification],
    ) -> list[ExperimentalRealizabilityReport]: ...
