from __future__ import annotations

from typing import Protocol, runtime_checkable

from dac_her.explorer_contracts import ExplorationReport, GraphExplorerPacket
from dac_her.explorer_validation import ExplorationReportValidator, ExplorationValidationResult


@runtime_checkable
class GraphExplorerBackend(Protocol):
    """Backend contract for the future v2.5.1 LLM Graph Explorer.

    v2.5.0 intentionally does not provide a generative implementation. A backend
    must return a schema-valid ExplorationReport; deterministic validation is then
    applied before downstream hypothesis generation is allowed.
    """

    def explore(self, packet: GraphExplorerPacket) -> ExplorationReport: ...


class GraphExplorerRunner:
    def __init__(
        self,
        backend: GraphExplorerBackend,
        *,
        validator: ExplorationReportValidator | None = None,
    ) -> None:
        self.backend = backend
        self.validator = validator or ExplorationReportValidator()

    def run(
        self,
        packet: GraphExplorerPacket,
    ) -> tuple[ExplorationReport, ExplorationValidationResult]:
        report = self.backend.explore(packet)
        if not isinstance(report, ExplorationReport):
            report = ExplorationReport.model_validate(report)
        validation = self.validator.validate(packet, report)
        return report, validation
