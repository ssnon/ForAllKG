from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.hypothesis_semantic_contracts import (
    HypothesisSemanticReview,
    HypothesisSemanticRunRecord,
)
from pipeline_core.discovery.hypothesis_semantic_disposition import (
    HypothesisSemanticDispositionV1,
    compile_hypothesis_semantic_disposition_v1,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pretty_json_bytes(value: object) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_exact_or_validate(path: Path, payload: object) -> str:
    expected = _pretty_json_bytes(payload)
    path = path.expanduser().resolve()
    if path.exists():
        observed = path.read_bytes()
        if observed != expected:
            raise ValueError(
                "existing write-once initial semantic gate artifact differs: "
                + str(path)
            )
        return hashlib.sha256(observed).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(expected)
    return hashlib.sha256(expected).hexdigest()


InitialSemanticGateStatusV1 = Literal[
    "SEMANTIC_HARD_GATE_FAILED",
    "SEMANTIC_REVIEW_INVALID",
    "SEMANTIC_INTERVENTION_REQUIRED",
    "PRE_N10_ENTRY_AUTHORIZED",
]


class PreN10InitialSemanticGateReportV1(StrictModel):
    schema_version: Literal[
        "pre-n10-initial-semantic-gate-report-v1"
    ] = "pre-n10-initial-semantic-gate-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_portfolio_path: str
    source_portfolio_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_portfolio_id: str
    source_portfolio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_semantic_run_path: str
    source_semantic_run_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_semantic_run_id: str

    source_semantic_review_path: str | None = None
    source_semantic_review_file_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_semantic_review_id: str | None = None

    semantic_disposition_path: str | None = None
    semantic_disposition_id: str | None = None
    semantic_disposition_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    semantic_disposition: str | None = None

    status: InitialSemanticGateStatusV1
    semantic_review_valid: bool
    semantic_admissible_for_pre_n10: bool
    pre_n10_entry_authorized: bool

    semantic_failed_dimensions: list[str] = Field(default_factory=list)
    semantic_warning_dimensions: list[str] = Field(default_factory=list)

    semantic_runtime_reexecuted: Literal[False] = False
    semantic_critic_llm_reinvoked: Literal[False] = False
    external_novelty_performed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    repair_performed: Literal[False] = False
    regeneration_performed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    verifier_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "PreN10InitialSemanticGateReportV1":
        authorized = self.status == "PRE_N10_ENTRY_AUTHORIZED"
        if self.pre_n10_entry_authorized != authorized:
            raise ValueError("pre-N10 entry authorization mismatch")
        if self.semantic_admissible_for_pre_n10 != authorized:
            raise ValueError("semantic admissibility/authorization mismatch")

        disposition_fields = (
            self.semantic_disposition_path,
            self.semantic_disposition_id,
            self.semantic_disposition_sha256,
            self.semantic_disposition,
        )
        if self.semantic_review_valid:
            if any(value is None for value in disposition_fields):
                raise ValueError("valid semantic review requires disposition artifact")
            if self.status not in {
                "SEMANTIC_INTERVENTION_REQUIRED",
                "PRE_N10_ENTRY_AUTHORIZED",
            }:
                raise ValueError("valid semantic review has incompatible status")
        else:
            if any(value is not None for value in disposition_fields):
                raise ValueError("invalid semantic review cannot carry disposition")
            if self.status not in {
                "SEMANTIC_HARD_GATE_FAILED",
                "SEMANTIC_REVIEW_INVALID",
            }:
                raise ValueError("invalid semantic review has incompatible status")

        if self.status == "PRE_N10_ENTRY_AUTHORIZED":
            if self.semantic_disposition != "PASS":
                raise ValueError("authorized pre-N10 entry requires PASS disposition")
            if self.semantic_failed_dimensions:
                raise ValueError("authorized pre-N10 entry cannot retain failures")
        if self.status == "SEMANTIC_INTERVENTION_REQUIRED":
            if self.semantic_disposition != "REQUIRES_SEMANTIC_INTERVENTION":
                raise ValueError("semantic intervention disposition mismatch")
            if not self.semantic_failed_dimensions:
                raise ValueError("semantic intervention requires failed dimension")

        review_fields = (
            self.source_semantic_review_path,
            self.source_semantic_review_file_sha256,
            self.source_semantic_review_id,
        )
        if self.semantic_review_valid and any(value is None for value in review_fields):
            raise ValueError("valid semantic review requires source review artifact")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("initial semantic gate SHA mismatch")
        if observed_id != "pre_n10_initial_semantic_gate_v1:" + expected_sha[:20]:
            raise ValueError("initial semantic gate ID mismatch")
        return self


def _validate_run_portfolio(
    *,
    run: HypothesisSemanticRunRecord,
    portfolio: HypothesisPortfolio,
) -> None:
    if run.portfolio_id != portfolio.portfolio_id:
        raise ValueError("semantic run/portfolio ID mismatch")
    portfolio_sha = _sha256_json(portfolio)
    if run.portfolio_sha256 != portfolio_sha:
        raise ValueError("semantic run/portfolio SHA mismatch")


def _validate_review_run(
    *,
    review: HypothesisSemanticReview,
    run: HypothesisSemanticRunRecord,
) -> None:
    if run.review_id != review.review_id:
        raise ValueError("semantic run/review ID mismatch")
    if review.source_context_id != run.context_id:
        raise ValueError("semantic review/run context ID mismatch")
    if review.source_context_sha256 != run.context_sha256:
        raise ValueError("semantic review/run context SHA mismatch")
    if review.source_portfolio_id != run.portfolio_id:
        raise ValueError("semantic review/run portfolio ID mismatch")
    if review.source_portfolio_sha256 != run.portfolio_sha256:
        raise ValueError("semantic review/run portfolio SHA mismatch")
    if review.critic_prompt_version != run.critic_prompt_version:
        raise ValueError("semantic review/run prompt version mismatch")
    if review.critic_prompt_sha256 != run.critic_prompt_sha256:
        raise ValueError("semantic review/run prompt SHA mismatch")
    if not run.hard_gate_passed or not review.source_hard_gate_passed:
        raise ValueError("accepted semantic review requires hard-gate pass")


def execute_pre_n10_initial_semantic_gate_v1(
    *,
    portfolio_path: Path,
    semantic_run_path: Path,
    semantic_review_path: Path | None,
    output_root: Path,
) -> tuple[
    PreN10InitialSemanticGateReportV1,
    HypothesisSemanticDispositionV1 | None,
]:
    portfolio_file = portfolio_path.expanduser().resolve()
    run_file = semantic_run_path.expanduser().resolve()
    review_file = (
        semantic_review_path.expanduser().resolve()
        if semantic_review_path is not None
        else None
    )
    root = output_root.expanduser().resolve()

    if not portfolio_file.is_file():
        raise ValueError("missing initial semantic source portfolio: " + str(portfolio_file))
    if not run_file.is_file():
        raise ValueError("missing initial semantic run artifact: " + str(run_file))

    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_file.read_text(encoding="utf-8")
    )
    run = HypothesisSemanticRunRecord.model_validate_json(
        run_file.read_text(encoding="utf-8")
    )
    _validate_run_portfolio(run=run, portfolio=portfolio)

    review: HypothesisSemanticReview | None = None
    disposition: HypothesisSemanticDispositionV1 | None = None
    review_file_sha: str | None = None

    if not run.accepted:
        if review_file is not None and review_file.exists():
            raise ValueError("non-accepted semantic run cannot carry accepted review artifact")
        status: InitialSemanticGateStatusV1 = (
            "SEMANTIC_HARD_GATE_FAILED"
            if not run.hard_gate_passed or run.failure_stage == "hard_gate"
            else "SEMANTIC_REVIEW_INVALID"
        )
        review_valid = False
        admissible = False
        failed_dimensions: list[str] = []
        warning_dimensions: list[str] = []
    else:
        if run.failure_stage != "none":
            raise ValueError("accepted semantic run must have failure_stage=none")
        if not run.hard_gate_passed:
            raise ValueError("accepted semantic run must pass hard gate")
        if run.review_id is None:
            raise ValueError("accepted semantic run lacks review ID")
        if review_file is None or not review_file.is_file():
            raise ValueError("accepted semantic run requires review artifact")

        review = HypothesisSemanticReview.model_validate_json(
            review_file.read_text(encoding="utf-8")
        )
        review_file_sha = _sha256_file(review_file)
        _validate_review_run(review=review, run=run)
        disposition = compile_hypothesis_semantic_disposition_v1(
            review=review,
            portfolio=portfolio,
        )
        disposition_path = root / "semantic.disposition.json"
        _write_exact_or_validate(disposition_path, disposition)

        review_valid = True
        admissible = disposition.semantic_admissible_for_pre_n10
        failed_dimensions = list(disposition.failed_dimensions)
        warning_dimensions = list(disposition.warning_dimensions)
        status = (
            "PRE_N10_ENTRY_AUTHORIZED"
            if admissible
            else "SEMANTIC_INTERVENTION_REQUIRED"
        )

    body = {
        "schema_version": "pre-n10-initial-semantic-gate-report-v1",
        "source_portfolio_path": str(portfolio_file),
        "source_portfolio_file_sha256": _sha256_file(portfolio_file),
        "source_portfolio_id": portfolio.portfolio_id,
        "source_portfolio_sha256": _sha256_json(portfolio),
        "source_semantic_run_path": str(run_file),
        "source_semantic_run_file_sha256": _sha256_file(run_file),
        "source_semantic_run_id": run.run_id,
        "source_semantic_review_path": (
            str(review_file) if review is not None and review_file is not None else None
        ),
        "source_semantic_review_file_sha256": review_file_sha,
        "source_semantic_review_id": review.review_id if review is not None else None,
        "semantic_disposition_path": (
            str(root / "semantic.disposition.json") if disposition is not None else None
        ),
        "semantic_disposition_id": (
            disposition.disposition_id if disposition is not None else None
        ),
        "semantic_disposition_sha256": (
            disposition.disposition_sha256 if disposition is not None else None
        ),
        "semantic_disposition": (
            disposition.disposition if disposition is not None else None
        ),
        "status": status,
        "semantic_review_valid": review_valid,
        "semantic_admissible_for_pre_n10": admissible,
        "pre_n10_entry_authorized": admissible,
        "semantic_failed_dimensions": failed_dimensions,
        "semantic_warning_dimensions": warning_dimensions,
        "semantic_runtime_reexecuted": False,
        "semantic_critic_llm_reinvoked": False,
        "external_novelty_performed": False,
        "n9_performed": False,
        "n10_performed": False,
        "repair_performed": False,
        "regeneration_performed": False,
        "endpoint_binding_performed": False,
        "verifier_performed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    report = PreN10InitialSemanticGateReportV1(
        **body,
        report_id="pre_n10_initial_semantic_gate_v1:" + digest[:20],
        report_sha256=digest,
    )
    _write_exact_or_validate(root / "initial_semantic_gate.report.json", report)
    return report, disposition


__all__ = [
    "InitialSemanticGateStatusV1",
    "PreN10InitialSemanticGateReportV1",
    "execute_pre_n10_initial_semantic_gate_v1",
]
