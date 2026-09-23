from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


InitialIntakeDisposition = Literal[
    "READY_FOR_RELATIONAL_PREVERIFIER",
    "UPSTREAM_NO_RELATIONAL_BINDING_INPUTS",
]


class ProspectiveRoutedInitialIntakeV2Report(StrictModel):
    schema_version: Literal[
        "prospective-routed-initial-intake-v2"
    ] = "prospective-routed-initial-intake-v2"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    case_id: Literal["P16", "P17", "P18", "P19", "P20"]
    run_dir: str

    main_e2e_manifest_present: bool
    main_e2e_manifest_status: str | None = None

    alpha4_portfolio_present: bool
    alpha4_hypothesis_count: int | None = Field(default=None, ge=0)

    n10_candidate_portfolio_present: bool
    n10_certification_present: bool

    disposition: InitialIntakeDisposition
    reason_codes: list[str]

    binding_plan_should_run: bool
    gate_v2_should_run: bool
    routed_dispatch_should_run: bool

    scientific_mutation_performed: Literal[False] = False
    llm_calls_performed: Literal[0] = 0
    retrieval_performed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    verifier_result_observed: Literal[False] = False

    legacy_semantics_source: Literal[
        "prospective_relational_campaign.UPSTREAM_NO_RELATIONAL_BINDING_INPUTS"
    ] = (
        "prospective_relational_campaign."
        "UPSTREAM_NO_RELATIONAL_BINDING_INPUTS"
    )

    @model_validator(mode="after")
    def validate_report(self) -> "ProspectiveRoutedInitialIntakeV2Report":
        ready = self.disposition == "READY_FOR_RELATIONAL_PREVERIFIER"
        if ready:
            if not (
                self.n10_candidate_portfolio_present
                and self.n10_certification_present
            ):
                raise ValueError(
                    "ready intake requires both frozen relational binding inputs"
                )
            if not (
                self.binding_plan_should_run
                and self.gate_v2_should_run
                and self.routed_dispatch_should_run
            ):
                raise ValueError(
                    "ready intake must permit binding/gate/dispatch"
                )
        else:
            if (
                self.binding_plan_should_run
                or self.gate_v2_should_run
                or self.routed_dispatch_should_run
            ):
                raise ValueError(
                    "upstream-no-input disposition must stop pre-verifier stages"
                )
            if not self.reason_codes:
                raise ValueError(
                    "upstream-no-input disposition requires reason codes"
                )

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("initial intake-v2 SHA mismatch")
        if observed_id != (
            "prospective_routed_initial_intake_v2:" + expected_sha[:20]
        ):
            raise ValueError("initial intake-v2 ID mismatch")
        return self


def _read_json_object(path: Path) -> dict | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: " + str(path))
    return value


def build_initial_intake_v2(
    *,
    case_id: str,
    run_dir: Path,
) -> ProspectiveRoutedInitialIntakeV2Report:
    run = run_dir.expanduser().resolve()

    manifest_path = run / "e2e_runner.manifest.json"
    alpha4_path = run / "hypothesis_axis_a4.portfolio.json"
    n10_candidate_path = (
        run / "novelty_refinement_a6.n10.candidate.portfolio.json"
    )
    n10_cert_path = run / "novelty_refinement_a6.n10.certification.json"

    manifest = _read_json_object(manifest_path)
    alpha4 = _read_json_object(alpha4_path)

    alpha4_count: int | None = None
    if alpha4 is not None:
        hypotheses = alpha4.get("hypotheses")
        if isinstance(hypotheses, list):
            alpha4_count = len(hypotheses)

    candidate_present = n10_candidate_path.is_file()
    certification_present = n10_cert_path.is_file()

    reasons: list[str] = []
    if not candidate_present:
        reasons.append("missing_n10_candidate_portfolio")
    if not certification_present:
        reasons.append("missing_n10_certification")
    if alpha4_count == 0:
        reasons.insert(0, "alpha4_no_surviving_hypotheses")

    ready = candidate_present and certification_present
    disposition: InitialIntakeDisposition = (
        "READY_FOR_RELATIONAL_PREVERIFIER"
        if ready
        else "UPSTREAM_NO_RELATIONAL_BINDING_INPUTS"
    )

    body = {
        "schema_version": "prospective-routed-initial-intake-v2",
        "case_id": case_id,
        "run_dir": str(run),
        "main_e2e_manifest_present": manifest is not None,
        "main_e2e_manifest_status": (
            str(manifest.get("status"))
            if manifest is not None and manifest.get("status") is not None
            else None
        ),
        "alpha4_portfolio_present": alpha4 is not None,
        "alpha4_hypothesis_count": alpha4_count,
        "n10_candidate_portfolio_present": candidate_present,
        "n10_certification_present": certification_present,
        "disposition": disposition,
        "reason_codes": reasons,
        "binding_plan_should_run": ready,
        "gate_v2_should_run": ready,
        "routed_dispatch_should_run": ready,
        "scientific_mutation_performed": False,
        "llm_calls_performed": 0,
        "retrieval_performed": False,
        "endpoint_binding_performed": False,
        "verifier_result_observed": False,
        "legacy_semantics_source": (
            "prospective_relational_campaign."
            "UPSTREAM_NO_RELATIONAL_BINDING_INPUTS"
        ),
    }
    digest = _sha256_json(body)
    return ProspectiveRoutedInitialIntakeV2Report(
        **body,
        report_id="prospective_routed_initial_intake_v2:" + digest[:20],
        report_sha256=digest,
    )


__all__ = [
    "InitialIntakeDisposition",
    "ProspectiveRoutedInitialIntakeV2Report",
    "build_initial_intake_v2",
]
