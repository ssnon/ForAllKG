from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from campaigns.sers_alpha4_epoch.alpha4.alpha4c5f_reserve import (
    sha256_file,
    sha256_json,
    stable_id,
)
from campaigns.sers_alpha4_epoch.reserve_b.alpha4c5h1_reserve_b import (
    ALPHA4C5H1_PROTOCOL_SEMANTICS_ID,
    EXPECTED_5H_FREEZE_ID,
    Alpha4c5h1Protocol,
    load_h1_protocol,
    verify_h1_protocol,
)
from dac_her.hypothesis_trend_directional_run_record import (
    HYPOTHESIS_TREND_DIRECTIONAL_RUNTIME_SEMANTICS_ID,
)
from dac_her.hypothesis_trend_directional_validator import (
    HYPOTHESIS_TREND_DIRECTIONAL_VALIDATOR_SEMANTICS_ID,
)


ALPHA4C5H2_POSTMORTEM_SEMANTICS_ID = (
    "sers_alpha4c5h2_reserve_b_postmortem_v1"
)
ALPHA4C5H2_POSTMORTEM_SCHEMA_VERSION = (
    "sers-alpha4c5h2-reserve-b-postmortem-v1"
)
EXPECTED_SOURCE_CAMPAIGN_ID = "sers_alpha4c5h1_reserve_b_v1"
EXPECTED_RESERVE_B_COUNT = 25
EXPECTED_FAILURE_ERROR_CODES = (
    "NONCANONICAL_TREND_DIRECTION_FRAME",
    "PARTIAL_PAPER_ABSENCE_CLAIM",
)

DEFAULT_SOURCE_PROTOCOL = Path(
    "evaluation/sers_alpha4c5h1/reserve_b_v1/control/"
    "execution_protocol.json"
)
DEFAULT_SOURCE_EVALUATION_ROOT = Path(
    "evaluation/sers_alpha4c5h1/reserve_b_v1"
)
DEFAULT_POSTMORTEM_ROOT = Path(
    "evaluation/sers_alpha4c5h2/reserve_b_postmortem_v1"
)
DEFAULT_POSTMORTEM_MANIFEST = (
    DEFAULT_POSTMORTEM_ROOT / "postmortem_manifest.json"
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FrozenArtifactBinding(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class FailureIssueSummary(StrictModel):
    severity: Literal["error", "warning"]
    code: str = Field(min_length=1)
    count: int = Field(ge=1)


class Alpha4c5h2PostmortemManifest(StrictModel):
    schema_version: Literal[
        "sers-alpha4c5h2-reserve-b-postmortem-v1"
    ] = ALPHA4C5H2_POSTMORTEM_SCHEMA_VERSION

    semantics_id: str
    postmortem_id: str
    postmortem_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_protocol_id: str
    source_protocol_sha256: str
    source_protocol_semantics_id: str
    source_campaign_id: Literal["sers_alpha4c5h1_reserve_b_v1"]
    source_partition: Literal["reserve_b"]
    source_domain_profile_id: Literal["sers_au_ag"]
    source_five_h_freeze_id: str

    reserve_paper_ids: list[str]
    paper_count: Literal[25]
    reserve_consumed: Literal[True]
    campaign_terminal_state: Literal["fail"]
    campaign_closed: Literal[True]
    failure_stage: Literal["validation"]

    expected_failure_error_codes: list[str]
    observed_error_codes: list[str]
    observed_issues: list[FailureIssueSummary]

    maker_generation_attempts: int = Field(ge=1)
    maker_repair_attempts: int = Field(ge=0, le=1)
    maker_final_validation_passed: Literal[False]
    maker_validation_errors: int = Field(ge=1)
    maker_validation_warnings: int = Field(ge=0)

    execution_protocol: FrozenArtifactBinding
    consumption_marker: FrozenArtifactBinding
    failure_marker: FrozenArtifactBinding
    campaign_manifest: FrozenArtifactBinding
    command_log: FrozenArtifactBinding
    maker_run: FrozenArtifactBinding
    maker_validation: FrozenArtifactBinding
    maker_draft_artifacts: list[FrozenArtifactBinding]
    maker_rejected_portfolio: FrozenArtifactBinding
    maker_auxiliary_artifacts: list[FrozenArtifactBinding] = Field(
        default_factory=list
    )

    pass_marker_absent: Literal[True]
    downstream_evaluation_absent: Literal[True]

    rerun_allowed: Literal[False]
    reserve_reuse_allowed: Literal[False]
    trend_tuning_authorized: Literal[False]
    precision_tuning_authorized: Literal[False]
    maker_acceptance_weakening_authorized: Literal[False]
    scientific_rollback_authorized: Literal[False]
    reserve_b_failure_authorizes_tuning: Literal[False]

    descriptive_only: Literal[True]
    scientific_transformation_performed: Literal[False]
    scientific_values_printed: Literal[False]
    llm_calls: Literal[0]

    @model_validator(mode="after")
    def _consistency(self) -> "Alpha4c5h2PostmortemManifest":
        if self.semantics_id != ALPHA4C5H2_POSTMORTEM_SEMANTICS_ID:
            raise ValueError("alpha4c.5h.2 semantics mismatch.")
        if self.source_protocol_semantics_id != (
            ALPHA4C5H1_PROTOCOL_SEMANTICS_ID
        ):
            raise ValueError("Unexpected source 5h.1 protocol semantics.")
        if self.source_five_h_freeze_id != EXPECTED_5H_FREEZE_ID:
            raise ValueError("Unexpected source 5h freeze ID.")
        if self.reserve_paper_ids != sorted(set(self.reserve_paper_ids)):
            raise ValueError("Reserve-B paper IDs must be sorted/unique.")
        if len(self.reserve_paper_ids) != EXPECTED_RESERVE_B_COUNT:
            raise ValueError("Reserve B must contain exactly 25 papers.")
        if self.expected_failure_error_codes != sorted(
            EXPECTED_FAILURE_ERROR_CODES
        ):
            raise ValueError("Expected failure signature drifted.")
        if self.observed_error_codes != self.expected_failure_error_codes:
            raise ValueError(
                "Observed Reserve-B error-code signature does not match "
                "the frozen alpha4c.5h.2 incident signature."
            )
        if not self.maker_draft_artifacts:
            raise ValueError("At least one Maker draft must be frozen.")
        if len(self.maker_draft_artifacts) != (
            1 + self.maker_repair_attempts
        ):
            raise ValueError(
                "Maker draft count must equal initial draft + repairs."
            )

        expected_id = stable_id(
            "sers_alpha4c5h2_reserve_b_postmortem",
            self.semantics_id,
            self.source_protocol_sha256,
            self.consumption_marker.sha256,
            self.failure_marker.sha256,
            self.campaign_manifest.sha256,
            self.maker_run.sha256,
            self.maker_validation.sha256,
            ",".join(self.observed_error_codes),
        )
        if self.postmortem_id != expected_id:
            raise ValueError("alpha4c.5h.2 postmortem_id is not stable.")

        payload = self.model_dump(mode="json")
        observed = str(payload.pop("postmortem_sha256", ""))
        expected = sha256_json(payload)
        if observed != expected:
            raise ValueError("alpha4c.5h.2 postmortem SHA mismatch.")
        return self


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _relative(root: Path, path: Path) -> str:
    root_resolved = root.resolve()
    path_resolved = path.resolve()
    try:
        return path_resolved.relative_to(root_resolved).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Artifact must remain inside repository root: {path}"
        ) from exc


def artifact_binding(
    *,
    root: Path,
    path: Path,
) -> FrozenArtifactBinding:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return FrozenArtifactBinding(
        path=_relative(root, path),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _assert_lineage(
    payload: Mapping[str, Any],
    *,
    artifact_name: str,
    protocol: Alpha4c5h1Protocol,
    require_protocol_sha: bool = True,
) -> None:
    if payload.get("campaign_id") != protocol.campaign_id:
        raise ValueError(
            f"{artifact_name} campaign_id does not match 5h.1 protocol."
        )
    if payload.get("protocol_id") != protocol.protocol_id:
        raise ValueError(
            f"{artifact_name} protocol_id does not match 5h.1 protocol."
        )
    if require_protocol_sha and payload.get("protocol_sha256") != (
        protocol.protocol_sha256
    ):
        raise ValueError(
            f"{artifact_name} protocol_sha256 does not match 5h.1 protocol."
        )


def _issue_summary(
    validation: Mapping[str, Any],
) -> tuple[list[str], list[FailureIssueSummary], int, int]:
    issues = validation.get("issues")
    if not isinstance(issues, list):
        raise ValueError("Maker validation issues must be a list.")

    counter: Counter[tuple[str, str]] = Counter()
    for index, row in enumerate(issues):
        if not isinstance(row, dict):
            raise ValueError(
                f"Maker validation issue[{index}] must be an object."
            )
        severity = str(row.get("severity", "")).strip()
        code = str(row.get("code", "")).strip()
        if severity not in {"error", "warning"}:
            raise ValueError(
                f"Maker validation issue[{index}] severity invalid."
            )
        if not code:
            raise ValueError(
                f"Maker validation issue[{index}] code is empty."
            )
        counter[(severity, code)] += 1

    summaries = [
        FailureIssueSummary(
            severity=severity,  # type: ignore[arg-type]
            code=code,
            count=count,
        )
        for (severity, code), count in sorted(counter.items())
    ]
    error_count = sum(
        row.count for row in summaries if row.severity == "error"
    )
    warning_count = sum(
        row.count for row in summaries if row.severity == "warning"
    )
    error_codes = sorted(
        row.code for row in summaries if row.severity == "error"
    )
    return error_codes, summaries, error_count, warning_count


def _validate_source_protocol(
    *,
    root: Path,
    protocol_path: Path,
) -> Alpha4c5h1Protocol:
    if not protocol_path.is_file():
        raise FileNotFoundError(
            f"5h.1 execution protocol missing: {protocol_path}"
        )
    protocol = load_h1_protocol(protocol_path)
    if protocol.campaign_id != EXPECTED_SOURCE_CAMPAIGN_ID:
        raise ValueError("Unexpected source campaign.")
    if protocol.reserve_partition != "reserve_b":
        raise ValueError("Source protocol is not Reserve B.")
    if protocol.domain_profile_id != "sers_au_ag":
        raise ValueError("Unexpected source domain profile.")
    if protocol.semantics_id != ALPHA4C5H1_PROTOCOL_SEMANTICS_ID:
        raise ValueError("Unexpected source 5h.1 semantics.")
    if protocol.five_h_freeze_id != EXPECTED_5H_FREEZE_ID:
        raise ValueError("Unexpected source 5h freeze.")
    if protocol.reserve_paper_ids != sorted(set(protocol.reserve_paper_ids)):
        raise ValueError("Source Reserve-B paper IDs are not sorted/unique.")
    if len(protocol.reserve_paper_ids) != EXPECTED_RESERVE_B_COUNT:
        raise ValueError("Source Reserve B must contain exactly 25 papers.")

    issues = verify_h1_protocol(root=root, protocol=protocol)
    if issues:
        raise ValueError(
            "Frozen 5h.1 protocol no longer verifies:\n- "
            + "\n- ".join(issues)
        )
    return protocol


def _collect_maker_artifacts(
    *,
    root: Path,
    hypothesis_root: Path,
    repair_attempts: int,
) -> tuple[
    FrozenArtifactBinding,
    FrozenArtifactBinding,
    list[FrozenArtifactBinding],
    FrozenArtifactBinding,
    list[FrozenArtifactBinding],
]:
    prefix = hypothesis_root / "reserve_maker"
    run_path = Path(str(prefix) + ".run.json")
    validation_path = Path(str(prefix) + ".validation.json")
    rejected_path = Path(str(prefix) + ".rejected_portfolio.json")
    accepted_path = Path(str(prefix) + ".portfolio.json")

    if accepted_path.exists():
        raise ValueError(
            "Accepted Maker portfolio exists; this is not the frozen "
            "Reserve-B validation-failure incident."
        )
    if not rejected_path.is_file():
        raise FileNotFoundError(
            "Rejected Maker portfolio missing for validation failure."
        )

    drafts = [Path(str(prefix) + ".draft.json")]
    drafts.extend(
        Path(str(prefix) + f".repair{index}.draft.json")
        for index in range(1, repair_attempts + 1)
    )
    for path in drafts:
        if not path.is_file():
            raise FileNotFoundError(f"Expected Maker draft missing: {path}")

    unexpected_repairs = sorted(
        path
        for path in hypothesis_root.glob("reserve_maker.repair*.draft.json")
        if path not in set(drafts[1:])
    )
    if unexpected_repairs:
        raise ValueError(
            "Unexpected Maker repair drafts exist: "
            + ", ".join(_relative(root, p) for p in unexpected_repairs)
        )

    dedicated = {
        run_path.resolve(),
        validation_path.resolve(),
        rejected_path.resolve(),
        *(path.resolve() for path in drafts),
    }
    auxiliary = []
    for path in sorted(hypothesis_root.glob("reserve_maker.*")):
        if not path.is_file():
            continue
        if path.resolve() in dedicated:
            continue
        auxiliary.append(artifact_binding(root=root, path=path))

    return (
        artifact_binding(root=root, path=run_path),
        artifact_binding(root=root, path=validation_path),
        [artifact_binding(root=root, path=path) for path in drafts],
        artifact_binding(root=root, path=rejected_path),
        auxiliary,
    )


def build_postmortem_manifest(
    *,
    root: Path,
    protocol_path: Path,
) -> Alpha4c5h2PostmortemManifest:
    root = root.resolve()
    protocol_path = (
        protocol_path
        if protocol_path.is_absolute()
        else root / protocol_path
    )
    protocol = _validate_source_protocol(
        root=root,
        protocol_path=protocol_path,
    )

    source_eval_root = root / protocol.evaluation_root
    if source_eval_root.resolve() != (
        root / DEFAULT_SOURCE_EVALUATION_ROOT
    ).resolve():
        raise ValueError(
            "Unexpected 5h.1 evaluation_root for the Reserve-B incident."
        )

    marker_path = source_eval_root / "consumption_started.json"
    fail_path = source_eval_root / "CAMPAIGN_FAIL.json"
    pass_path = source_eval_root / "CAMPAIGN_PASS.json"
    campaign_path = source_eval_root / "campaign_manifest.json"
    command_log_path = source_eval_root / "command_log.jsonl"
    hypothesis_root = source_eval_root / "hypothesis"
    run_path = hypothesis_root / "reserve_maker.run.json"
    validation_path = hypothesis_root / "reserve_maker.validation.json"
    downstream_evaluation = hypothesis_root / "reserve_evaluation.json"

    if pass_path.exists():
        raise ValueError(
            "CAMPAIGN_PASS.json exists; refusing failure postmortem freeze."
        )
    if downstream_evaluation.exists():
        raise ValueError(
            "Downstream 5e evaluation exists; expected Maker validation "
            "failure before downstream evaluation."
        )

    marker = read_json(marker_path)
    failure = read_json(fail_path)
    campaign = read_json(campaign_path)
    run = read_json(run_path)
    validation = read_json(validation_path)

    _assert_lineage(
        marker,
        artifact_name="consumption marker",
        protocol=protocol,
    )
    _assert_lineage(
        failure,
        artifact_name="failure marker",
        protocol=protocol,
    )
    _assert_lineage(
        campaign,
        artifact_name="campaign manifest",
        protocol=protocol,
    )

    if marker.get("reserve_consumed") is not True:
        raise ValueError("Reserve-B consumption marker is not consumed=true.")
    if sorted(map(str, marker.get("paper_ids", []))) != (
        protocol.reserve_paper_ids
    ):
        raise ValueError("Consumption marker paper set mismatch.")
    if marker.get("reserve_partition") != "reserve_b":
        raise ValueError("Consumption marker partition mismatch.")
    if marker.get("trend_semantics_id") != protocol.trend_semantics_id:
        raise ValueError("Consumption marker Trend semantics mismatch.")

    if failure.get("reserve_consumed") is not True:
        raise ValueError("Failure marker must preserve consumed=true.")
    if failure.get("accepted") is not False:
        raise ValueError("Failure marker accepted must be false.")
    if failure.get("rerun_allowed") is not False:
        raise ValueError("Failure marker must forbid rerun.")
    if failure.get("reserve_b_failure_authorizes_tuning") is not False:
        raise ValueError("Failure marker must forbid tuning authorization.")
    if failure.get("automatic_scientific_output_rollback") is not False:
        raise ValueError("Failure marker must forbid scientific rollback.")
    if campaign.get("state") != "fail":
        raise ValueError("Campaign manifest is not terminal state=fail.")
    if campaign.get("reserve_consumed") is not True:
        raise ValueError("Campaign manifest must preserve consumed=true.")

    if run.get("runtime_semantics_id") != (
        HYPOTHESIS_TREND_DIRECTIONAL_RUNTIME_SEMANTICS_ID
    ):
        raise ValueError("Unexpected Maker runtime semantics.")
    if run.get("final_validation_passed") is not False:
        raise ValueError("Maker run must have final_validation_passed=false.")
    if run.get("failure_stage") != "validation":
        raise ValueError(
            "Maker failure_stage must be exactly 'validation'."
        )
    generation_attempts = int(run.get("generation_attempts", -1))
    repair_attempts = int(run.get("repair_attempts", -1))
    if generation_attempts < 1:
        raise ValueError("Maker generation_attempts must be >= 1.")
    if repair_attempts not in {0, 1}:
        raise ValueError("Maker repair_attempts must be 0 or 1.")
    if int(run.get("max_repairs", -1)) != 1:
        raise ValueError("Maker max_repairs drifted from frozen policy.")

    if validation.get("semantics_id") != (
        HYPOTHESIS_TREND_DIRECTIONAL_VALIDATOR_SEMANTICS_ID
    ):
        raise ValueError("Unexpected Maker validator semantics.")
    if validation.get("passes") is not False:
        raise ValueError("Maker validation must have passes=false.")

    (
        error_codes,
        issue_summaries,
        error_count,
        warning_count,
    ) = _issue_summary(validation)

    expected_codes = sorted(EXPECTED_FAILURE_ERROR_CODES)
    if error_codes != expected_codes:
        raise ValueError(
            "Reserve-B validation error signature mismatch: "
            f"expected={expected_codes}, observed={error_codes}"
        )
    if error_count < 1:
        raise ValueError("Maker validation must contain an error.")
    if int(validation.get("errors", -1)) != error_count:
        raise ValueError("Maker validation error count drifted.")
    if int(validation.get("warnings", -1)) != warning_count:
        raise ValueError("Maker validation warning count drifted.")
    if int(run.get("validation_errors", -1)) != error_count:
        raise ValueError("Maker run validation_errors mismatch.")
    if int(run.get("validation_warnings", -1)) != warning_count:
        raise ValueError("Maker run validation_warnings mismatch.")

    (
        run_binding,
        validation_binding,
        draft_bindings,
        rejected_binding,
        auxiliary_bindings,
    ) = _collect_maker_artifacts(
        root=root,
        hypothesis_root=hypothesis_root,
        repair_attempts=repair_attempts,
    )

    values: dict[str, Any] = {
        "schema_version": ALPHA4C5H2_POSTMORTEM_SCHEMA_VERSION,
        "semantics_id": ALPHA4C5H2_POSTMORTEM_SEMANTICS_ID,
        "postmortem_id": "",
        "postmortem_sha256": "",
        "source_protocol_id": protocol.protocol_id,
        "source_protocol_sha256": protocol.protocol_sha256,
        "source_protocol_semantics_id": protocol.semantics_id,
        "source_campaign_id": protocol.campaign_id,
        "source_partition": protocol.reserve_partition,
        "source_domain_profile_id": protocol.domain_profile_id,
        "source_five_h_freeze_id": protocol.five_h_freeze_id,
        "reserve_paper_ids": protocol.reserve_paper_ids,
        "paper_count": len(protocol.reserve_paper_ids),
        "reserve_consumed": True,
        "campaign_terminal_state": "fail",
        "campaign_closed": True,
        "failure_stage": "validation",
        "expected_failure_error_codes": expected_codes,
        "observed_error_codes": error_codes,
        "observed_issues": issue_summaries,
        "maker_generation_attempts": generation_attempts,
        "maker_repair_attempts": repair_attempts,
        "maker_final_validation_passed": False,
        "maker_validation_errors": error_count,
        "maker_validation_warnings": warning_count,
        "execution_protocol": artifact_binding(
            root=root, path=protocol_path
        ),
        "consumption_marker": artifact_binding(
            root=root, path=marker_path
        ),
        "failure_marker": artifact_binding(
            root=root, path=fail_path
        ),
        "campaign_manifest": artifact_binding(
            root=root, path=campaign_path
        ),
        "command_log": artifact_binding(
            root=root, path=command_log_path
        ),
        "maker_run": run_binding,
        "maker_validation": validation_binding,
        "maker_draft_artifacts": draft_bindings,
        "maker_rejected_portfolio": rejected_binding,
        "maker_auxiliary_artifacts": auxiliary_bindings,
        "pass_marker_absent": True,
        "downstream_evaluation_absent": True,
        "rerun_allowed": False,
        "reserve_reuse_allowed": False,
        "trend_tuning_authorized": False,
        "precision_tuning_authorized": False,
        "maker_acceptance_weakening_authorized": False,
        "scientific_rollback_authorized": False,
        "reserve_b_failure_authorizes_tuning": False,
        "descriptive_only": True,
        "scientific_transformation_performed": False,
        "scientific_values_printed": False,
        "llm_calls": 0,
    }

    values["postmortem_id"] = stable_id(
        "sers_alpha4c5h2_reserve_b_postmortem",
        values["semantics_id"],
        values["source_protocol_sha256"],
        values["consumption_marker"].sha256,
        values["failure_marker"].sha256,
        values["campaign_manifest"].sha256,
        values["maker_run"].sha256,
        values["maker_validation"].sha256,
        ",".join(values["observed_error_codes"]),
    )

    provisional = Alpha4c5h2PostmortemManifest.model_construct(**values)
    payload = provisional.model_dump(mode="json")
    payload.pop("postmortem_sha256", None)
    values["postmortem_sha256"] = sha256_json(payload)
    return Alpha4c5h2PostmortemManifest.model_validate(values)


def load_postmortem_manifest(
    path: Path,
) -> Alpha4c5h2PostmortemManifest:
    return Alpha4c5h2PostmortemManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def iter_frozen_bindings(
    manifest: Alpha4c5h2PostmortemManifest,
) -> Iterable[FrozenArtifactBinding]:
    yield manifest.execution_protocol
    yield manifest.consumption_marker
    yield manifest.failure_marker
    yield manifest.campaign_manifest
    yield manifest.command_log
    yield manifest.maker_run
    yield manifest.maker_validation
    yield from manifest.maker_draft_artifacts
    yield manifest.maker_rejected_portfolio
    yield from manifest.maker_auxiliary_artifacts


def verify_postmortem_manifest(
    *,
    root: Path,
    manifest: Alpha4c5h2PostmortemManifest,
) -> list[str]:
    root = root.resolve()
    issues: list[str] = []
    seen: set[str] = set()
    for binding in iter_frozen_bindings(manifest):
        if binding.path in seen:
            issues.append(f"duplicate frozen artifact: {binding.path}")
            continue
        seen.add(binding.path)
        path = root / binding.path
        if not path.is_file():
            issues.append(f"frozen artifact missing: {binding.path}")
            continue
        observed = sha256_file(path)
        if observed != binding.sha256:
            issues.append(
                "frozen artifact SHA drift: "
                f"{binding.path}: expected={binding.sha256}, "
                f"observed={observed}"
            )
        if path.stat().st_size != binding.size_bytes:
            issues.append(
                f"frozen artifact size drift: {binding.path}"
            )

    source_eval_root = root / DEFAULT_SOURCE_EVALUATION_ROOT
    if (source_eval_root / "CAMPAIGN_PASS.json").exists():
        issues.append("closed Reserve-B campaign unexpectedly has PASS marker")
    if (
        source_eval_root
        / "hypothesis"
        / "reserve_evaluation.json"
    ).exists():
        issues.append(
            "closed Reserve-B campaign unexpectedly has downstream evaluation"
        )
    return sorted(set(issues))
