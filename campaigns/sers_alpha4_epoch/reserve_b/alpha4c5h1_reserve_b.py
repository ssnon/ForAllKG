from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from pydantic import BaseModel, ConfigDict, model_validator

from campaigns.sers_alpha4_epoch.alpha4.alpha4c5f_reserve import (
    Alpha4c5fArtifactIds,
    Alpha4c5fExplorerPolicy,
    Alpha4c5fMakerPolicy,
    Alpha4c5fTraversalPolicy,
    sha256_file,
    sha256_json,
    stable_id,
)
from campaigns.sers_alpha4_epoch.readiness.canonical_readiness import (
    CANONICAL_READINESS_SEMANTICS_ID,
    load_and_verify_readiness_lock,
)
from dac_her.hypothesis_trend_evaluation import (
    load_protocol as load_5e_protocol,
    load_reserve_manifest,
    verify_protocol_integrity as verify_5e_protocol_integrity,
)


ALPHA4C5H1_PROTOCOL_SEMANTICS_ID = (
    "sers_alpha4c5h1_guarded_reserve_b_v1"
)
ALPHA4C5H1_RUNNER_CONTRACT_ID = (
    "alpha4c5h1_guarded_reserve_b_runner_v1"
)
EXPECTED_5H_FREEZE_ID = (
    "sers_alpha4c5h_v6r2_freeze:710786859181e21535f2"
)
EXPECTED_5H_CONFIRMATION_PROTOCOL_ID = (
    "sers_alpha4c5h_reserve_b_confirmation:7e71422f94caadf9161a"
)
EXPECTED_TREND_SEMANTICS_ID = (
    "sers_au_ag_trend_v6r2_alpha4c5g2r2"
)
EXPECTED_5E_PROTOCOL_ID = (
    "trend_hypothesis_evaluation_protocol:b97b65fe4bc66c4f5695"
)
RUNTIME_PRECISION_SEMANTICS_ID = (
    "sers_au_ag_trend_precision_v5_alpha4c21211"
)
EXPECTED_RESERVE_B_COUNT = 25

DEFAULT_5H_FREEZE_MANIFEST = Path(
    "evaluation/sers_alpha4c5h/freeze_v1/freeze_manifest.json"
)
DEFAULT_5H_CONFIRMATION_PROTOCOL = Path(
    "evaluation/sers_alpha4c5h/freeze_v1/"
    "reserve_b_confirmation_protocol.json"
)
DEFAULT_5E_PROTOCOL = Path(
    "configs/heldout/"
    "sers_alpha4c5e_trend_hypothesis_evaluation_protocol.json"
)
DEFAULT_CONTROL_ROOT = Path(
    "evaluation/sers_alpha4c5h1/reserve_b_v1/control"
)
DEFAULT_EVALUATION_ROOT = Path(
    "evaluation/sers_alpha4c5h1/reserve_b_v1"
)
DEFAULT_DATA_ROOT = Path(
    "evaluation/sers_alpha4c5h1/reserve_b_v1/work_data_sers"
)
DEFAULT_RESERVE_A_TEMPLATE_PROTOCOL = Path(
    "configs/heldout/sers_alpha4c5f2_reserve_a_v1_protocol.json"
)
EXPECTED_RESERVE_A_TEMPLATE_PROTOCOL_ID = (
    "sers_alpha4c5f2_reserve_protocol:c82a1337fb18247cb0fb"
)
EXPECTED_RESERVE_A_TEMPLATE_SEMANTICS_ID = (
    "sers_alpha4c5f2_readiness_locked_blind_reserve_e2e_v1"
)

EXECUTION_COMPONENT_PATHS = (
    "dac_her/alpha4c5h1_reserve_b.py",
    "dac_her/alpha4c5h1_runtime_bindings.py",
    "scripts/build_trend_evidence_alpha4c5h1.py",
    "scripts/build_trend_precision_alpha4c5h1.py",
    "scripts/build_cross_context_profiles_alpha4c5h1.py",
    "scripts/build_cross_context_assessments_alpha4c5h1.py",
    "scripts/run_sers_alpha4c5h1_dev_compatibility.py",
    "scripts/prepare_sers_alpha4c5h1_reserve_b_readiness.py",
    "scripts/freeze_sers_alpha4c5h1_execution_protocol.py",
    "scripts/run_sers_alpha4c5h1_reserve_b.py",
    "scripts/run_sers_alpha4c5f_reserve.py",
    "dac_her/domains/sers_au_ag_trend_alpha4c5g2r2.py",
    "dac_her/domains/sers_au_ag_trend_precision_alpha4c21211.py",
    "scripts/build_trend_evidence.py",
    "scripts/build_trend_precision.py",
    "scripts/build_cross_context_profiles.py",
    "scripts/build_cross_context_assessments.py",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def raw_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def verify_5h_freeze_command(root: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.verify_sers_alpha4c5h_freeze",
        ],
        cwd=root,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(
            "alpha4c.5h freeze verification failed before 5h.1:\n"
            + result.stdout
            + result.stderr
        )


def load_and_verify_5h_binding(
    *,
    root: Path,
    freeze_manifest_path: Path,
    confirmation_protocol_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    verify_5h_freeze_command(root)

    freeze = read_json(freeze_manifest_path)
    confirmation = read_json(confirmation_protocol_path)

    if freeze.get("freeze_id") != EXPECTED_5H_FREEZE_ID:
        raise ValueError(
            "Unexpected alpha4c.5h freeze ID: "
            f"{freeze.get('freeze_id')!r}."
        )
    if (
        confirmation.get("confirmation_protocol_id")
        != EXPECTED_5H_CONFIRMATION_PROTOCOL_ID
    ):
        raise ValueError(
            "Unexpected alpha4c.5h confirmation protocol ID."
        )
    if confirmation.get("freeze_id") != freeze.get("freeze_id"):
        raise ValueError("5h freeze/confirmation binding mismatch.")
    if freeze.get("trend_semantics_id") != EXPECTED_TREND_SEMANTICS_ID:
        raise ValueError("5h frozen Trend semantics mismatch.")
    if confirmation.get("trend_semantics_id") != EXPECTED_TREND_SEMANTICS_ID:
        raise ValueError("5h confirmation Trend semantics mismatch.")
    if (
        confirmation.get("acceptance_protocol_id")
        != EXPECTED_5E_PROTOCOL_ID
    ):
        raise ValueError("5h confirmation 5e protocol mismatch.")

    paper_ids = confirmation.get("reserve_b_paper_ids")
    if not isinstance(paper_ids, list):
        raise ValueError("5h confirmation lacks Reserve-B paper IDs.")
    normalized = sorted(str(value) for value in paper_ids)
    if (
        len(normalized) != EXPECTED_RESERVE_B_COUNT
        or len(set(normalized)) != EXPECTED_RESERVE_B_COUNT
    ):
        raise ValueError("Reserve B must contain exactly 25 unique papers.")
    if int(confirmation.get("reserve_b_paper_count", -1)) != 25:
        raise ValueError("Reserve-B paper_count drift.")
    if confirmation.get(
        "execution_policy",
        {},
    ).get("one_shot_confirmation") is not True:
        raise ValueError("Reserve-B one-shot policy drift.")
    return freeze, confirmation


def component_hashes(
    *,
    root: Path,
    paths: Iterable[str] = EXECUTION_COMPONENT_PATHS,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for rel in sorted(set(map(str, paths))):
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(
                f"5h.1 execution component missing: {rel}"
            )
        result[rel] = sha256_file(path)
    return dict(sorted(result.items()))


def verify_component_hashes(
    *,
    root: Path,
    expected: Mapping[str, str],
) -> list[str]:
    issues: list[str] = []
    for rel, wanted in sorted(expected.items()):
        path = root / str(rel)
        if not path.exists():
            issues.append(f"execution component missing: {rel}")
            continue
        observed = sha256_file(path)
        if observed != wanted:
            issues.append(
                f"execution component drift: {rel}: "
                f"expected={wanted}, observed={observed}"
            )
    return issues


class Alpha4c5h1ExecutionPolicy(StrictModel):
    exact_reserve_set_required: Literal[True] = True
    reserve_partition: Literal["reserve_b"] = "reserve_b"
    paper_override_allowed: Literal[False] = False
    evidence_mode_only: Literal[True] = True
    bridge_required: Literal[False] = False
    new_extraction_llm_allowed: Literal[False] = False
    canonical_source_copy_isolated: Literal[True] = True
    allow_critical_partial_projection: Literal[True] = True
    count_thresholds_used_for_acceptance: Literal[False] = False
    zero_trend_yield_is_execution_failure: Literal[False] = False
    zero_hypotheses_is_evaluation_failure: Literal[False] = False
    canonical_readiness_required_before_consumption: Literal[True] = True
    readiness_revalidated_immediately_before_consumption: Literal[
        True
    ] = True
    five_h_freeze_revalidated_before_consumption: Literal[True] = True
    execution_component_hashes_revalidated_before_consumption: Literal[
        True
    ] = True
    direct_consumption_marker_write_allowed: Literal[False] = False
    reserve_consumed_before_first_scientific_transformation: Literal[
        True
    ] = True
    rerun_after_consumption_allowed: Literal[False] = False
    automatic_scientific_output_rollback: Literal[False] = False
    scientific_semantics_patch_after_reserve_b_allowed: Literal[
        False
    ] = False
    reserve_b_failure_authorizes_tuning: Literal[False] = False


class Alpha4c5h1Protocol(StrictModel):
    schema_version: Literal[
        "sers-alpha4c5h1-reserve-b-protocol-v1"
    ] = "sers-alpha4c5h1-reserve-b-protocol-v1"

    protocol_id: str
    protocol_sha256: str
    semantics_id: str
    runner_contract_id: str

    campaign_id: Literal["sers_alpha4c5h1_reserve_b_v1"]
    reserve_partition: Literal["reserve_b"]
    domain_profile_id: Literal["sers_au_ag"] = "sers_au_ag"
    data_root: str
    evaluation_root: str

    five_h_freeze_manifest_path: str
    five_h_freeze_manifest_sha256: str
    five_h_freeze_id: str
    five_h_confirmation_protocol_path: str
    five_h_confirmation_protocol_sha256: str
    five_h_confirmation_protocol_id: str

    reserve_paper_ids: list[str]

    canonical_readiness_lock_path: str
    canonical_readiness_lock_file_sha256: str
    canonical_readiness_lock_payload_sha256: str
    canonical_readiness_semantics_id: str

    development_compatibility_path: str
    development_compatibility_sha256: str

    evaluation_protocol_path: str
    evaluation_protocol_id: str
    evaluation_protocol_file_sha256: str

    reserve_manifest_path: str
    reserve_manifest_id: str
    reserve_manifest_sha256: str
    reserve_manifest_file_sha256: str

    template_reserve_a_protocol_path: str
    template_reserve_a_protocol_id: str
    template_reserve_a_protocol_sha256: str

    trend_semantics_id: str
    precision_semantics_id: str

    artifact_ids: Alpha4c5fArtifactIds
    traversal: Alpha4c5fTraversalPolicy
    explorer: Alpha4c5fExplorerPolicy
    maker: Alpha4c5fMakerPolicy

    execution_component_sha256: dict[str, str]
    execution_policy: Alpha4c5h1ExecutionPolicy

    reserve_consumed_at_protocol_freeze: Literal[False] = False
    llm_calls_at_protocol_freeze: Literal[0] = 0

    @model_validator(mode="after")
    def _consistency(self) -> "Alpha4c5h1Protocol":
        if self.semantics_id != ALPHA4C5H1_PROTOCOL_SEMANTICS_ID:
            raise ValueError("alpha4c.5h.1 protocol semantics mismatch.")
        if self.runner_contract_id != ALPHA4C5H1_RUNNER_CONTRACT_ID:
            raise ValueError("alpha4c.5h.1 runner contract mismatch.")
        if self.five_h_freeze_id != EXPECTED_5H_FREEZE_ID:
            raise ValueError("5h freeze ID mismatch.")
        if (
            self.five_h_confirmation_protocol_id
            != EXPECTED_5H_CONFIRMATION_PROTOCOL_ID
        ):
            raise ValueError("5h confirmation protocol ID mismatch.")
        if self.evaluation_protocol_id != EXPECTED_5E_PROTOCOL_ID:
            raise ValueError("5e evaluation protocol ID mismatch.")
        if self.trend_semantics_id != EXPECTED_TREND_SEMANTICS_ID:
            raise ValueError("v6r2 Trend semantics mismatch.")
        if self.precision_semantics_id != RUNTIME_PRECISION_SEMANTICS_ID:
            raise ValueError("runtime precision semantics mismatch.")
        if self.reserve_paper_ids != sorted(set(self.reserve_paper_ids)):
            raise ValueError("Reserve-B paper IDs must be sorted/unique.")
        if len(self.reserve_paper_ids) != EXPECTED_RESERVE_B_COUNT:
            raise ValueError("Reserve B must contain exactly 25 papers.")
        if (
            self.canonical_readiness_semantics_id
            != CANONICAL_READINESS_SEMANTICS_ID
        ):
            raise ValueError("Canonical readiness semantics mismatch.")
        if list(self.execution_component_sha256) != sorted(
            self.execution_component_sha256
        ):
            raise ValueError("Execution component map must be sorted.")

        expected_id = stable_id(
            "sers_alpha4c5h1_reserve_b_protocol",
            self.semantics_id,
            self.runner_contract_id,
            self.campaign_id,
            self.five_h_freeze_manifest_sha256,
            self.five_h_confirmation_protocol_sha256,
            self.canonical_readiness_lock_payload_sha256,
            self.development_compatibility_sha256,
            self.evaluation_protocol_file_sha256,
            self.reserve_manifest_sha256,
            sha256_json(self.execution_component_sha256),
            ",".join(self.reserve_paper_ids),
            self.explorer.question,
            self.traversal.source_query,
            self.traversal.target_query,
        )
        if self.protocol_id != expected_id:
            raise ValueError("alpha4c.5h.1 protocol_id is not stable.")

        payload = self.model_dump(mode="json")
        observed = str(payload.pop("protocol_sha256", ""))
        expected = sha256_json(payload)
        if observed != expected:
            raise ValueError("alpha4c.5h.1 protocol SHA mismatch.")
        return self


def load_h1_protocol(path: Path) -> Alpha4c5h1Protocol:
    return Alpha4c5h1Protocol.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def protocol_sha_payload(
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact JSON-mode payload used for protocol SHA binding."""
    payload = Alpha4c5h1Protocol.model_construct(
        **dict(values)
    ).model_dump(mode="json")
    payload.pop("protocol_sha256", None)
    return payload


def discover_reserve_a_template_protocol(
    *,
    root: Path,
) -> tuple[Path, Any]:
    from campaigns.sers_alpha4_epoch.alpha4.alpha4c5f2_reserve import load_5f2_protocol

    path = root / DEFAULT_RESERVE_A_TEMPLATE_PROTOCOL
    if not path.exists():
        raise FileNotFoundError(
            "Frozen alpha4c.5f.2 Reserve-A protocol template missing: "
            f"{path}"
        )

    protocol = load_5f2_protocol(path)
    if protocol.protocol_id != EXPECTED_RESERVE_A_TEMPLATE_PROTOCOL_ID:
        raise ValueError(
            "Unexpected frozen alpha4c.5f.2 Reserve-A protocol ID: "
            f"{protocol.protocol_id!r}."
        )
    if protocol.semantics_id != EXPECTED_RESERVE_A_TEMPLATE_SEMANTICS_ID:
        raise ValueError(
            "Unexpected frozen alpha4c.5f.2 Reserve-A semantics ID: "
            f"{protocol.semantics_id!r}."
        )
    if protocol.campaign_id != "sers_alpha4c5f2_reserve_a_v1":
        raise ValueError(
            "Unexpected frozen alpha4c.5f.2 Reserve-A campaign ID: "
            f"{protocol.campaign_id!r}."
        )
    if protocol.reserve_partition != "reserve_a":
        raise ValueError(
            "Frozen alpha4c.5f.2 template is not Reserve A."
        )
    if len(protocol.reserve_paper_ids) != 25:
        raise ValueError(
            "Frozen alpha4c.5f.2 Reserve-A template must contain "
            "exactly 25 papers."
        )
    return path, protocol


def make_h1_protocol(
    *,
    root: Path,
    freeze_manifest_path: Path,
    confirmation_protocol_path: Path,
    readiness_lock_path: Path,
    development_compatibility_path: Path,
    evaluation_protocol_path: Path,
    reserve_manifest_path: Path,
    reserve_manifest: Any,
    reserve_manifest_file_sha256: str,
) -> Alpha4c5h1Protocol:
    freeze, confirmation = load_and_verify_5h_binding(
        root=root,
        freeze_manifest_path=freeze_manifest_path,
        confirmation_protocol_path=confirmation_protocol_path,
    )
    reserve_ids = sorted(
        str(value)
        for value in confirmation["reserve_b_paper_ids"]
    )

    compatibility = read_json(development_compatibility_path)
    if compatibility.get("passes_downstream_compatibility") is not True:
        raise ValueError("5h.1 DEV downstream compatibility is not PASS.")
    if compatibility.get("trend_semantics_id") != EXPECTED_TREND_SEMANTICS_ID:
        raise ValueError("DEV compatibility Trend semantics mismatch.")
    if compatibility.get("scientific_semantics_modified") is not False:
        raise ValueError("DEV compatibility reports scientific mutation.")
    if compatibility.get("llm_calls") != 0:
        raise ValueError("DEV compatibility must perform zero LLM calls.")

    lock = load_and_verify_readiness_lock(
        root=root,
        lock_path=readiness_lock_path,
        expected_paper_ids=reserve_ids,
        expected_domain_profile_id="sers_au_ag",
    )

    evaluation_protocol = load_5e_protocol(evaluation_protocol_path)
    issues = verify_5e_protocol_integrity(
        evaluation_protocol,
        root=root,
    )
    if issues:
        raise ValueError(
            "Frozen 5e protocol integrity failed:\n- "
            + "\n- ".join(issues)
        )
    if evaluation_protocol.protocol_id != EXPECTED_5E_PROTOCOL_ID:
        raise ValueError("Unexpected 5e protocol ID.")

    if reserve_manifest.protocol_id != evaluation_protocol.protocol_id:
        raise ValueError("Reserve-B manifest protocol binding mismatch.")
    if (
        reserve_manifest.protocol_sha256
        != evaluation_protocol.protocol_sha256
    ):
        raise ValueError("Reserve-B manifest protocol SHA mismatch.")
    if reserve_manifest.paper_ids != reserve_ids:
        raise ValueError("Reserve-B manifest paper set mismatch.")
    if reserve_manifest.reserve_consumed_at_registration is not False:
        raise ValueError("Reserve B was consumed at registration.")

    template_path, template = discover_reserve_a_template_protocol(
        root=root
    )

    prefix = "sers_alpha4c5h1_reserve_b_v1"
    artifact_ids = Alpha4c5fArtifactIds(
        corpus=f"{prefix}_corpus",
        measurement_result_identity=(
            f"{prefix}_measurement_identity"
        ),
        metric_definition=f"{prefix}_metric_definition",
        comparison=f"{prefix}_comparison",
        trend=f"{prefix}_trend_v6r2",
        precision=f"{prefix}_precision",
        context=f"{prefix}_context",
        assessment=f"{prefix}_assessment",
    )

    component_map = component_hashes(root=root)

    values: dict[str, Any] = {
        "protocol_id": "",
        "protocol_sha256": "",
        "semantics_id": ALPHA4C5H1_PROTOCOL_SEMANTICS_ID,
        "runner_contract_id": ALPHA4C5H1_RUNNER_CONTRACT_ID,
        "campaign_id": "sers_alpha4c5h1_reserve_b_v1",
        "reserve_partition": "reserve_b",
        "domain_profile_id": "sers_au_ag",
        "data_root": str(DEFAULT_DATA_ROOT),
        "evaluation_root": str(DEFAULT_EVALUATION_ROOT),
        "five_h_freeze_manifest_path":
            str(freeze_manifest_path.relative_to(root)),
        "five_h_freeze_manifest_sha256":
            sha256_file(freeze_manifest_path),
        "five_h_freeze_id": freeze["freeze_id"],
        "five_h_confirmation_protocol_path":
            str(confirmation_protocol_path.relative_to(root)),
        "five_h_confirmation_protocol_sha256":
            sha256_file(confirmation_protocol_path),
        "five_h_confirmation_protocol_id":
            confirmation["confirmation_protocol_id"],
        "reserve_paper_ids": reserve_ids,
        "canonical_readiness_lock_path":
            str(readiness_lock_path.relative_to(root)),
        "canonical_readiness_lock_file_sha256":
            sha256_file(readiness_lock_path),
        "canonical_readiness_lock_payload_sha256":
            lock["lock_sha256"],
        "canonical_readiness_semantics_id":
            lock["semantics_id"],
        "development_compatibility_path":
            str(development_compatibility_path.relative_to(root)),
        "development_compatibility_sha256":
            sha256_file(development_compatibility_path),
        "evaluation_protocol_path":
            str(evaluation_protocol_path.relative_to(root)),
        "evaluation_protocol_id":
            evaluation_protocol.protocol_id,
        "evaluation_protocol_file_sha256":
            sha256_file(evaluation_protocol_path),
        "reserve_manifest_path":
            str(reserve_manifest_path.relative_to(root)),
        "reserve_manifest_id": reserve_manifest.manifest_id,
        "reserve_manifest_sha256": reserve_manifest.manifest_sha256,
        "reserve_manifest_file_sha256":
            reserve_manifest_file_sha256,
        "template_reserve_a_protocol_path":
            str(template_path.relative_to(root)),
        "template_reserve_a_protocol_id": template.protocol_id,
        "template_reserve_a_protocol_sha256":
            template.protocol_sha256,
        "trend_semantics_id": EXPECTED_TREND_SEMANTICS_ID,
        "precision_semantics_id": RUNTIME_PRECISION_SEMANTICS_ID,
        "artifact_ids": artifact_ids,
        "traversal": template.traversal,
        "explorer": template.explorer,
        "maker": template.maker,
        "execution_component_sha256": component_map,
        "execution_policy": Alpha4c5h1ExecutionPolicy(),
        "reserve_consumed_at_protocol_freeze": False,
        "llm_calls_at_protocol_freeze": 0,
    }

    provisional = Alpha4c5h1Protocol.model_construct(**values)
    values["protocol_id"] = stable_id(
        "sers_alpha4c5h1_reserve_b_protocol",
        values["semantics_id"],
        values["runner_contract_id"],
        values["campaign_id"],
        values["five_h_freeze_manifest_sha256"],
        values["five_h_confirmation_protocol_sha256"],
        values["canonical_readiness_lock_payload_sha256"],
        values["development_compatibility_sha256"],
        values["evaluation_protocol_file_sha256"],
        values["reserve_manifest_sha256"],
        sha256_json(values["execution_component_sha256"]),
        ",".join(values["reserve_paper_ids"]),
        values["explorer"].question,
        values["traversal"].source_query,
        values["traversal"].target_query,
    )
    payload_for_sha = protocol_sha_payload(values)
    values["protocol_sha256"] = sha256_json(payload_for_sha)
    return Alpha4c5h1Protocol.model_validate(values)


def verify_h1_protocol(
    *,
    root: Path,
    protocol: Alpha4c5h1Protocol,
) -> list[str]:
    issues: list[str] = []

    try:
        freeze_path = root / protocol.five_h_freeze_manifest_path
        confirmation_path = (
            root / protocol.five_h_confirmation_protocol_path
        )
        freeze, confirmation = load_and_verify_5h_binding(
            root=root,
            freeze_manifest_path=freeze_path,
            confirmation_protocol_path=confirmation_path,
        )
        if sha256_file(freeze_path) != (
            protocol.five_h_freeze_manifest_sha256
        ):
            issues.append("5h freeze manifest file SHA drifted")
        if sha256_file(confirmation_path) != (
            protocol.five_h_confirmation_protocol_sha256
        ):
            issues.append("5h confirmation protocol file SHA drifted")
        if freeze["freeze_id"] != protocol.five_h_freeze_id:
            issues.append("5h freeze ID drifted")
        if confirmation["confirmation_protocol_id"] != (
            protocol.five_h_confirmation_protocol_id
        ):
            issues.append("5h confirmation protocol ID drifted")
        if sorted(confirmation["reserve_b_paper_ids"]) != (
            protocol.reserve_paper_ids
        ):
            issues.append("Reserve-B paper set drifted")
    except Exception as exc:
        issues.append(f"5h freeze verification: {exc}")

    try:
        lock_path = root / protocol.canonical_readiness_lock_path
        if sha256_file(lock_path) != (
            protocol.canonical_readiness_lock_file_sha256
        ):
            issues.append("readiness lock file SHA drifted")
        lock = load_and_verify_readiness_lock(
            root=root,
            lock_path=lock_path,
            expected_paper_ids=protocol.reserve_paper_ids,
            expected_domain_profile_id=protocol.domain_profile_id,
        )
        if lock["lock_sha256"] != (
            protocol.canonical_readiness_lock_payload_sha256
        ):
            issues.append("readiness lock payload SHA drifted")
    except Exception as exc:
        issues.append(f"canonical readiness verification: {exc}")

    compatibility_path = root / protocol.development_compatibility_path
    if not compatibility_path.exists():
        issues.append("DEV compatibility summary missing")
    else:
        if sha256_file(compatibility_path) != (
            protocol.development_compatibility_sha256
        ):
            issues.append("DEV compatibility summary SHA drifted")
        try:
            compatibility = read_json(compatibility_path)
            if compatibility.get(
                "passes_downstream_compatibility"
            ) is not True:
                issues.append("DEV downstream compatibility no longer PASS")
        except Exception as exc:
            issues.append(f"DEV compatibility parse: {exc}")

    try:
        evaluation_path = root / protocol.evaluation_protocol_path
        if sha256_file(evaluation_path) != (
            protocol.evaluation_protocol_file_sha256
        ):
            issues.append("5e protocol file SHA drifted")
        evaluation = load_5e_protocol(evaluation_path)
        evaluation_issues = verify_5e_protocol_integrity(
            evaluation,
            root=root,
        )
        issues.extend(
            f"5e protocol integrity: {value}"
            for value in evaluation_issues
        )
        if evaluation.protocol_id != protocol.evaluation_protocol_id:
            issues.append("5e protocol ID drifted")

        manifest_path = root / protocol.reserve_manifest_path
        if sha256_file(manifest_path) != (
            protocol.reserve_manifest_file_sha256
        ):
            issues.append("Reserve-B manifest file SHA drifted")
        manifest = load_reserve_manifest(manifest_path)
        if manifest.manifest_id != protocol.reserve_manifest_id:
            issues.append("Reserve-B manifest ID drifted")
        if manifest.manifest_sha256 != protocol.reserve_manifest_sha256:
            issues.append("Reserve-B manifest semantic SHA drifted")
        if manifest.paper_ids != protocol.reserve_paper_ids:
            issues.append("Reserve-B manifest paper set drifted")
        if manifest.reserve_consumed_at_registration is not False:
            issues.append("Reserve-B manifest registration state drifted")
    except Exception as exc:
        issues.append(f"5e/Reserve-B manifest verification: {exc}")

    issues.extend(
        verify_component_hashes(
            root=root,
            expected=protocol.execution_component_sha256,
        )
    )
    return sorted(set(issues))
