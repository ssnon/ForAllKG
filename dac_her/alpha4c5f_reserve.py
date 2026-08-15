from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dac_her.hypothesis_trend_evaluation import (
    TrendHypothesisEvaluationProtocol,
    TrendHypothesisReserveManifest,
    load_protocol,
    load_reserve_manifest,
    verify_protocol_integrity,
)


ALPHA4C5F_PROTOCOL_SEMANTICS_ID = (
    "sers_alpha4c5f_frozen_reserve_e2e_protocol_v1"
)

EXPECTED_SOURCE_SPLIT_GIT_BLOB = (
    "5bacf1a4c68f5a23f59303a79e90951fc8d355a6"
)
EXPECTED_SOURCE_SPLIT_SHA256 = (
    "6eebae74732070408e920154ba898841c879a8e97cb866a8971cb9feef526966"
)
EXPECTED_TREND_BASELINE_GIT_BLOB = (
    "1a251befb3e4c363d37c95f3bdec2154222f6572"
)
EXPECTED_5A_CONFIG_GIT_BLOB = (
    "147ad90dad6ff7093a9f6f8818223cb83355c9e4"
)
EXPECTED_5B_CONFIG_GIT_BLOB = (
    "18e9f2b6671e82953101b1a5f51c5d0a28b86110"
)
EXPECTED_5C_CONFIG_GIT_BLOB = (
    "b50e1e59a71d3f0c1fb82387dda39f8d84e113d2"
)

EXPECTED_5E_PROTOCOL_ID = (
    "trend_hypothesis_evaluation_protocol:b97b65fe4bc66c4f5695"
)
EXPECTED_5E_PROTOCOL_SHA256 = (
    "3674946fad1b6c867eee0e7eeb39dc7999e9b9a3392e18d7c7aaabf72744975f"
)
EXPECTED_RESERVE_MANIFEST_ID = (
    "trend_hypothesis_reserve_manifest:e2faa42699ff8dc97d0f"
)
EXPECTED_RESERVE_MANIFEST_SHA256 = (
    "1a57bde4e2ed45b4e6427e1a60cdb2f7ebc8e419b55b781fd711cbcf39375e26"
)

SOURCE_SPLIT_PATH = Path(
    "configs/heldout/sers_alpha4c4d1_trend_holdout_v2.json"
)
TREND_BASELINE_PATH = Path(
    "configs/heldout/sers_alpha4c4d2_trend_holdout_v2_run.json"
)
CONFIG_5A_PATH = Path(
    "configs/heldout/sers_alpha4c5a_trend_hypothesis_grounding.json"
)
CONFIG_5B_PATH = Path(
    "configs/heldout/sers_alpha4c5b_trend_aware_hypothesis_input.json"
)
CONFIG_5C_PATH = Path(
    "configs/heldout/sers_alpha4c5c_trend_reference_contract.json"
)
DEFAULT_5E_PROTOCOL_PATH = Path(
    "configs/heldout/sers_alpha4c5e_trend_hypothesis_evaluation_protocol.json"
)
DEFAULT_RESERVE_MANIFEST_PATH = Path(
    "configs/heldout/sers_alpha4c5e_reserve_v1.json"
)
DEFAULT_5F_PROTOCOL_PATH = Path(
    "configs/heldout/sers_alpha4c5f_reserve_protocol.json"
)

EXPECTED_RESERVE_SET = {
    "Kiwook_SERS_36",
    "Kiwook_SERS_32",
    "Kiwook_SERS_7",
    "Kiwook_SERS_20",
    "Kiwook_SERS_3",
    "Kiwook_SERS_15",
    "Kiwook_SERS_24",
    "Kiwook_SERS_29",
    "Kiwook_SERS_33",
    "Kiwook_SERS_27",
    "Kiwook_SERS_26",
    "Kiwook_SERS_31",
    "Kiwook_SERS_14",
    "Kiwook_SERS_9",
}

EXTRA_FROZEN_COMPONENT_PATHS = (
    # Upstream Trend/Comparison orchestration.
    "scripts/build_graphagents_projection.py",
    "scripts/build_corpus_graph.py",
    "scripts/build_measurement_result_identities.py",
    "scripts/build_metric_definition_contexts.py",
    "scripts/build_comparison_contexts.py",
    "scripts/build_trend_evidence.py",
    "scripts/build_trend_precision.py",
    "scripts/build_cross_context_profiles.py",
    "scripts/build_cross_context_assessments.py",
    # Graph navigation / Explorer.
    "scripts/build_navigation_graph.py",
    "scripts/build_node_index.py",
    "scripts/run_graph_traversal.py",
    "scripts/build_explorer_packet.py",
    "scripts/run_graph_explorer.py",
    "dac_her/navigation_graph.py",
    "dac_her/node_mapping.py",
    "dac_her/traversal_engine.py",
    "dac_her/endpoint_selection.py",
    "dac_her/path_bundle.py",
    "dac_her/path_quality.py",
    "dac_her/direct_concept.py",
    "dac_her/waypoint_selection.py",
    "dac_her/explorer_packet.py",
    "dac_her/explorer_contracts.py",
    "dac_her/explorer_draft.py",
    "dac_her/explorer_llm.py",
    "dac_her/explorer_prompt.py",
    "dac_her/explorer_compiler.py",
    "dac_her/explorer_normalization.py",
    "dac_her/explorer_validation.py",
    "dac_her/explorer_run_record.py",
    "dac_her/explorer_runtime.py",
    # Hypothesis context / Trend grounding / 5b input.
    "scripts/build_hypothesis_context.py",
    "scripts/build_hypothesis_trend_grounding.py",
    "scripts/build_hypothesis_trend_input.py",
    "dac_her/hypothesis_context.py",
    "dac_her/hypothesis_trend_grounding.py",
    # Frozen reserve Maker/evaluator consumers.
    "scripts/run_direction_aware_trend_hypothesis_maker.py",
    "scripts/evaluate_direction_aware_trend_hypothesis_run.py",
    # 5f itself. Freeze after installation, before reserve execution.
    "dac_her/alpha4c5f_reserve.py",
    "scripts/run_sers_alpha4c5f_reserve.py",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def git_blob(root: Path, path: Path) -> str:
    return subprocess.check_output(
        ["git", "hash-object", str(path)],
        cwd=root,
        text=True,
    ).strip()


def stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return (
        f"{prefix}:"
        f"{hashlib.sha256(raw).hexdigest()[:length]}"
    )


class Alpha4c5fArtifactIds(StrictModel):
    corpus: str
    measurement_result_identity: str
    metric_definition: str
    comparison: str
    trend: str
    precision: str
    context: str
    assessment: str


class Alpha4c5fTraversalPolicy(StrictModel):
    mode: Literal["evidence"] = "evidence"
    algorithm: Literal["top_n"] = "top_n"
    source_query: str
    target_query: str
    max_depth: int = 8
    top_k: int = 8
    node_map_k: int = 20
    endpoint_pair_k: int = 12
    reverse_penalty: float = 0.6
    node_index_model: str
    include_alignment_hubs_in_index: Literal[False] = False


class Alpha4c5fExplorerPolicy(StrictModel):
    question: str
    objective: Literal["map_evidence"] = "map_evidence"
    model: str
    temperature: Literal[0.0] = 0.0
    max_repairs: Literal[1] = 1
    parse_retries: int
    instructor_mode: str
    base_url: str | None = None
    api_key_env: Literal["OPENAI_API_KEY"] = "OPENAI_API_KEY"


class Alpha4c5fMakerPolicy(StrictModel):
    model: str
    temperature: Literal[0.0] = 0.0
    max_hypotheses: Literal[1] = 1
    max_repairs: Literal[1] = 1
    parse_retries: int
    instructor_mode: str
    base_url: str | None = None
    api_key_env: Literal["OPENAI_API_KEY"] = "OPENAI_API_KEY"


class Alpha4c5fExecutionPolicy(StrictModel):
    exact_reserve_set_required: Literal[True] = True
    paper_override_allowed: Literal[False] = False
    evidence_mode_only: Literal[True] = True
    bridge_required: Literal[False] = False
    new_extraction_llm_allowed: Literal[False] = False
    canonical_source_copy_isolated: Literal[True] = True
    allow_critical_partial_projection: Literal[True] = True
    count_thresholds_used_for_acceptance: Literal[False] = False
    zero_trend_yield_is_execution_failure: Literal[False] = False
    zero_hypotheses_is_evaluation_failure: Literal[False] = False
    reserve_consumed_before_first_scientific_transformation: Literal[
        True
    ] = True
    rerun_after_consumption_allowed: Literal[False] = False
    automatic_scientific_output_rollback: Literal[False] = False
    semantic_patch_requires_new_protocol_epoch: Literal[True] = True


class Alpha4c5fProtocol(StrictModel):
    schema_version: Literal[
        "sers-alpha4c5f-reserve-protocol-v1"
    ] = "sers-alpha4c5f-reserve-protocol-v1"

    protocol_id: str
    protocol_sha256: str
    semantics_id: str
    campaign_id: Literal["sers_alpha4c5f_reserve_v1"]
    domain_profile_id: Literal["sers_au_ag"]
    data_root: str
    evaluation_root: str

    source_split_path: str
    source_split_git_blob: str
    source_split_sha256: str
    reserve_paper_ids: list[str]

    trend_baseline_path: str
    trend_baseline_git_blob: str
    frozen_trend_semantics: dict[str, str]

    evaluation_protocol_path: str
    evaluation_protocol_id: str
    evaluation_protocol_sha256: str

    reserve_manifest_path: str
    reserve_manifest_id: str
    reserve_manifest_sha256: str

    artifact_ids: Alpha4c5fArtifactIds
    traversal: Alpha4c5fTraversalPolicy
    explorer: Alpha4c5fExplorerPolicy
    maker: Alpha4c5fMakerPolicy
    frozen_component_sha256: dict[str, str]
    execution_policy: Alpha4c5fExecutionPolicy

    reserve_consumed_at_protocol_freeze: Literal[False] = False
    llm_calls_at_protocol_freeze: Literal[0] = 0

    @model_validator(mode="after")
    def _consistency(self) -> "Alpha4c5fProtocol":
        if self.semantics_id != ALPHA4C5F_PROTOCOL_SEMANTICS_ID:
            raise ValueError("alpha4c.5f semantics mismatch.")
        if set(self.reserve_paper_ids) != EXPECTED_RESERVE_SET:
            raise ValueError(
                "alpha4c.5f reserve set is not the frozen v3 set."
            )
        if self.reserve_paper_ids != sorted(
            self.reserve_paper_ids
        ):
            raise ValueError(
                "alpha4c.5f reserve IDs must use manifest-sorted order."
            )
        if self.source_split_git_blob != (
            EXPECTED_SOURCE_SPLIT_GIT_BLOB
        ):
            raise ValueError("source split Git blob mismatch.")
        if self.source_split_sha256 != (
            EXPECTED_SOURCE_SPLIT_SHA256
        ):
            raise ValueError("source split semantic SHA mismatch.")
        if self.trend_baseline_git_blob != (
            EXPECTED_TREND_BASELINE_GIT_BLOB
        ):
            raise ValueError("Trend baseline Git blob mismatch.")
        if self.evaluation_protocol_id != (
            EXPECTED_5E_PROTOCOL_ID
        ):
            raise ValueError("5e protocol ID mismatch.")
        if self.evaluation_protocol_sha256 != (
            EXPECTED_5E_PROTOCOL_SHA256
        ):
            raise ValueError("5e protocol SHA mismatch.")
        if self.reserve_manifest_id != (
            EXPECTED_RESERVE_MANIFEST_ID
        ):
            raise ValueError("reserve manifest ID mismatch.")
        if self.reserve_manifest_sha256 != (
            EXPECTED_RESERVE_MANIFEST_SHA256
        ):
            raise ValueError("reserve manifest SHA mismatch.")
        if list(self.frozen_component_sha256) != sorted(
            self.frozen_component_sha256
        ):
            raise ValueError(
                "frozen component paths must be sorted."
            )

        expected_id = stable_id(
            "sers_alpha4c5f_reserve_protocol",
            self.semantics_id,
            self.campaign_id,
            self.source_split_sha256,
            self.evaluation_protocol_sha256,
            self.reserve_manifest_sha256,
            sha256_json(self.frozen_component_sha256),
            self.explorer.question,
            self.traversal.source_query,
            self.traversal.target_query,
        )
        if self.protocol_id != expected_id:
            raise ValueError("alpha4c.5f protocol_id is not stable.")

        payload = self.model_dump(mode="json")
        observed = str(payload.pop("protocol_sha256", ""))
        expected = sha256_json(payload)
        if observed != expected:
            raise ValueError("alpha4c.5f protocol SHA mismatch.")
        return self


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_5f_protocol(path: Path) -> Alpha4c5fProtocol:
    return Alpha4c5fProtocol.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def current_extra_component_hashes(
    root: Path,
) -> dict[str, str]:
    values: dict[str, str] = {}
    for rel in EXTRA_FROZEN_COMPONENT_PATHS:
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(
                f"alpha4c.5f component missing: {rel}"
            )
        values[rel] = sha256_file(path)
    return dict(sorted(values.items()))


def verify_source_split(
    root: Path,
) -> tuple[dict[str, Any], list[str]]:
    path = root / SOURCE_SPLIT_PATH
    if not path.exists():
        raise FileNotFoundError(path)
    observed_blob = git_blob(root, SOURCE_SPLIT_PATH)
    if observed_blob != EXPECTED_SOURCE_SPLIT_GIT_BLOB:
        raise ValueError(
            "Frozen v3 source split Git blob drifted: "
            f"{observed_blob}"
        )
    value = read_json(path)
    selection = value.get("selection", {})
    if not isinstance(selection, dict):
        raise ValueError("source split selection missing.")
    if str(selection.get("split_sha256", "")) != (
        EXPECTED_SOURCE_SPLIT_SHA256
    ):
        raise ValueError("source split SHA drifted.")
    if selection.get("scientific_content_inspected_for_split") is not False:
        raise ValueError(
            "source split no longer records blind scientific-content selection."
        )
    if selection.get("trend_outputs_inspected_for_split") is not False:
        raise ValueError(
            "source split no longer records blind Trend-output selection."
        )
    papers = value.get("papers", {})
    if not isinstance(papers, dict):
        raise ValueError("source split papers missing.")
    reserve = [
        str(item)
        for item in papers.get("reserved_future_v3", [])
    ]
    if set(reserve) != EXPECTED_RESERVE_SET:
        raise ValueError(
            "reserved_future_v3 does not equal frozen 14-paper reserve."
        )
    return value, reserve


def verify_trend_baseline(
    root: Path,
) -> dict[str, Any]:
    path = root / TREND_BASELINE_PATH
    if git_blob(root, TREND_BASELINE_PATH) != (
        EXPECTED_TREND_BASELINE_GIT_BLOB
    ):
        raise ValueError("Trend baseline config Git blob drifted.")
    value = read_json(path)
    policy = value.get("holdout_acceptance_policy", {})
    if not isinstance(policy, dict):
        raise ValueError("Trend baseline acceptance policy missing.")
    if policy.get("count_thresholds_used") is not False:
        raise ValueError(
            "Trend baseline unexpectedly uses count thresholds."
        )
    frozen = value.get("frozen_semantics", {})
    if not isinstance(frozen, dict):
        raise ValueError("Trend baseline semantics missing.")

    # Historical alpha4c.4d.2 freezes semantic IDs and its config blob,
    # but it does NOT contain per-file implementation blob locks. Do not
    # invent a stronger historical guarantee than the source artifact
    # actually records. alpha4c.5f establishes a fresh implementation
    # freeze boundary below by hashing every component it will execute
    # before the v3 reserve is consumed.
    expected_semantics = {
        "comparison": "sers_au_ag_comparison_v7_alpha4b3b321",
        "cross_context_assessment":
            "cross_context_trend_assessment_v1_alpha4c3c",
        "measurement_merge_invariant":
            "measurement_payload_isolation_v1_alpha4b4a",
        "measurement_result_identity":
            "measurement_result_identity_v1_alpha4b4a1",
        "method": "sers_au_ag_method_v4_alpha4b3b321",
        "metric_definition":
            "sers_au_ag_metric_definition_v3_alpha4c4c1",
        "trend": "sers_au_ag_trend_v5_alpha4c2121",
        "trend_context":
            "sers_au_ag_trend_context_v1_alpha4c3b",
        "trend_precision":
            "sers_au_ag_trend_precision_v5_alpha4c21211",
    }
    normalized = {
        str(key): str(item)
        for key, item in frozen.items()
    }
    if normalized != expected_semantics:
        raise ValueError(
            "alpha4c.4d.2 frozen semantic IDs drifted: "
            f"{normalized!r}"
        )
    return value


def verify_fixed_config_blobs(root: Path) -> None:
    expected = {
        CONFIG_5A_PATH: EXPECTED_5A_CONFIG_GIT_BLOB,
        CONFIG_5B_PATH: EXPECTED_5B_CONFIG_GIT_BLOB,
        CONFIG_5C_PATH: EXPECTED_5C_CONFIG_GIT_BLOB,
    }
    for path, wanted in expected.items():
        observed = git_blob(root, path)
        if observed != wanted:
            raise ValueError(
                f"Frozen config blob drifted for {path}: "
                f"{observed} != {wanted}"
            )


def verify_5e_and_reserve(
    root: Path,
    *,
    protocol_path: Path = DEFAULT_5E_PROTOCOL_PATH,
    reserve_path: Path = DEFAULT_RESERVE_MANIFEST_PATH,
) -> tuple[
    TrendHypothesisEvaluationProtocol,
    TrendHypothesisReserveManifest,
]:
    eval_protocol = load_protocol(root / protocol_path)
    if eval_protocol.protocol_id != EXPECTED_5E_PROTOCOL_ID:
        raise ValueError("unexpected 5e protocol ID.")
    if (
        eval_protocol.protocol_sha256
        != EXPECTED_5E_PROTOCOL_SHA256
    ):
        raise ValueError("unexpected 5e protocol SHA.")
    drift = verify_protocol_integrity(
        eval_protocol,
        root=root,
    )
    if drift:
        raise ValueError(
            "5e frozen implementation drift:\n"
            + "\n".join(drift)
        )

    reserve = load_reserve_manifest(root / reserve_path)
    if reserve.manifest_id != EXPECTED_RESERVE_MANIFEST_ID:
        raise ValueError("unexpected reserve manifest ID.")
    if reserve.manifest_sha256 != (
        EXPECTED_RESERVE_MANIFEST_SHA256
    ):
        raise ValueError("unexpected reserve manifest SHA.")
    if reserve.protocol_id != eval_protocol.protocol_id:
        raise ValueError("reserve is bound to another 5e protocol.")
    if reserve.protocol_sha256 != (
        eval_protocol.protocol_sha256
    ):
        raise ValueError(
            "reserve is bound to another 5e protocol SHA."
        )
    if set(reserve.paper_ids) != EXPECTED_RESERVE_SET:
        raise ValueError("reserve manifest paper set drifted.")
    return eval_protocol, reserve


def verify_5f_protocol(
    root: Path,
    protocol: Alpha4c5fProtocol,
    *,
    check_canonical_presence: bool = True,
) -> list[str]:
    issues: list[str] = []

    try:
        verify_source_split(root)
    except Exception as exc:
        issues.append(f"source_split:{exc}")
    try:
        baseline = verify_trend_baseline(root)
        observed_semantics = baseline.get(
            "frozen_semantics", {}
        )
        if observed_semantics != (
            protocol.frozen_trend_semantics
        ):
            issues.append(
                "trend_baseline:frozen semantics drift"
            )
    except Exception as exc:
        issues.append(f"trend_baseline:{exc}")
    try:
        verify_fixed_config_blobs(root)
    except Exception as exc:
        issues.append(f"alpha4c5abc:{exc}")
    try:
        eval_protocol, reserve = verify_5e_and_reserve(
            root,
            protocol_path=Path(
                protocol.evaluation_protocol_path
            ),
            reserve_path=Path(
                protocol.reserve_manifest_path
            ),
        )
        if (
            eval_protocol.protocol_id
            != protocol.evaluation_protocol_id
            or eval_protocol.protocol_sha256
            != protocol.evaluation_protocol_sha256
        ):
            issues.append("5e:protocol binding mismatch")
        if (
            reserve.manifest_id
            != protocol.reserve_manifest_id
            or reserve.manifest_sha256
            != protocol.reserve_manifest_sha256
        ):
            issues.append("reserve:manifest binding mismatch")
        if reserve.paper_ids != protocol.reserve_paper_ids:
            issues.append(
                "reserve:paper ordering/set mismatch"
            )
    except Exception as exc:
        issues.append(f"5e_or_reserve:{exc}")

    try:
        observed = current_extra_component_hashes(root)
        if observed != protocol.frozen_component_sha256:
            all_paths = sorted(
                set(observed)
                | set(protocol.frozen_component_sha256)
            )
            for rel in all_paths:
                if observed.get(rel) != (
                    protocol.frozen_component_sha256.get(rel)
                ):
                    issues.append(
                        "component_drift:"
                        f"{rel}:"
                        f"{observed.get(rel)}!="
                        f"{protocol.frozen_component_sha256.get(rel)}"
                    )
    except Exception as exc:
        issues.append(f"component_hashes:{exc}")

    if check_canonical_presence:
        for paper_id in protocol.reserve_paper_ids:
            source = (
                root
                / "data_sers"
                / "extracted"
                / paper_id
                / f"{paper_id}.graphml"
            )
            if not source.exists():
                issues.append(
                    f"canonical_missing:{paper_id}:{source}"
                )

    return issues
