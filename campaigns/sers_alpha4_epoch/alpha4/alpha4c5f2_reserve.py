from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from campaigns.sers_alpha4_epoch.alpha4.alpha4c5f_reserve import (
    Alpha4c5fArtifactIds,
    Alpha4c5fExplorerPolicy,
    Alpha4c5fMakerPolicy,
    Alpha4c5fTraversalPolicy,
)
from campaigns.sers_alpha4_epoch.readiness.canonical_readiness import (
    CANONICAL_READINESS_SEMANTICS_ID,
    load_and_verify_readiness_lock,
)
from dac_her.hypothesis_trend_evaluation import (
    TrendHypothesisReserveManifest,
    load_protocol as load_5e_protocol,
    load_reserve_manifest,
    verify_protocol_integrity as verify_5e_protocol_integrity,
)


POOL_SEMANTICS_ID = "sers_alpha4c5f2_pool_v1"
BLIND_SPLIT_SEMANTICS_ID = "sers_alpha4c5f2_blind_split_v1"
ALPHA4C5F2_PROTOCOL_SEMANTICS_ID = (
    "sers_alpha4c5f2_readiness_locked_blind_reserve_e2e_v1"
)
RUNNER_CONTRACT_ID = "alpha4c5f2_readiness_locked_reserve_runner_v1"

EXPECTED_DOMAIN_PROFILE_ID = "sers_au_ag"
EXPECTED_SOURCE_MODE = "mechanism"
EXPECTED_POOL_SIZE = 103
DEVELOPMENT_COUNT = 53
RESERVE_A_COUNT = 25
RESERVE_B_COUNT = 25
EXPECTED_RESERVE_PARTITION = "reserve_a"

DEFAULT_5E_PROTOCOL_PATH = Path(
    "configs/heldout/sers_alpha4c5e_trend_hypothesis_evaluation_protocol.json"
)
DEFAULT_LEGACY_5F_PROTOCOL_PATH = Path(
    "configs/heldout/sers_alpha4c5f_reserve_protocol.json"
)

NEW_FROZEN_COMPONENT_PATHS = (
    "dac_her/canonical_readiness.py",
    "dac_her/alpha4c5f1_sers_readiness.py",
    "scripts/prepare_sers_canonical_readiness.py",
    "dac_her/alpha4c5f2_strict_source.py",
    "dac_her/alpha4c5f2_readiness.py",
    "scripts/prepare_sers_alpha4c5f2_canonical_readiness.py",
    "dac_her/alpha4c5f2_reserve.py",
    "scripts/register_sers_alpha4c5f2_pool.py",
    "scripts/register_sers_alpha4c5f2_reserve.py",
    "scripts/freeze_sers_alpha4c5f2_reserve_protocol.py",
    "scripts/run_sers_alpha4c5f2_reserve.py",
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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


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


def _payload_sha(
    payload: Mapping[str, Any],
    sha_field: str,
) -> str:
    value = dict(payload)
    value.pop(sha_field, None)
    return sha256_json(value)


def _require_exact_payload_sha(
    payload: Mapping[str, Any],
    sha_field: str,
    label: str,
) -> None:
    observed = str(payload.get(sha_field) or "")
    expected = _payload_sha(payload, sha_field)
    if observed != expected:
        raise ValueError(
            f"{label} semantic SHA mismatch: "
            f"{observed!r} != {expected!r}"
        )


def _unique_paper_ids(values: Iterable[object]) -> list[str]:
    paper_ids = [str(value) for value in values]
    if not paper_ids:
        raise ValueError("Paper list must not be empty.")
    if len(set(paper_ids)) != len(paper_ids):
        raise ValueError("Paper list contains duplicates.")
    return paper_ids


def make_pool_manifest(
    *,
    source_manifest_path: Path,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    domain = str(source_manifest.get("domain_profile_id") or "")
    mode = str(source_manifest.get("mode") or "")
    paper_ids = _unique_paper_ids(
        source_manifest.get("paper_ids") or []
    )
    paper_count = int(source_manifest.get("paper_count", -1))

    if domain != EXPECTED_DOMAIN_PROFILE_ID:
        raise ValueError(
            f"Source corpus domain mismatch: {domain!r}."
        )
    if mode != EXPECTED_SOURCE_MODE:
        raise ValueError(
            f"Source corpus mode mismatch: {mode!r}."
        )
    if paper_count != len(paper_ids):
        raise ValueError(
            "Source manifest paper_count does not match paper_ids."
        )
    if paper_count != EXPECTED_POOL_SIZE:
        raise ValueError(
            f"alpha4c.5f.2 pool requires exactly "
            f"{EXPECTED_POOL_SIZE} papers; observed {paper_count}."
        )
    corpus_id = str(source_manifest.get("corpus_id") or "")
    if not corpus_id:
        raise ValueError("Source manifest lacks corpus_id.")

    payload: dict[str, Any] = {
        "schema_version": "sers-alpha4c5f2-pool-manifest-v1",
        "semantics_id": POOL_SEMANTICS_ID,
        "pool_id": stable_id(
            "sers_alpha4c5f2_pool",
            POOL_SEMANTICS_ID,
            corpus_id,
            ",".join(sorted(paper_ids)),
        ),
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_corpus_id": corpus_id,
        "source_domain_profile_id": domain,
        "source_mode": mode,
        "paper_count": len(paper_ids),
        "paper_ids": sorted(paper_ids),
        "split_input_fields": ["paper_id"],
        "scientific_fields_used_for_split": False,
        "comparison_results_used_for_split": False,
        "trend_results_used_for_split": False,
        "cross_context_results_used_for_split": False,
        "explorer_results_used_for_split": False,
        "maker_results_used_for_split": False,
        "reserve_consumed_at_pool_registration": False,
        "llm_calls_at_pool_registration": 0,
    }
    payload["manifest_sha256"] = _payload_sha(
        payload,
        "manifest_sha256",
    )
    return payload


def validate_pool_manifest(
    *,
    root: Path,
    pool_path: Path,
    verify_source_manifest: bool = True,
) -> dict[str, Any]:
    pool = read_json(pool_path)
    if pool.get("semantics_id") != POOL_SEMANTICS_ID:
        raise ValueError("Pool semantics mismatch.")
    _require_exact_payload_sha(
        pool,
        "manifest_sha256",
        "pool manifest",
    )
    paper_ids = _unique_paper_ids(pool.get("paper_ids") or [])
    if paper_ids != sorted(paper_ids):
        raise ValueError("Pool paper_ids must be sorted.")
    if len(paper_ids) != EXPECTED_POOL_SIZE:
        raise ValueError("Pool paper count mismatch.")
    if pool.get("paper_count") != len(paper_ids):
        raise ValueError("Pool paper_count mismatch.")
    if pool.get("source_domain_profile_id") != EXPECTED_DOMAIN_PROFILE_ID:
        raise ValueError("Pool domain mismatch.")
    if pool.get("source_mode") != EXPECTED_SOURCE_MODE:
        raise ValueError("Pool source mode mismatch.")
    for key in (
        "scientific_fields_used_for_split",
        "comparison_results_used_for_split",
        "trend_results_used_for_split",
        "cross_context_results_used_for_split",
        "explorer_results_used_for_split",
        "maker_results_used_for_split",
        "reserve_consumed_at_pool_registration",
    ):
        if pool.get(key) is not False:
            raise ValueError(f"Pool safety flag changed: {key}.")
    if pool.get("llm_calls_at_pool_registration") != 0:
        raise ValueError("Pool registration must perform zero LLM calls.")
    if pool.get("split_input_fields") != ["paper_id"]:
        raise ValueError("Blind split may consume paper_id only.")

    if verify_source_manifest:
        raw = Path(str(pool["source_manifest_path"]))
        source_path = raw if raw.is_absolute() else root / raw
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if sha256_file(source_path) != pool["source_manifest_sha256"]:
            raise ValueError("Source corpus manifest SHA drifted.")
        source = read_json(source_path)
        if sorted(source.get("paper_ids") or []) != paper_ids:
            raise ValueError("Source corpus paper IDs drifted.")
        if int(source.get("paper_count", -1)) != EXPECTED_POOL_SIZE:
            raise ValueError("Source corpus paper_count drifted.")
        if source.get("domain_profile_id") != EXPECTED_DOMAIN_PROFILE_ID:
            raise ValueError("Source corpus domain drifted.")
        if source.get("mode") != EXPECTED_SOURCE_MODE:
            raise ValueError("Source corpus mode drifted.")
    return pool


def _split_score(paper_id: str) -> str:
    raw = (
        BLIND_SPLIT_SEMANTICS_ID + "\0" + paper_id
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def make_blind_split(
    pool: Mapping[str, Any],
) -> dict[str, Any]:
    paper_ids = _unique_paper_ids(pool.get("paper_ids") or [])
    if len(paper_ids) != EXPECTED_POOL_SIZE:
        raise ValueError("Unexpected pool size for blind split.")
    ranked = sorted(
        (
            {
                "paper_id": paper_id,
                "score_sha256": _split_score(paper_id),
            }
            for paper_id in paper_ids
        ),
        key=lambda row: (
            row["score_sha256"],
            row["paper_id"],
        ),
    )
    dev_ranked = ranked[:DEVELOPMENT_COUNT]
    a_ranked = ranked[
        DEVELOPMENT_COUNT:
        DEVELOPMENT_COUNT + RESERVE_A_COUNT
    ]
    b_ranked = ranked[
        DEVELOPMENT_COUNT + RESERVE_A_COUNT:
    ]
    if len(b_ranked) != RESERVE_B_COUNT:
        raise AssertionError("Blind split count arithmetic failed.")

    payload: dict[str, Any] = {
        "schema_version": "sers-alpha4c5f2-blind-split-v1",
        "semantics_id": BLIND_SPLIT_SEMANTICS_ID,
        "split_id": stable_id(
            "sers_alpha4c5f2_blind_split",
            BLIND_SPLIT_SEMANTICS_ID,
            str(pool.get("manifest_sha256") or ""),
        ),
        "pool_id": pool["pool_id"],
        "pool_manifest_sha256": pool["manifest_sha256"],
        "paper_count": len(ranked),
        "development_count": len(dev_ranked),
        "reserve_a_count": len(a_ranked),
        "reserve_b_count": len(b_ranked),
        "development": sorted(
            row["paper_id"] for row in dev_ranked
        ),
        "reserve_a": sorted(
            row["paper_id"] for row in a_ranked
        ),
        "reserve_b": sorted(
            row["paper_id"] for row in b_ranked
        ),
        "assignment_records": [
            {
                **row,
                "rank": index,
                "partition": (
                    "development"
                    if index <= DEVELOPMENT_COUNT
                    else (
                        "reserve_a"
                        if index <= (
                            DEVELOPMENT_COUNT + RESERVE_A_COUNT
                        )
                        else "reserve_b"
                    )
                ),
            }
            for index, row in enumerate(ranked, start=1)
        ],
        "split_algorithm": (
            "sort ascending by SHA256("
            "sers_alpha4c5f2_blind_split_v1\\0 + paper_id)"
        ),
        "split_input_fields": ["paper_id"],
        "scientific_fields_used": False,
        "reserve_a_consumed_at_split": False,
        "reserve_b_consumed_at_split": False,
        "reserve_b_sealed_for_future_confirmation": True,
        "llm_calls_at_split": 0,
    }
    payload["split_sha256"] = _payload_sha(
        payload,
        "split_sha256",
    )
    return payload


def validate_blind_split(
    *,
    pool: Mapping[str, Any],
    split: Mapping[str, Any],
) -> dict[str, Any]:
    if split.get("semantics_id") != BLIND_SPLIT_SEMANTICS_ID:
        raise ValueError("Blind split semantics mismatch.")
    _require_exact_payload_sha(
        split,
        "split_sha256",
        "blind split",
    )
    if split.get("pool_id") != pool.get("pool_id"):
        raise ValueError("Blind split pool_id mismatch.")
    if split.get("pool_manifest_sha256") != pool.get(
        "manifest_sha256"
    ):
        raise ValueError("Blind split pool SHA mismatch.")
    recomputed = make_blind_split(pool)
    if canonical_json(recomputed) != canonical_json(split):
        raise ValueError(
            "Blind split is not the deterministic ID-only split."
        )

    dev = set(split["development"])
    reserve_a = set(split["reserve_a"])
    reserve_b = set(split["reserve_b"])
    if dev & reserve_a or dev & reserve_b or reserve_a & reserve_b:
        raise ValueError("Blind split partitions overlap.")
    if dev | reserve_a | reserve_b != set(pool["paper_ids"]):
        raise ValueError("Blind split does not cover the pool exactly.")
    if (
        len(dev) != DEVELOPMENT_COUNT
        or len(reserve_a) != RESERVE_A_COUNT
        or len(reserve_b) != RESERVE_B_COUNT
    ):
        raise ValueError("Blind split partition counts drifted.")
    if split.get("scientific_fields_used") is not False:
        raise ValueError("Scientific fields were used for split.")
    if split.get("reserve_b_sealed_for_future_confirmation") is not True:
        raise ValueError("Reserve B is not sealed.")
    if split.get("llm_calls_at_split") != 0:
        raise ValueError("Blind split must perform zero LLM calls.")
    return dict(split)


def load_pool_and_split(
    *,
    root: Path,
    pool_path: Path,
    split_path: Path,
    verify_source_manifest: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pool = validate_pool_manifest(
        root=root,
        pool_path=pool_path,
        verify_source_manifest=verify_source_manifest,
    )
    split = read_json(split_path)
    validate_blind_split(pool=pool, split=split)
    return pool, split


class Alpha4c5f2ExecutionPolicy(StrictModel):
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
    canonical_readiness_required_before_consumption: Literal[True] = True
    readiness_revalidated_immediately_before_consumption: Literal[
        True
    ] = True
    direct_consumption_marker_write_allowed: Literal[False] = False
    reserve_consumed_before_first_scientific_transformation: Literal[
        True
    ] = True
    rerun_after_consumption_allowed: Literal[False] = False
    automatic_scientific_output_rollback: Literal[False] = False
    semantic_patch_requires_new_protocol_epoch: Literal[True] = True
    reserve_b_execution_allowed_in_this_epoch: Literal[False] = False


class Alpha4c5f2Protocol(StrictModel):
    schema_version: Literal[
        "sers-alpha4c5f2-reserve-protocol-v1"
    ] = "sers-alpha4c5f2-reserve-protocol-v1"

    protocol_id: str
    protocol_sha256: str
    semantics_id: str
    runner_contract_id: str

    campaign_id: str
    reserve_partition: Literal["reserve_a"]
    domain_profile_id: Literal["sers_au_ag"] = "sers_au_ag"
    data_root: str
    evaluation_root: str

    pool_manifest_path: str
    pool_manifest_sha256: str
    blind_split_path: str
    blind_split_sha256: str
    reserve_paper_ids: list[str]

    canonical_readiness_lock_path: str
    canonical_readiness_lock_file_sha256: str
    canonical_readiness_lock_payload_sha256: str
    canonical_readiness_semantics_id: str

    trend_baseline_path: str
    trend_baseline_sha256: str
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
    execution_policy: Alpha4c5f2ExecutionPolicy

    reserve_consumed_at_protocol_freeze: Literal[False] = False
    llm_calls_at_protocol_freeze: Literal[0] = 0

    @model_validator(mode="after")
    def _consistency(self) -> "Alpha4c5f2Protocol":
        if self.semantics_id != ALPHA4C5F2_PROTOCOL_SEMANTICS_ID:
            raise ValueError("alpha4c.5f.2 semantics mismatch.")
        if self.runner_contract_id != RUNNER_CONTRACT_ID:
            raise ValueError("alpha4c.5f.2 runner contract mismatch.")
        if not self.campaign_id.startswith(
            "sers_alpha4c5f2_reserve_a_"
        ):
            raise ValueError("Unexpected alpha4c.5f.2 campaign ID.")
        if self.reserve_paper_ids != sorted(
            set(self.reserve_paper_ids)
        ):
            raise ValueError(
                "alpha4c.5f.2 reserve IDs must be sorted/unique."
            )
        if len(self.reserve_paper_ids) != RESERVE_A_COUNT:
            raise ValueError("Reserve A must contain exactly 25 papers.")
        if self.canonical_readiness_semantics_id != (
            CANONICAL_READINESS_SEMANTICS_ID
        ):
            raise ValueError("Canonical readiness semantics mismatch.")
        if list(self.frozen_component_sha256) != sorted(
            self.frozen_component_sha256
        ):
            raise ValueError(
                "Frozen component paths must be sorted."
            )

        expected_id = stable_id(
            "sers_alpha4c5f2_reserve_protocol",
            self.semantics_id,
            self.runner_contract_id,
            self.campaign_id,
            self.pool_manifest_sha256,
            self.blind_split_sha256,
            self.evaluation_protocol_sha256,
            self.reserve_manifest_sha256,
            self.canonical_readiness_lock_payload_sha256,
            sha256_json(self.frozen_component_sha256),
            self.explorer.question,
            self.traversal.source_query,
            self.traversal.target_query,
        )
        if self.protocol_id != expected_id:
            raise ValueError(
                "alpha4c.5f.2 protocol_id is not stable."
            )

        payload = self.model_dump(mode="json")
        observed = str(payload.pop("protocol_sha256", ""))
        expected = sha256_json(payload)
        if observed != expected:
            raise ValueError(
                "alpha4c.5f.2 protocol SHA mismatch."
            )
        return self


def load_5f2_protocol(path: Path) -> Alpha4c5f2Protocol:
    return Alpha4c5f2Protocol.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def current_component_hashes(
    *,
    root: Path,
    paths: Iterable[str],
) -> dict[str, str]:
    values: dict[str, str] = {}
    for rel in sorted(set(str(value) for value in paths)):
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(
                f"Frozen alpha4c.5f.2 component missing: {rel}"
            )
        values[rel] = sha256_file(path)
    return dict(sorted(values.items()))


def verify_legacy_frozen_components(
    *,
    root: Path,
    legacy_protocol: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []
    expected = legacy_protocol.get("frozen_component_sha256")
    if not isinstance(expected, Mapping):
        return ["Legacy 5f frozen component map missing."]
    for rel, wanted in sorted(expected.items()):
        path = root / str(rel)
        if not path.exists():
            issues.append(f"legacy frozen component missing: {rel}")
            continue
        observed = sha256_file(path)
        if observed != wanted:
            issues.append(
                f"legacy frozen component drift: {rel}: "
                f"expected={wanted}, observed={observed}"
            )
    return issues


def make_5f2_protocol(
    *,
    root: Path,
    campaign_id: str,
    pool_path: Path,
    split_path: Path,
    reserve_manifest_path: Path,
    readiness_lock_path: Path,
    evaluation_protocol_path: Path,
    legacy_5f_protocol_path: Path,
) -> Alpha4c5f2Protocol:
    pool, split = load_pool_and_split(
        root=root,
        pool_path=pool_path,
        split_path=split_path,
        verify_source_manifest=True,
    )
    reserve_ids = sorted(split["reserve_a"])

    legacy = read_json(legacy_5f_protocol_path)
    legacy_issues = verify_legacy_frozen_components(
        root=root,
        legacy_protocol=legacy,
    )
    if legacy_issues:
        raise ValueError(
            "Historical 5f scientific component freeze drifted:\n- "
            + "\n- ".join(legacy_issues)
        )

    evaluation_protocol = load_5e_protocol(
        evaluation_protocol_path
    )
    evaluation_issues = verify_5e_protocol_integrity(
        evaluation_protocol,
        root=root,
    )
    if evaluation_issues:
        raise ValueError(
            "Frozen 5e protocol integrity failed:\n- "
            + "\n- ".join(evaluation_issues)
        )

    reserve_manifest = load_reserve_manifest(
        reserve_manifest_path
    )
    if reserve_manifest.protocol_id != evaluation_protocol.protocol_id:
        raise ValueError("5e reserve manifest protocol_id mismatch.")
    if (
        reserve_manifest.protocol_sha256
        != evaluation_protocol.protocol_sha256
    ):
        raise ValueError("5e reserve manifest protocol SHA mismatch.")
    if reserve_manifest.domain_profile_id != EXPECTED_DOMAIN_PROFILE_ID:
        raise ValueError("5e reserve manifest domain mismatch.")
    if reserve_manifest.paper_ids != reserve_ids:
        raise ValueError(
            "5e reserve manifest does not exactly bind Reserve A."
        )
    if reserve_manifest.reserve_consumed_at_registration is not False:
        raise ValueError("Reserve was consumed at 5e registration.")

    readiness_lock = load_and_verify_readiness_lock(
        root=root,
        lock_path=readiness_lock_path,
        expected_paper_ids=reserve_ids,
        expected_domain_profile_id=EXPECTED_DOMAIN_PROFILE_ID,
    )

    # Preserve the previously frozen scientific settings. 5f.2 changes only
    # the reserve orchestration boundary.
    old_components = dict(
        legacy["frozen_component_sha256"]
    )
    all_component_paths = set(old_components) | set(
        NEW_FROZEN_COMPONENT_PATHS
    )
    frozen_components = current_component_hashes(
        root=root,
        paths=all_component_paths,
    )
    # The old map was checked above; this assertion makes the preservation
    # explicit in the new protocol.
    for rel, wanted in old_components.items():
        if frozen_components[rel] != wanted:
            raise ValueError(
                f"Frozen scientific component changed during freeze: {rel}"
            )

    trend_baseline_path = root / str(
        legacy["trend_baseline_path"]
    )
    if not trend_baseline_path.exists():
        raise FileNotFoundError(trend_baseline_path)

    artifact_prefix = campaign_id
    artifact_ids = {
        "corpus": f"{artifact_prefix}_corpus",
        "measurement_result_identity": (
            f"{artifact_prefix}_measurement_identity"
        ),
        "metric_definition": (
            f"{artifact_prefix}_metric_definition"
        ),
        "comparison": f"{artifact_prefix}_comparison",
        "trend": f"{artifact_prefix}_trend",
        "precision": f"{artifact_prefix}_precision",
        "context": f"{artifact_prefix}_trend_context",
        "assessment": f"{artifact_prefix}_assessment",
    }
    evaluation_root = (
        f"evaluation/sers_alpha4c5f2/{campaign_id}"
    )
    data_root = f"{evaluation_root}/work_data_sers"

    payload: dict[str, Any] = {
        "schema_version":
            "sers-alpha4c5f2-reserve-protocol-v1",
        "protocol_id": "",
        "protocol_sha256": "",
        "semantics_id": ALPHA4C5F2_PROTOCOL_SEMANTICS_ID,
        "runner_contract_id": RUNNER_CONTRACT_ID,
        "campaign_id": campaign_id,
        "reserve_partition": EXPECTED_RESERVE_PARTITION,
        "domain_profile_id": EXPECTED_DOMAIN_PROFILE_ID,
        "data_root": data_root,
        "evaluation_root": evaluation_root,
        "pool_manifest_path": str(pool_path),
        "pool_manifest_sha256": pool["manifest_sha256"],
        "blind_split_path": str(split_path),
        "blind_split_sha256": split["split_sha256"],
        "reserve_paper_ids": reserve_ids,
        "canonical_readiness_lock_path": str(
            readiness_lock_path
        ),
        "canonical_readiness_lock_file_sha256": sha256_file(
            readiness_lock_path
        ),
        "canonical_readiness_lock_payload_sha256": (
            readiness_lock["lock_sha256"]
        ),
        "canonical_readiness_semantics_id": (
            CANONICAL_READINESS_SEMANTICS_ID
        ),
        "trend_baseline_path": legacy["trend_baseline_path"],
        "trend_baseline_sha256": sha256_file(
            trend_baseline_path
        ),
        "frozen_trend_semantics": dict(
            legacy["frozen_trend_semantics"]
        ),
        "evaluation_protocol_path": str(
            evaluation_protocol_path
        ),
        "evaluation_protocol_id": evaluation_protocol.protocol_id,
        "evaluation_protocol_sha256": (
            evaluation_protocol.protocol_sha256
        ),
        "reserve_manifest_path": str(
            reserve_manifest_path
        ),
        "reserve_manifest_id": reserve_manifest.manifest_id,
        "reserve_manifest_sha256": (
            reserve_manifest.manifest_sha256
        ),
        "artifact_ids": artifact_ids,
        "traversal": dict(legacy["traversal"]),
        "explorer": dict(legacy["explorer"]),
        "maker": dict(legacy["maker"]),
        "frozen_component_sha256": frozen_components,
        "execution_policy": {
            "exact_reserve_set_required": True,
            "paper_override_allowed": False,
            "evidence_mode_only": True,
            "bridge_required": False,
            "new_extraction_llm_allowed": False,
            "canonical_source_copy_isolated": True,
            "allow_critical_partial_projection": True,
            "count_thresholds_used_for_acceptance": False,
            "zero_trend_yield_is_execution_failure": False,
            "zero_hypotheses_is_evaluation_failure": False,
            "canonical_readiness_required_before_consumption": True,
            "readiness_revalidated_immediately_before_consumption": True,
            "direct_consumption_marker_write_allowed": False,
            "reserve_consumed_before_first_scientific_transformation": True,
            "rerun_after_consumption_allowed": False,
            "automatic_scientific_output_rollback": False,
            "semantic_patch_requires_new_protocol_epoch": True,
            "reserve_b_execution_allowed_in_this_epoch": False,
        },
        "reserve_consumed_at_protocol_freeze": False,
        "llm_calls_at_protocol_freeze": 0,
    }

    payload["protocol_id"] = stable_id(
        "sers_alpha4c5f2_reserve_protocol",
        payload["semantics_id"],
        payload["runner_contract_id"],
        payload["campaign_id"],
        payload["pool_manifest_sha256"],
        payload["blind_split_sha256"],
        payload["evaluation_protocol_sha256"],
        payload["reserve_manifest_sha256"],
        payload["canonical_readiness_lock_payload_sha256"],
        sha256_json(payload["frozen_component_sha256"]),
        payload["explorer"]["question"],
        payload["traversal"]["source_query"],
        payload["traversal"]["target_query"],
    )
    payload["protocol_sha256"] = sha256_json(
        {
            key: value
            for key, value in payload.items()
            if key != "protocol_sha256"
        }
    )
    return Alpha4c5f2Protocol.model_validate(payload)


def verify_5f2_protocol(
    *,
    root: Path,
    protocol: Alpha4c5f2Protocol,
    verify_source_manifest: bool = True,
) -> list[str]:
    issues: list[str] = []

    try:
        pool_path = _resolve(root, protocol.pool_manifest_path)
        split_path = _resolve(root, protocol.blind_split_path)
        pool, split = load_pool_and_split(
            root=root,
            pool_path=pool_path,
            split_path=split_path,
            verify_source_manifest=verify_source_manifest,
        )
        if pool["manifest_sha256"] != protocol.pool_manifest_sha256:
            issues.append("pool manifest SHA binding mismatch")
        if split["split_sha256"] != protocol.blind_split_sha256:
            issues.append("blind split SHA binding mismatch")
        if sorted(split["reserve_a"]) != protocol.reserve_paper_ids:
            issues.append("protocol does not exactly bind Reserve A")
    except Exception as exc:
        issues.append(f"pool/split verification: {exc}")

    try:
        evaluation_path = _resolve(
            root,
            protocol.evaluation_protocol_path,
        )
        evaluation_protocol = load_5e_protocol(evaluation_path)
        if (
            evaluation_protocol.protocol_id
            != protocol.evaluation_protocol_id
        ):
            issues.append("5e protocol ID binding mismatch")
        if (
            evaluation_protocol.protocol_sha256
            != protocol.evaluation_protocol_sha256
        ):
            issues.append("5e protocol SHA binding mismatch")
        issues.extend(
            "5e frozen component drift: " + issue
            for issue in verify_5e_protocol_integrity(
                evaluation_protocol,
                root=root,
            )
        )
    except Exception as exc:
        issues.append(f"5e protocol verification: {exc}")

    try:
        reserve_path = _resolve(
            root,
            protocol.reserve_manifest_path,
        )
        reserve = load_reserve_manifest(reserve_path)
        if reserve.manifest_id != protocol.reserve_manifest_id:
            issues.append("reserve manifest ID binding mismatch")
        if (
            reserve.manifest_sha256
            != protocol.reserve_manifest_sha256
        ):
            issues.append("reserve manifest SHA binding mismatch")
        if reserve.paper_ids != protocol.reserve_paper_ids:
            issues.append("reserve manifest paper set/order mismatch")
        if reserve.protocol_id != protocol.evaluation_protocol_id:
            issues.append("reserve manifest 5e protocol ID mismatch")
        if (
            reserve.protocol_sha256
            != protocol.evaluation_protocol_sha256
        ):
            issues.append("reserve manifest 5e protocol SHA mismatch")
    except Exception as exc:
        issues.append(f"reserve manifest verification: {exc}")

    try:
        lock_path = _resolve(
            root,
            protocol.canonical_readiness_lock_path,
        )
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
        if lock.get("lock_sha256") != (
            protocol.canonical_readiness_lock_payload_sha256
        ):
            issues.append("readiness lock payload SHA drifted")
    except Exception as exc:
        issues.append(f"canonical readiness verification: {exc}")

    try:
        observed = current_component_hashes(
            root=root,
            paths=protocol.frozen_component_sha256,
        )
        if observed != protocol.frozen_component_sha256:
            for rel in sorted(
                set(observed)
                | set(protocol.frozen_component_sha256)
            ):
                expected = protocol.frozen_component_sha256.get(rel)
                actual = observed.get(rel)
                if expected != actual:
                    issues.append(
                        f"frozen component drift: {rel}: "
                        f"expected={expected}, observed={actual}"
                    )
    except Exception as exc:
        issues.append(f"frozen component verification: {exc}")

    try:
        baseline = _resolve(
            root,
            protocol.trend_baseline_path,
        )
        if sha256_file(baseline) != protocol.trend_baseline_sha256:
            issues.append("Trend baseline SHA drifted")
    except Exception as exc:
        issues.append(f"Trend baseline verification: {exc}")

    expected_eval_root = (
        f"evaluation/sers_alpha4c5f2/{protocol.campaign_id}"
    )
    expected_data_root = f"{expected_eval_root}/work_data_sers"
    if protocol.evaluation_root != expected_eval_root:
        issues.append("unexpected alpha4c.5f.2 evaluation_root")
    if protocol.data_root != expected_data_root:
        issues.append("unexpected alpha4c.5f.2 data_root")

    return issues
