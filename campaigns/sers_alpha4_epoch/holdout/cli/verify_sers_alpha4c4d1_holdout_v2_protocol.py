from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from campaigns.sers_alpha4_epoch.holdout.trend_holdout import rank_candidate_papers, validate_protocol_split


from campaigns.sers_alpha4_epoch.paths import PROJECT_ROOT as ROOT
DEFAULT_PROTOCOL = (
    ROOT / "configs" / "heldout" /
    "sers_alpha4c4d1_trend_holdout_v2.json"
)


class FrozenV2HoldoutError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FrozenV2HoldoutError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git_blob(path: Path) -> str:
    proc = subprocess.run(
        ["git", "hash-object", str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise FrozenV2HoldoutError(proc.stderr.strip())
    return proc.stdout.strip()


def _require(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise FrozenV2HoldoutError(
            f"{label}: {observed!r} != {expected!r}"
        )


def _epoch_split_sha(protocol: Mapping[str, Any]) -> str:
    selection = protocol["selection"]
    papers = protocol["papers"]
    payload = {
        "split_semantics_id": selection["split_semantics_id"],
        "selection_algorithm": selection["algorithm"],
        "source_pool_role": selection["source_pool_role"],
        "namespace": selection["namespace"],
        "holdout_count": selection["holdout_count"],
        "candidate_papers": papers["candidate_papers"],
        "ranked_candidates": papers["ranked_candidates"],
        "frozen_holdout_v2": papers["frozen_holdout_v2"],
        "reserved_future_v3": papers["reserved_future_v3"],
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _verify_runtime_semantics(expected: Mapping[str, str]) -> dict[str, str]:
    from dac_her.cross_context_trend import (
        CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
    )
    from dac_her.cross_context_trend_assessment import (
        CROSS_CONTEXT_TREND_ASSESSMENT_SEMANTICS_ID,
    )
    from dac_her.domains.comparison_registry import get_comparison_adapter
    from dac_her.domains.metric_definition_registry import (
        get_metric_definition_adapter,
    )
    from dac_her.domains.registry import get_domain_profile
    from campaigns.sers_alpha4_epoch.legacy.trend.sers_au_ag_cross_context_trend import (
        SERS_AU_AG_TREND_CONTEXT_SEMANTICS_ID,
    )
    from dac_her.domains.trend_precision_registry import (
        get_trend_precision_adapter,
    )
    from dac_her.domains.trend_registry import get_trend_adapter
    from dac_her.measurement_merge_invariants import (
        MEASUREMENT_MERGE_INVARIANT_ID,
    )
    from dac_her.measurement_result_identity import (
        MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID,
    )
    from dac_her.quality_aware_comparison import (
        QUALITY_AWARE_NUMERIC_GATE_SEMANTICS_ID,
    )
    from dac_her.trend_domain import TREND_EVIDENCE_CONTRACT_SEMANTICS_ID

    profile = get_domain_profile("sers_au_ag")
    metric = get_metric_definition_adapter(profile)
    comparison = get_comparison_adapter(profile)
    trend = get_trend_adapter(profile)
    precision = get_trend_precision_adapter(profile)

    observed = {
        "projection": str(profile.projection.semantics_id),
        "corpus": str(profile.corpus.semantics_id),
        "measurement_merge_invariant":
            str(MEASUREMENT_MERGE_INVARIANT_ID),
        "measurement_result_identity":
            str(MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID),
        "metric_definition": str(metric.semantics_id),
        "comparison": str(comparison.semantics_id),
        "method": str(
            comparison.method_semantics.semantics_id
            if comparison.method_semantics else ""
        ),
        "quality_gate":
            str(QUALITY_AWARE_NUMERIC_GATE_SEMANTICS_ID),
        "trend_contract":
            str(TREND_EVIDENCE_CONTRACT_SEMANTICS_ID),
        "trend": str(trend.semantics_id),
        "trend_precision": str(precision.precision_semantics_id),
        "cross_context_contract":
            str(CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID),
        "trend_context":
            str(SERS_AU_AG_TREND_CONTEXT_SEMANTICS_ID),
        "cross_context_assessment":
            str(CROSS_CONTEXT_TREND_ASSESSMENT_SEMANTICS_ID),
    }
    _require("runtime semantics", observed, dict(expected))
    return observed


def verify(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = _read_json(protocol_path)

    _require("phase", protocol.get("phase"), "alpha4c.4d.1")
    _require(
        "state",
        protocol.get("state"),
        "frozen_v2_split_ready_for_alpha4c4d2",
    )
    _require("domain", protocol.get("domain_profile"), "sers_au_ag")
    _require("acquisition snapshot used",
             protocol.get("acquisition_snapshot_used"), False)

    # Bind exactly to alpha4a, the retired v1 campaign, and the successful
    # repair replay.  These metadata files are read only; no paper content is.
    src = protocol["source_state"]

    alpha4a_path = ROOT / src["alpha4a_protocol"]["path"]
    _require(
        "alpha4a protocol sha256",
        _sha256(alpha4a_path),
        src["alpha4a_protocol"]["sha256"],
    )
    alpha4a = _read_json(alpha4a_path)
    split4a = validate_protocol_split(alpha4a)
    _require(
        "alpha4a split sha",
        split4a.split_sha256,
        src["alpha4a_protocol"]["split_sha256"],
    )

    retirement_path = ROOT / src["v1_retirement"]["path"]
    _require(
        "v1 retirement sha256",
        _sha256(retirement_path),
        src["v1_retirement"]["sha256"],
    )
    retirement = _read_json(retirement_path)
    _require(
        "v1 retirement state",
        retirement.get("state"),
        "retired_scientific_contract_failure",
    )
    _require(
        "v1 blind holdout resumable",
        retirement.get("blind_holdout_may_not_resume"),
        True,
    )
    _require(
        "consumed v1 papers",
        retirement.get("consumed_papers_promoted_to_seen"),
        protocol["papers"]["consumed_v1_seen"],
    )

    pass_path = ROOT / src["repair_replay_pass"]["path"]
    _require(
        "repair pass sha256",
        _sha256(pass_path),
        src["repair_replay_pass"]["sha256"],
    )
    replay_pass = _read_json(pass_path)
    _require(
        "repair replay state",
        replay_pass.get("state"),
        "repair_replay_passed",
    )
    _require(
        "v2 split allowed by repair replay",
        replay_pass.get("v2_holdout_split_allowed"),
        True,
    )
    _require(
        "repair replay metric semantics",
        replay_pass.get("metric_definition_semantics_id"),
        "sers_au_ag_metric_definition_v3_alpha4c4c1",
    )
    _require(
        "consumed v1 is seen",
        replay_pass.get("consumed_holdout_v1_is_seen"),
        True,
    )

    # The v2 candidate pool must be exactly the untouched 22-paper reserve
    # from alpha4a. No paper may be added, omitted, or reintroduced.
    _require(
        "candidate pool",
        protocol["papers"]["candidate_papers"],
        list(split4a.reserved_future_papers),
    )

    seen = (
        set(protocol["papers"]["development_calibration"])
        | set(protocol["papers"]["development_seen_regression"])
        | set(protocol["papers"]["consumed_v1_seen"])
    )
    candidates = set(protocol["papers"]["candidate_papers"])
    if seen & candidates:
        raise FrozenV2HoldoutError(
            f"seen/candidate overlap: {sorted(seen & candidates)!r}"
        )

    selection = protocol["selection"]
    _require(
        "selection inputs",
        selection.get("selection_inputs"),
        ["paper_id"],
    )
    _require(
        "scientific content inspected for split",
        selection.get("scientific_content_inspected_for_split"),
        False,
    )
    _require(
        "trend output inspected for split",
        selection.get("trend_outputs_inspected_for_split"),
        False,
    )
    _require(
        "paper metadata used for split",
        selection.get("paper_metadata_used_for_split"),
        False,
    )

    ranked = list(
        rank_candidate_papers(
            protocol["papers"]["candidate_papers"],
            namespace=selection["namespace"],
        )
    )
    _require(
        "ranked candidates",
        protocol["papers"]["ranked_candidates"],
        ranked,
    )
    count = int(selection["holdout_count"])
    _require(
        "frozen v2 holdout",
        protocol["papers"]["frozen_holdout_v2"],
        [row["paper_id"] for row in ranked[:count]],
    )
    _require(
        "future v3 reserve",
        protocol["papers"]["reserved_future_v3"],
        [row["paper_id"] for row in ranked[count:]],
    )
    _require(
        "v2 split sha256",
        _epoch_split_sha(protocol),
        selection["split_sha256"],
    )

    if len(protocol["papers"]["frozen_holdout_v2"]) != 8:
        raise FrozenV2HoldoutError("v2 must freeze exactly 8 papers.")
    if len(protocol["papers"]["reserved_future_v3"]) != 14:
        raise FrozenV2HoldoutError("v3 reserve must contain 14 papers.")

    # Count-free / zero-yield behavior is frozen before v2 runs.
    acceptance = protocol["alpha4c4d2_acceptance_policy"]
    _require(
        "count thresholds used",
        acceptance.get("count_thresholds_used"),
        False,
    )
    for key in (
        "minimum_trend_evidence_count",
        "minimum_local_result_count",
        "minimum_cross_paper_pair_count",
        "minimum_repeated_count",
        "minimum_reversed_count",
        "minimum_context_specific_count",
        "maximum_insufficient_count",
    ):
        if acceptance.get(key) is not None:
            raise FrozenV2HoldoutError(
                f"forbidden output target populated: {key}"
            )
    _require(
        "zero TrendEvidence valid",
        acceptance.get("zero_trend_evidence_valid"),
        True,
    )
    _require(
        "zero local result terminal",
        acceptance.get("zero_local_results_terminal_status"),
        "not_applicable_zero_local_results",
    )

    for rel, expected_blob in sorted(
        protocol["frozen_implementation_blobs"].items()
    ):
        path = ROOT / rel
        if not path.exists():
            raise FrozenV2HoldoutError(
                f"frozen implementation file missing: {rel}"
            )
        _require(
            f"implementation blob {rel}",
            _git_blob(path),
            expected_blob,
        )

    runtime = _verify_runtime_semantics(protocol["frozen_semantics"])

    return {
        "holdout": protocol["papers"]["frozen_holdout_v2"],
        "reserve": protocol["papers"]["reserved_future_v3"],
        "split_sha256": selection["split_sha256"],
        "runtime_semantics": runtime,
    }


def main() -> int:
    result = verify()
    print("alpha4c.4d.1 frozen v2 Trend holdout protocol: PASS")
    print("Candidate pool:", 22)
    print("Frozen v2 holdout:", ", ".join(result["holdout"]))
    print("Future v3 reserve:", len(result["reserve"]))
    print("Split SHA256:", result["split_sha256"])
    print("MetricDefinition semantics:",
          result["runtime_semantics"]["metric_definition"])
    print("Count thresholds used for alpha4c.4d.2 acceptance: False")
    print(
        "Zero local TrendResults terminal status: "
        "not_applicable_zero_local_results"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
