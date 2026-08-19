from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from campaigns.sers_alpha4_epoch.holdout.trend_holdout import (
    TREND_HOLDOUT_SELECTION_ALGORITHM,
    TREND_HOLDOUT_SPLIT_SEMANTICS_ID,
    validate_protocol_split,
)


from campaigns.sers_alpha4_epoch.paths import PROJECT_ROOT
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "configs"
    / "heldout"
    / "sers_alpha4c4_trend_holdout.json"
)


class FrozenTrendHoldoutProtocolError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FrozenTrendHoldoutProtocolError(
            f"Expected JSON object: {path}"
        )
    return value


def _git_blob(path: Path) -> str:
    result = subprocess.run(
        ["git", "hash-object", str(path)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise FrozenTrendHoldoutProtocolError(
            f"Cannot hash {path}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _paper_blocks_from_config(path: Path) -> dict[str, bool]:
    text = path.read_text(encoding="utf-8")
    matches = list(
        re.finditer(
            r"(?m)^  (Kiwook_SERS_\d+):\s*$",
            text,
        )
    )
    result: dict[str, bool] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )
        block = text[start:end]
        result[match.group(1)] = bool(
            re.search(r"(?m)^    enabled:\s*true\s*$", block)
        )
    return result


def _verify_runtime_semantics(
    expected: dict[str, str],
) -> dict[str, str]:
    from dac_her.cross_context_trend import (
        CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
    )
    from dac_her.cross_context_trend_assessment import (
        CROSS_CONTEXT_TREND_ASSESSMENT_SEMANTICS_ID,
    )
    from campaigns.sers_alpha4_epoch.legacy.trend.sers_au_ag_cross_context_trend import (
        SERS_AU_AG_TREND_CONTEXT_SEMANTICS_ID,
    )
    from dac_her.domains.trend_precision_registry import (
        get_trend_precision_adapter,
    )
    from dac_her.domains.trend_registry import get_trend_adapter
    from dac_her.measurement_result_identity import (
        MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID,
    )
    from dac_her.trend_domain import (
        TREND_EVIDENCE_CONTRACT_SEMANTICS_ID,
    )

    trend = get_trend_adapter("sers_au_ag")
    precision = get_trend_precision_adapter("sers_au_ag")

    # Upstream comparison/method IDs are intentionally frozen as protocol
    # values even though this alpha4c.4a verifier does not rebuild them.
    observed = {
        "measurement_result_identity":
            MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID,
        "comparison": expected["comparison"],
        "method": expected["method"],
        "trend_contract":
            TREND_EVIDENCE_CONTRACT_SEMANTICS_ID,
        "trend": trend.semantics_id,
        "trend_precision": precision.precision_semantics_id,
        "cross_context_contract":
            CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
        "trend_context":
            SERS_AU_AG_TREND_CONTEXT_SEMANTICS_ID,
        "cross_context_assessment":
            CROSS_CONTEXT_TREND_ASSESSMENT_SEMANTICS_ID,
    }
    if observed != expected:
        raise FrozenTrendHoldoutProtocolError(
            "Frozen semantic IDs drifted: "
            f"expected={expected!r}, observed={observed!r}"
        )
    return observed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the alpha4c.4a curated-corpus unseen Trend "
            "holdout split and frozen protocol."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_path = (
        args.protocol
        if args.protocol.is_absolute()
        else PROJECT_ROOT / args.protocol
    ).resolve()
    protocol = _read_json(protocol_path)

    if protocol.get("phase") != "alpha4c.4a":
        raise FrozenTrendHoldoutProtocolError(
            "Unexpected holdout protocol phase."
        )
    if protocol.get("state") != "frozen_split_ready_for_alpha4c4b":
        raise FrozenTrendHoldoutProtocolError(
            "alpha4c.4a protocol is not frozen/ready."
        )
    if protocol.get("acquisition_snapshot_used") is not False:
        raise FrozenTrendHoldoutProtocolError(
            "alpha4c.4a must use the curated Kiwook_SERS corpus, "
            "not the production acquisition snapshot."
        )

    split = validate_protocol_split(protocol)

    source = protocol["selection_source"]
    config_path = PROJECT_ROOT / source["paper_config_path"]
    if not config_path.exists():
        raise FrozenTrendHoldoutProtocolError(
            f"Curated paper config is missing: {config_path}"
        )
    if _git_blob(config_path) != source["paper_config_git_blob"]:
        raise FrozenTrendHoldoutProtocolError(
            "Curated paper config drifted after split freeze."
        )

    blocks = _paper_blocks_from_config(config_path)
    expected = list(split.all_paper_ids)
    missing = [
        paper_id
        for paper_id in expected
        if paper_id not in blocks
    ]
    disabled = [
        paper_id
        for paper_id in expected
        if blocks.get(paper_id) is False
    ]
    unexpected = sorted(
        set(blocks) - set(expected)
    )
    if missing or disabled or unexpected:
        raise FrozenTrendHoldoutProtocolError(
            "Curated Kiwook_SERS_1..38 config membership drifted: "
            f"missing={missing!r}, disabled={disabled!r}, "
            f"unexpected={unexpected!r}"
        )

    frozen_blobs = protocol["frozen_implementation_blobs"]
    observed_blobs: dict[str, str] = {}
    for relative_path, expected_blob in sorted(
        frozen_blobs.items()
    ):
        path = PROJECT_ROOT / relative_path
        if not path.exists():
            raise FrozenTrendHoldoutProtocolError(
                f"Frozen implementation file missing: {relative_path}"
            )
        actual = _git_blob(path)
        observed_blobs[relative_path] = actual
        if actual != expected_blob:
            raise FrozenTrendHoldoutProtocolError(
                f"Frozen implementation drift: {relative_path}: "
                f"{actual} != {expected_blob}"
            )

    observed_semantics = _verify_runtime_semantics(
        dict(protocol["frozen_semantics"])
    )

    if (
        protocol["selection"]["split_semantics_id"]
        != TREND_HOLDOUT_SPLIT_SEMANTICS_ID
        or protocol["selection"]["algorithm"]
        != TREND_HOLDOUT_SELECTION_ALGORITHM
    ):
        raise FrozenTrendHoldoutProtocolError(
            "Split implementation semantics drifted."
        )

    print("alpha4c.4a frozen Trend holdout protocol: PASS")
    print(
        "Development calibration:",
        ", ".join(split.development_calibration),
    )
    print(
        "Development seen regression:",
        ", ".join(split.development_seen_regression),
    )
    print("Candidate pool:", len(split.candidate_papers))
    print(
        "Frozen holdout:",
        ", ".join(split.holdout_papers),
    )
    print(
        "Future reserve:",
        len(split.reserved_future_papers),
    )
    print("Split SHA256:", split.split_sha256)
    print(
        "Count thresholds used for alpha4c.4b acceptance:",
        protocol["alpha4c4b_acceptance_policy"][
            "count_thresholds_used"
        ],
    )
    print("Runtime semantics:", json.dumps(
        observed_semantics,
        sort_keys=True,
    ))
    print("Frozen implementation files:", len(observed_blobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
