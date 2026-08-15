from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.alpha4c5f_reserve import (
    ALPHA4C5F_PROTOCOL_SEMANTICS_ID,
    CONFIG_5A_PATH,
    CONFIG_5B_PATH,
    CONFIG_5C_PATH,
    DEFAULT_5E_PROTOCOL_PATH,
    DEFAULT_RESERVE_MANIFEST_PATH,
    EXPECTED_RESERVE_SET,
    EXPECTED_SOURCE_SPLIT_GIT_BLOB,
    EXPECTED_SOURCE_SPLIT_SHA256,
    EXPECTED_TREND_BASELINE_GIT_BLOB,
    SOURCE_SPLIT_PATH,
    TREND_BASELINE_PATH,
    Alpha4c5fArtifactIds,
    Alpha4c5fExecutionPolicy,
    Alpha4c5fExplorerPolicy,
    Alpha4c5fMakerPolicy,
    Alpha4c5fProtocol,
    Alpha4c5fTraversalPolicy,
    current_extra_component_hashes,
    sha256_json,
    stable_id,
    verify_5e_and_reserve,
    verify_fixed_config_blobs,
    verify_source_split,
    verify_trend_baseline,
)


QUESTION = (
    "How do Au/Ag nanostructure design variables and local "
    "experimental context relate to SERS performance?"
)
SOURCE_QUERY = "nanostructure design"
TARGET_QUERY = "SERS performance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path.cwd()

    _, source_reserve_order = verify_source_split(root)
    baseline = verify_trend_baseline(root)
    verify_fixed_config_blobs(root)
    eval_protocol, reserve = verify_5e_and_reserve(root)

    if set(source_reserve_order) != set(reserve.paper_ids):
        raise RuntimeError(
            "5e reserve manifest does not equal reserved_future_v3."
        )
    # Use manifest-sorted order downstream because 5e requires exact
    # equality with Trend corpus paper_ids, not merely set equality.
    paper_ids = list(reserve.paper_ids)
    if paper_ids != sorted(paper_ids):
        raise RuntimeError(
            "5e reserve manifest paper IDs are not sorted."
        )
    if set(paper_ids) != EXPECTED_RESERVE_SET:
        raise RuntimeError("unexpected v3 reserve set.")

    from dac_her.node_mapping import DEFAULT_EMBED_MODEL

    maker = eval_protocol.maker_settings
    if maker.max_hypotheses != 1:
        raise RuntimeError(
            "5f requires frozen 5e max_hypotheses=1."
        )
    if maker.max_repairs != 1:
        raise RuntimeError(
            "5f requires frozen 5e max_repairs=1."
        )
    if float(maker.temperature) != 0.0:
        raise RuntimeError(
            "5f requires frozen 5e temperature=0."
        )

    parse_retries = (
        int(maker.parse_retries)
        if maker.parse_retries is not None
        else 1
    )
    instructor_mode = maker.backend_mode or "JSON"

    artifact_ids = Alpha4c5fArtifactIds(
        corpus="sers_alpha4c5f_reserve_v1_corpus",
        measurement_result_identity=(
            "sers_alpha4c5f_reserve_v1_measurement_identity"
        ),
        metric_definition=(
            "sers_alpha4c5f_reserve_v1_metric_definition"
        ),
        comparison="sers_alpha4c5f_reserve_v1_comparison",
        trend="sers_alpha4c5f_reserve_v1_trend",
        precision="sers_alpha4c5f_reserve_v1_precision",
        context="sers_alpha4c5f_reserve_v1_trend_context",
        assessment="sers_alpha4c5f_reserve_v1_assessment",
    )

    frozen_components = current_extra_component_hashes(root)
    execution_policy = Alpha4c5fExecutionPolicy()
    traversal = Alpha4c5fTraversalPolicy(
        source_query=SOURCE_QUERY,
        target_query=TARGET_QUERY,
        node_index_model=DEFAULT_EMBED_MODEL,
    )
    explorer = Alpha4c5fExplorerPolicy(
        question=QUESTION,
        model=maker.model,
        temperature=0.0,
        max_repairs=1,
        parse_retries=parse_retries,
        instructor_mode=instructor_mode,
        base_url=maker.base_url,
    )
    maker_policy = Alpha4c5fMakerPolicy(
        model=maker.model,
        temperature=0.0,
        max_hypotheses=1,
        max_repairs=1,
        parse_retries=parse_retries,
        instructor_mode=instructor_mode,
        base_url=maker.base_url,
    )

    frozen_trend_semantics = baseline.get(
        "frozen_semantics", {}
    )
    if not isinstance(frozen_trend_semantics, dict):
        raise RuntimeError(
            "baseline frozen_semantics must be an object."
        )
    frozen_trend_semantics = {
        str(key): str(value)
        for key, value in frozen_trend_semantics.items()
    }

    protocol_id = stable_id(
        "sers_alpha4c5f_reserve_protocol",
        ALPHA4C5F_PROTOCOL_SEMANTICS_ID,
        "sers_alpha4c5f_reserve_v1",
        EXPECTED_SOURCE_SPLIT_SHA256,
        eval_protocol.protocol_sha256,
        reserve.manifest_sha256,
        sha256_json(frozen_components),
        QUESTION,
        SOURCE_QUERY,
        TARGET_QUERY,
    )
    payload = {
        "schema_version":
            "sers-alpha4c5f-reserve-protocol-v1",
        "protocol_id": protocol_id,
        "semantics_id":
            ALPHA4C5F_PROTOCOL_SEMANTICS_ID,
        "campaign_id": "sers_alpha4c5f_reserve_v1",
        "domain_profile_id": "sers_au_ag",
        "data_root":
            "evaluation/sers_alpha4c5f/reserve_v1/work_data_sers",
        "evaluation_root":
            "evaluation/sers_alpha4c5f/reserve_v1",
        "source_split_path": str(SOURCE_SPLIT_PATH),
        "source_split_git_blob":
            EXPECTED_SOURCE_SPLIT_GIT_BLOB,
        "source_split_sha256":
            EXPECTED_SOURCE_SPLIT_SHA256,
        "reserve_paper_ids": paper_ids,
        "trend_baseline_path": str(TREND_BASELINE_PATH),
        "trend_baseline_git_blob":
            EXPECTED_TREND_BASELINE_GIT_BLOB,
        "frozen_trend_semantics": frozen_trend_semantics,
        "evaluation_protocol_path":
            str(DEFAULT_5E_PROTOCOL_PATH),
        "evaluation_protocol_id":
            eval_protocol.protocol_id,
        "evaluation_protocol_sha256":
            eval_protocol.protocol_sha256,
        "reserve_manifest_path":
            str(DEFAULT_RESERVE_MANIFEST_PATH),
        "reserve_manifest_id": reserve.manifest_id,
        "reserve_manifest_sha256":
            reserve.manifest_sha256,
        "artifact_ids":
            artifact_ids.model_dump(mode="json"),
        "traversal":
            traversal.model_dump(mode="json"),
        "explorer":
            explorer.model_dump(mode="json"),
        "maker":
            maker_policy.model_dump(mode="json"),
        "frozen_component_sha256": frozen_components,
        "execution_policy":
            execution_policy.model_dump(mode="json"),
        "reserve_consumed_at_protocol_freeze": False,
        "llm_calls_at_protocol_freeze": 0,
    }
    payload["protocol_sha256"] = sha256_json(payload)
    protocol = Alpha4c5fProtocol.model_validate(payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        observed = args.output.read_text(encoding="utf-8")
        desired = (
            json.dumps(
                protocol.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        if observed != desired:
            raise RuntimeError(
                "Refusing to replace an existing different 5f protocol."
            )
    else:
        args.output.write_text(
            json.dumps(
                protocol.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print("alpha4c.5f reserve execution protocol freeze")
    print("Protocol ID:", protocol.protocol_id)
    print("Protocol SHA256:", protocol.protocol_sha256)
    print("Campaign:", protocol.campaign_id)
    print("Reserve papers:", len(protocol.reserve_paper_ids))
    print("Mode:", protocol.traversal.mode)
    print("Bridge required: False")
    print("Question:", protocol.explorer.question)
    print(
        "Traversal:",
        protocol.traversal.source_query,
        "->",
        protocol.traversal.target_query,
        protocol.traversal.algorithm,
    )
    print("Explorer model:", protocol.explorer.model)
    print("Maker model:", protocol.maker.model)
    print("Max hypotheses:", protocol.maker.max_hypotheses)
    print("Max repairs:", protocol.maker.max_repairs)
    print("Temperature:", protocol.maker.temperature)
    print("Count thresholds used: False")
    print("Reserve consumed: False")
    print("LLM calls: 0")
    print("Output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
