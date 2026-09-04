import json
from pathlib import Path

import pytest

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxisSynthesisReport,
)
from pipeline_core.discovery.n10_alpha6_lineage_loader import (
    load_alpha6_lineage_input,
)
from pipeline_core.discovery.n10_post_generation_lineage_projection import (
    N10PostGenerationLineageProjection,
)


def _write(
    tmp_path: Path,
    payload: dict,
) -> Path:
    path = tmp_path / "lineage.json"

    path.write_text(
        json.dumps(
            payload
        ),
        encoding="utf-8",
    )

    return path


def _synthesis_payload():
    return {
        "schema_version":
            "discovery-axis-synthesis-report-v1",

        "report_id":
            "report:1",

        "report_sha256":
            "report-sha",

        "source_dual_context_id":
            "dual:1",

        "source_dual_context_sha256":
            "dual-sha",

        "axis_plan_id":
            "plan:1",

        "axis_plan_sha256":
            "plan-sha",

        "final_portfolio_id":
            "portfolio:h0",

        "final_portfolio_sha256":
            "portfolio-sha",

        "attempted_axis_count":
            1,

        "accepted_hypothesis_count":
            1,

        "lineages": [
            {
                "hypothesis_id":
                    "hypothesis:h0",

                "axis_id":
                    "axis:1",

                "inspiration_id":
                    "inspiration:1",

                "candidate_unit_id":
                    "unit:1",

                "axis_fidelity_status":
                    "pass",

                "inference_status":
                    "pass",

                "internal_novelty_status":
                    "corpus_distinct_candidate",
            }
        ],

        "attempts":
            [],

        "external_novelty_status":
            "not_assessed",

        "policy_version":
            "discovery-axis-synthesis-policy-v2",
    }


def _projection_payload():
    body = {
        "schema_version":
            "n10-post-generation-lineage-projection-v1",

        "projection_id":
            "projection:1",

        "source_lineage_report_id":
            "report:1",

        "source_lineage_report_sha256":
            "report-sha",

        "source_axis_plan_id":
            "plan:1",

        "source_axis_plan_sha256":
            "plan-sha",

        "source_dual_context_id":
            "dual:1",

        "source_dual_context_sha256":
            "dual-sha",

        "source_hypothesis_id":
            "hypothesis:h0",

        "projected_hypothesis_id":
            "hypothesis:h1",

        "projected_portfolio_id":
            "portfolio:h1",

        "projected_portfolio_sha256":
            "portfolio-h1-sha",

        "axis_id":
            "axis:1",

        "lineages": [
            {
                "hypothesis_id":
                    "hypothesis:h1",

                "axis_id":
                    "axis:1",

                "inspiration_id":
                    "inspiration:1",

                "candidate_unit_id":
                    "unit:1",

                "axis_fidelity_status":
                    "pass",

                "inference_status":
                    "pass",

                "internal_novelty_status":
                    "corpus_distinct_candidate",
            }
        ],

        "identity_projection_only":
            True,

        "production_authority":
            False,
    }

    # This test only exercises loading; projection SHA is a required
    # provenance field but is validated as an opaque digest string.
    body[
        "projection_sha256"
    ] = "projection-sha"

    return body


def test_loads_frozen_synthesis_report(
    tmp_path,
):
    result = load_alpha6_lineage_input(
        _write(
            tmp_path,
            _synthesis_payload(),
        )
    )

    assert isinstance(
        result,
        DiscoveryAxisSynthesisReport,
    )

    assert (
        result.lineages[0].hypothesis_id
        == "hypothesis:h0"
    )


def test_loads_identity_only_n10_projection(
    tmp_path,
):
    result = load_alpha6_lineage_input(
        _write(
            tmp_path,
            _projection_payload(),
        )
    )

    assert isinstance(
        result,
        N10PostGenerationLineageProjection,
    )

    assert (
        result.production_authority
        is False
    )

    assert (
        result.lineages[0].hypothesis_id
        == "hypothesis:h1"
    )


def test_unknown_schema_fails_closed(
    tmp_path,
):
    with pytest.raises(
        ValueError,
        match="unsupported Alpha6 lineage schema",
    ):
        load_alpha6_lineage_input(
            _write(
                tmp_path,
                {
                    "schema_version":
                        "unknown-v999"
                },
            )
        )


def test_projection_identity_mismatch_fails_closed(
    tmp_path,
):
    payload = _projection_payload()

    payload[
        "lineages"
    ][0][
        "hypothesis_id"
    ] = "hypothesis:wrong"

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        load_alpha6_lineage_input(
            _write(
                tmp_path,
                payload,
            )
        )


def test_projection_requires_exactly_one_row(
    tmp_path,
):
    payload = _projection_payload()

    payload[
        "lineages"
    ] = []

    with pytest.raises(
        ValueError,
        match="exactly one",
    ):
        load_alpha6_lineage_input(
            _write(
                tmp_path,
                payload,
            )
        )


def test_entrypoint_uses_explicit_loader():
    text = Path(
        "scripts/discovery/"
        "run_novelty_refinement.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "load_alpha6_lineage_input"
        in text
    )

    assert (
        "DiscoveryAxisSynthesisReport.model_validate_json"
        not in text
    )


def test_runtime_still_consumes_lineages_only():
    text = Path(
        "pipeline_core/discovery/"
        "novelty_refinement_runtime.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "lineage_by_h = "
        "{x.hypothesis_id: x for x in lineage.lineages}"
        in text
    )

    forbidden = [
        "lineage.projection_id",
        "lineage.projection_sha256",
        "lineage.production_authority",
        "lineage.projected_hypothesis_id",
        "lineage.source_hypothesis_id",
    ]

    for token in forbidden:
        assert token not in text
