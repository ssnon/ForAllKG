import ast
from pathlib import Path

import pytest

from pipeline_core.discovery.novelty_refinement_runtime import (
    PerHypothesisExternalArtifacts,
    _fresh_specification_sanitization_slice,
)


def test_fresh_slice_excludes_stale_prior_call_records():
    records = [
        {
            "schema_version":
                "novelty-claim-specification-sanitization-v1",
            "diagnostic_only":
                True,
            "hypothesis_id":
                "hypothesis:old",
            "claim_id":
                "claim:old",
        },
        {
            "schema_version":
                "novelty-claim-specification-sanitization-v1",
            "diagnostic_only":
                True,
            "hypothesis_id":
                "hypothesis:new",
            "claim_id":
                "claim:new-1",
        },
        {
            "schema_version":
                "novelty-claim-specification-sanitization-v1",
            "diagnostic_only":
                True,
            "hypothesis_id":
                "hypothesis:new",
            "claim_id":
                "claim:new-2",
        },
    ]

    result = (
        _fresh_specification_sanitization_slice(
            records=records,
            start_index=1,
        )
    )

    assert [
        row["claim_id"]
        for row in result
    ] == [
        "claim:new-1",
        "claim:new-2",
    ]


def test_fresh_slice_copies_record_objects():
    records = [
        {
            "claim_id":
                "claim:1",
            "diagnostic_only":
                True,
        }
    ]

    result = (
        _fresh_specification_sanitization_slice(
            records=records,
            start_index=0,
        )
    )

    assert result[0] == records[0]
    assert result[0] is not records[0]


@pytest.mark.parametrize(
    "start_index",
    [
        -1,
        2,
    ],
)
def test_invalid_slice_boundary_fails_closed(
    start_index,
):
    with pytest.raises(
        ValueError,
        match="invalid specification-sanitization",
    ):
        _fresh_specification_sanitization_slice(
            records=[
                {
                    "claim_id":
                        "claim:1",
                }
            ],
            start_index=start_index,
        )


def test_external_artifact_default_preserves_backward_compatibility():
    fields = (
        PerHypothesisExternalArtifacts
        .__dataclass_fields__
    )

    assert (
        fields[
            "specification_sanitization_records"
        ].default
        == ()
    )


def test_fresh_external_snapshots_records_before_decomposition():
    text = Path(
        "pipeline_core/discovery/"
        "novelty_refinement_runtime.py"
    ).read_text(
        encoding="utf-8"
    )

    start = text.index(
        "sanitization_start = len("
    )

    decompose = text.index(
        "decompositions = "
        "self.external_assessor.decompose_portfolio(portfolio)"
    )

    sliced = text.index(
        "_fresh_specification_sanitization_slice("
        ,
        decompose,
    )

    assert start < decompose < sliced


def _source_string_constants(
    path: str,
) -> set[str]:
    tree = ast.parse(
        Path(path).read_text(
            encoding="utf-8"
        )
    )

    return {
        node.value
        for node in ast.walk(tree)
        if (
            isinstance(
                node,
                ast.Constant,
            )
            and isinstance(
                node.value,
                str,
            )
        )
    }


def test_nested_writer_reuses_existing_sidecar_schema():
    refinement_path = (
        "scripts/discovery/"
        "run_novelty_refinement.py"
    )

    external_path = (
        "scripts/discovery/"
        "run_external_novelty.py"
    )

    refinement = Path(
        refinement_path
    ).read_text(
        encoding="utf-8"
    )

    refinement_constants = (
        _source_string_constants(
            refinement_path
        )
    )

    external_constants = (
        _source_string_constants(
            external_path
        )
    )

    schema = (
        "novelty-specification-"
        "sanitization-audit-v1"
    )

    # Adjacent Python string literals are semantically one
    # constant even when their indentation/line wrapping differs.
    # Compare the parsed value rather than source formatting.
    assert schema in refinement_constants
    assert schema in external_constants

    assert (
        ".specification_sanitization.json"
        in refinement
    )

    assert (
        '"diagnostic_only": True'
        in refinement
    )

    assert (
        '"decomposition_executed": True'
        in refinement
    )


def test_nested_sidecar_is_not_added_to_query_plan_or_n9_contract():
    runtime = Path(
        "pipeline_core/discovery/"
        "novelty_refinement_runtime.py"
    ).read_text(
        encoding="utf-8"
    )

    cli = Path(
        "scripts/discovery/"
        "run_novelty_refinement.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "specification_sanitization_records"
        in runtime
    )

    assert (
        ".specification_sanitization.json"
        in cli
    )

    # The new field is confined to the external-artifact transport
    # and output writer; no new N9/N10 argument or authority flag
    # is introduced by this patch.
    assert (
        "specification_sanitization_records="
        in runtime
    )

    assert (
        "production_authority"
        not in (
            cli[
                cli.index(
                    ".specification_sanitization.json"
                ):
                cli.index(
                    ".claims_queries.json",
                    cli.index(
                        ".specification_sanitization.json"
                    ),
                )
            ]
        )
    )
