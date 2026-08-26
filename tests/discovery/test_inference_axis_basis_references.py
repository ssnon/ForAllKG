from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.discovery_axis_inference_prompt import (
    INFERENCE_PROMPT_VERSION,
    SYSTEM_PROMPT,
    allowed_axis_basis,
    axis_basis_reference_map,
    resolve_axis_basis_reference,
)


def _axis():
    return SimpleNamespace(
        label=(
            "Porous-network hotspot density "
            "promotes SERS performance"
        ),
        proposed_subject=(
            "porous network microstructure"
        ),
        proposed_relation="PROMOTES",
        proposed_object="SERS performance",
    )


def test_axis_basis_reference_map_is_stable_and_explicit() -> None:
    axis = _axis()

    mapping = (
        axis_basis_reference_map(
            axis
        )
    )

    assert mapping == {
        "axis_basis:label":
            (
                "Porous-network hotspot density "
                "promotes SERS performance"
            ),

        "axis_basis:triple":
            (
                "porous network microstructure "
                "| PROMOTES | SERS performance"
            ),
    }


def test_reference_ids_resolve_to_authoritative_strings() -> None:
    axis = _axis()

    assert (
        resolve_axis_basis_reference(
            axis,
            "axis_basis:label",
        )
        == (
            "Porous-network hotspot density "
            "promotes SERS performance"
        )
    )

    assert (
        resolve_axis_basis_reference(
            axis,
            "axis_basis:triple",
        )
        == (
            "porous network microstructure "
            "| PROMOTES | SERS performance"
        )
    )


def test_legacy_exact_strings_remain_accepted() -> None:
    axis = _axis()

    label = (
        allowed_axis_basis(
            axis
        )[0]
    )

    assert (
        resolve_axis_basis_reference(
            axis,
            label,
        )
        == label
    )


def test_abbreviated_or_paraphrased_basis_remains_invalid() -> None:
    axis = _axis()

    # This reproduces the failure class observed in historical Q06:
    # the semantic phrase is recognizable but it is not authoritative
    # provenance.
    assert (
        resolve_axis_basis_reference(
            axis,
            (
                "Porous-network hotspot density "
                "promotes SERS"
            ),
        )
        is None
    )


def test_prompt_contract_uses_reference_ids() -> None:
    assert (
        INFERENCE_PROMPT_VERSION
        == (
            "axis-inference-critic-prompt-"
            "v1.2-contract-repair"
        )
    )

    assert (
        "axis_basis may contain ONLY basis_id values"
        in SYSTEM_PROMPT
    )
