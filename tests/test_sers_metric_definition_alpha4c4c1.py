from __future__ import annotations

from domains.sers.metric_definition import (
    SERS_METRIC_DEFINITION_SEMANTICS_ID,
    _finalize_definition_interpretation,
)
from dac_her.metric_definition_domain import MetricDefinitionContext


def test_new_semantic_epoch():
    assert SERS_METRIC_DEFINITION_SEMANTICS_ID == 'sers_au_ag_metric_definition_v3_alpha4c4c1'


def test_unknown_clears_interpreted_definition_evidence():
    observed = _finalize_definition_interpretation(
        status="unknown",
        criterion="interpreted criterion",
        formula_text="EF=(Ifoo/Ibar)",
        normalization_basis="molecule_count",
        reference_basis="normal_raman",
    )
    assert observed == ("", "", "unspecified", "unspecified")


def test_known_definition_is_unchanged():
    expected = (
        "criterion",
        "formula",
        "molecule_count",
        "normal_raman",
    )
    assert _finalize_definition_interpretation(
        status="known",
        criterion=expected[0],
        formula_text=expected[1],
        normalization_basis=expected[2],
        reference_basis=expected[3],
    ) == expected


def test_generic_contract_accepts_sanitized_unknown_and_keeps_raw_source():
    criterion, formula, normalization, reference = (
        _finalize_definition_interpretation(
            status="unknown",
            criterion="",
            formula_text="EF=(Ifoo/Ibar)",
            normalization_basis="molecule_count",
            reference_basis="normal_raman",
        )
    )
    context = MetricDefinitionContext(
        context_id="ctx",
        domain_profile_id="sers_au_ag",
        metric_definition_semantics_id='sers_au_ag_metric_definition_v3_alpha4c4c1',
        paper_id="P",
        measurement_id="m",
        observable_key="sers_enhancement_factor",
        definition_status="unknown",
        definition_family="reported_ef_unspecified",
        aggregation_scope="unspecified",
        normalization_basis=normalization,
        reference_basis=reference,
        criterion=criterion,
        formula_text=formula,
        source_expression="raw source expression survives",
        source_measurement_ids=("m",),
        source_node_ids=("m",),
    )
    assert context.formula_text == ""
    assert context.source_expression == "raw source expression survives"
