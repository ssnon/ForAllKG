from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

import pipeline_core.discovery.nonobviousness_shadow as shadow_module
from pipeline_core.discovery.nonobviousness_shadow import (
    _json_safe,
    build_nonobviousness_shadow,
    reconcile_intake_required_bridge,
)
from pipeline_core.discovery.novelty_residue import (
    HypothesisNoveltyResidue,
    NoveltyResidueClaim,
)
from pipeline_core.discovery.typed_required_bridge_binding import (
    TYPED_REQUIRED_BRIDGE_PROVENANCE,
    validate_typed_required_bridge_binding,
)


def _claim(
    *,
    required_bridge: str = "",
) -> NoveltyResidueClaim:
    return NoveltyResidueClaim(
        hypothesis_id="h1",
        claim_id="c1",
        claim_text=(
            "Fe coordination controls H adsorption "
            "through the paired active site."
        ),
        claim_kind="relation",
        prior_art_status="NO_DIRECT_MATCH_FOUND",
        disposition="RESIDUAL",
        is_residue=True,
        distinguishing_terms=("Fe coordination",),
        prior_art_identity_terms=("Fe coordination",),
        relation_nucleus_terms=("controls",),
        required_bridge=required_bridge,
        predicted_observation=(
            "H adsorption changes with Fe coordination."
        ),
        falsification_condition=(
            "H adsorption remains unchanged across Fe coordination states."
        ),
        direct_or_partial_work_ids=(),
        lower_order_work_ids=(),
        component_work_ids=(),
    )


def _hypothesis() -> SimpleNamespace:
    return SimpleNamespace(
        hypothesis_id="h1",
        inferential_bridge=(
            "Fe coordination controls H adsorption "
            "through the paired active site."
        ),
    )


def _binding_and_contract(
    hypothesis: SimpleNamespace,
) -> tuple[dict[str, object], dict[str, object]]:
    quote = hypothesis.inferential_bridge
    source_sha = hashlib.sha256(
        quote.encode("utf-8")
    ).hexdigest()
    endpoints = [
        "Fe coordination",
        "H adsorption",
    ]
    scopes: list[str] = []
    directions: list[str] = []

    binding = {
        "schema_version":
            "novelty-required-bridge-source-binding-diagnostic-v1",
        "binding_semantics":
            "TYPED_RELATION_REFERENCE_TO_EXISTING_SOURCE_SPAN",
        "source_path": "inferential_bridge",
        "source_sha256": source_sha,
        "start": 0,
        "end": len(quote),
        "quote": quote,
        "relation_endpoint_anchors": endpoints,
        "scope_qualifier_spans": scopes,
        "directional_qualifier_spans": directions,
        "diagnostic_only": True,
        "production_authority": False,
        "free_text_bridge_generated": False,
        "novelty_assessed": False,
        "scientific_truth_assessed": False,
        "scientific_equivalence_assessed": False,
    }
    contract = {
        "hypothesis_id": "h1",
        "claim_id": "c1",
        "source_bridge_sha256": source_sha,
        "relation_endpoint_anchors": endpoints,
        "scope_qualifier_spans": scopes,
        "directional_qualifier_spans": directions,
        "semantic_warnings": {
            "bridge_alignment_shadow_warning": False,
            "direction_shadow_warning": False,
            "endpoint_shadow_warning": False,
            "scope_shadow_warning": False,
        },
    }
    return binding, contract


def test_validate_typed_binding_returns_exact_existing_source_quote() -> None:
    hypothesis = _hypothesis()
    claim = _claim()
    binding, contract = _binding_and_contract(
        hypothesis
    )

    recovered = validate_typed_required_bridge_binding(
        hypothesis=hypothesis,
        claim=claim,
        binding=binding,
        contract=contract,
    )

    assert recovered == hypothesis.inferential_bridge


def test_validate_typed_binding_rejects_source_span_drift() -> None:
    hypothesis = _hypothesis()
    claim = _claim()
    binding, contract = _binding_and_contract(
        hypothesis
    )
    binding = dict(binding)
    binding["quote"] = (
        "Fe coordination controls H adsorption."
    )

    with pytest.raises(
        ValueError,
        match="quote/offset mismatch",
    ):
        validate_typed_required_bridge_binding(
            hypothesis=hypothesis,
            claim=claim,
            binding=binding,
            contract=contract,
        )


def test_validate_typed_binding_rejects_multi_sentence_source_span() -> None:
    hypothesis = SimpleNamespace(
        hypothesis_id="h1",
        inferential_bridge=(
            "Background context is stated first. "
            "Fe coordination controls H adsorption "
            "through the paired active site."
        ),
    )
    claim = _claim()
    quote = hypothesis.inferential_bridge
    source_sha = hashlib.sha256(
        quote.encode("utf-8")
    ).hexdigest()
    endpoints = [
        "Fe coordination",
        "H adsorption",
    ]

    binding = {
        "schema_version":
            "novelty-required-bridge-source-binding-diagnostic-v1",
        "binding_semantics":
            "TYPED_RELATION_REFERENCE_TO_EXISTING_SOURCE_SPAN",
        "source_path": "inferential_bridge",
        "source_sha256": source_sha,
        "start": 0,
        "end": len(quote),
        "quote": quote,
        "relation_endpoint_anchors": endpoints,
        "scope_qualifier_spans": [],
        "directional_qualifier_spans": [],
        "diagnostic_only": True,
        "production_authority": False,
        "free_text_bridge_generated": False,
        "novelty_assessed": False,
        "scientific_truth_assessed": False,
        "scientific_equivalence_assessed": False,
    }
    contract = {
        "hypothesis_id": "h1",
        "claim_id": "c1",
        "source_bridge_sha256": source_sha,
        "relation_endpoint_anchors": endpoints,
        "scope_qualifier_spans": [],
        "directional_qualifier_spans": [],
        "semantic_warnings": {
            "bridge_alignment_shadow_warning": False,
            "direction_shadow_warning": False,
            "endpoint_shadow_warning": False,
            "scope_shadow_warning": False,
        },
    }

    with pytest.raises(
        ValueError,
        match="exactly one existing source sentence",
    ):
        validate_typed_required_bridge_binding(
            hypothesis=hypothesis,
            claim=claim,
            binding=binding,
            contract=contract,
        )


def test_validate_typed_binding_rejects_nonunique_qualifying_sentence() -> None:
    sentence = (
        "Fe coordination controls H adsorption "
        "through the paired active site."
    )
    hypothesis = SimpleNamespace(
        hypothesis_id="h1",
        inferential_bridge=(
            sentence
            + " "
            + sentence
        ),
    )
    claim = _claim()
    source_bridge = hypothesis.inferential_bridge
    source_sha = hashlib.sha256(
        source_bridge.encode("utf-8")
    ).hexdigest()
    endpoints = [
        "Fe coordination",
        "H adsorption",
    ]

    binding = {
        "schema_version":
            "novelty-required-bridge-source-binding-diagnostic-v1",
        "binding_semantics":
            "TYPED_RELATION_REFERENCE_TO_EXISTING_SOURCE_SPAN",
        "source_path": "inferential_bridge",
        "source_sha256": source_sha,
        "start": 0,
        "end": len(sentence),
        "quote": sentence,
        "relation_endpoint_anchors": endpoints,
        "scope_qualifier_spans": [],
        "directional_qualifier_spans": [],
        "diagnostic_only": True,
        "production_authority": False,
        "free_text_bridge_generated": False,
        "novelty_assessed": False,
        "scientific_truth_assessed": False,
        "scientific_equivalence_assessed": False,
    }
    contract = {
        "hypothesis_id": "h1",
        "claim_id": "c1",
        "source_bridge_sha256": source_sha,
        "relation_endpoint_anchors": endpoints,
        "scope_qualifier_spans": [],
        "directional_qualifier_spans": [],
        "semantic_warnings": {
            "bridge_alignment_shadow_warning": False,
            "direction_shadow_warning": False,
            "endpoint_shadow_warning": False,
            "scope_shadow_warning": False,
        },
    }

    with pytest.raises(
        ValueError,
        match="not uniquely qualified",
    ):
        validate_typed_required_bridge_binding(
            hypothesis=hypothesis,
            claim=claim,
            binding=binding,
            contract=contract,
        )


def test_reconcile_typed_binding_materializes_only_validated_source_text() -> None:
    hypothesis = _hypothesis()
    claim = _claim()
    binding, contract = _binding_and_contract(
        hypothesis
    )
    intake_claim = _json_safe(claim)
    assert isinstance(intake_claim, dict)
    assert intake_claim["required_bridge"] == ""

    reconciled = reconcile_intake_required_bridge(
        claim,
        intake_claim=intake_claim,
        specification_provenance={
            "required_bridge":
                TYPED_REQUIRED_BRIDGE_PROVENANCE,
        },
        hypothesis=hypothesis,
        sibling_claims=(claim,),
        required_bridge_binding=binding,
        required_bridge_binding_contract=contract,
    )

    assert (
        reconciled.required_bridge
        == hypothesis.inferential_bridge
    )
    assert claim.required_bridge == ""


def test_reconcile_typed_binding_cannot_replace_existing_query_plan_bridge() -> None:
    hypothesis = _hypothesis()
    claim = _claim(
        required_bridge=hypothesis.inferential_bridge
    )
    binding, contract = _binding_and_contract(
        hypothesis
    )
    intake_claim = _json_safe(claim)
    assert isinstance(intake_claim, dict)

    with pytest.raises(
        ValueError,
        match="cannot replace existing query-plan bridge",
    ):
        reconcile_intake_required_bridge(
            claim,
            intake_claim=intake_claim,
            specification_provenance={
                "required_bridge":
                    TYPED_REQUIRED_BRIDGE_PROVENANCE,
            },
            hypothesis=hypothesis,
            sibling_claims=(claim,),
            required_bridge_binding=binding,
            required_bridge_binding_contract=contract,
        )


def test_shadow_typed_binding_preserves_raw_claim_and_only_unlocks_specification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hypothesis = _hypothesis()
    claim = _claim()
    binding, contract = _binding_and_contract(
        hypothesis
    )
    residue = HypothesisNoveltyResidue(
        hypothesis_id="h1",
        external_status="NO_DIRECT_MATCH_FOUND",
        claims=(claim,),
    )

    monkeypatch.setattr(
        shadow_module,
        "extract_novelty_residue",
        lambda plan, report: [residue],
    )
    monkeypatch.setattr(
        shadow_module,
        "recover_uniquely_attributed_required_bridge",
        lambda **kwargs: "",
    )

    plan = SimpleNamespace(
        source_portfolio_id="p1",
        plan_id="plan1",
        plan_sha256="plan-sha",
    )
    report = SimpleNamespace(
        source_portfolio_id="p1",
        report_id="report1",
        report_sha256="report-sha",
        source_prior_art_packet_id="packet1",
        cards=[],
    )
    portfolio = SimpleNamespace(
        portfolio_id="p1",
        hypotheses=[hypothesis],
    )

    result = build_nonobviousness_shadow(
        plan=plan,
        report=report,
        source_portfolio=portfolio,
        required_bridge_bindings={
            "c1": {
                "binding": binding,
                "contract": contract,
            }
        },
    )

    decision = result["hypotheses"][0]["claims"][0]

    assert decision["shadow_state"] == "READY_FOR_CLOSURE"
    assert decision["claim"]["required_bridge"] == ""
    assert (
        decision["specification_provenance"]["required_bridge"]
        == TYPED_REQUIRED_BRIDGE_PROVENANCE
    )
    assert decision["required_bridge_binding"] == binding
    assert (
        decision["required_bridge_binding_contract"]
        == contract
    )
    assert result["scientific_selection_changed"] is False


def test_shadow_without_typed_binding_retains_fail_closed_specification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hypothesis = _hypothesis()
    claim = _claim()
    residue = HypothesisNoveltyResidue(
        hypothesis_id="h1",
        external_status="NO_DIRECT_MATCH_FOUND",
        claims=(claim,),
    )

    monkeypatch.setattr(
        shadow_module,
        "extract_novelty_residue",
        lambda plan, report: [residue],
    )
    monkeypatch.setattr(
        shadow_module,
        "recover_uniquely_attributed_required_bridge",
        lambda **kwargs: "",
    )

    plan = SimpleNamespace(
        source_portfolio_id="p1",
        plan_id="plan1",
        plan_sha256="plan-sha",
    )
    report = SimpleNamespace(
        source_portfolio_id="p1",
        report_id="report1",
        report_sha256="report-sha",
        source_prior_art_packet_id="packet1",
        cards=[],
    )
    portfolio = SimpleNamespace(
        portfolio_id="p1",
        hypotheses=[hypothesis],
    )

    result = build_nonobviousness_shadow(
        plan=plan,
        report=report,
        source_portfolio=portfolio,
    )

    decision = result["hypotheses"][0]["claims"][0]
    assert decision["shadow_state"] == "NEEDS_REFINEMENT"
    assert decision["claim"]["required_bridge"] == ""
    assert (
        decision["specification_provenance"]["required_bridge"]
        == "UNRESOLVED"
    )
