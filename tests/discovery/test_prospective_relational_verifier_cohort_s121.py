from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipeline_core.discovery.prospective_relational_verifier_cohort import (
    ProspectiveRelationalVerifierPoolSpec,
    build_prospective_relational_verifier_cohort_freeze,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingClaimPlan,
    RelationalAtomicBindingHypothesisPlan,
    RelationalAtomicBindingPlan,
)
from pipeline_core.discovery.relational_atomic_endpoint_binding import (
    CompiledLiteralEndpointBinding,
    RelationalAtomicEndpointBindingReport,
)
from pipeline_core.discovery.reframing.atomic_cross_lane_synthesis import (
    AtomicCrossLaneSynthesisReport,
    CompiledAtomicCrossLaneHypothesis,
    CompiledAtomicSpecification,
)
from pipeline_core.discovery.reframing.production_candidate_contract import (
    ProductionCandidateFalsifier,
    ProductionCandidateLineageValidation,
    ProductionCandidatePrediction,
    ProductionFacingScientificCandidatePortfolio,
    ProductionScientificCandidate,
)
from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimScientificStructure,
)


def _sha_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _candidate(candidate_id: str, lane: str) -> ProductionScientificCandidate:
    return ProductionScientificCandidate(
        candidate_id=candidate_id,
        source_entry_id="entry:" + candidate_id,
        source_lane=lane,
        source_object_kind=(
            "relational_hypothesis"
            if lane == "RELATIONAL_DISCOVERY"
            else "scientific_reframe"
        ),
        source_schema_version="test-v1",
        source_object_id="source:" + candidate_id,
        source_epistemic_status="test",
        title="Candidate",
        reasoning_label="test",
        scientific_proposal="proposal",
        reasoning_rationale="rationale",
        predictions=[
            ProductionCandidatePrediction(
                observable="signal",
                expected_direction="increase",
                rationale="test",
            )
        ] if lane == "RELATIONAL_DISCOVERY" else [
            ProductionCandidatePrediction(
                observable="signal",
                baseline_expectation="baseline",
                alternative_expectation="alternative",
                discriminating_outcome="different",
            )
        ],
        falsifiers=[
            ProductionCandidateFalsifier(
                falsifying_outcome="no difference"
            )
        ],
    )


def _write_case(
    root: Path,
    case_key: str,
    *,
    lane_profile: str,
) -> dict:
    case = root / case_key
    case.mkdir(parents=True)

    rel_id = "production_candidate:rel:" + case_key
    ref_id = "production_candidate:ref:" + case_key
    candidates = [
        _candidate(rel_id, "RELATIONAL_DISCOVERY"),
        _candidate(ref_id, "SCIENTIFIC_REFRAMING"),
    ]
    production = ProductionFacingScientificCandidatePortfolio(
        portfolio_id="production_portfolio:" + case_key,
        source_cross_lane_portfolio_id="cross:" + case_key,
        source_cross_lane_portfolio_file_sha256="a" * 64,
        source_task_id="task:" + case_key,
        source_context_id="context:" + case_key,
        source_context_sha256="b" * 64,
        question="question",
        domain_profile_id="sers_au_ag",
        candidates=candidates,
        lineage_validation=ProductionCandidateLineageValidation(),
        candidate_count=2,
        relational_candidate_count=1,
        reframing_candidate_count=1,
        source_kind_counts={
            "relational_hypothesis": 1,
            "scientific_reframe": 1,
        },
    )

    if lane_profile == "RELATIONAL_ONLY":
        spec_sources = [rel_id]
    elif lane_profile == "REFRAMING_ONLY":
        spec_sources = [ref_id]
    else:
        spec_sources = [rel_id, ref_id]

    original_id = "hypothesis:original:" + case_key
    candidate_id = "hypothesis:candidate:" + case_key
    final_id = "hypothesis:final:" + case_key
    claim_id = "claim:" + case_key

    spec = CompiledAtomicSpecification(
        local_id="a1",
        claim_id=claim_id,
        kind="moderator_interaction",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text=(
            "Catalyst composition changes Raman intensity "
            "with nanostructure spacing."
        ),
        rationale="test",
        source_candidate_ids=spec_sources,
        premise_statement_ids=[],
        gap_statement_ids=[],
        prior_art_identity_terms=["Catalyst composition"],
        relation_endpoint_anchors=[
            "Raman intensity",
            "nanostructure spacing",
        ],
        scope_qualifier_spans=[],
        directional_qualifier_spans=[],
        relation_nucleus_terms=[
            "Raman intensity",
            "nanostructure spacing",
        ],
        distinguishing_terms=[],
        required_bridge=(
            "Catalyst composition changes Raman intensity "
            "with nanostructure spacing."
        ),
        observable="Raman intensity",
        predicted_observation=(
            "Catalyst composition changes Raman intensity "
            "with nanostructure spacing."
        ),
        falsification_condition=(
            "Catalyst composition does not change Raman intensity "
            "with nanostructure spacing."
        ),
        prediction_observation_id="prediction:" + case_key,
        falsification_criterion_id="falsifier:" + case_key,
        search_concepts=[],
        search_queries=["Raman intensity nanostructure spacing"],
        scientific_structure=NoveltyClaimScientificStructure(),
        scientific_structure_reason_codes=[],
    )
    atomic = AtomicCrossLaneSynthesisReport(
        report_id="atomic_report:" + case_key,
        source_candidate_portfolio_id=production.portfolio_id,
        source_task_id=production.source_task_id,
        source_context_id=production.source_context_id,
        question="question",
        backend_name="test",
        model_name="test",
        hypotheses=[
            CompiledAtomicCrossLaneHypothesis(
                hypothesis_id=original_id,
                source_candidate_ids=[rel_id, ref_id],
                source_lanes=[
                    "RELATIONAL_DISCOVERY",
                    "SCIENTIFIC_REFRAMING",
                ],
                synthesis_kind="conditional_relation_refinement",
                title="test",
                hypothesis_statement=spec.text,
                premise_statement_ids=[],
                gap_statement_ids=[],
                assumptions=[],
                atomic_specifications=[spec],
            )
        ],
        hypothesis_count=1,
        atomic_specification_count=1,
        llm_calls_performed=1,
    )

    claim = RelationalAtomicBindingClaimPlan(
        candidate_hypothesis_id=candidate_id,
        final_hypothesis_id=final_id,
        claim_id=claim_id,
        claim_rank=1,
        kind="moderator_interaction",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        claim_text=spec.text,
        rationale=spec.rationale,
        prior_art_identity_terms=["Catalyst composition"],
        relation_nucleus_terms=list(spec.relation_nucleus_terms),
        required_bridge=spec.required_bridge,
        predicted_observation=spec.predicted_observation,
        falsification_condition=spec.falsification_condition,
        search_concepts=[],
        search_queries=list(spec.search_queries),
        source_claim_sha256="c" * 64,
        binding_status="READY_FOR_LITERAL_ENDPOINT_BINDING",
    )
    hyp = RelationalAtomicBindingHypothesisPlan(
        original_hypothesis_id=original_id,
        candidate_hypothesis_id=candidate_id,
        final_hypothesis_id=final_id,
        alpha6_decision="ignored_by_selector",
        certification_status="ignored_by_selector",
        n10_selection_class="ignored_by_selector",
        source_candidate_portfolio=str(case / "candidate.json"),
        source_candidate_portfolio_sha256="d" * 64,
        source_query_plan=str(case / "query.json"),
        source_query_plan_sha256="e" * 64,
        claim_count=1,
        binding_ready_claim_count=1,
        novelty_bearing_binding_ready_claim_count=1,
        binding_status="READY_FOR_LITERAL_ENDPOINT_BINDING",
        claims=[claim],
    )
    plan_body = {
        "schema_version": "relational-atomic-binding-plan-v1",
        "run_dir": str(case),
        "source_alpha6_candidate_portfolio": str(case / "final.json"),
        "source_alpha6_candidate_portfolio_sha256": "f" * 64,
        "source_certification_report": str(case / "cert.json"),
        "source_certification_report_sha256": "1" * 64,
        "hypotheses": [hyp.model_dump(mode="json")],
        "hypothesis_count": 1,
        "ready_hypothesis_count": 1,
        "not_ready_hypothesis_count": 0,
        "claim_count": 1,
        "binding_ready_claim_count": 1,
        "novelty_bearing_binding_ready_claim_count": 1,
        "hypothesis_status_counts": {
            "READY_FOR_LITERAL_ENDPOINT_BINDING": 1
        },
        "claim_status_counts": {
            "READY_FOR_LITERAL_ENDPOINT_BINDING": 1
        },
        "certification_only_source_required": True,
        "candidate_final_authority_equivalence_required": True,
        "endpoint_binding_performed": False,
        "verifier_result_observed": False,
        "production_selection_authority": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    plan_sha = _sha_json(plan_body)
    plan = RelationalAtomicBindingPlan(
        **plan_body,
        plan_id="relational_atomic_binding_plan:" + plan_sha[:20],
        plan_sha256=plan_sha,
    )

    endpoint = RelationalAtomicEndpointBindingReport(
        report_id="endpoint:" + case_key,
        source_binding_plan_id=plan.plan_id,
        source_binding_plan_sha256=plan.plan_sha256,
        backend_name="test",
        model_name="test",
        bindings=[
            CompiledLiteralEndpointBinding(
                candidate_hypothesis_id=candidate_id,
                final_hypothesis_id=final_id,
                claim_id=claim_id,
                novelty_selection_role="NOVELTY_BEARING",
                source_claim_sha256=claim.source_claim_sha256,
                relation_endpoint_anchors=[
                    "Raman intensity",
                    "nanostructure spacing",
                ],
                outcome="BOUND_LITERAL_ENDPOINTS",
            )
        ],
        selected_hypothesis_count=1,
        selected_claim_count=1,
        bound_claim_count=1,
        abstained_claim_count=0,
        novelty_bearing_bound_claim_count=1,
        llm_calls_performed=1,
    )

    paths = {
        "production_candidate_portfolio": case / "production.json",
        "atomic_synthesis_report": case / "atomic.json",
        "binding_plan": case / "plan.json",
        "endpoint_binding_report": case / "endpoint.json",
    }
    paths["production_candidate_portfolio"].write_text(
        production.model_dump_json(indent=2),
        encoding="utf-8",
    )
    paths["atomic_synthesis_report"].write_text(
        atomic.model_dump_json(indent=2),
        encoding="utf-8",
    )
    paths["binding_plan"].write_text(
        plan.model_dump_json(indent=2),
        encoding="utf-8",
    )
    paths["endpoint_binding_report"].write_text(
        endpoint.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return {key: str(path) for key, path in paths.items()}


def _spec(root: Path) -> ProspectiveRelationalVerifierPoolSpec:
    layouts = [
        ("C01", "RELATIONAL_ONLY"),
        ("C02", "RELATIONAL_ONLY"),
        ("C03", "REFRAMING_ONLY"),
        ("C04", "REFRAMING_ONLY"),
        ("C05", "MIXED"),
        ("C06", "MIXED"),
    ]
    cases = []
    for case_key, profile in layouts:
        paths = _write_case(root, case_key, lane_profile=profile)
        cases.append({"case_key": case_key, **paths})
    return ProspectiveRelationalVerifierPoolSpec(
        cases=cases,
        requested_case_count=5,
        requested_relational_only=2,
        requested_reframing_only=2,
        requested_mixed=1,
    )


def test_freeze_selects_requested_lane_structure_without_outcome_fields(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    freeze = build_prospective_relational_verifier_cohort_freeze(
        pool_spec=spec,
        pool_spec_sha256="a" * 64,
        base_dir=tmp_path,
    )

    assert [row.prospective_case_id for row in freeze.frozen_cases] == [
        "P06", "P07", "P08", "P09", "P10"
    ]
    assert freeze.frozen_lane_profile_counts == {
        "MIXED": 1,
        "REFRAMING_ONLY": 2,
        "RELATIONAL_ONLY": 2,
    }
    assert freeze.prior_art_retrieval_used_for_selection is False
    assert freeze.external_novelty_outcomes_used_for_selection is False
    assert freeze.final_certification_used_for_selection is False
    assert freeze.old_n10_outcome_fields_used_for_selection is False
    assert freeze.new_verifier_result_observed_before_freeze is False


def test_selector_is_deterministic(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    first = build_prospective_relational_verifier_cohort_freeze(
        pool_spec=spec,
        pool_spec_sha256="b" * 64,
        base_dir=tmp_path,
    )
    second = build_prospective_relational_verifier_cohort_freeze(
        pool_spec=spec,
        pool_spec_sha256="b" * 64,
        base_dir=tmp_path,
    )
    assert first.freeze_id == second.freeze_id
    assert [
        row.final_hypothesis_id for row in first.frozen_cases
    ] == [
        row.final_hypothesis_id for row in second.frozen_cases
    ]


def test_selector_ignores_old_n10_status_strings(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    baseline = build_prospective_relational_verifier_cohort_freeze(
        pool_spec=spec,
        pool_spec_sha256="c" * 64,
        base_dir=tmp_path,
    )

    # Mutation of old-N10 descriptive fields must not affect structural
    # selection. Recompute plan hashes because the schema is hash-bound.
    first_case = spec.cases[0]
    plan_path = Path(first_case.binding_plan)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["hypotheses"][0]["alpha6_decision"] = "wildly_different"
    payload["hypotheses"][0]["certification_status"] = "also_different"
    payload["hypotheses"][0]["n10_selection_class"] = "different_again"
    payload.pop("plan_id")
    payload.pop("plan_sha256")
    digest = _sha_json(payload)
    payload["plan_id"] = "relational_atomic_binding_plan:" + digest[:20]
    payload["plan_sha256"] = digest
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    endpoint_path = Path(first_case.endpoint_binding_report)
    endpoint_payload = json.loads(endpoint_path.read_text(encoding="utf-8"))
    endpoint_payload["source_binding_plan_id"] = payload["plan_id"]
    endpoint_payload["source_binding_plan_sha256"] = digest
    endpoint_path.write_text(
        json.dumps(endpoint_payload),
        encoding="utf-8",
    )

    mutated = build_prospective_relational_verifier_cohort_freeze(
        pool_spec=spec,
        pool_spec_sha256="c" * 64,
        base_dir=tmp_path,
    )
    assert [
        (row.pool_case_key, row.final_hypothesis_id)
        for row in baseline.frozen_cases
    ] == [
        (row.pool_case_key, row.final_hypothesis_id)
        for row in mutated.frozen_cases
    ]


def test_unbound_novelty_claim_is_excluded(tmp_path: Path) -> None:
    paths = _write_case(
        tmp_path,
        "C01",
        lane_profile="RELATIONAL_ONLY",
    )
    endpoint_path = Path(paths["endpoint_binding_report"])
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    endpoint["bindings"][0]["relation_endpoint_anchors"] = []
    endpoint["bindings"][0]["outcome"] = "ABSTAINED_UNBINDABLE"
    endpoint["bindings"][0]["abstention_reason"] = "no literal pair"
    endpoint["bound_claim_count"] = 0
    endpoint["abstained_claim_count"] = 1
    endpoint["novelty_bearing_bound_claim_count"] = 0
    endpoint_path.write_text(json.dumps(endpoint), encoding="utf-8")

    spec = ProspectiveRelationalVerifierPoolSpec(
        cases=[{"case_key": "C01", **paths}],
        requested_case_count=1,
        requested_relational_only=1,
        requested_reframing_only=0,
        requested_mixed=0,
    )

    with pytest.raises(
        ValueError,
        match="insufficient eligible prospective verifier hypotheses",
    ):
        build_prospective_relational_verifier_cohort_freeze(
            pool_spec=spec,
            pool_spec_sha256="d" * 64,
            base_dir=tmp_path,
        )
