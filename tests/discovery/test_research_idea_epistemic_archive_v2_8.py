from __future__ import annotations

from pipeline_core.discovery.research_idea_epistemic_archive import (
    EpistemicDecompositionReport,
    EpistemicRealizationRecord,
    EvidenceRecoveryPlan,
    EvidenceRecoveryRequest,
    MissingEvidenceRequirement,
    build_epistemic_decomposition_report,
    build_evidence_recovery_plan,
    build_multi_realization_archive,
    build_recovery_relevance_gate,
)


def idea(idea_id: str = "idea:1", generation: int = 2):
    return {
        "idea_id": idea_id,
        "generation_index": generation,
        "source_context_id": "ctx:1",
        "kernel": {
            "canonical_intent": "Test whether hotspot redistribution mediates orientation dependent Raman response",
            "core_scientific_commitments": [
                "molecular orientation --MODULATES--> hotspot redistribution",
                "hotspot redistribution --MODULATES--> Raman intensity ratio",
            ],
            "scope_commitments": ["Au Ag SERS"],
            "contrastive_commitments": ["direct orientation-only explanation"],
        },
        "differential_prediction": "Raman mode ratios should shift with orientation even at matched loading.",
        "falsification_condition": "Matched orientation states show no reproducible ratio shift.",
        "discriminating_observation": "orientation-conditioned Raman ratio",
    }


def lifecycle(*links):
    return {"links": list(links)}


def link(
    *,
    realization_id: str,
    hypothesis_id: str | None,
    status: str,
    attempt: int,
    issues=None,
):
    return {
        "realization_id": realization_id,
        "idea_id": "idea:1",
        "generation_index": 2,
        "hypothesis_id": hypothesis_id,
        "source_context_id": "ctx:1",
        "realization_kind": "INITIAL" if attempt == 1 else "ALTERNATE",
        "attempt_index": attempt,
        "materialization_status": status,
        "issue_codes": list(issues or []),
    }


def card(hypothesis_id: str):
    return {
        "hypothesis_id": hypothesis_id,
        "premise_statement_ids": ["stmt:1", "stmt:2"],
        "source_paper_ids": ["paper:1"],
        "inferential_bridge": "Orientation-dependent hotspot redistribution mediates mode-ratio changes.",
        "predicted_observations": [
            {"observable": "Raman mode ratio", "expected_direction": "shift", "rationale": "bridge"}
        ],
        "falsification_criteria": [
            {"observable": "Raman mode ratio", "falsifying_outcome": "No ratio difference at matched loading"}
        ],
        "assumptions": ["loading can be matched"],
    }


def feedback(hypothesis_id: str, *, residual=None, identifiable="PROSPECTIVELY_IDENTIFIABLE", evidence="SUPPORTED", measurement="COMPATIBLE"):
    return {
        "records": [
            {
                "hypothesis_id": hypothesis_id,
                "idea_id": "idea:1",
                "prospective_identifiability": identifiable,
                "current_evidence_status": evidence,
                "directionality_mode": "SUPPORTED",
                "measurement_compatibility_mode": measurement,
                "residual_epistemic_state": residual,
                "residual_state_reason": None,
            }
        ]
    }


def test_ungrounded_research_idea_is_preserved_as_partial_object():
    report = build_epistemic_decomposition_report(
        research_ideas=[idea()],
        lifecycles=[
            lifecycle(
                link(
                    realization_id="r:1",
                    hypothesis_id=None,
                    status="ABSTAINED",
                    attempt=1,
                )
            )
        ],
        portfolios=[{"hypotheses": []}],
    )
    row = report.records[0]
    assert row.idea_id == "idea:1"
    assert row.hypothesis_id is None
    assert row.research_idea_preserved_when_not_groundable is True
    assert row.hypothesis_card_contract_relaxed is False
    assert row.epistemic_maturity == "EVIDENCE_SEEKING"
    assert {req.kind for req in row.missing_evidence_requirements} == {"GROUNDING_PREMISE_COVERAGE"}


def test_generation_failure_is_not_automatically_misdiagnosed_as_evidence_gap():
    report = build_epistemic_decomposition_report(
        research_ideas=[idea()],
        lifecycles=[
            lifecycle(
                link(
                    realization_id="r:1",
                    hypothesis_id=None,
                    status="GENERATION_FAILED",
                    attempt=1,
                    issues=["ProviderTimeout"],
                )
            )
        ],
        portfolios=[{"hypotheses": []}],
    )
    assert report.records[0].missing_evidence_requirements == []


def test_strict_grounded_hypothesis_keeps_bridge_separate_from_premises():
    report = build_epistemic_decomposition_report(
        research_ideas=[idea()],
        lifecycles=[lifecycle(link(realization_id="r:1", hypothesis_id="h:1", status="MATERIALIZED", attempt=1))],
        portfolios=[{"hypotheses": [card("h:1")]}],
        feedback_reports=[feedback("h:1", residual="RESIDUAL_AUTHORITY_CANDIDATE_SHADOW")],
    )
    row = report.records[0]
    assert row.grounding_coverage == "STRICT"
    assert row.epistemic_maturity == "OPERATIONAL_GROUNDED"
    assert row.grounded_premise_statement_ids == ["stmt:1", "stmt:2"]
    assert "hotspot redistribution" in (row.hypothetical_bridge or "")
    assert row.speculative_components_are_not_positive_premises is True


def test_residual_evidence_gap_routes_to_literature_before_idea_mutation():
    report = build_epistemic_decomposition_report(
        research_ideas=[idea()],
        lifecycles=[lifecycle(link(realization_id="r:1", hypothesis_id="h:1", status="MATERIALIZED", attempt=1))],
        portfolios=[{"hypotheses": [card("h:1")]}],
        feedback_reports=[feedback("h:1", residual="UNRESOLVED_EVIDENCE_GAP")],
    )
    req = next(row for row in report.records[0].missing_evidence_requirements if row.kind == "RELATION_OR_MECHANISM_EVIDENCE")
    assert "LITERATURE_ACQUISITION" in req.recovery_routes
    assert "CONTEXT_REBUILD" in req.recovery_routes


def test_topology_gap_routes_to_kg_retraversal_not_literature_by_default():
    report = build_epistemic_decomposition_report(
        research_ideas=[idea()],
        lifecycles=[lifecycle(link(realization_id="r:1", hypothesis_id="h:1", status="MATERIALIZED", attempt=1))],
        portfolios=[{"hypotheses": [card("h:1")]}],
        feedback_reports=[feedback("h:1", residual="UNRESOLVED_TOPOLOGY_GAP")],
    )
    req = next(row for row in report.records[0].missing_evidence_requirements if row.kind == "GRAPH_TOPOLOGY_COVERAGE")
    assert "KG_RETRAVERSAL" in req.recovery_routes
    assert "LITERATURE_ACQUISITION" not in req.recovery_routes


def test_not_operationalizable_is_local_gap_not_idea_deletion():
    report = build_epistemic_decomposition_report(
        research_ideas=[idea()],
        lifecycles=[lifecycle(link(realization_id="r:1", hypothesis_id="h:1", status="MATERIALIZED", attempt=1))],
        portfolios=[{"hypotheses": [card("h:1")]}],
        feedback_reports=[feedback("h:1", identifiable="NOT_OPERATIONALIZABLE")],
    )
    row = report.records[0]
    req = next(value for value in row.missing_evidence_requirements if value.kind == "MEASUREMENT_OR_OBSERVABLE_SUPPORT")
    assert req.recovery_routes == ["OPERATIONALIZATION_REVIEW"]
    assert "NOT_OPERATIONALIZABLE" in row.operationalization_gap_codes
    assert row.research_idea_preserved_when_not_groundable is True


def _epistemic_record(
    eid: str,
    *,
    realization: str,
    residual: str,
    materialized: bool,
    operational: bool,
    evaluated: bool = True,
) -> EpistemicRealizationRecord:
    return EpistemicRealizationRecord(
        epistemic_realization_id=eid,
        idea_id="idea:1",
        generation_index=2,
        realization_id=realization,
        hypothesis_id=(f"h:{realization}" if materialized else None),
        source_context_id="ctx:1",
        attempt_index=int(realization.split(":")[-1]),
        realization_kind="ALTERNATE",
        materialization_status="MATERIALIZED" if materialized else "ABSTAINED",
        canonical_intent="intent",
        scientific_commitments=["A --CAUSES--> B"],
        grounded_premise_statement_ids=["stmt:1"] if materialized else [],
        prediction_texts=["P"],
        falsifier_texts=["F"],
        prospective_identifiability=("PROSPECTIVELY_IDENTIFIABLE" if operational else "NOT_OPERATIONALIZABLE") if evaluated else None,
        residual_epistemic_state=residual,
        grounding_coverage="STRICT" if materialized else "NONE",
        epistemic_maturity=("OPERATIONAL_GROUNDED" if materialized and operational else "STRICT_GROUNDED" if materialized else "SPECULATIVE_BUT_FALSIFIABLE"),
    )


def _decomposition(rows):
    return EpistemicDecompositionReport(
        report_id="d:1",
        report_sha256="a" * 64,
        records=rows,
        record_count=len(rows),
        idea_count=len({row.idea_id for row in rows}),
        maturity_counts={},
        requirement_kind_counts={},
        literature_recovery_requirement_count=0,
        kg_retraversal_requirement_count=0,
    )


def test_archive_preserves_epistemically_distinct_tradeoff_realizations():
    strong_operational = _epistemic_record(
        "e:1", realization="r:1", residual="UNRESOLVED_EVIDENCE_GAP", materialized=True, operational=True
    )
    residually_distinct_but_weaker = _epistemic_record(
        "e:2", realization="r:2", residual="RESIDUAL_AUTHORITY_CANDIDATE_SHADOW", materialized=True, operational=False
    )
    archive = build_multi_realization_archive(_decomposition([strong_operational, residually_distinct_but_weaker]))
    entries = {row.epistemic_realization_id: row for row in archive.archives[0].entries}
    assert entries["e:1"].archive_retained is True
    assert entries["e:2"].archive_retained is True
    assert archive.destructive_replacement_applied is False


def test_archive_keeps_dominated_attempt_for_audit_without_selecting_it():
    good = _epistemic_record(
        "e:1", realization="r:1", residual="UNRESOLVED_EVIDENCE_GAP", materialized=True, operational=True
    )
    weak = _epistemic_record(
        "e:2", realization="r:2", residual="UNRESOLVED_EVIDENCE_GAP", materialized=False, operational=False
    )
    archive = build_multi_realization_archive(_decomposition([good, weak]))
    entries = {row.epistemic_realization_id: row for row in archive.archives[0].entries}
    assert entries["e:2"].archive_retained is False
    assert entries["e:2"].archive_disposition == "DIAGNOSTIC_DOMINATED"
    assert len(archive.archives[0].entries) == 2


def test_evidence_recovery_collapses_same_idea_requirement_across_realizations():
    req1 = MissingEvidenceRequirement(
        requirement_id="req:1",
        idea_id="idea:1",
        generation_index=2,
        source_realization_ids=["r:1"],
        kind="RELATION_OR_MECHANISM_EVIDENCE",
        description="gap",
        search_terms=["hotspot", "orientation"],
        proposed_queries=["hotspot orientation"],
        recovery_routes=["LITERATURE_ACQUISITION", "CONTEXT_REBUILD"],
        reason_codes=["A"],
    )
    req2 = req1.model_copy(
        update={
            "requirement_id": "req:2",
            "source_realization_ids": ["r:2"],
            "proposed_queries": ["orientation Raman mechanism"],
            "reason_codes": ["B"],
        }
    )
    base = _epistemic_record(
        "e:1", realization="r:1", residual="UNRESOLVED_EVIDENCE_GAP", materialized=True, operational=True
    )
    second = _epistemic_record(
        "e:2", realization="r:2", residual="UNRESOLVED_EVIDENCE_GAP", materialized=True, operational=True
    )
    base = base.model_copy(update={"missing_evidence_requirements": [req1]})
    second = second.model_copy(update={"missing_evidence_requirements": [req2]})
    report = _decomposition([base, second])
    plan = build_evidence_recovery_plan(report, domain_profile_id="sers", results_per_query=20, target_total=4)
    literature = [row for row in plan.requests if row.route == "LITERATURE_ACQUISITION"]
    assert len(literature) == 1
    assert set(literature[0].source_realization_ids) == {"r:1", "r:2"}
    assert len(literature[0].query_strings) == 2
    assert plan.external_literature_direct_premise_injection is False
    profile = plan.targeted_acquisition_profile
    assert profile is not None
    assert profile["schema_version"] == "corpus-acquisition-profile-v1"
    assert profile["selection"]["target_total"] == len(profile["axes"])
    assert profile["selection"]["target_total"] <= 4


def test_recovery_target_budget_caps_axes_before_literature_discovery():
    rows = []
    for index in range(12):
        requirement = MissingEvidenceRequirement(
            requirement_id=f"req:{index}",
            idea_id=f"idea:{index}",
            generation_index=2,
            source_realization_ids=[f"r:{index + 1}"],
            kind=(
                "MEASUREMENT_OR_OBSERVABLE_SUPPORT"
                if index % 3 == 0
                else "RELATION_OR_MECHANISM_EVIDENCE"
            ),
            description="targeted evidence gap",
            search_terms=[f"material{index}", f"mediator{index}", f"observable{index}"],
            proposed_queries=[f"material{index} mediator{index} observable{index} spectroscopy{index}"],
            recovery_routes=["LITERATURE_ACQUISITION", "CONTEXT_REBUILD"],
            reason_codes=["RESIDUAL_UNRESOLVED_EVIDENCE_GAP"],
        )
        row = _epistemic_record(
            f"e:{index}",
            realization=f"r:{index + 1}",
            residual="UNRESOLVED_EVIDENCE_GAP",
            materialized=True,
            operational=True,
        ).model_copy(
            update={
                "idea_id": f"idea:{index}",
                "missing_evidence_requirements": [requirement],
            }
        )
        rows.append(row)

    plan = build_evidence_recovery_plan(
        _decomposition(rows),
        domain_profile_id="sers",
        results_per_query=20,
        target_total=4,
    )
    profile = plan.targeted_acquisition_profile
    assert profile is not None
    assert plan.literature_request_count == 12
    assert plan.literature_cluster_count == 12
    assert plan.targeted_literature_axis_count == 4
    assert plan.recovery_target_budget == 4
    assert plan.literature_requests_collapsed_by_clustering_count == 0
    assert plan.literature_clusters_suppressed_by_budget_count == 8
    assert len(profile["axes"]) == plan.targeted_literature_axis_count
    assert profile["selection"]["target_total"] == len(profile["axes"])
    assert profile["selection"]["target_total"] <= 4


def test_similar_literature_gaps_cluster_before_budget_selection():
    rows = []
    for index, query in enumerate(
        [
            "plasmonic hotspot orientation Raman mechanism",
            "plasmonic hotspot orientation Raman coupling mechanism",
            "carrier density semiconductor plasmon frequency tuning",
        ]
    ):
        requirement = MissingEvidenceRequirement(
            requirement_id=f"req:{index}",
            idea_id=f"idea:{index}",
            generation_index=2,
            source_realization_ids=[f"r:{index + 1}"],
            kind="RELATION_OR_MECHANISM_EVIDENCE",
            description="mechanism evidence gap",
            search_terms=query.split(),
            proposed_queries=[query],
            recovery_routes=["LITERATURE_ACQUISITION", "CONTEXT_REBUILD"],
            reason_codes=["RESIDUAL_UNRESOLVED_EVIDENCE_GAP"],
        )
        rows.append(
            _epistemic_record(
                f"e:{index}",
                realization=f"r:{index + 1}",
                residual="UNRESOLVED_EVIDENCE_GAP",
                materialized=True,
                operational=True,
            ).model_copy(
                update={
                    "idea_id": f"idea:{index}",
                    "missing_evidence_requirements": [requirement],
                }
            )
        )

    plan = build_evidence_recovery_plan(
        _decomposition(rows),
        domain_profile_id="sers",
        results_per_query=20,
        target_total=8,
    )
    profile = plan.targeted_acquisition_profile
    assert profile is not None
    assert plan.literature_request_count == 3
    assert plan.literature_cluster_count == 2
    assert plan.targeted_literature_axis_count == 2
    assert len(profile["axes"]) == 2
    assert all(
        any(" " in indicator for indicator in axis["indicators"])
        for axis in profile["axes"]
    )



def test_recovery_relevance_gate_separates_strict_adjacent_and_off_domain():
    profile = {
        "profile_id": "p",
        "domain_profile_id": "sers_au_ag",
        "selection": {"target_total": 8, "include_manual_review": True},
        "axes": [
            {
                "axis_id": "a1",
                "queries": ["plasmonic nanostructure electromagnetic hotspot SERS"],
                "indicators": ["plasmonic nanostructure", "electromagnetic hotspot"],
            },
            {
                "axis_id": "a2",
                "queries": ["plasmonic grating surface plasmon propagation length"],
                "indicators": ["plasmonic grating", "propagation length"],
            },
            {
                "axis_id": "a3",
                "queries": ["excitation polarization molecular orientation imaging"],
                "indicators": ["excitation polarization", "molecular orientation"],
            },
        ],
    }
    catalog = {
        "catalog_id": "catalog:1",
        "works": [
            {
                "work_id": "w1",
                "title": "Structure dependent SERS activity of plasmonic nanostructures with hotspots",
                "abstract": "Electromagnetic hotspots in plasmonic nanostructures control SERS response.",
                "citation_count": 50,
            },
            {
                "work_id": "w2",
                "title": "Propagation length of surface plasmon polaritons excited by a plasmonic grating",
                "abstract": "Propagation length is measured for a plasmonic grating.",
                "citation_count": 30,
            },
            {
                "work_id": "w3",
                "title": "Influence of excitation polarization on single molecule 3D orientation imaging",
                "abstract": "Excitation polarization changes orientation imaging accuracy.",
                "citation_count": 10,
            },
            {
                "work_id": "w4",
                "title": "Spatial distribution and hotspots of mammals in Canada",
                "abstract": "Spatial distribution hotspots are mapped across mammal species.",
                "citation_count": 5,
            },
        ],
    }
    assessments = [
        {"work_id": "w1", "eligibility_status": "eligible", "matched_axes": ["a1"], "total_score": 3.0},
        {"work_id": "w2", "eligibility_status": "eligible", "matched_axes": ["a2"], "total_score": 2.5},
        {"work_id": "w3", "eligibility_status": "eligible", "matched_axes": ["a3"], "total_score": 2.0},
        {"work_id": "w4", "eligibility_status": "eligible", "matched_axes": ["a1"], "total_score": 2.0},
    ]
    report = build_recovery_relevance_gate(
        acquisition_profile=profile,
        catalog=catalog,
        assessments=assessments,
        max_strict_total=8,
    )
    by_id = {row.work_id: row for row in report.records}
    assert by_id["w1"].relevance_class == "STRICT_DOMAIN_RELEVANT"
    assert by_id["w2"].relevance_class == "STRICT_DOMAIN_RELEVANT"
    assert by_id["w3"].relevance_class == "ADJACENT_METHOD_OR_MECHANISM"
    assert by_id["w4"].relevance_class == "OFF_DOMAIN_REJECT"
    assert report.strict_selected_count == 2
    assert report.max_strict_acquisition_total == 8
    assert report.strict_selected_work_ids == ["w1", "w2"]


def test_recovery_relevance_gate_budget_is_maximum_not_fill_target():
    profile = {
        "profile_id": "p",
        "domain_profile_id": "sers_au_ag",
        "selection": {"target_total": 5, "include_manual_review": False},
        "axes": [
            {
                "axis_id": "a1",
                "queries": ["plasmonic hotspot SERS nanostructure"],
                "indicators": ["plasmonic hotspot"],
            },
            {
                "axis_id": "a2",
                "queries": ["plasmonic coupling nanostructure Raman"],
                "indicators": ["plasmonic coupling"],
            },
        ],
    }
    catalog = {
        "catalog_id": "catalog:2",
        "works": [
            {"work_id": "w1", "title": "Plasmonic hotspot SERS", "abstract": "plasmonic hotspot", "citation_count": 2},
            {"work_id": "w2", "title": "Mammal hotspot distribution", "abstract": "spatial hotspot distribution", "citation_count": 100},
        ],
    }
    assessments = [
        {"work_id": "w1", "eligibility_status": "eligible", "matched_axes": ["a1"], "total_score": 2.0},
        {"work_id": "w2", "eligibility_status": "eligible", "matched_axes": ["a1"], "total_score": 5.0},
    ]
    report = build_recovery_relevance_gate(
        acquisition_profile=profile,
        catalog=catalog,
        assessments=assessments,
        max_strict_total=5,
    )
    assert report.strict_candidate_count == 1
    assert report.strict_selected_count == 1
    assert report.strict_selected_work_ids == ["w1"]
    assert report.strict_selection_is_max_budget_not_fill_target is True



def test_recovery_relevance_gate_is_conditioned_on_specific_recovery_request():
    profile = {
        "profile_id": "p_gap_conditioned",
        "domain_profile_id": "sers_au_ag",
        "selection": {"target_total": 8, "include_manual_review": True},
        "axes": [
            {
                "axis_id": "a1",
                "queries": ["plasmonic nanostructure geometry controls electromagnetic hotspot intensity SERS"],
                "indicators": ["plasmonic nanostructure", "electromagnetic hotspot"],
            },
            {
                "axis_id": "a2",
                "queries": ["plasmonic nanostructure excitation wavelength hotspot enhancement SERS"],
                "indicators": ["plasmonic nanostructure", "excitation wavelength"],
            },
            {
                "axis_id": "a3",
                "queries": ["plasmonic nanostructure geometry hotspot intensity SERS"],
                "indicators": ["plasmonic nanostructure", "hotspot intensity"],
            },
        ],
    }
    requests = [
        EvidenceRecoveryRequest(
            request_id="r1",
            requirement_id="req1",
            requirement_kind="RELATION_OR_MECHANISM_EVIDENCE",
            idea_id="i1",
            generation_index=2,
            route="LITERATURE_ACQUISITION",
            query_strings=[profile["axes"][0]["queries"][0]],
            search_terms=["geometry", "electromagnetic", "hotspot", "intensity", "control"],
            status="DISCOVERY_READY",
            positive_evidence_promotion_required_before_retry=True,
        ),
        EvidenceRecoveryRequest(
            request_id="r2",
            requirement_id="req2",
            requirement_kind="MEASUREMENT_OR_OBSERVABLE_SUPPORT",
            idea_id="i2",
            generation_index=2,
            route="LITERATURE_ACQUISITION",
            query_strings=[profile["axes"][1]["queries"][0]],
            search_terms=["excitation", "wavelength", "hotspot", "enhancement"],
            status="DISCOVERY_READY",
            positive_evidence_promotion_required_before_retry=True,
        ),
        EvidenceRecoveryRequest(
            request_id="r3",
            requirement_id="req3",
            requirement_kind="RELATION_OR_MECHANISM_EVIDENCE",
            idea_id="i3",
            generation_index=2,
            route="LITERATURE_ACQUISITION",
            query_strings=[profile["axes"][2]["queries"][0]],
            search_terms=["geometry", "hotspot", "intensity", "control"],
            status="DISCOVERY_READY",
            positive_evidence_promotion_required_before_retry=True,
        ),
    ]
    recovery = EvidenceRecoveryPlan(
        plan_id="plan:1",
        plan_sha256="sha",
        requests=requests,
        request_count=3,
        route_counts={"LITERATURE_ACQUISITION": 3},
        literature_request_count=3,
        literature_cluster_count=3,
        targeted_literature_axis_count=3,
        targeted_literature_representative_request_ids=["r1", "r2", "r3"],
        recovery_target_budget=8,
        kg_retraversal_request_count=0,
        operationalization_review_request_count=0,
        representation_review_request_count=0,
        targeted_acquisition_profile=profile,
    )
    catalog = {
        "catalog_id": "catalog:gap",
        "works": [
            {
                "work_id": "strict",
                "title": "Structure-dependent SERS activity of plasmonic nanostructures with electromagnetic hotspots",
                "abstract": "Nanostructure geometry controls electromagnetic hotspot intensity and SERS response.",
                "citation_count": 30,
            },
            {
                "work_id": "adjacent_luminescence",
                "title": "Excitation wavelength dependent luminescence quantum yields of plasmonic nanostructures",
                "abstract": "Quantum yield changes with excitation wavelength.",
                "citation_count": 20,
            },
            {
                "work_id": "adjacent_sensor",
                "title": "Au graphene hybrid plasmonic nanostructure sensor based on intensity shift",
                "abstract": "A plasmonic sensor reports an optical intensity shift.",
                "citation_count": 10,
            },
            {
                "work_id": "off",
                "title": "Spatial distribution and hotspots of mammals in Canada",
                "abstract": "Mammal hotspots are mapped geographically.",
                "citation_count": 5,
            },
        ],
    }
    assessments = [
        {"work_id": "strict", "eligibility_status": "eligible", "matched_axes": ["a1"], "total_score": 3.0},
        {"work_id": "adjacent_luminescence", "eligibility_status": "eligible", "matched_axes": ["a2"], "total_score": 3.0},
        {"work_id": "adjacent_sensor", "eligibility_status": "eligible", "matched_axes": ["a3"], "total_score": 3.0},
        {"work_id": "off", "eligibility_status": "eligible", "matched_axes": ["a3"], "total_score": 3.0},
    ]
    report = build_recovery_relevance_gate(
        acquisition_profile=profile,
        catalog=catalog,
        assessments=assessments,
        max_strict_total=8,
        recovery_plan=recovery,
    )
    by_id = {row.work_id: row for row in report.records}
    assert report.request_conditioning_applied is True
    assert by_id["strict"].relevance_class == "STRICT_DOMAIN_RELEVANT"
    assert by_id["strict"].request_conditioned_strict_match is True
    assert by_id["adjacent_luminescence"].relevance_class == "ADJACENT_METHOD_OR_MECHANISM"
    assert by_id["adjacent_sensor"].relevance_class == "ADJACENT_METHOD_OR_MECHANISM"
    assert by_id["off"].relevance_class == "OFF_DOMAIN_REJECT"
    assert report.strict_selected_work_ids == ["strict"]


def test_recovery_relevance_calibration_keeps_direct_sers_hotspot_papers_without_role_gate():
    profile = {
        "profile_id": "p_gap_calibrated",
        "domain_profile_id": "sers_au_ag",
        "selection": {"target_total": 8, "include_manual_review": True},
        "axes": [
            {
                "axis_id": "a1",
                "queries": ["plasmonic nanostructure geometry electromagnetic hotspot intensity SERS"],
                "indicators": ["plasmonic nanostructure", "electromagnetic hotspot"],
            },
            {
                "axis_id": "a2",
                "queries": ["plasmonic nanostructure excitation wavelength hotspot enhancement SERS"],
                "indicators": ["plasmonic nanostructure", "excitation wavelength"],
            },
            {
                "axis_id": "a3",
                "queries": ["plasmonic nanostructure geometry hotspot intensity SERS"],
                "indicators": ["plasmonic nanostructure", "hotspot intensity"],
            },
        ],
    }
    requests = [
        EvidenceRecoveryRequest(
            request_id="r1",
            requirement_id="req1",
            requirement_kind="RELATION_OR_MECHANISM_EVIDENCE",
            idea_id="i1",
            generation_index=2,
            route="LITERATURE_ACQUISITION",
            query_strings=[profile["axes"][0]["queries"][0]],
            search_terms=["geometry", "hotspot", "intensity", "distribution", "structure"],
            status="DISCOVERY_READY",
            positive_evidence_promotion_required_before_retry=True,
        ),
        EvidenceRecoveryRequest(
            request_id="r2",
            requirement_id="req2",
            requirement_kind="MEASUREMENT_OR_OBSERVABLE_SUPPORT",
            idea_id="i2",
            generation_index=2,
            route="LITERATURE_ACQUISITION",
            query_strings=[profile["axes"][1]["queries"][0]],
            search_terms=["excitation", "wavelength", "hotspot", "enhancement"],
            status="DISCOVERY_READY",
            positive_evidence_promotion_required_before_retry=True,
        ),
        EvidenceRecoveryRequest(
            request_id="r3",
            requirement_id="req3",
            requirement_kind="RELATION_OR_MECHANISM_EVIDENCE",
            idea_id="i3",
            generation_index=2,
            route="LITERATURE_ACQUISITION",
            query_strings=[profile["axes"][2]["queries"][0]],
            search_terms=["geometry", "hotspot", "intensity", "control"],
            status="DISCOVERY_READY",
            positive_evidence_promotion_required_before_retry=True,
        ),
    ]
    recovery = EvidenceRecoveryPlan(
        plan_id="plan:calibrated",
        plan_sha256="sha",
        requests=requests,
        request_count=3,
        route_counts={"LITERATURE_ACQUISITION": 3},
        literature_request_count=3,
        literature_cluster_count=3,
        targeted_literature_axis_count=3,
        targeted_literature_representative_request_ids=["r1", "r2", "r3"],
        recovery_target_budget=8,
        kg_retraversal_request_count=0,
        operationalization_review_request_count=0,
        representation_review_request_count=0,
        targeted_acquisition_profile=profile,
    )
    catalog = {
        "catalog_id": "catalog:calibrated",
        "works": [
            {
                "work_id": "direct_detection",
                "title": "Plasmonic hollow fibers with distributed inner-wall hotspots for direct SERS detection",
                "abstract": "Distributed hotspots enable sensitive SERS detection in a plasmonic structure.",
                "citation_count": 30,
            },
            {
                "work_id": "direct_structure",
                "title": "Structure-dependent SERS activity of plasmonic nanorattles with built-in electromagnetic hotspots",
                "abstract": "Structure changes hotspot intensity and SERS activity.",
                "citation_count": 40,
            },
            {
                "work_id": "direct_no_role",
                "title": "Plasmonic nanostructure geometry and hotspot distribution in SERS",
                "abstract": "Geometry and hotspot distribution are resolved across the nanostructure.",
                "citation_count": 12,
            },
            {
                "work_id": "luminescence",
                "title": "Excitation wavelength dependent luminescence quantum yields of plasmonic nanostructures",
                "abstract": "Luminescence yield varies with excitation wavelength.",
                "citation_count": 20,
            },
            {
                "work_id": "sensor",
                "title": "Au graphene hybrid plasmonic nanostructure sensor based on intensity shift",
                "abstract": "A plasmonic sensor reports optical intensity shift.",
                "citation_count": 10,
            },
            {
                "work_id": "off",
                "title": "Spatial distribution and hotspots of mammals in Canada",
                "abstract": "Mammal hotspots are mapped geographically.",
                "citation_count": 5,
            },
        ],
    }
    assessments = [
        {"work_id": "direct_detection", "eligibility_status": "eligible", "matched_axes": ["a1"], "total_score": 3.0},
        {"work_id": "direct_structure", "eligibility_status": "eligible", "matched_axes": ["a1"], "total_score": 3.0},
        {"work_id": "direct_no_role", "eligibility_status": "eligible", "matched_axes": ["a1"], "total_score": 2.8},
        {"work_id": "luminescence", "eligibility_status": "eligible", "matched_axes": ["a2"], "total_score": 3.0},
        {"work_id": "sensor", "eligibility_status": "eligible", "matched_axes": ["a3"], "total_score": 3.0},
        {"work_id": "off", "eligibility_status": "eligible", "matched_axes": ["a3"], "total_score": 3.0},
    ]
    report = build_recovery_relevance_gate(
        acquisition_profile=profile,
        catalog=catalog,
        assessments=assessments,
        max_strict_total=8,
        recovery_plan=recovery,
    )
    by_id = {row.work_id: row for row in report.records}
    assert by_id["direct_detection"].relevance_class == "STRICT_DOMAIN_RELEVANT"
    assert by_id["direct_structure"].relevance_class == "STRICT_DOMAIN_RELEVANT"
    assert by_id["direct_no_role"].relevance_class == "STRICT_DOMAIN_RELEVANT"
    assert by_id["direct_no_role"].scientific_role_hits == []
    assert by_id["luminescence"].relevance_class == "ADJACENT_METHOD_OR_MECHANISM"
    assert by_id["sensor"].relevance_class == "ADJACENT_METHOD_OR_MECHANISM"
    assert by_id["off"].relevance_class == "OFF_DOMAIN_REJECT"
    assert report.scientific_role_match_is_confidence_booster_not_strict_gate is True
    assert report.request_conditioned_strict_requires_role_match is False
