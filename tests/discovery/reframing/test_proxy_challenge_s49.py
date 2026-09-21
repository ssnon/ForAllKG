from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipeline_core.corpus.semantic_ir.annotation import SemanticAnnotation
from pipeline_core.corpus.semantic_ir.schema import SemanticObjectRef
from pipeline_core.discovery.reframing.proxy_challenge import (
    ProxyChallengeBatchDraft,
    ProxyChallengeDraft,
    ProxyChallengeGeneration,
    ProxyChallengeShadowRuntime,
    build_proxy_challenge_input,
    compile_proxy_challenge_candidate,
)
from pipeline_core.discovery.reframing.proxy_enrichment import (
    EXTRACTOR_VERSION,
    PROMPT_VERSION,
    ProxySemanticEnrichmentReport,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    DifferentialPredictionDraft,
    DiscriminatingTestDraft,
    ReframeFalsifierDraft,
    ScientificModelDraft,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    ReframeEvidenceStatement,
    ScientificReframeEvidencePacket,
)


def _evidence() -> ScientificReframeEvidencePacket:
    return ScientificReframeEvidencePacket(
        task_id="task:1",
        question=(
            "How does dielectric environment alter the mismatch between far-field "
            "LSPR and SERS-optimal excitation?"
        ),
        source_context_id="context:1",
        source_context_sha256="a" * 64,
        premise_statements=[
            ReframeEvidenceStatement(
                statement_id="p1",
                text="Far-field LSPR matching is associated with stronger SERS in one context.",
                epistemic_role="reported",
                claim_kind="association",
                paper_ids=["P1"],
            ),
            ReframeEvidenceStatement(
                statement_id="p2",
                text=(
                    "Maximum SERS enhancement can occur at an excitation wavelength "
                    "offset from the measured plasmon resonance."
                ),
                epistemic_role="reported",
                claim_kind="observation",
                paper_ids=["P2"],
            ),
        ],
        gap_statements=[
            ReframeEvidenceStatement(
                statement_id="g1",
                text="The packet does not establish whether far-field LSPR is sufficient to predict the local useful field.",
                epistemic_role="unresolved",
                claim_kind="scope_limit",
                paper_ids=["P1", "P2"],
            )
        ],
        grounded_source_chunk_count=3,
        unresolved_grounded_node_count=0,
    )


def _annotation(
    *,
    annotation_id: str,
    measurement_id: str,
    metric: str,
    target: str,
    summary: str,
    paper_id: str = "P3",
) -> SemanticAnnotation:
    chunk_id = f"{paper_id}:main:c1"
    return SemanticAnnotation(
        annotation_id=annotation_id,
        capability="proxy_semantics",
        subject_ref=SemanticObjectRef(
            paper_id=paper_id,
            chunk_id=chunk_id,
            object_kind="measurement",
            object_id=measurement_id,
            source_path=f"chunks/{paper_id}.json",
        ),
        value={
            "schema_version": "proxy-semantic-annotation-value-v1.1",
            "measurement_id": measurement_id,
            "metric": metric,
            "subject_id": "subject:1",
            "source_expression": "The LSPR band appeared at 800 nm.",
            "proxy_role": "surrogate_for_construct",
            "target_construct": target,
            "interchangeability": "context_limited",
            "interpretation_summary": summary,
            "distinct_target_rationale": "The observable and construct are distinct.",
            "support_kind": "explicit_inference_to_distinct_construct",
            "support_quotes": ["Band positions indicated coupling."],
            "limitations": [],
            "context_dependencies": [],
            "evidence_basis": "explicit_source_statement",
            "hypothesis_generation_authority": False,
            "negative_evidence_authority": False,
            "scientific_equivalence_authority": False,
        },
        source_refs=[
            SemanticObjectRef(
                paper_id=paper_id,
                chunk_id=chunk_id,
                object_kind="chunk",
                object_id=chunk_id,
                source_path=f"source_chunks/{paper_id}.json",
            )
        ],
        extractor_version=EXTRACTOR_VERSION,
        epistemic_status="source_interpretation",
    )


def _report(annotation_count: int = 1) -> ProxySemanticEnrichmentReport:
    return ProxySemanticEnrichmentReport(
        report_id="report:1",
        scope_id="task:1",
        source_plan_id="plan:1",
        annotation_path="proxy.jsonl",
        review_path="reviews.jsonl",
        plan_target_count=3,
        considered_target_count=3,
        completed_target_count=3,
        resumed_target_count=0,
        new_target_count=3,
        pending_target_count=0,
        failed_target_count=0,
        new_annotation_count=annotation_count,
        total_annotation_count=annotation_count,
        new_rejected_candidate_count=0,
        total_rejected_candidate_count=0,
        llm_calls_performed=3,
        errors=[],
        coverage_complete_for_plan=True,
        execution_performed=True,
    )


def _annotation_file(tmp_path: Path, rows: list[SemanticAnnotation]) -> Path:
    path = tmp_path / "proxy.jsonl"
    path.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _draft(seed_id: str = "ann:lspr") -> ProxyChallengeDraft:
    return ProxyChallengeDraft(
        local_id="proxy1",
        title="Far-field resonance may be an incomplete proxy for the useful local field",
        semantic_seed_annotation_ids=[seed_id],
        premise_statement_ids=["p1", "p2"],
        gap_statement_ids=["g1"],
        baseline_model=ScientificModelDraft(
            summary="Far-field LSPR position is a sufficient indicator of the condition maximizing SERS.",
            assumptions=["Far-field and locally useful resonance track together."],
            explained_statement_ids=["p1"],
            expected_observations=["SERS optimum should track the far-field LSPR."],
        ),
        alternative_model=ScientificModelDraft(
            summary="Far-field LSPR captures only part of the local optical response sampled by the adsorbate.",
            assumptions=["Local-field and extinction spectra may diverge."],
            explained_statement_ids=["p1", "p2"],
            expected_observations=["The SERS optimum may remain offset from far-field LSPR."],
        ),
        challenged_proxy_assumption="Far-field LSPR is an interchangeable proxy for the spectrally useful local field.",
        challenged_observable="far-field LSPR wavelength",
        target_construct="spectrally useful local near field experienced by the adsorbate",
        proxy_failure_mode="construct_undercoverage",
        differential_predictions=[
            DifferentialPredictionDraft(
                local_id="dp1",
                observable="relative shift of far-field LSPR and SERS-optimal excitation",
                baseline_expectation="the two shifts track closely",
                alternative_expectation="the shifts can diverge systematically",
                discriminating_outcome="independent divergence favors the proxy-challenge model",
            )
        ],
        falsifiers=[
            ReframeFalsifierDraft(
                local_id="f1",
                falsifying_outcome="SERS optimum remains fully predicted by far-field LSPR under controlled perturbations.",
            )
        ],
        discriminating_test=DiscriminatingTestDraft(
            test_design="Measure far-field extinction and local-field-sensitive response under matched dielectric perturbations.",
            primary_observables=["far-field LSPR", "SERS excitation maximum"],
            baseline_favoring_outcome="the observables shift together",
            alternative_favoring_outcome="the observables reproducibly diverge",
        ),
    )


def _input(tmp_path: Path, annotations: list[SemanticAnnotation]):
    path = _annotation_file(tmp_path, annotations)
    return build_proxy_challenge_input(
        evidence=_evidence(),
        enrichment_report=_report(len(annotations)),
        annotations=annotations,
        annotation_path=path,
    )


def test_input_requires_complete_enrichment(tmp_path: Path):
    ann = _annotation(
        annotation_id="ann:lspr",
        measurement_id="m1",
        metric="LSPR wavelength",
        target="metal coupling",
        summary="LSPR band position is used to infer coupling.",
    )
    report = _report(1).model_copy(update={"coverage_complete_for_plan": False})
    with pytest.raises(ValueError, match="complete targeted enrichment coverage"):
        build_proxy_challenge_input(
            evidence=_evidence(),
            enrichment_report=report,
            annotations=[ann],
            annotation_path=_annotation_file(tmp_path, [ann]),
        )


def test_lspr_semantic_seed_is_task_relevant(tmp_path: Path):
    proxy_input = _input(
        tmp_path,
        [
            _annotation(
                annotation_id="ann:lspr",
                measurement_id="m1",
                metric="LSPR wavelength",
                target="metal coupling",
                summary="LSPR band positions are used to infer coupling.",
            )
        ],
    )
    assert proxy_input.task_relevant_seed_ids == ["ann:lspr"]
    assert "lspr" in proxy_input.seeds[0].matched_task_terms


def test_biocompatibility_seed_is_not_task_relevant(tmp_path: Path):
    proxy_input = _input(
        tmp_path,
        [
            _annotation(
                annotation_id="ann:bio",
                measurement_id="m2",
                metric="cell viability",
                target="in vitro biocompatibility",
                summary="Cell viability is used to indicate biocompatibility.",
            )
        ],
    )
    assert proxy_input.task_relevant_seed_ids == []


def test_annotation_sidecar_does_not_gain_premise_authority(tmp_path: Path):
    proxy_input = _input(
        tmp_path,
        [
            _annotation(
                annotation_id="ann:lspr",
                measurement_id="m1",
                metric="LSPR wavelength",
                target="metal coupling",
                summary="LSPR band positions are used to infer coupling.",
            )
        ],
    )
    assert proxy_input.annotation_is_positive_premise is False
    assert proxy_input.annotation_is_scientific_equivalence_authority is False


def test_compile_rejects_unknown_or_irrelevant_seed(tmp_path: Path):
    proxy_input = _input(
        tmp_path,
        [
            _annotation(
                annotation_id="ann:lspr",
                measurement_id="m1",
                metric="LSPR wavelength",
                target="metal coupling",
                summary="LSPR band positions are used to infer coupling.",
            )
        ],
    )
    with pytest.raises(ValueError, match="unknown or task-irrelevant"):
        compile_proxy_challenge_candidate(
            evidence=_evidence(),
            proxy_input=proxy_input,
            draft=_draft("ann:missing"),
        )


def test_compile_requires_two_grounded_premises(tmp_path: Path):
    proxy_input = _input(
        tmp_path,
        [
            _annotation(
                annotation_id="ann:lspr",
                measurement_id="m1",
                metric="LSPR wavelength",
                target="metal coupling",
                summary="LSPR band positions are used to infer coupling.",
            )
        ],
    )
    draft = _draft().model_copy(
        update={
            "premise_statement_ids": ["p1"],
            "alternative_model": _draft().alternative_model.model_copy(
                update={"explained_statement_ids": ["p1"]}
            ),
        }
    )
    with pytest.raises(ValueError, match="at least two grounded premises"):
        compile_proxy_challenge_candidate(
            evidence=_evidence(), proxy_input=proxy_input, draft=draft
        )


def test_compile_valid_candidate_is_hypothesis_only(tmp_path: Path):
    proxy_input = _input(
        tmp_path,
        [
            _annotation(
                annotation_id="ann:lspr",
                measurement_id="m1",
                metric="LSPR wavelength",
                target="metal coupling",
                summary="LSPR band positions are used to infer coupling.",
            )
        ],
    )
    candidate = compile_proxy_challenge_candidate(
        evidence=_evidence(), proxy_input=proxy_input, draft=_draft()
    )
    assert candidate.operator_id == "PROXY_CHALLENGE"
    assert candidate.epistemic_status == "hypothesis_only"
    assert candidate.proxy_annotation_is_positive_premise is False
    assert candidate.production_selection_authority is False


def test_question_anchored_relevance_rejects_generic_overlap_from_premises(tmp_path: Path):
    ann = _annotation(
        annotation_id="ann:hemin",
        measurement_id="m-hemin",
        metric="Raman intensity",
        target="molecular orientation",
        summary=(
            "The signal is associated with a specific orientation because the "
            "measured band changes on the substrate."
        ),
        paper_id="P-hemin",
    )
    proxy_input = _input(tmp_path, [ann])
    assert proxy_input.task_relevant_seed_ids == []
    assert proxy_input.seeds[0].matched_task_terms == []


def test_compile_repairs_unique_one_edit_opaque_seed_id_copy_error(tmp_path: Path):
    canonical_id = "semantic_annotation:73a5b9bf101dbc2681f5"
    typo_id = "semantic_annotation:73a5b9bf101dbc2685f5"
    proxy_input = _input(
        tmp_path,
        [
            _annotation(
                annotation_id=canonical_id,
                measurement_id="m1",
                metric="LSPR wavelength",
                target="metal coupling",
                summary="LSPR band positions are used to infer coupling.",
            )
        ],
    )
    candidate = compile_proxy_challenge_candidate(
        evidence=_evidence(),
        proxy_input=proxy_input,
        draft=_draft(typo_id),
    )
    assert candidate.semantic_seed_annotation_ids == [canonical_id]
    assert any(
        "one-edit semantic seed ID copy error" in row
        for row in candidate.provenance_normalizations
    )


def test_compile_does_not_repair_exact_task_irrelevant_seed(tmp_path: Path):
    relevant = _annotation(
        annotation_id="semantic_annotation:aaaaaaaaaaaaaaaaaaaa",
        measurement_id="m1",
        metric="LSPR wavelength",
        target="metal coupling",
        summary="LSPR band positions are used to infer coupling.",
    )
    irrelevant = _annotation(
        annotation_id="semantic_annotation:aaaaaaaaaaaaaaaaaaab",
        measurement_id="m2",
        metric="cell viability",
        target="in vitro biocompatibility",
        summary="Cell viability is used to indicate biocompatibility.",
    )
    proxy_input = _input(tmp_path, [relevant, irrelevant])
    assert proxy_input.task_relevant_seed_ids == [relevant.annotation_id]
    with pytest.raises(ValueError, match="task-irrelevant semantic seed ID"):
        compile_proxy_challenge_candidate(
            evidence=_evidence(),
            proxy_input=proxy_input,
            draft=_draft(irrelevant.annotation_id),
        )


class _Backend:
    backend_name = "fake"
    model_name = "fake-model"

    def __init__(self, draft: ProxyChallengeBatchDraft):
        self.draft = draft
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return ProxyChallengeGeneration(draft=self.draft)


def test_runtime_skips_without_relevant_seed_and_calls_zero_llm(tmp_path: Path):
    proxy_input = _input(
        tmp_path,
        [
            _annotation(
                annotation_id="ann:bio",
                measurement_id="m2",
                metric="cell viability",
                target="in vitro biocompatibility",
                summary="Cell viability is used to indicate biocompatibility.",
            )
        ],
    )
    backend = _Backend(ProxyChallengeBatchDraft(candidates=[]))
    report, prompt = ProxyChallengeShadowRuntime(backend).run(
        evidence=_evidence(), proxy_input=proxy_input
    )
    assert backend.calls == 0
    assert prompt is None
    assert report.decision == "skipped_no_task_relevant_seed"
    assert report.llm_calls_performed == 0


def test_runtime_generates_one_shadow_candidate(tmp_path: Path):
    proxy_input = _input(
        tmp_path,
        [
            _annotation(
                annotation_id="ann:lspr",
                measurement_id="m1",
                metric="LSPR wavelength",
                target="metal coupling",
                summary="LSPR band positions are used to infer coupling.",
            )
        ],
    )
    backend = _Backend(ProxyChallengeBatchDraft(candidates=[_draft()]))
    report, prompt = ProxyChallengeShadowRuntime(backend).run(
        evidence=_evidence(), proxy_input=proxy_input
    )
    assert backend.calls == 1
    assert prompt is not None
    assert report.decision == "generated"
    assert report.llm_calls_performed == 1
    assert len(report.candidates) == 1
    assert report.canonical_graph_mutated is False
    assert report.production_selection_changed is False


def test_annotation_file_hash_binds_candidate_input(tmp_path: Path):
    ann = _annotation(
        annotation_id="ann:lspr",
        measurement_id="m1",
        metric="LSPR wavelength",
        target="metal coupling",
        summary="LSPR band positions are used to infer coupling.",
    )
    path = _annotation_file(tmp_path, [ann])
    proxy_input = build_proxy_challenge_input(
        evidence=_evidence(),
        enrichment_report=_report(1),
        annotations=[ann],
        annotation_path=path,
    )
    assert proxy_input.source_annotation_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
