from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_core.corpus.semantic_ir.backfill import BackfillTarget, build_backfill_plan
from pipeline_core.corpus.semantic_ir.capability import CapabilityRequirement
from pipeline_core.corpus.semantic_ir.schema import (
    SemanticIRBundle,
    SemanticIRRecord,
    SemanticObjectRef,
)
from pipeline_core.discovery.reframing.proxy_enrichment import (
    ProxyDraftGeneration,
    ProxySemanticChunkDraft,
    ProxySemanticRelationDraft,
    build_proxy_semantic_prompt,
    compile_proxy_semantic_annotations,
    execute_proxy_semantic_enrichment,
    load_source_chunk_payload,
    measurement_records_for_chunk,
)


def _source_ref(paper: str = "p1", chunk: str = "c1") -> SemanticObjectRef:
    return SemanticObjectRef(
        paper_id=paper,
        chunk_id=chunk,
        object_kind="chunk",
        object_id=chunk,
        source_path=f"source_chunks/{chunk}.json",
    )


def _measurement(
    source: SemanticObjectRef,
    measurement_id: str,
    metric: str = "LSPR peak",
) -> SemanticIRRecord:
    return SemanticIRRecord(
        ref=SemanticObjectRef(
            paper_id=source.paper_id,
            chunk_id=source.chunk_id,
            object_kind="measurement",
            object_id=measurement_id,
            source_path=f"chunks/{source.chunk_id}.json",
        ),
        payload={
            "id": measurement_id,
            "metric_id": f"metric:{measurement_id}",
            "metric": metric,
            "subject_id": "sample:1",
            "source_expression": metric,
            "value": 700,
            "unit": "nm",
            "conditions": {"medium": "water"},
        },
        source_chunk_ref=source,
    )


def _bundle(tmp_path: Path, *, two_measurements: bool = False) -> tuple[SemanticIRBundle, SemanticObjectRef]:
    attempt = tmp_path / "attempt"
    source_dir = attempt / "source_chunks"
    source_dir.mkdir(parents=True)
    source = _source_ref()
    payload = {
        "paper_id": "p1",
        "chunk_id": "c1",
        "section": "Results",
        "left_context": "Far-field extinction was measured first.",
        "core_text": (
            "The LSPR peak was used as an indicator of resonance matching, "
            "although local fields can diverge from the far-field maximum."
        ),
        "right_context": "Raman enhancement was measured separately.",
        "asset_context": [],
    }
    (source_dir / "c1.json").write_text(json.dumps(payload), encoding="utf-8")
    records = [_measurement(source, "m1")]
    if two_measurements:
        records.append(_measurement(source, "m2", metric="Raman intensity"))
    bundle = SemanticIRBundle(
        bundle_id="bundle:p1",
        paper_id="p1",
        source_attempt_directory=str(attempt),
        source_chunks=[source],
        records=records,
    )
    return bundle, source


def _plan(source: SemanticObjectRef):
    return build_backfill_plan(
        plan_id="plan:proxy",
        requirements=[
            CapabilityRequirement(
                capability="proxy_semantics",
                minimum_state="complete",
                reason="proxy challenge requires proxy semantics",
            )
        ],
        targets=[
            BackfillTarget(
                source_chunk_ref=source,
                missing_capabilities=["proxy_semantics"],
                reasons=["targeted measurement-bearing chunk"],
            )
        ],
    )


def _draft(measurement_id: str = "m1") -> ProxySemanticChunkDraft:
    return ProxySemanticChunkDraft(
        items=[
            ProxySemanticRelationDraft(
                measurement_id=measurement_id,
                proxy_role="proxy_limitation",
                target_construct="spectrally useful local near field",
                interchangeability="context_limited",
                interpretation_summary=(
                    "The far-field LSPR peak is used to indicate resonance matching, "
                    "but the source distinguishes it from the local field maximum."
                ),
                distinct_target_rationale=(
                    "The measured far-field LSPR peak and the local-field maximum are "
                    "distinct optical observables and the source says they can diverge."
                ),
                support_kind="explicit_non_equivalence_or_limitation",
                support_quotes=[
                    "although local fields can diverge from the far-field maximum."
                ],
                limitations=["Far-field and local-field maxima can diverge."],
                context_dependencies=["dielectric environment"],
                evidence_basis="explicit_source_statement",
            )
        ]
    )


def _source_payload(bundle: SemanticIRBundle, source: SemanticObjectRef) -> dict:
    payload, _ = load_source_chunk_payload(bundle=bundle, source_chunk_ref=source)
    return payload


class FakeBackend:
    backend_name = "fake"
    model_name = "fake-model"

    def __init__(self, drafts: list[ProxySemanticChunkDraft]):
        self.drafts = list(drafts)
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return ProxyDraftGeneration(
            draft=self.drafts.pop(0),
            input_tokens=10,
            output_tokens=5,
            response_id=f"r{self.calls}",
        )


def test_source_chunk_is_recovered_from_existing_attempt(tmp_path: Path):
    bundle, source = _bundle(tmp_path)
    payload, digest = load_source_chunk_payload(bundle=bundle, source_chunk_ref=source)
    assert payload["chunk_id"] == "c1"
    assert len(digest) == 64


def test_measurements_are_chunk_local(tmp_path: Path):
    bundle, source = _bundle(tmp_path, two_measurements=True)
    unrelated_source = SemanticObjectRef(
        paper_id="p1",
        chunk_id="c2",
        object_kind="chunk",
        object_id="c2",
        source_path="source_chunks/c2.json",
    )
    bundle.records.append(_measurement(unrelated_source, "m9"))
    rows = measurement_records_for_chunk(bundle=bundle, source_chunk_ref=source)
    assert [row.ref.object_id for row in rows] == ["m1", "m2"]


def test_prompt_allows_empty_items_and_forbids_negative_inference(tmp_path: Path):
    bundle, source = _bundle(tmp_path)
    payload, _ = load_source_chunk_payload(bundle=bundle, source_chunk_ref=source)
    rows = measurement_records_for_chunk(bundle=bundle, source_chunk_ref=source)
    prompt = build_proxy_semantic_prompt(
        source_chunk_ref=source,
        source_payload=payload,
        measurements=rows,
    )
    assert 'It is valid to return {"items": []}' in prompt.user_prompt
    assert "Do not label absence of a proxy relation" in prompt.system_prompt


def test_compiler_rejects_invented_measurement_id(tmp_path: Path):
    bundle, source = _bundle(tmp_path)
    rows = measurement_records_for_chunk(bundle=bundle, source_chunk_ref=source)
    with pytest.raises(ValueError, match="unknown measurement_id"):
        compile_proxy_semantic_annotations(
            scope_id="task:1",
            source_chunk_ref=source,
            source_payload=_source_payload(bundle, source),
            measurements=rows,
            draft=_draft("invented"),
        )


def test_compiled_annotation_has_no_evidence_or_selection_authority(tmp_path: Path):
    bundle, source = _bundle(tmp_path)
    rows = measurement_records_for_chunk(bundle=bundle, source_chunk_ref=source)
    annotations = compile_proxy_semantic_annotations(
        scope_id="task:1",
        source_chunk_ref=source,
        source_payload=_source_payload(bundle, source),
        measurements=rows,
        draft=_draft(),
    )
    assert len(annotations) == 1
    annotation = annotations[0]
    assert annotation.capability == "proxy_semantics"
    assert annotation.subject_ref.object_id == "m1"
    assert annotation.canonical_graph_mutated is False
    assert annotation.positive_premise_authority is False
    assert annotation.novelty_authority is False
    assert annotation.selection_authority is False
    assert annotation.value["scientific_equivalence_authority"] is False
    assert annotation.value["support_kind"] == "explicit_non_equivalence_or_limitation"
    assert annotation.value["support_quotes"]


def test_compiler_rejects_support_quote_not_present_in_source(tmp_path: Path):
    bundle, source = _bundle(tmp_path)
    rows = measurement_records_for_chunk(bundle=bundle, source_chunk_ref=source)
    draft = _draft().model_copy(deep=True)
    draft.items[0].support_quotes = ["This wording never appears in the source."]
    with pytest.raises(ValueError, match="not an exact source substring"):
        compile_proxy_semantic_annotations(
            scope_id="task:1",
            source_chunk_ref=source,
            source_payload=_source_payload(bundle, source),
            measurements=rows,
            draft=draft,
        )


def test_compiler_rejects_generic_measurement_reporting_as_proxy_support(tmp_path: Path):
    bundle, source = _bundle(tmp_path)
    source_path = Path(bundle.source_attempt_directory) / source.source_path
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["core_text"] = "The enhancement factor was 1.2 × 10^7 for the alloy substrate."
    source_path.write_text(json.dumps(payload), encoding="utf-8")
    rows = measurement_records_for_chunk(bundle=bundle, source_chunk_ref=source)
    draft = ProxySemanticChunkDraft(
        items=[
            ProxySemanticRelationDraft(
                measurement_id="m1",
                proxy_role="surrogate_for_construct",
                target_construct="substrate enhancement performance",
                interchangeability="context_limited",
                interpretation_summary="The EF represents substrate enhancement performance.",
                distinct_target_rationale="Performance is broader than a single reported EF.",
                support_kind="explicit_inference_to_distinct_construct",
                support_quotes=["The enhancement factor was 1.2 × 10^7 for the alloy substrate."],
                limitations=[],
                context_dependencies=[],
                evidence_basis="source_interpretation",
            )
        ]
    )
    with pytest.raises(ValueError, match="does not substantiate support_kind"):
        compile_proxy_semantic_annotations(
            scope_id="task:1",
            source_chunk_ref=source,
            source_payload=_source_payload(bundle, source),
            measurements=rows,
            draft=draft,
        )


def test_prompt_surfaces_explicit_limitation_as_attention_hint(tmp_path: Path):
    bundle, source = _bundle(tmp_path)
    source_path = Path(bundle.source_attempt_directory) / source.source_path
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["core_text"] = (
        "The apparent shell thickness was 9 nm. "
        "The absolute shell thickness is difficult to determine because alloying occurs at the interface."
    )
    source_path.write_text(json.dumps(payload), encoding="utf-8")
    rows = measurement_records_for_chunk(bundle=bundle, source_chunk_ref=source)
    prompt = build_proxy_semantic_prompt(
        source_chunk_ref=source,
        source_payload=_source_payload(bundle, source),
        measurements=rows,
    )
    assert "LIMITATION_CANDIDATE_HINTS" in prompt.user_prompt
    assert "absolute shell thickness is difficult to determine" in prompt.user_prompt
    assert "attention aids copied from the source" in prompt.system_prompt


def test_prompt_does_not_turn_generic_measurement_reporting_into_limitation_hint(tmp_path: Path):
    bundle, source = _bundle(tmp_path)
    source_path = Path(bundle.source_attempt_directory) / source.source_path
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["core_text"] = "The enhancement factor was 1.2 × 10^7 for the alloy substrate."
    source_path.write_text(json.dumps(payload), encoding="utf-8")
    rows = measurement_records_for_chunk(bundle=bundle, source_chunk_ref=source)
    prompt = build_proxy_semantic_prompt(
        source_chunk_ref=source,
        source_payload=_source_payload(bundle, source),
        measurements=rows,
    )
    marker = "LIMITATION_CANDIDATE_HINTS\n==========================\n"
    tail = prompt.user_prompt.split(marker, 1)[1]
    assert tail.lstrip().startswith("[]")


def test_compiler_accepts_explicit_source_limitation(tmp_path: Path):
    bundle, source = _bundle(tmp_path)
    source_path = Path(bundle.source_attempt_directory) / source.source_path
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["core_text"] = (
        "The apparent shell thickness was 9 nm, but the absolute shell thickness "
        "is difficult to determine because alloying occurs at the interface."
    )
    source_path.write_text(json.dumps(payload), encoding="utf-8")
    rows = measurement_records_for_chunk(bundle=bundle, source_chunk_ref=source)
    draft = ProxySemanticChunkDraft(
        items=[
            ProxySemanticRelationDraft(
                measurement_id="m1",
                proxy_role="proxy_limitation",
                target_construct="absolute shell thickness",
                interchangeability="context_limited",
                interpretation_summary="The apparent thickness need not equal absolute shell thickness.",
                distinct_target_rationale="The source distinguishes apparent measured thickness from absolute shell thickness.",
                support_kind="explicit_non_equivalence_or_limitation",
                support_quotes=[
                    "the absolute shell thickness is difficult to determine because alloying occurs at the interface."
                ],
                limitations=["Interfacial alloying limits absolute shell-thickness determination."],
                context_dependencies=["interfacial alloying"],
                evidence_basis="explicit_source_statement",
            )
        ]
    )
    annotations = compile_proxy_semantic_annotations(
        scope_id="task:1",
        source_chunk_ref=source,
        source_payload=_source_payload(bundle, source),
        measurements=rows,
        draft=draft,
    )
    assert len(annotations) == 1
    assert annotations[0].value["proxy_role"] == "proxy_limitation"


def test_empty_draft_completes_review_without_negative_annotation(tmp_path: Path):
    bundle, source = _bundle(tmp_path)
    backend = FakeBackend([ProxySemanticChunkDraft(items=[])])
    annotations = tmp_path / "annotations.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    report = execute_proxy_semantic_enrichment(
        scope_id="task:1",
        plan=_plan(source),
        bundles={"p1": bundle},
        backend=backend,
        annotation_path=annotations,
        review_path=reviews,
    )
    assert report.coverage_complete_for_plan is True
    assert report.new_annotation_count == 0
    assert not annotations.exists()
    review = json.loads(reviews.read_text(encoding="utf-8").strip())
    assert review["emitted_annotation_count"] == 0
    assert review["absence_of_annotation_is_not_negative_evidence"] is True


def test_resume_skips_completed_chunk_without_second_llm_call(tmp_path: Path):
    bundle, source = _bundle(tmp_path)
    annotations = tmp_path / "annotations.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    first = FakeBackend([_draft()])
    report1 = execute_proxy_semantic_enrichment(
        scope_id="task:1",
        plan=_plan(source),
        bundles={"p1": bundle},
        backend=first,
        annotation_path=annotations,
        review_path=reviews,
    )
    assert report1.llm_calls_performed == 1
    second = FakeBackend([])
    report2 = execute_proxy_semantic_enrichment(
        scope_id="task:1",
        plan=_plan(source),
        bundles={"p1": bundle},
        backend=second,
        annotation_path=annotations,
        review_path=reviews,
    )
    assert report2.llm_calls_performed == 0
    assert report2.resumed_target_count == 1
    assert second.calls == 0
    assert len(annotations.read_text(encoding="utf-8").splitlines()) == 1


def test_resume_fails_closed_if_reviewed_source_changes(tmp_path: Path):
    bundle, source = _bundle(tmp_path)
    annotations = tmp_path / "annotations.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    first = FakeBackend([_draft()])
    execute_proxy_semantic_enrichment(
        scope_id="task:1",
        plan=_plan(source),
        bundles={"p1": bundle},
        backend=first,
        annotation_path=annotations,
        review_path=reviews,
    )
    source_path = Path(bundle.source_attempt_directory) / source.source_path
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["core_text"] += " changed"
    source_path.write_text(json.dumps(payload), encoding="utf-8")
    second = FakeBackend([])
    report = execute_proxy_semantic_enrichment(
        scope_id="task:1",
        plan=_plan(source),
        bundles={"p1": bundle},
        backend=second,
        annotation_path=annotations,
        review_path=reviews,
    )
    assert report.failed_target_count == 1
    assert report.coverage_complete_for_plan is False
    assert report.errors[0].error_type == "ValueError"
    assert "stale" in report.errors[0].error_message
    assert second.calls == 0


def test_max_calls_limits_new_work_without_mutating_plan(tmp_path: Path):
    bundle1, source1 = _bundle(tmp_path / "a")
    bundle2, source2 = _bundle(tmp_path / "b")
    source2 = source2.model_copy(update={
        "paper_id": "p2",
        "chunk_id": "c2",
        "object_id": "c2",
        "source_path": "source_chunks/c1.json",
    })
    # Rebuild the second bundle around a distinct paper/chunk while reusing its temp file.
    payload_path = Path(bundle2.source_attempt_directory) / "source_chunks/c1.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["paper_id"] = "p2"
    payload["chunk_id"] = "c2"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    m2 = _measurement(source2, "m2")
    bundle2 = SemanticIRBundle(
        bundle_id="bundle:p2",
        paper_id="p2",
        source_attempt_directory=bundle2.source_attempt_directory,
        source_chunks=[source2],
        records=[m2],
    )
    plan = build_backfill_plan(
        plan_id="plan:two",
        requirements=[],
        targets=[
            BackfillTarget(
                source_chunk_ref=source1,
                missing_capabilities=["proxy_semantics"],
                reasons=["r"],
            ),
            BackfillTarget(
                source_chunk_ref=source2,
                missing_capabilities=["proxy_semantics"],
                reasons=["r"],
            ),
        ],
    )
    backend = FakeBackend([_draft("m1")])
    report = execute_proxy_semantic_enrichment(
        scope_id="task:two",
        plan=plan,
        bundles={"p1": bundle1, "p2": bundle2},
        backend=backend,
        annotation_path=tmp_path / "annotations.jsonl",
        review_path=tmp_path / "reviews.jsonl",
        max_new_calls=1,
    )
    assert report.llm_calls_performed == 1
    assert report.completed_target_count == 1
    assert report.pending_target_count == 1
    assert report.coverage_complete_for_plan is False
    assert plan.target_count == 2


def test_executor_rejects_mixed_capability_backfill_target(tmp_path: Path):
    bundle, source = _bundle(tmp_path)
    plan = build_backfill_plan(
        plan_id="plan:mixed",
        requirements=[],
        targets=[
            BackfillTarget(
                source_chunk_ref=source,
                missing_capabilities=["proxy_semantics", "measurement_condition_coverage"],
                reasons=["r"],
            )
        ],
    )
    with pytest.raises(ValueError, match="pure proxy_semantics"):
        execute_proxy_semantic_enrichment(
            scope_id="task:1",
            plan=plan,
            bundles={"p1": bundle},
            backend=FakeBackend([]),
            annotation_path=tmp_path / "annotations.jsonl",
            review_path=tmp_path / "reviews.jsonl",
        )


def test_executor_isolates_invalid_candidate_and_completes_chunk(tmp_path: Path):
    bundle, source = _bundle(tmp_path)
    invalid = _draft().model_copy(deep=True)
    invalid.items[0].support_quotes = ["This wording never appears in the source."]
    backend = FakeBackend([invalid])
    annotations = tmp_path / "annotations.jsonl"
    reviews = tmp_path / "reviews.jsonl"

    report = execute_proxy_semantic_enrichment(
        scope_id="task:1",
        plan=_plan(source),
        bundles={"p1": bundle},
        backend=backend,
        annotation_path=annotations,
        review_path=reviews,
    )

    assert report.failed_target_count == 0
    assert report.coverage_complete_for_plan is True
    assert report.new_annotation_count == 0
    assert report.new_rejected_candidate_count == 1
    assert report.total_rejected_candidate_count == 1
    assert not annotations.exists()
    review = json.loads(reviews.read_text(encoding="utf-8").strip())
    assert review["completed"] is True
    assert review["draft_candidate_count"] == 1
    assert review["emitted_annotation_count"] == 0
    assert review["rejected_candidate_count"] == 1
    assert review["candidate_rejection_is_not_negative_evidence"] is True
    rejection = review["rejected_candidates"][0]
    assert rejection["candidate_rejected_not_negative_evidence"] is True
    assert "not an exact source substring" in rejection["error_message"]


def test_executor_preserves_valid_sibling_when_another_candidate_is_rejected(tmp_path: Path):
    bundle, source = _bundle(tmp_path, two_measurements=True)
    valid = _draft("m1").items[0]
    invalid = _draft("m2").items[0].model_copy(deep=True)
    invalid.support_quotes = ["Invented support quote."]
    backend = FakeBackend([ProxySemanticChunkDraft(items=[valid, invalid])])
    annotations = tmp_path / "annotations.jsonl"
    reviews = tmp_path / "reviews.jsonl"

    report = execute_proxy_semantic_enrichment(
        scope_id="task:1",
        plan=_plan(source),
        bundles={"p1": bundle},
        backend=backend,
        annotation_path=annotations,
        review_path=reviews,
    )

    assert report.failed_target_count == 0
    assert report.new_annotation_count == 1
    assert report.new_rejected_candidate_count == 1
    annotation_rows = [json.loads(row) for row in annotations.read_text(encoding="utf-8").splitlines()]
    assert len(annotation_rows) == 1
    assert annotation_rows[0]["subject_ref"]["object_id"] == "m1"
    review = json.loads(reviews.read_text(encoding="utf-8").strip())
    assert review["draft_candidate_count"] == 2
    assert review["emitted_annotation_count"] == 1
    assert review["rejected_candidate_count"] == 1


def test_existing_v1_2_review_without_candidate_fields_remains_resumable(tmp_path: Path):
    bundle, source = _bundle(tmp_path)
    annotations = tmp_path / "annotations.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    first = FakeBackend([_draft()])
    execute_proxy_semantic_enrichment(
        scope_id="task:1",
        plan=_plan(source),
        bundles={"p1": bundle},
        backend=first,
        annotation_path=annotations,
        review_path=reviews,
    )

    legacy = json.loads(reviews.read_text(encoding="utf-8").strip())
    for key in (
        "draft_candidate_count",
        "rejected_candidate_count",
        "rejected_candidates",
        "candidate_rejection_is_not_negative_evidence",
    ):
        legacy.pop(key, None)
    reviews.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    second = FakeBackend([])
    report = execute_proxy_semantic_enrichment(
        scope_id="task:1",
        plan=_plan(source),
        bundles={"p1": bundle},
        backend=second,
        annotation_path=annotations,
        review_path=reviews,
    )
    assert report.resumed_target_count == 1
    assert report.llm_calls_performed == 0
    assert report.failed_target_count == 0
    assert second.calls == 0
