from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxis,
    DiscoveryAxisPlan,
    DiscoveryAxisPlannerPolicy,
)
from pipeline_core.discovery.external_novelty_contracts import (
    PriorArtWork,
)
from pipeline_core.discovery.open_world_discovery_axis import (
    ExternalAxisDraft,
    ExternalAxisDraftBatch,
)
from pipeline_core.discovery.open_world_discovery_axis_runtime import (
    ExternalAxisDraftGeneration,
    InstructorOpenAICompatibleExternalAxisBackend,
    NodeMapperAxisSimilarity,
    OpenWorldDiscoveryAxisRuntime,
    build_external_axis_messages,
)


def _statement(
    statement_id: str,
    text: str,
    *,
    premise: bool = False,
    gap: bool = False,
):
    return SimpleNamespace(
        statement_id=statement_id,
        text=text,
        claim_kind="reported_relation",
        eligible_as_premise=premise,
        eligible_as_gap=gap,
    )


def _dual():
    grounded = SimpleNamespace(
        task_id="task:1",
        question="How does factor X alter response Y?",
        corpus_id="corpus:test",
        evidence_statements=[
            _statement(
                "stmt:p1",
                "Grounded premise one.",
                premise=True,
            ),
            _statement(
                "stmt:g1",
                "The packet leaves relation R unresolved.",
                gap=True,
            ),
        ],
    )
    return SimpleNamespace(
        dual_context_id="dual:test",
        dual_context_sha256="a" * 64,
        grounded_context=grounded,
    )


def _control_plan() -> DiscoveryAxisPlan:
    axis = DiscoveryAxis(
        axis_id="axis:control",
        axis_rank=1,
        inspiration_id="insp:control",
        source_path_id="path:control",
        candidate_unit_id="unit:control",
        label="control",
        entry_anchor_id="stmt:p1",
        entry_anchor_label="Grounded premise one.",
        exit_anchor_id="stmt:p1",
        exit_anchor_label="Grounded premise one.",
        proposed_subject="factor A",
        proposed_relation="MODULATES",
        proposed_object="response B",
        rendered_path="factor A --MODULATES--> response B",
        source_mode="persistent_kg",
        exploration_score=1.0,
        candidate_unit_score=1.0,
        planner_score=1.0,
        mechanistic_continuity_band="same_domain",
        requires_verification=False,
    )
    body = {
        "schema_version": "discovery-axis-plan-v1",
        "plan_id": "plan:control",
        "source_dual_context_id": "dual:test",
        "source_dual_context_sha256": "a" * 64,
        "source_bundle_id": "bundle:control",
        "source_bundle_sha256": "b" * 64,
        "corpus_id": "corpus:test",
        "axes": [axis.model_dump(mode="json")],
        "excluded_inspiration_ids": [],
        "policy": DiscoveryAxisPlannerPolicy().model_dump(
            mode="json"
        ),
    }
    import hashlib
    import json

    digest = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return DiscoveryAxisPlan(
        **body,
        plan_sha256=digest,
    )


class _Provider:
    provider_name = "openalex"

    def __init__(self, *, fail_second: bool = False) -> None:
        self.fail_second = fail_second

    def search(self, query, *, limit: int):
        if (
            self.fail_second
            and query.query_id == "open_world_query:2"
        ):
            raise RuntimeError("synthetic provider failure")
        return [
            PriorArtWork(
                work_id="work:ext",
                title="External work",
                doi="10.1/ext",
                abstract=(
                    "Neighboring-site activation changes "
                    "the reaction pathway."
                ),
                citation_count=10,
            )
        ]


class _Backend:
    backend_name = "fake"
    model_name = "fake-model"

    def __init__(self, *, duplicate: bool = False) -> None:
        self.calls = 0
        self.payload = None
        self.duplicate = duplicate

    def generate(self, payload):
        self.calls += 1
        self.payload = payload
        if self.duplicate:
            subject = "factor A"
            relation = "MODULATES"
            obj = "response B"
        else:
            subject = "neighboring-site activation"
            relation = "RECONFIGURES"
            obj = "reaction pathway"
        return ExternalAxisDraftGeneration(
            draft=ExternalAxisDraftBatch(
                axes=[
                    ExternalAxisDraft(
                        local_id="ext1",
                        label="external relation",
                        proposed_subject=subject,
                        proposed_relation=relation,
                        proposed_object=obj,
                        source_work_ids=["work:ext"],
                        source_evidence_spans=[
                            "Neighboring-site activation changes "
                            "the reaction pathway."
                        ],
                        compatible_grounded_statement_ids=[
                            "stmt:p1"
                        ],
                        requires_verification=True,
                    )
                ],
                interpretation="bounded test",
            )
        )


class _Encoder:
    def encode_query(self, text: str):
        if (
            "factor A" in text
            or "response B" in text
        ):
            return np.asarray(
                [1.0, 0.0],
                dtype=np.float32,
            )
        return np.asarray(
            [0.0, 1.0],
            dtype=np.float32,
        )


class _Mapper:
    encoder = _Encoder()


def test_prompt_contract_keeps_axis_synthesis_non_authoritative() -> None:
    messages = build_external_axis_messages(
        {
            "research_question": "question",
            "selected_positive_premises": [],
            "research_gaps": [],
            "current_control_axes": [],
            "retrieved_abstract_backed_works": [],
        }
    )
    combined = "\n".join(
        row["content"]
        for row in messages
    )
    assert "NOT generating hypotheses" in combined
    assert "NOT deciding novelty" in combined
    assert "MUST NOT be promoted to a positive premise" in combined
    assert "Absence from the persistent KG is not novelty" in combined
    assert "Do not generate hypotheses or novelty verdicts" in combined


def test_live_runtime_builds_external_plan_without_promoting_literature() -> None:
    backend = _Backend()
    runtime = OpenWorldDiscoveryAxisRuntime(
        providers=[_Provider()],
        backend=backend,
        mapper=_Mapper(),
    )

    outcome = runtime.run(
        dual=_dual(),
        control_plan=_control_plan(),
    )

    assert backend.calls == 1
    assert outcome.retrieval.complete is True
    assert len(outcome.selected_works) == 1
    assert len(outcome.validation.accepted_axes) == 1
    assert len(outcome.axis_plan.plan.axes) == 1

    axis = outcome.axis_plan.plan.axes[0]
    assert axis.source_mode == "external_open_world"
    assert axis.requires_verification is True
    assert "NOT_POSITIVE_PREMISE" in axis.reason_codes

    supplied_premises = backend.payload[
        "selected_positive_premises"
    ]
    assert {
        row["statement_id"]
        for row in supplied_premises
    } == {"stmt:p1"}


def test_provider_matrix_failure_blocks_llm_call() -> None:
    backend = _Backend()
    runtime = OpenWorldDiscoveryAxisRuntime(
        providers=[_Provider(fail_second=True)],
        backend=backend,
        mapper=_Mapper(),
    )

    with pytest.raises(
        RuntimeError,
        match="retrieval is incomplete",
    ):
        runtime.run(
            dual=_dual(),
            control_plan=_control_plan(),
        )

    assert backend.calls == 0


def test_node_mapper_similarity_rejects_control_duplicate() -> None:
    backend = _Backend(duplicate=True)
    runtime = OpenWorldDiscoveryAxisRuntime(
        providers=[_Provider()],
        backend=backend,
        mapper=_Mapper(),
    )

    outcome = runtime.run(
        dual=_dual(),
        control_plan=_control_plan(),
    )

    assert outcome.axis_plan.plan.axes == []
    assert outcome.axis_plan.rejected_axes[0].reason_codes == [
        "CONTROL_AXIS_SEMANTIC_DUPLICATE",
        "EXACT_CONTROL_TRIPLE_DUPLICATE",
    ]


def test_similarity_adapter_normalizes_encoder_vectors() -> None:
    class Encoder:
        def encode_query(self, text: str):
            if text == "left":
                return np.asarray([2.0, 0.0])
            if text == "same":
                return np.asarray([7.0, 0.0])
            return np.asarray([0.0, 3.0])

    similarity = NodeMapperAxisSimilarity(
        SimpleNamespace(encoder=Encoder())
    )

    assert similarity("left", "same") == pytest.approx(1.0)
    assert similarity("left", "orthogonal") == pytest.approx(0.0)


def test_openai_backend_keeps_one_call_no_retry_contract() -> None:
    backend = InstructorOpenAICompatibleExternalAxisBackend(
        model="test-model",
        api_key="not-used",
    )
    assert backend.temperature == 0.0
    assert backend.parse_retries == 0

    with pytest.raises(
        ValueError,
        match="parse_retries must remain 0",
    ):
        InstructorOpenAICompatibleExternalAxisBackend(
            model="test-model",
            api_key="not-used",
            parse_retries=1,
        )
