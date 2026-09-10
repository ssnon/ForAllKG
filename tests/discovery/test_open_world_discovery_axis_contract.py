from __future__ import annotations

from types import SimpleNamespace

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
    build_external_axis_plan,
    build_external_axis_prompt_payload,
    build_outcome_blind_retrieval_seeds,
    retrieve_open_world_sources,
    select_open_world_abstracts,
    validate_external_axis_drafts,
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
                "The current packet does not establish relation R.",
                gap=True,
            ),
            _statement(
                "stmt:p2",
                "Grounded premise two.",
                premise=True,
            ),
        ],
    )
    return SimpleNamespace(
        dual_context_id="dual:test",
        dual_context_sha256="a" * 64,
        grounded_context=grounded,
    )


def _work(
    work_id: str,
    title: str,
    *,
    doi: str,
    abstract: str | None,
    citation_count: int = 0,
) -> PriorArtWork:
    return PriorArtWork(
        work_id=work_id,
        title=title,
        doi=doi,
        abstract=abstract,
        citation_count=citation_count,
    )


class _Provider:
    def __init__(
        self,
        name: str,
        rows_by_query: dict[str, list[PriorArtWork]],
        *,
        fail_query: str | None = None,
    ) -> None:
        self.provider_name = name
        self.rows_by_query = rows_by_query
        self.fail_query = fail_query

    def search(self, query, *, limit: int):
        if query.query_id == self.fail_query:
            raise RuntimeError("synthetic provider failure")
        return list(self.rows_by_query.get(query.query_id, []))[:limit]


def _control_plan() -> DiscoveryAxisPlan:
    axis = DiscoveryAxis(
        axis_id="axis:control",
        axis_rank=1,
        inspiration_id="insp:control",
        source_path_id="path:control",
        candidate_unit_id="unit:control",
        label="control relation",
        entry_anchor_id="stmt:p1",
        entry_anchor_label="Grounded premise one.",
        exit_anchor_id="stmt:p2",
        exit_anchor_label="Grounded premise two.",
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


def test_retrieval_seeds_are_question_plus_frozen_gaps_only() -> None:
    seeds = build_outcome_blind_retrieval_seeds(_dual())

    assert [row.source_kind for row in seeds] == [
        "FROZEN_RESEARCH_QUESTION",
        "FROZEN_GAP_STATEMENT",
    ]
    assert [row.source_id for row in seeds] == [
        "task:1",
        "stmt:g1",
    ]
    assert seeds[0].query_text == (
        "How does factor X alter response Y?"
    )
    assert seeds[1].query_text == (
        "The current packet does not establish relation R."
    )


def test_retrieval_is_complete_matrix_and_ranks_outcome_blind() -> None:
    shared_a = _work(
        "work:a-openalex",
        "Shared high coverage work",
        doi="10.1/shared",
        abstract="Exact external relation evidence.",
        citation_count=5,
    )
    shared_b = _work(
        "work:a-crossref",
        "Shared high coverage work",
        doi="10.1/shared",
        abstract="Exact external relation evidence.",
        citation_count=5,
    )
    unique = _work(
        "work:unique",
        "Unique work",
        doi="10.1/unique",
        abstract="Other evidence.",
        citation_count=100,
    )

    providers = [
        _Provider(
            "openalex",
            {
                "open_world_query:1": [shared_a, unique],
                "open_world_query:2": [shared_a],
            },
        ),
        _Provider(
            "crossref",
            {
                "open_world_query:1": [shared_b],
                "open_world_query:2": [shared_b],
            },
        ),
    ]

    result = retrieve_open_world_sources(
        seeds=build_outcome_blind_retrieval_seeds(_dual()),
        providers=providers,
        results_per_query=20,
    )

    assert result.complete is True
    assert len(result.executions) == 4
    assert result.canonical_work_count == 2

    selected = select_open_world_abstracts(
        result,
        max_selected=2,
    )
    assert selected[0].work.doi == "10.1/shared"
    assert selected[0].distinct_seed_count == 2
    assert selected[0].distinct_provider_count == 2


def test_incomplete_provider_matrix_cannot_feed_axis_synthesis() -> None:
    provider = _Provider(
        "openalex",
        {},
        fail_query="open_world_query:2",
    )
    result = retrieve_open_world_sources(
        seeds=build_outcome_blind_retrieval_seeds(_dual()),
        providers=[provider],
    )

    assert result.complete is False
    with pytest.raises(
        RuntimeError,
        match="retrieval is incomplete",
    ):
        select_open_world_abstracts(result)


def test_prompt_payload_keeps_external_literature_out_of_positive_premises() -> None:
    ranked = retrieve_open_world_sources(
        seeds=build_outcome_blind_retrieval_seeds(_dual()),
        providers=[
            _Provider(
                "openalex",
                {
                    "open_world_query:1": [
                        _work(
                            "work:ext",
                            "External work",
                            doi="10.1/ext",
                            abstract="External span.",
                        )
                    ],
                    "open_world_query:2": [],
                },
            )
        ],
    )
    selected = select_open_world_abstracts(ranked)
    payload = build_external_axis_prompt_payload(
        dual=_dual(),
        control_plan=_control_plan(),
        selected_works=selected,
    )

    assert {
        row["statement_id"]
        for row in payload["selected_positive_premises"]
    } == {"stmt:p1", "stmt:p2"}
    assert payload["research_gaps"] == [
        {
            "statement_id": "stmt:g1",
            "text": (
                "The current packet does not establish relation R."
            ),
        }
    ]
    assert payload["retrieved_abstract_backed_works"][0][
        "work_id"
    ] == "work:ext"


def test_axis_validation_requires_exact_source_and_positive_premise() -> None:
    result = retrieve_open_world_sources(
        seeds=build_outcome_blind_retrieval_seeds(_dual()),
        providers=[
            _Provider(
                "openalex",
                {
                    "open_world_query:1": [
                        _work(
                            "work:ext",
                            "External work",
                            doi="10.1/ext",
                            abstract=(
                                "Neighboring-site activation changes "
                                "the reaction pathway."
                            ),
                        )
                    ],
                    "open_world_query:2": [],
                },
            )
        ],
    )
    selected = select_open_world_abstracts(result)

    good = ExternalAxisDraft(
        local_id="ext1",
        label="neighboring-site activation",
        proposed_subject="neighboring-site activation",
        proposed_relation="RECONFIGURES",
        proposed_object="reaction pathway",
        source_work_ids=["work:ext"],
        source_evidence_spans=[
            "Neighboring-site activation changes the reaction pathway."
        ],
        compatible_grounded_statement_ids=["stmt:p1"],
        requires_verification=True,
    )
    bad = ExternalAxisDraft(
        local_id="ext2",
        label="bad",
        proposed_subject="x",
        proposed_relation="Y",
        proposed_object="z",
        source_work_ids=["work:missing"],
        source_evidence_spans=["not an exact span"],
        compatible_grounded_statement_ids=["stmt:g1"],
        requires_verification=True,
    )

    checked = validate_external_axis_drafts(
        batch=ExternalAxisDraftBatch(
            axes=[good, bad],
            interpretation="test",
        ),
        dual=_dual(),
        selected_works=selected,
    )

    assert [row.local_id for row in checked.accepted_axes] == [
        "ext1"
    ]
    assert len(checked.rejected_axes) == 1
    reasons = checked.rejected_axes[0].reason_codes
    assert any(
        row.startswith("UNKNOWN_SOURCE_WORK_IDS")
        for row in reasons
    )
    assert any(
        row.startswith("INVALID_GROUNDED_STATEMENT_IDS")
        for row in reasons
    )
    assert any(
        row.startswith("SOURCE_SPAN_NOT_EXACT")
        for row in reasons
    )


def test_external_plan_preserves_inspiration_only_authority() -> None:
    result = retrieve_open_world_sources(
        seeds=build_outcome_blind_retrieval_seeds(_dual()),
        providers=[
            _Provider(
                "openalex",
                {
                    "open_world_query:1": [
                        _work(
                            "work:ext",
                            "External work",
                            doi="10.1/ext",
                            abstract="Exact external evidence.",
                        )
                    ],
                    "open_world_query:2": [],
                },
            )
        ],
    )
    selected = select_open_world_abstracts(result)

    draft = ExternalAxisDraft(
        local_id="ext1",
        label="external regime",
        proposed_subject="factor X",
        proposed_relation="RECONFIGURES",
        proposed_object="response Y",
        source_work_ids=["work:ext"],
        source_evidence_spans=["Exact external evidence."],
        compatible_grounded_statement_ids=["stmt:p1"],
        requires_verification=True,
    )

    built = build_external_axis_plan(
        dual=_dual(),
        control_plan=_control_plan(),
        selected_works=selected,
        validated_axes=[draft],
        similarity=lambda left, right: 0.2,
    )

    assert len(built.plan.axes) == 1
    axis = built.plan.axes[0]
    assert axis.source_mode == "external_open_world"
    assert axis.requires_verification is True
    assert "EXTERNAL_INSPIRATION_ONLY" in axis.reason_codes
    assert "NOT_POSITIVE_PREMISE" in axis.reason_codes
    assert built.bundle.provenance[0].source_evidence_spans == [
        "Exact external evidence."
    ]


def test_external_plan_rejects_control_duplicate() -> None:
    result = retrieve_open_world_sources(
        seeds=build_outcome_blind_retrieval_seeds(_dual()),
        providers=[
            _Provider(
                "openalex",
                {
                    "open_world_query:1": [
                        _work(
                            "work:ext",
                            "External work",
                            doi="10.1/ext",
                            abstract="Exact external evidence.",
                        )
                    ],
                    "open_world_query:2": [],
                },
            )
        ],
    )
    selected = select_open_world_abstracts(result)

    duplicate = ExternalAxisDraft(
        local_id="dup",
        label="duplicate",
        proposed_subject="factor A",
        proposed_relation="MODULATES",
        proposed_object="response B",
        source_work_ids=["work:ext"],
        source_evidence_spans=["Exact external evidence."],
        compatible_grounded_statement_ids=["stmt:p1"],
        requires_verification=True,
    )

    built = build_external_axis_plan(
        dual=_dual(),
        control_plan=_control_plan(),
        selected_works=selected,
        validated_axes=[duplicate],
        similarity=lambda left, right: 1.0,
    )

    assert built.plan.axes == []
    assert built.rejected_axes[0].reason_codes == [
        "CONTROL_AXIS_SEMANTIC_DUPLICATE",
        "EXACT_CONTROL_TRIPLE_DUPLICATE",
    ]
