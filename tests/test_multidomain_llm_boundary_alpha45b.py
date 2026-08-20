from types import SimpleNamespace

from pipeline_core.discovery.explorer_draft import (
    ExplorationDraft,
    ExplorerStatementDraft,
    MechanisticMotifDraft,
)
from dac_her.explorer_normalization import (
    ExplorerDraftNormalizer,
)
from pipeline_core.discovery.hypothesis_semantic_contracts import (
    HypothesisSemanticDimensionDraft,
    HypothesisSemanticReviewDraft,
    SEMANTIC_DIMENSIONS,
)
from dac_her.hypothesis_semantic_reference import (
    HypothesisSemanticReferenceSanitizer,
)


def _packet():
    return SimpleNamespace(
        domain_profile_id="sers_au_ag",
        evidence_catalog=SimpleNamespace(
            nodes={},
            edges={},
        ),
        paths=[],
        direct_concept_hits=[],
    )


def test_explorer_normalizer_only_weakens_noncausal_unsupported_mechanism():
    draft = ExplorationDraft(
        statements=[
            ExplorerStatementDraft(
                local_id="s1",
                text=(
                    "The two reported quantities are associated "
                    "under the supplied conditions."
                ),
                epistemic_role="evidence_synthesis",
                claim_kind="mechanism",
            ),
        ],
        recurring_mechanistic_motifs=[],
    )
    result = ExplorerDraftNormalizer().normalize(
        _packet(),
        draft,
    )
    assert (
        result.draft.statements[0].claim_kind
        == "association"
    )
    assert result.draft.statements[0].text == (
        draft.statements[0].text
    )
    assert result.audit.action_count == 1
    assert result.audit.blocked_count == 0


def test_explorer_normalizer_drops_unsupported_strong_causal_statement():
    draft = ExplorationDraft(
        statements=[
            ExplorerStatementDraft(
                local_id="s1",
                text=(
                    "The nanogap causes stronger SERS "
                    "enhancement."
                ),
                epistemic_role="evidence_synthesis",
                claim_kind="mechanism",
            ),
        ],
        recurring_mechanistic_motifs=[],
    )
    result = ExplorerDraftNormalizer().normalize(
        _packet(),
        draft,
    )

    assert result.draft.statements == []
    assert result.audit.applied is True
    assert result.audit.action_count == 1
    assert result.audit.blocked_count == 0
    assert result.audit.actions[0].action == (
        "drop_unsupported_strong_causal_statement"
    )


def test_explorer_normalizer_drops_unsupported_mechanistic_motif():
    draft = ExplorationDraft(
        statements=[
            ExplorerStatementDraft(
                local_id="s1",
                text="A packet-scoped observation.",
                epistemic_role="reported",
                claim_kind="observation",
            ),
        ],
        recurring_mechanistic_motifs=[
            MechanisticMotifDraft(
                local_id="m1",
                label="Unsupported motif",
                statement_local_ids=["s1"],
            )
        ],
    )
    result = ExplorerDraftNormalizer().normalize(
        _packet(),
        draft,
    )
    assert result.draft.recurring_mechanistic_motifs == []
    assert result.audit.action_count == 1


def _semantic_draft(
    *,
    hypothesis_ids=None,
    statement_ids=None,
    reference_dimension=None,
):
    hypothesis_ids = hypothesis_ids or []
    statement_ids = statement_ids or []
    reference_dimension = (
        reference_dimension or SEMANTIC_DIMENSIONS[0]
    )
    rows = []
    for dimension in SEMANTIC_DIMENSIONS:
        rows.append(
            HypothesisSemanticDimensionDraft(
                dimension=dimension,
                verdict="pass",
                rationale="Grounded review rationale.",
                hypothesis_ids=(
                    list(hypothesis_ids)
                    if dimension == reference_dimension else []
                ),
                statement_ids=(
                    list(statement_ids)
                    if dimension == reference_dimension else []
                ),
            )
        )
    return HypothesisSemanticReviewDraft(
        dimensions=rows,
        overall_summary="Summary.",
    )


def test_semantic_reference_sanitizer_safe_drops_mixed_unknown_ids():
    draft = _semantic_draft(
        hypothesis_ids=["h1", "hallucinated_h"],
        statement_ids=["s1", "hallucinated_s"],
    )
    result = (
        HypothesisSemanticReferenceSanitizer()
        .sanitize(
            draft,
            valid_hypothesis_ids={"h1"},
            valid_statement_ids={"s1"},
        )
    )
    first = result.draft.dimensions[0]
    assert first.hypothesis_ids == ["h1"]
    assert first.statement_ids == ["s1"]
    assert result.audit.drop_count == 2
    assert result.audit.fatal is False


def test_semantic_reference_sanitizer_fails_closed_if_required_namespace_lost():
    draft = _semantic_draft(
        hypothesis_ids=["hallucinated_h"],
        statement_ids=["s1"],
        reference_dimension="inferential_proportionality",
    )
    result = (
        HypothesisSemanticReferenceSanitizer()
        .sanitize(
            draft,
            valid_hypothesis_ids={"h1"},
            valid_statement_ids={"s1"},
        )
    )
    assert result.audit.fatal is True
    assert result.audit.fatal_reasons
