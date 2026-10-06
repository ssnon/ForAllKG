from __future__ import annotations

import re
from typing import Sequence

from pipeline_core.discovery.research_idea_contracts import (
    ConceptualFamilyAssessment,
    IdeaFacetAssessment,
    IdeaIdentityRelation,
    IdeaParentComparison,
    IdeaTransitionAssessment,
    ResearchIdeaKernel,
    ResearchIdeaNode,
)


_TOKEN_RE = re.compile(r"[A-Za-z0-9α-ωΑ-Ω가-힣]+")

_EXPECT_SAME_IDEA = {
    "SAME_PREMISE_SHARPEN",
    "EVIDENCE_REAXIS",
    "REQUEST_GRAPH_RETRAVERSAL",
    "GRAPH_RETRAVERSAL",
}

_EXPECT_CHILD_IDEA = {
    "CROSS_SOURCE_BRIDGE",
    "BACKBONE_MUTATION",
    "CANDIDATE_INTERPRETATION",
    "LATENT_VARIABLE",
    "REGIME_BOUNDARY",
    "PROXY_CHALLENGE",
}


def _norm(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[‐‑‒–—−]+", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _norm_set(values: Sequence[str]) -> set[str]:
    return {_norm(value) for value in values if _norm(value)}


def _tokens(values: Sequence[str] | str | None) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        text = values
    else:
        text = " ".join(str(value) for value in values)
    return {token.casefold() for token in _TOKEN_RE.findall(text)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _exact_set_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    return _jaccard(_norm_set(left), _norm_set(right))


def _relation_triplet(value: str) -> tuple[str, str, str] | None:
    match = re.match(r"^\s*(.+?)\s*--(.+?)-->\s*(.+?)\s*$", str(value or ""))
    if match is None:
        return None
    subject, relation, obj = (_norm(part) for part in match.groups())
    if not subject or not relation or not obj:
        return None
    return subject, relation, obj


def _structured_relation_set(values: Sequence[str]) -> set[tuple[str, str, str]] | None:
    if not values:
        return None
    parsed = [_relation_triplet(value) for value in values]
    if any(row is None for row in parsed):
        return None
    return {row for row in parsed if row is not None}


def _token_similarity(left: Sequence[str] | str | None, right: Sequence[str] | str | None) -> float:
    return _jaccard(_tokens(left), _tokens(right))


def _facet_assessment(
    *,
    facet: str,
    left: Sequence[str],
    right: Sequence[str],
) -> IdeaFacetAssessment:
    a = _norm_set(left)
    b = _norm_set(right)
    similarity = _jaccard(a, b)
    if a == b:
        relation = "PRESERVED"
        rationale = "Normalized commitments are unchanged."
    elif not a and b:
        relation = "ADDED"
        rationale = "The proposed idea adds commitments to an empty parent facet."
    elif a and not b:
        relation = "REMOVED"
        rationale = "The proposed idea removes all commitments from this parent facet."
    elif a <= b or b <= a:
        relation = "REFINED"
        rationale = "One normalized commitment set is a strict subset of the other."
    else:
        relation = "REPLACED"
        rationale = "The normalized commitment sets differ without a subset relation."
    return IdeaFacetAssessment(
        facet=facet,
        relation=relation,
        similarity=similarity,
        rationale=rationale,
    )


def _question_assessment(
    parent: ResearchIdeaKernel,
    proposed: ResearchIdeaKernel,
) -> IdeaFacetAssessment:
    left = [parent.question_commitment] if parent.question_commitment else []
    right = [proposed.question_commitment] if proposed.question_commitment else []
    row = _facet_assessment(facet="QUESTION", left=left, right=right)
    if row.relation == "REPLACED":
        token_similarity = _token_similarity(
            parent.question_commitment,
            proposed.question_commitment,
        )
        if token_similarity >= 0.85:
            return row.model_copy(
                update={
                    "relation": "INDETERMINATE",
                    "similarity": token_similarity,
                    "rationale": (
                        "Question wording changed but remains lexically close; "
                        "semantic identity is not asserted."
                    ),
                }
            )
    return row


def compare_idea_kernels(
    parent: ResearchIdeaKernel,
    proposed: ResearchIdeaKernel,
) -> tuple[IdeaIdentityRelation, list[IdeaFacetAssessment]]:
    """Conservative deterministic identity diagnostic.

    SAME_IDEA is intentionally narrow: normalized core, contrast, question, and
    scope commitments must all be preserved. Changes to identity-bearing facets
    become DIFFERENT_IDEA; ambiguous scope refinements and near-rewordings remain
    INDETERMINATE rather than being forced into an identity decision.
    """

    facets = [
        _facet_assessment(
            facet="CORE_COMMITMENTS",
            left=parent.core_scientific_commitments,
            right=proposed.core_scientific_commitments,
        ),
        _facet_assessment(
            facet="SCOPE",
            left=parent.scope_commitments,
            right=proposed.scope_commitments,
        ),
        _facet_assessment(
            facet="CONTRAST",
            left=parent.contrastive_commitments,
            right=proposed.contrastive_commitments,
        ),
        _question_assessment(parent, proposed),
    ]
    by_facet = {row.facet: row for row in facets}
    core = by_facet["CORE_COMMITMENTS"]
    scope = by_facet["SCOPE"]
    contrast = by_facet["CONTRAST"]
    question = by_facet["QUESTION"]

    if all(
        row.relation == "PRESERVED"
        for row in (core, scope, contrast, question)
    ):
        return "SAME_IDEA", facets

    if contrast.relation not in {"PRESERVED", "INDETERMINATE"}:
        return "DIFFERENT_IDEA", facets
    if question.relation not in {"PRESERVED", "INDETERMINATE"}:
        return "DIFFERENT_IDEA", facets

    if core.relation == "PRESERVED":
        # Scope changes often distinguish a refinement from a genuinely new
        # scientific program. V1 deliberately refuses to force that boundary.
        return "INDETERMINATE", facets

    parent_relations = _structured_relation_set(parent.core_scientific_commitments)
    proposed_relations = _structured_relation_set(proposed.core_scientific_commitments)
    if (
        parent_relations is not None
        and proposed_relations is not None
        and parent_relations != proposed_relations
    ):
        # Structured scientific relation commitments are stronger than lexical
        # similarity. Reversing direction or substituting a mediator is a real
        # commitment change even when most tokens are shared.
        return "DIFFERENT_IDEA", facets

    core_token_similarity = _token_similarity(
        parent.core_scientific_commitments,
        proposed.core_scientific_commitments,
    )
    if core_token_similarity >= 0.85:
        return "INDETERMINATE", facets
    return "DIFFERENT_IDEA", facets


def assess_conceptual_family(
    idea_a: ResearchIdeaNode,
    idea_b: ResearchIdeaNode,
) -> ConceptualFamilyAssessment:
    """Soft, versioned family diagnostic; never a permanent family identity."""

    if idea_a.kernel_sha256 == idea_b.kernel_sha256:
        return ConceptualFamilyAssessment(
            idea_id_a=idea_a.idea_id,
            idea_id_b=idea_b.idea_id,
            relation="SAME_FAMILY",
            similarity=1.0,
            confidence=1.0,
        )

    a = idea_a.kernel
    b = idea_b.kernel
    core = _token_similarity(
        a.core_scientific_commitments,
        b.core_scientific_commitments,
    )
    scope = _token_similarity(a.scope_commitments, b.scope_commitments)
    contrast = _token_similarity(
        a.contrastive_commitments,
        b.contrastive_commitments,
    )
    intent = _token_similarity(a.canonical_intent, b.canonical_intent)
    similarity = 0.65 * core + 0.10 * scope + 0.10 * contrast + 0.15 * intent

    if similarity >= 0.78:
        relation = "SAME_FAMILY"
        confidence = min(0.95, 0.70 + (similarity - 0.78))
    elif similarity >= 0.42:
        relation = "ADJACENT_FAMILY"
        confidence = 0.60
    else:
        relation = "DISTINCT_FAMILY"
        confidence = min(0.90, 0.60 + (0.42 - similarity))

    return ConceptualFamilyAssessment(
        idea_id_a=idea_a.idea_id,
        idea_id_b=idea_b.idea_id,
        relation=relation,
        similarity=similarity,
        confidence=confidence,
    )


def _comparison(
    parent: ResearchIdeaNode,
    proposed: ResearchIdeaNode,
) -> IdeaParentComparison:
    identity, facets = compare_idea_kernels(parent.kernel, proposed.kernel)
    return IdeaParentComparison(
        parent_idea_id=parent.idea_id,
        identity_relation=identity,
        facet_assessments=facets,
        family_assessment=assess_conceptual_family(parent, proposed),
    )


def _overall_identity(
    comparisons: Sequence[IdeaParentComparison],
) -> IdeaIdentityRelation:
    if not comparisons:
        return "INDETERMINATE"
    relations = [row.identity_relation for row in comparisons]
    if len(relations) == 1:
        return relations[0]
    if all(value == "SAME_IDEA" for value in relations):
        return "SAME_IDEA"
    if all(value == "DIFFERENT_IDEA" for value in relations):
        return "DIFFERENT_IDEA"
    return "INDETERMINATE"


def _genealogy(
    *,
    parent_count: int,
    identity: IdeaIdentityRelation,
) -> str:
    if parent_count == 0:
        return "IMPORTED_ROOT"
    if parent_count > 1:
        return "COMPOSED_FROM"
    if identity == "SAME_IDEA":
        return "REFINEMENT_OF"
    if identity == "DIFFERENT_IDEA":
        return "CHILD_OF"
    return "DERIVED_FROM"


def _operator_diagnostics(
    *,
    operator_id: str | None,
    identity: IdeaIdentityRelation,
) -> tuple[bool | None, list[str]]:
    if not operator_id:
        return None, []

    diagnostics: list[str] = []
    if operator_id == "AXIS_MUTATION":
        if identity == "DIFFERENT_IDEA":
            diagnostics.append("LOCAL_REPAIR_ESCALATED_TO_IDEA_MUTATION")
        return None, diagnostics

    if operator_id in _EXPECT_SAME_IDEA:
        if identity == "INDETERMINATE":
            return None, diagnostics
        consistent = identity == "SAME_IDEA"
        if not consistent:
            diagnostics.append("LOCAL_REPAIR_CROSSED_IDEA_BOUNDARY")
        return consistent, diagnostics

    if operator_id in _EXPECT_CHILD_IDEA:
        if identity == "INDETERMINATE":
            return None, diagnostics
        consistent = identity == "DIFFERENT_IDEA"
        if not consistent:
            diagnostics.append("EXPECTED_IDEA_MUTATION_BUT_SEMANTIC_NOOP")
        return consistent, diagnostics

    return None, diagnostics


def assess_idea_transition(
    *,
    parents: Sequence[ResearchIdeaNode],
    proposed: ResearchIdeaNode,
    operator_id: str | None = None,
) -> IdeaTransitionAssessment:
    parent_ids = [row.idea_id for row in parents]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("transition parents must be unique")
    if proposed.idea_id in set(parent_ids):
        raise ValueError("proposed idea must differ from parent IDs")

    comparisons = [_comparison(parent, proposed) for parent in parents]
    identity = _overall_identity(comparisons)
    genealogy = _genealogy(parent_count=len(parents), identity=identity)
    expectation, diagnostics = _operator_diagnostics(
        operator_id=operator_id,
        identity=identity,
    )

    if len(parents) > 1 and identity == "SAME_IDEA":
        diagnostics.append("MULTI_PARENT_COMPOSITION_SEMANTIC_NOOP")

    return IdeaTransitionAssessment(
        proposed_idea_id=proposed.idea_id,
        parent_idea_ids=parent_ids,
        parent_comparisons=comparisons,
        identity_relation=identity,
        genealogy_relation=genealogy,
        operator_id=operator_id,
        operator_expectation_consistent=expectation,
        diagnostic_codes=diagnostics,
    )


__all__ = [
    "assess_conceptual_family",
    "assess_idea_transition",
    "compare_idea_kernels",
]
