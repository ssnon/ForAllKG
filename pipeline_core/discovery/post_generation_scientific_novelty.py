from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipeline_core.discovery.scientific_distinctiveness import (
    ScientificDistinctivenessAnalyzer,
)
from pipeline_core.discovery.semantic_distinctiveness import (
    compile_semantic_distinctiveness_review,
)
from pipeline_core.discovery.semantic_distinctiveness_prompt import (
    SemanticDistinctivenessPromptAssembler,
)
from pipeline_core.discovery.scientific_novelty_action_policy import (
    ScientificNoveltyActionPolicy,
)


_REPAIRABLE_REFERENCE_PREFIXES = (
    "semantic dimension references unknown work IDs:",
    "semantic dimension references unknown claim IDs:",
)


@dataclass(frozen=True)
class PostGenerationScientificNoveltyAssessment:
    scientific_report: Any
    semantic_pass_1: Any
    semantic_pass_2: Any
    action_decision: Any


def _semantic_pass(
    *,
    scientific_report: Any,
    scientific_review: Any,
    external_card: Any,
    packet: Any,
    backend: Any,
    review_pass_index: int,
) -> Any:
    prompt = (
        SemanticDistinctivenessPromptAssembler()
        .build(
            scientific_review,
            external_card,
            packet,
        )
    )

    generation = backend.review(
        prompt,
        review_pass_index=review_pass_index,
    )

    active_prompt = prompt
    repair_count = 0
    repair_issues: list[str] = []

    try:
        return compile_semantic_distinctiveness_review(
            scientific_report=scientific_report,
            scientific_review=scientific_review,
            prompt=active_prompt,
            draft=generation.draft,
            backend_name=backend.backend_name,
            requested_model=generation.requested_model,
            served_model=generation.served_model,
            review_pass_index=review_pass_index,
            reference_contract_repair_count=repair_count,
            reference_contract_repair_issues=repair_issues,
        )
    except ValueError as exc:
        issue = str(exc)

        if not issue.startswith(
            _REPAIRABLE_REFERENCE_PREFIXES
        ):
            raise

        repair_count = 1
        repair_issues = [issue]

        repair_prompt = (
            SemanticDistinctivenessPromptAssembler()
            .build_reference_validation_repair(
                original_prompt=prompt,
                previous_draft=generation.draft,
                issues=repair_issues,
            )
        )

        generation = backend.review(
            repair_prompt,
            review_pass_index=review_pass_index,
        )

        # Deliberately no second catch:
        # a second reference-contract failure remains fail-closed.
        return compile_semantic_distinctiveness_review(
            scientific_report=scientific_report,
            scientific_review=scientific_review,
            prompt=repair_prompt,
            draft=generation.draft,
            backend_name=backend.backend_name,
            requested_model=generation.requested_model,
            served_model=generation.served_model,
            review_pass_index=review_pass_index,
            reference_contract_repair_count=repair_count,
            reference_contract_repair_issues=repair_issues,
        )


def evaluate_post_generation_scientific_novelty(
    *,
    hypothesis_id: str,
    report: Any,
    plan: Any,
    packet: Any,
    backend: Any,
) -> PostGenerationScientificNoveltyAssessment:
    scientific_report = (
        ScientificDistinctivenessAnalyzer()
        .build(
            report,
            plan,
            packet,
        )
    )

    scientific_by_id = {
        row.hypothesis_id: row
        for row in scientific_report.reviews
    }

    external_by_id = {
        row.hypothesis_id: row
        for row in report.cards
    }

    if hypothesis_id not in scientific_by_id:
        raise ValueError(
            "hypothesis absent from scientific distinctiveness report: "
            + hypothesis_id
        )

    if hypothesis_id not in external_by_id:
        raise ValueError(
            "hypothesis absent from external novelty report: "
            + hypothesis_id
        )

    scientific_review = scientific_by_id[hypothesis_id]
    external_card = external_by_id[hypothesis_id]

    pass_1 = _semantic_pass(
        scientific_report=scientific_report,
        scientific_review=scientific_review,
        external_card=external_card,
        packet=packet,
        backend=backend,
        review_pass_index=1,
    )

    pass_2 = _semantic_pass(
        scientific_report=scientific_report,
        scientific_review=scientific_review,
        external_card=external_card,
        packet=packet,
        backend=backend,
        review_pass_index=2,
    )

    action_decision = (
        ScientificNoveltyActionPolicy()
        .evaluate(
            external_status=external_card.status,
            semantic_pass_1=pass_1.overall_tier,
            semantic_pass_2=pass_2.overall_tier,
        )
    )

    return PostGenerationScientificNoveltyAssessment(
        scientific_report=scientific_report,
        semantic_pass_1=pass_1,
        semantic_pass_2=pass_2,
        action_decision=action_decision,
    )
