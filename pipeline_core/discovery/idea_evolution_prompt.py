from __future__ import annotations

import json
from dataclasses import dataclass

from pipeline_core.discovery.frontier_idea_population import (
    FrontierIdeaPopulation,
)
from pipeline_core.discovery.idea_evolution import (
    IdeaEvolutionOperatorPlan,
    NativeEvolutionOperatorId,
    frontier_idea_payloads,
)


@dataclass(frozen=True)
class IdeaEvolutionPrompt:
    operator_id: NativeEvolutionOperatorId
    system_prompt: str
    user_prompt: str


_COMMON = """You perform shadow-only scientific idea evolution over an already-audited Frontier population.

You are not judging truth, novelty, priority, or production suitability. You are constructing explicit, testable exploratory transformations from supplied parent ideas.

Hard authority rules:
- Parent ideas are inspiration material, not automatically positive evidence.
- OPEN_WORLD_AXIS material remains external-literature inspiration only. Never present it as a grounded premise or verified mechanism.
- Candidate/unverified lineage remains candidate/unverified after evolution.
- Do not claim novelty, literature absence, empirical confirmation, causal truth, or superiority.
- Use only supplied parent_idea_ids exactly. Do not invent IDs.
- Every emitted idea must include a differential_prediction, falsification_condition, and discriminating_observation.
- scientific_intent must be a concrete scientific proposal, not a generic recommendation to study something.
- core_relations must contain 1-4 concise relation statements central to the evolved idea.
- Keep native evolution centered on the supplied original task, but do not claim a certified DIRECT/SUBORDINATE task-semantic class; that classification is outside this stage.
- If the operator constraints cannot be satisfied without unsupported invention, abstain.
""".strip()


_OPERATOR = {
    "CROSS_SOURCE_BRIDGE": """Operator: CROSS_SOURCE_BRIDGE.

Construct source-bounded bridges that combine genuinely different Frontier source families.
Each candidate MUST:
- cite at least two parent ideas from at least two source_kind values;
- include at least one OPEN_WORLD_AXIS parent and at least one non-OPEN_WORLD_AXIS parent;
- state what each source contributes to the bridge;
- stay centered on the original task rather than replacing it;
- not treat the external literature parent as positive evidence.

The objective is not to concatenate two statements. The bridge must propose a testable scientific dependency, mediator, condition, or mechanism that becomes visible only when the parent ideas are considered together.
Set mutation_kind=null, challenged_assumption=null, alternative_explanations=[], and transformed_question=null.
""".strip(),
    "BACKBONE_MUTATION": """Operator: BACKBONE_MUTATION.

Change the internal conceptual structure of an existing higher-order backbone. Do NOT merely replace or add a modifier while retaining the same backbone.
At least one parent must be a HIGHER_ORDER_TOPOLOGY idea. Additional primitive parents may be cited as inspiration.
Choose exactly one mutation_kind from:
- MEDIATOR_SUBSTITUTION
- MECHANISM_INSERTION
- PROXY_TARGET_SEPARATION
- CONDITION_INVERSION
- CAUSAL_DIRECTION_CHALLENGE
- MULTI_MECHANISM_COMPETITION

core_relations must describe the mutated backbone and must differ from the parent backbone relation set. The evolved idea must stay centered on the original task, while task-semantic classification remains uncertified in this stage.
Set challenged_assumption=null, alternative_explanations=[], and transformed_question=null.
""".strip(),
    "CANDIDATE_INTERPRETATION": """Operator: CANDIDATE_INTERPRETATION.

Continue candidate/unverified higher-order material into an explicit scientific interpretive fork.
Every candidate must cite at least one parent carrying candidate_or_unverified_lineage=true.
State one challenged_assumption and exactly two distinct alternative_explanations that would make different observations under a discriminating test.
Do not promote the candidate relation into evidence. The point is to expose what would need to be true for the candidate to matter scientifically.
Set mutation_kind=null and transformed_question=null.
""".strip(),
}


def build_idea_evolution_prompt(
    *,
    population: FrontierIdeaPopulation,
    operator_plan: IdeaEvolutionOperatorPlan,
) -> IdeaEvolutionPrompt:
    if not operator_plan.enabled:
        raise ValueError("cannot build a prompt for a disabled evolution operator")
    parent_payloads = frontier_idea_payloads(
        population=population,
        idea_ids=operator_plan.parent_pool_idea_ids,
    )
    payload = {
        "research_question": population.research_question,
        "task_source": population.task_source,
        "task_target": population.task_target,
        "operator_id": operator_plan.operator_id,
        "max_output_count": operator_plan.max_output_count,
        "parent_pool": parent_payloads,
    }
    user = (
        "Generate up to max_output_count valid evolved ideas under the operator contract. "
        "Return an abstention_reason if no valid transformation is justified.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return IdeaEvolutionPrompt(
        operator_id=operator_plan.operator_id,
        system_prompt=_COMMON + "\n\n" + _OPERATOR[operator_plan.operator_id],
        user_prompt=user,
    )


__all__ = ["IdeaEvolutionPrompt", "build_idea_evolution_prompt"]
