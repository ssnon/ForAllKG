from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from pipeline_core.discovery.adaptive_discovery_controller import (
    safe_premise_ids,
)
from pipeline_core.discovery.hypothesis_compiler import (
    HypothesisCompileError,
    HypothesisCompiler,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
    HypothesisPortfolioDraft,
)
from pipeline_core.discovery.hypothesis_llm import (
    InstructorOpenAICompatibleHypothesisBackend,
)
from pipeline_core.discovery.hypothesis_prompt import (
    HypothesisPrompt,
    HypothesisPromptAssembler,
)
from pipeline_core.discovery.hypothesis_validation import (
    HypothesisValidator,
)
from pipeline_core.discovery.question_axis_responsiveness_llm import (
    OpenRouterQuestionAxisResponsivenessBackend,
)
from pipeline_core.discovery.question_hypothesis_responsiveness import (
    evaluate_hypothesis_task_preservation,
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _prompt_sha(version: str, system: str, user: str) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "prompt_version": version,
                "system_prompt": system,
                "user_prompt": user,
            }
        ).encode("utf-8")
    ).hexdigest()


def _norm(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _compile(
    *,
    context: HypothesisContext,
    draft: HypothesisPortfolioDraft,
) -> tuple[HypothesisPortfolio | None, list[str]]:
    try:
        portfolio = HypothesisCompiler().compile(context, draft)
    except HypothesisCompileError as exc:
        return None, [
            f"{row.code}:{row.location}:{row.message}"
            for row in exc.issues
        ]
    except Exception as exc:
        return None, [f"{type(exc).__name__}:{exc}"]

    validation = HypothesisValidator().validate(context, portfolio)
    if not validation.passes:
        return None, [
            f"{row.code}:{row.location}:{row.message}"
            for row in validation.issues
            if row.severity == "error"
        ]
    return portfolio, []


class GraphRetraversalHypothesisPromptAssembler:
    prompt_version = "adaptive-graph-retraversal-hypothesis-prompt-v1"

    def __init__(
        self,
        *,
        original: Any,
        request: Mapping[str, Any],
        allowed_premise_ids: Sequence[str],
        required_new_premise_ids: Sequence[str],
        attempt_history: Sequence[Mapping[str, Any]],
    ) -> None:
        self.original = original
        self.request = dict(request)
        self.allowed_premise_ids = list(
            dict.fromkeys(map(str, allowed_premise_ids))
        )
        self.required_new_premise_ids = list(
            dict.fromkeys(map(str, required_new_premise_ids))
        )
        self.attempt_history = list(attempt_history)[-12:]

    def build(self, context: HypothesisContext) -> HypothesisPrompt:
        base = HypothesisPromptAssembler(max_hypotheses=1).build(context)

        system = base.system_prompt + """

ADAPTIVE DISCOVERY V2 — GROUNDED CONTEXT RESET
===============================================
The previous grounded context was exhausted by bounded local search.
The supplied HypothesisContext was rebuilt from a different, validated
neighborhood of the SAME persistent scientific KG and SAME research task.

Generate exactly ONE scientifically meaningful, falsifiable hypothesis from
the NEW grounded context, or abstain.

Hard rules:
1. Positive premises may come ONLY from ALLOWED NEW-CONTEXT PREMISE IDS.
2. The hypothesis MUST use at least one REQUIRED NEW PREMISE ID. This proves
   that graph retraversal changed the positive evidence basis.
3. Previous hypotheses, failed attempts, and external prior art are negative
   search-boundary information only. NEVER cite or encode them as positive
   premise_statement_ids.
4. Do not cosmetically restate the previous rejected/known hypothesis.
5. Stay DIRECT or SUBORDINATE to the original research question.
6. Do not claim literature-wide novelty or absence.
7. Return exactly ONE hypothesis or abstain.
8. Predictions and falsifiers must test the new relation/backbone itself.
"""
        user = base.user_prompt + "\n\n" + "\n".join(
            [
                "GRAPH-RETRAVERSAL CONTEXT RESET",
                "===============================",
                f"source_context_id: {self.request.get('source_context_id')}",
                f"new_context_id: {context.context_id}",
                f"source_hypothesis_id: {self.original.hypothesis_id}",
                f"source_hypothesis: {self.original.hypothesis_statement}",
                "",
                "ALREADY-KNOWN / EXCLUDED REGION",
                "================================",
                *(
                    [
                        f"- {row}"
                        for row in self.request.get(
                            "already_known_boundary",
                            [],
                        )
                    ]
                    or ["- NONE"]
                ),
                "",
                "UNRESOLVED PRIOR REGION",
                "=======================",
                *(
                    [
                        f"- {row}"
                        for row in self.request.get(
                            "unresolved_boundary",
                            [],
                        )
                    ]
                    or ["- NONE"]
                ),
                "",
                "PRIOR LOCAL SEARCH HISTORY",
                "==========================",
                json.dumps(
                    self.attempt_history,
                    ensure_ascii=False,
                    indent=2,
                ),
                "",
                "ALLOWED NEW-CONTEXT PREMISE IDS",
                "===============================",
                *[f"- {row}" for row in self.allowed_premise_ids],
                "",
                "REQUIRED NEW PREMISE IDS",
                "========================",
                *[f"- {row}" for row in self.required_new_premise_ids],
                "",
                "TASK",
                "====",
                (
                    "Use the newly grounded evidence neighborhood to form one "
                    "different relation/backbone hypothesis."
                ),
                (
                    "External literature remains exclusion-boundary "
                    "information only."
                ),
            ]
        )
        return HypothesisPrompt(
            prompt_version=self.prompt_version,
            system_prompt=system,
            user_prompt=user,
            prompt_sha256=_prompt_sha(
                self.prompt_version,
                system,
                user,
            ),
        )


def generate_retraversal_hypothesis(
    *,
    old_context: HypothesisContext,
    new_context: HypothesisContext,
    original: Any,
    request: Mapping[str, Any],
    attempt_history: Sequence[Mapping[str, Any]],
    required_new_premise_ids: Sequence[str] | None = None,
    model: str,
    critic_model: str,
    api_key_env: str,
    base_url: str | None,
    output_prefix: str,
) -> tuple[HypothesisPortfolio | None, dict[str, Any]]:
    allowed = set(safe_premise_ids(new_context))
    old_allowed = set(safe_premise_ids(old_context))
    raw_required_new = (
        set(map(str, required_new_premise_ids))
        if required_new_premise_ids is not None
        else (allowed - old_allowed)
    )
    required_new = sorted(
        raw_required_new.intersection(allowed)
    )

    record: dict[str, Any] = {
        "source_hypothesis_id": str(original.hypothesis_id),
        "route": "GRAPH_RETRAVERSAL_CONTEXT_RESET",
        "decision": None,
        "generated_hypothesis_id": None,
        "source_context_id": str(old_context.context_id),
        "output_context_id": str(new_context.context_id),
        "required_new_premise_ids": required_new,
        "reason_codes": [],
        "external_prior_art_as_positive_premise": False,
    }

    if not required_new:
        record["decision"] = "CONTEXT_RESET_HAS_NO_NEW_SAFE_PREMISE"
        record["reason_codes"] = [
            "new_context_added_no_structurally_new_eligible_positive_premise"
        ]
        return None, record

    prompt = GraphRetraversalHypothesisPromptAssembler(
        original=original,
        request=request,
        allowed_premise_ids=sorted(allowed),
        required_new_premise_ids=required_new,
        attempt_history=attempt_history,
    ).build(new_context)

    backend = InstructorOpenAICompatibleHypothesisBackend(
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
        parse_retries=3,
        telemetry_path=output_prefix + ".telemetry.jsonl",
        telemetry_context={
            "pipeline": "adaptive_discovery_controller_v2",
            "stage": "graph_retraversal_context_reset_generation",
            "source_hypothesis_id": str(original.hypothesis_id),
            "source_context_id": str(old_context.context_id),
            "output_context_id": str(new_context.context_id),
        },
    )

    generation = backend.generate(prompt)
    draft = generation.draft

    if not draft.hypotheses:
        record["decision"] = "ABSTAINED"
        record["reason_codes"] = [
            "model_abstained_after_context_reset"
        ]
        return None, record

    if len(draft.hypotheses) != 1:
        record["decision"] = "REJECTED_CARDINALITY"
        record["reason_codes"] = [
            "context_reset_requires_exactly_one_hypothesis"
        ]
        return None, record

    compiled, issues = _compile(context=new_context, draft=draft)

    if compiled is None:
        feedback = "\n".join(
            [
                "GRAPH-RETRAVERSAL GENERATION REPAIR",
                "Use only the new context's allowed grounded premise IDs.",
                "Use at least one REQUIRED NEW PREMISE ID.",
                "Previous hypothesis and external prior art are boundary-only.",
                "Issues:",
                *[f"- {row}" for row in issues],
                "Return exactly one corrected hypothesis or abstain.",
            ]
        )
        repaired = backend.repair(prompt, draft, feedback).draft
        compiled, issues = _compile(
            context=new_context,
            draft=repaired,
        )

    if compiled is None:
        record["decision"] = "COMPILE_OR_VALIDATION_REJECTED"
        record["reason_codes"] = issues[:12]
        return None, record

    candidate = compiled.hypotheses[0]
    premises = set(map(str, candidate.premise_statement_ids))

    if not premises.issubset(allowed):
        record["decision"] = "GROUNDING_DRIFT_REJECTED"
        record["generated_hypothesis_id"] = str(candidate.hypothesis_id)
        record["reason_codes"] = [
            "context_reset_used_noneligible_positive_premise"
        ]
        return None, record

    if not premises.intersection(required_new):
        record["decision"] = "NEW_CONTEXT_NOT_USED_REJECTED"
        record["generated_hypothesis_id"] = str(candidate.hypothesis_id)
        record["reason_codes"] = [
            "candidate_did_not_use_new_retraversed_evidence"
        ]
        return None, record

    if _norm(candidate.hypothesis_statement) == _norm(
        original.hypothesis_statement
    ):
        record["decision"] = "UNCHANGED_HYPOTHESIS_REJECTED"
        record["generated_hypothesis_id"] = str(candidate.hypothesis_id)
        record["reason_codes"] = [
            "context_reset_reproduced_original_statement"
        ]
        return None, record

    task_backend = OpenRouterQuestionAxisResponsivenessBackend(
        model=critic_model,
        temperature=0.0,
        reasoning_effort="medium",
        telemetry_path=output_prefix + ".task.telemetry.jsonl",
        telemetry_context={
            "pipeline": "adaptive_discovery_controller_v2",
            "stage": "graph_retraversal_task_preservation",
        },
    )
    task, stability = evaluate_hypothesis_task_preservation(
        question=new_context.question,
        hypothesis=candidate,
        backend=task_backend,
        debug_path_prefix=output_prefix + ".task",
    )
    task_ok = (
        task.decision_stable
        and task.task_class in {"DIRECT", "SUBORDINATE"}
    )
    if not task_ok:
        record["decision"] = "TASK_PRESERVATION_REJECTED"
        record["generated_hypothesis_id"] = str(candidate.hypothesis_id)
        record["task_preservation"] = task.task_class
        record["task_stability"] = stability.model_dump(mode="json")
        record["reason_codes"] = [
            "context_reset_lost_original_task"
        ]
        return None, record

    record.update(
        {
            "decision": "ACCEPTED_GENERATION_SHADOW",
            "generated_hypothesis_id": str(candidate.hypothesis_id),
            "generated_title": str(candidate.title),
            "used_new_premise_ids": sorted(
                premises.intersection(required_new)
            ),
            "task_preservation": task.task_class,
            "task_decision_stable": task.decision_stable,
            "task_stability": stability.model_dump(mode="json"),
            "reason_codes": [
                "new_grounded_context_used",
                "new_positive_premise_used",
                "task_preservation_passed",
                "external_prior_art_used_as_boundary_only",
            ],
        }
    )
    return compiled, record
