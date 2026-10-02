
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

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
from pipeline_core.llm.llm_telemetry import (
    run_instructor_structured_call,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


AdaptiveAction = Literal[
    "KEEP",
    "RETRIEVE_MORE",
    "SAME_PREMISE_SHARPEN",
    "EVIDENCE_REAXIS",
    "AXIS_MUTATION",
    "REQUEST_GRAPH_RETRAVERSAL",
    "STOP",
]


_ACTION_SCOPE = {
    "RETRIEVE_MORE": 0,
    "SAME_PREMISE_SHARPEN": 1,
    "EVIDENCE_REAXIS": 2,
    "AXIS_MUTATION": 3,
    "REQUEST_GRAPH_RETRAVERSAL": 4,
    "STOP": 5,
    "KEEP": -1,
}


class AdaptiveControllerAdvice(StrictModel):
    recommended_action: AdaptiveAction
    rationale: str = Field(min_length=1)
    expected_information_gain: str = Field(min_length=1)
    lower_scope_exhausted_reason: str | None = None


class AdaptiveControllerDecision(StrictModel):
    root_hypothesis_id: str
    current_hypothesis_id: str
    current_epistemic_state: str
    current_external_status: str | None = None

    action: AdaptiveAction
    allowed_actions: list[AdaptiveAction] = Field(min_length=1)
    action_scope_rank: int

    rationale: str
    expected_information_gain: str
    lower_scope_exhausted_reason: str | None = None

    safe_unused_premise_statement_ids: list[str] = Field(default_factory=list)
    target_claim_ids: list[str] = Field(default_factory=list)
    already_known_boundary: list[str] = Field(default_factory=list)
    unresolved_boundary: list[str] = Field(default_factory=list)

    prior_action_counts: dict[str, int] = Field(default_factory=dict)
    prior_attempt_count: int = 0

    llm_advisory_used: bool = False
    llm_advisory_accepted: bool = False
    deterministic_fallback_used: bool = False

    external_prior_art_as_positive_premise: Literal[False] = False
    controller_has_novelty_authority: Literal[False] = False
    controller_has_generation_authority: Literal[False] = False


class AdaptiveControllerPlan(StrictModel):
    schema_version: Literal[
        "adaptive-discovery-controller-plan-v1"
    ] = "adaptive-discovery-controller-plan-v1"

    plan_id: str
    source_portfolio_id: str
    round_index: int
    decisions: list[AdaptiveControllerDecision] = Field(default_factory=list)
    action_counts: dict[str, int] = Field(default_factory=dict)

    max_local_attempts_per_lineage: int
    llm_advisory_enabled: bool

    external_prior_art_as_positive_premise: Literal[False] = False
    controller_has_novelty_authority: Literal[False] = False
    controller_has_generation_authority: Literal[False] = False
    graph_retraversal_executes_in_v1: Literal[False] = False


def canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(canonical_json(x) for x in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def _norm(text: object) -> str:
    return " ".join(str(text or "").casefold().split())


def safe_premise_ids(
    context: HypothesisContext,
) -> list[str]:
    return sorted(
        str(row.statement_id)
        for row in context.evidence_statements
        if (
            row.eligible_as_premise
            and not row.requires_verification
            and not row.premise_restrictions
            and row.epistemic_role in {"reported", "evidence_synthesis"}
        )
    )


def safe_unused_premise_ids(
    context: HypothesisContext,
    card: Any,
) -> list[str]:
    used = set(map(str, card.premise_statement_ids))
    return [
        statement_id
        for statement_id in safe_premise_ids(context)
        if statement_id not in used
    ]


def external_boundaries(
    external_card: Mapping[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    known: list[str] = []
    unresolved: list[str] = []
    targets: list[str] = []

    for review in external_card.get("claim_reviews", []):
        status = str(review.get("status") or "")
        claim_id = str(review.get("claim_id") or "")
        claim_text = str(review.get("claim_text") or "")
        importance = str(review.get("importance") or "")

        base = f"{status}: {claim_text}".strip()
        titles = [
            str(row.get("title") or "").strip()
            for row in review.get("matches", [])[:2]
            if str(row.get("title") or "").strip()
        ]
        if titles:
            base += " | prior art: " + " ; ".join(titles)

        if status in {"DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"}:
            known.append(base)
            if claim_id and importance == "core":
                targets.append(claim_id)
        else:
            unresolved.append(base)

    if not targets:
        targets = [
            str(row.get("claim_id"))
            for row in external_card.get("claim_reviews", [])
            if (
                str(row.get("claim_id") or "")
                and str(row.get("status") or "")
                in {"DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"}
            )
        ]

    return known[:10], unresolved[:10], targets[:6]


def normalize_seed_action(route: str) -> AdaptiveAction | None:
    mapping: dict[str, AdaptiveAction] = {
        "KEEP_RESIDUAL": "KEEP",
        "SAME_PREMISE_GAP_SHARPEN": "SAME_PREMISE_SHARPEN",
        "FRESH_CONTEXT_REAXIS": "EVIDENCE_REAXIS",
        "HOLD_FOR_EVIDENCE": "RETRIEVE_MORE",
    }
    return mapping.get(str(route))


def _action_counts(
    history: Sequence[Mapping[str, Any]],
) -> Counter[str]:
    return Counter(
        str(row.get("action"))
        for row in history
        if str(row.get("action") or "")
    )


def _allowed_actions(
    *,
    epistemic_state: str,
    history: Sequence[Mapping[str, Any]],
    safe_unused: Sequence[str],
    target_claim_ids: Sequence[str],
    max_local_attempts: int,
) -> list[AdaptiveAction]:
    counts = _action_counts(history)
    attempts = sum(
        1
        for row in history
        if str(row.get("action") or "")
        not in {"", "KEEP", "STOP", "REQUEST_GRAPH_RETRAVERSAL"}
    )

    if epistemic_state == "RESIDUAL_AUTHORITY_CANDIDATE_SHADOW":
        return ["KEEP"]

    if attempts >= max_local_attempts:
        if epistemic_state in {
            "UNRESOLVED_EVIDENCE_GAP",
            "UNRESOLVED_TOPOLOGY_GAP",
        } and counts["REQUEST_GRAPH_RETRAVERSAL"] == 0:
            return ["REQUEST_GRAPH_RETRAVERSAL"]
        return ["STOP"]

    if epistemic_state == "UNRESOLVED_EVIDENCE_GAP":
        if counts["RETRIEVE_MORE"] == 0:
            return ["RETRIEVE_MORE"]
        if safe_unused and counts["EVIDENCE_REAXIS"] == 0:
            return ["EVIDENCE_REAXIS", "AXIS_MUTATION"]
        if counts["AXIS_MUTATION"] == 0:
            return ["AXIS_MUTATION", "REQUEST_GRAPH_RETRAVERSAL"]
        if counts["REQUEST_GRAPH_RETRAVERSAL"] == 0:
            return ["REQUEST_GRAPH_RETRAVERSAL"]
        return ["STOP"]

    if epistemic_state == "UNRESOLVED_TOPOLOGY_GAP":
        if safe_unused and counts["EVIDENCE_REAXIS"] == 0:
            options: list[AdaptiveAction] = ["EVIDENCE_REAXIS"]
            if target_claim_ids and counts["SAME_PREMISE_SHARPEN"] == 0:
                options.append("SAME_PREMISE_SHARPEN")
            return options
        if target_claim_ids and counts["SAME_PREMISE_SHARPEN"] == 0:
            return ["SAME_PREMISE_SHARPEN", "AXIS_MUTATION"]
        if counts["AXIS_MUTATION"] == 0:
            return ["AXIS_MUTATION", "REQUEST_GRAPH_RETRAVERSAL"]
        if counts["REQUEST_GRAPH_RETRAVERSAL"] == 0:
            return ["REQUEST_GRAPH_RETRAVERSAL"]
        return ["STOP"]

    if epistemic_state == "PRIOR_ART_BACKED_OR_NO_RESIDUAL":
        if target_claim_ids and counts["SAME_PREMISE_SHARPEN"] == 0:
            options = ["SAME_PREMISE_SHARPEN"]
            if safe_unused and counts["EVIDENCE_REAXIS"] == 0:
                options.append("EVIDENCE_REAXIS")
            return options
        if safe_unused and counts["EVIDENCE_REAXIS"] == 0:
            return ["EVIDENCE_REAXIS", "AXIS_MUTATION"]
        if counts["AXIS_MUTATION"] == 0:
            return ["AXIS_MUTATION"]
        return ["STOP"]

    if counts["RETRIEVE_MORE"] == 0:
        return ["RETRIEVE_MORE", "STOP"]
    return ["STOP"]


class InstructorOpenAICompatibleAdaptiveControllerBackend:
    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str | None = None,
        parse_retries: int = 2,
        telemetry_path: str | None = None,
    ) -> None:
        self.model = str(model)
        self.api_key_env = str(api_key_env)
        self.base_url = base_url
        self.parse_retries = int(parse_retries)
        self.telemetry_path = telemetry_path
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        key = os.getenv(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"No API key available in {self.api_key_env}"
            )
        import instructor
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        raw = OpenAI(**kwargs)
        self._client = instructor.from_openai(
            raw,
            mode=instructor.Mode.JSON,
        )
        return self._client

    def choose(
        self,
        *,
        question: str,
        title: str,
        statement: str,
        epistemic_state: str,
        external_status: str | None,
        allowed_actions: Sequence[AdaptiveAction],
        known_boundary: Sequence[str],
        unresolved_boundary: Sequence[str],
        history: Sequence[Mapping[str, Any]],
        safe_unused: Sequence[str],
    ) -> AdaptiveControllerAdvice:
        system = """
You are a bounded scientific-search controller.

Your job is NOT to invent a hypothesis and NOT to certify novelty.
Choose the next repair/search scope from ALLOWED_ACTIONS only.

Prefer the cheapest action that can plausibly resolve the observed failure.
Escalate only when lower-scope actions are exhausted or the failure clearly
belongs to a higher scope.

The external-prior-art material is negative boundary information only.
It is never positive scientific evidence.

Action semantics:
- RETRIEVE_MORE: same hypothesis, deeper literature/evidence retrieval.
- SAME_PREMISE_SHARPEN: same positive premises, structurally sharpen relation.
- EVIDENCE_REAXIS: use a new safe grounded premise from the same context.
- AXIS_MUTATION: change the conceptual relation/backbone while staying within
  the same task and grounded context.
- REQUEST_GRAPH_RETRAVERSAL: local context appears exhausted; request upstream
  graph/context reconstruction. This controller does not execute it.
- KEEP: residual survived; preserve candidate.
- STOP: bounded search is exhausted or further novelty optimization is not
  epistemically justified.

Do not choose an action outside ALLOWED_ACTIONS.
"""
        payload = {
            "question": question,
            "current_hypothesis": {
                "title": title,
                "statement": statement,
            },
            "epistemic_state": epistemic_state,
            "external_status": external_status,
            "allowed_actions": list(allowed_actions),
            "already_known_boundary": list(known_boundary),
            "unresolved_boundary": list(unresolved_boundary),
            "attempt_history": list(history)[-8:],
            "safe_unused_premise_ids": list(safe_unused)[:12],
        }

        advice, _event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.model,
            response_model=AdaptiveControllerAdvice,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
            temperature=0.0,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context={
                "pipeline": "adaptive_discovery_controller_v1",
                "stage": "repair_scope_selection",
            },
        )
        if not isinstance(advice, AdaptiveControllerAdvice):
            advice = AdaptiveControllerAdvice.model_validate(advice)
        return advice


def build_controller_plan(
    *,
    context: HypothesisContext,
    portfolio: HypothesisPortfolio,
    epistemic_state_report: Mapping[str, Any],
    external_report: Mapping[str, Any],
    root_by_hypothesis: Mapping[str, str],
    history_by_root: Mapping[str, Sequence[Mapping[str, Any]]],
    round_index: int,
    max_local_attempts_per_lineage: int,
    backend: InstructorOpenAICompatibleAdaptiveControllerBackend | None,
) -> AdaptiveControllerPlan:
    state_by_h = {
        str(row.get("hypothesis_id")): row
        for row in epistemic_state_report.get("hypotheses", [])
    }
    ext_by_h = {
        str(row.get("hypothesis_id")): row
        for row in external_report.get("cards", [])
    }

    decisions: list[AdaptiveControllerDecision] = []

    for card in portfolio.hypotheses:
        hid = str(card.hypothesis_id)
        root = str(root_by_hypothesis.get(hid, hid))
        state_row = state_by_h.get(hid, {})
        ext = ext_by_h.get(hid, {})

        state = str(
            state_row.get("final_epistemic_state")
            or "UNRESOLVED_EVIDENCE_GAP"
        )
        known, unresolved, target_claim_ids = external_boundaries(ext)
        safe_unused = safe_unused_premise_ids(context, card)
        history = list(history_by_root.get(root, []))
        counts = _action_counts(history)

        allowed = _allowed_actions(
            epistemic_state=state,
            history=history,
            safe_unused=safe_unused,
            target_claim_ids=target_claim_ids,
            max_local_attempts=max_local_attempts_per_lineage,
        )

        chosen = allowed[0]
        rationale = (
            "Deterministic bounded escalation selected the lowest admissible "
            "repair scope for the current failure state."
        )
        expected = "Resolve the current failure without unnecessary search expansion."
        lower_reason = None
        advisory_used = False
        advisory_accepted = False
        fallback = False

        if backend is not None and len(allowed) > 1:
            advisory_used = True
            try:
                advice = backend.choose(
                    question=context.question,
                    title=str(card.title),
                    statement=str(card.hypothesis_statement),
                    epistemic_state=state,
                    external_status=(
                        str(ext.get("status"))
                        if ext.get("status") is not None
                        else None
                    ),
                    allowed_actions=allowed,
                    known_boundary=known,
                    unresolved_boundary=unresolved,
                    history=history,
                    safe_unused=safe_unused,
                )
                if advice.recommended_action in allowed:
                    chosen = advice.recommended_action
                    rationale = advice.rationale
                    expected = advice.expected_information_gain
                    lower_reason = advice.lower_scope_exhausted_reason
                    advisory_accepted = True
                else:
                    fallback = True
            except Exception as exc:
                fallback = True
                rationale += (
                    " Controller LLM advisory failed closed: "
                    + type(exc).__name__
                )

        decisions.append(
            AdaptiveControllerDecision(
                root_hypothesis_id=root,
                current_hypothesis_id=hid,
                current_epistemic_state=state,
                current_external_status=(
                    str(ext.get("status"))
                    if ext.get("status") is not None
                    else None
                ),
                action=chosen,
                allowed_actions=list(allowed),
                action_scope_rank=_ACTION_SCOPE[chosen],
                rationale=rationale,
                expected_information_gain=expected,
                lower_scope_exhausted_reason=lower_reason,
                safe_unused_premise_statement_ids=safe_unused,
                target_claim_ids=target_claim_ids,
                already_known_boundary=known,
                unresolved_boundary=unresolved,
                prior_action_counts=dict(sorted(counts.items())),
                prior_attempt_count=len(history),
                llm_advisory_used=advisory_used,
                llm_advisory_accepted=advisory_accepted,
                deterministic_fallback_used=fallback,
            )
        )

    counts = Counter(row.action for row in decisions)
    provisional = {
        "source_portfolio_id": portfolio.portfolio_id,
        "round_index": round_index,
        "decisions": [
            row.model_dump(mode="json")
            for row in decisions
        ],
    }
    return AdaptiveControllerPlan(
        plan_id=stable_id(
            "adaptive_discovery_controller_plan",
            provisional,
        ),
        source_portfolio_id=portfolio.portfolio_id,
        round_index=round_index,
        decisions=decisions,
        action_counts=dict(sorted(counts.items())),
        max_local_attempts_per_lineage=max_local_attempts_per_lineage,
        llm_advisory_enabled=backend is not None,
    )


def _prompt_sha(version: str, system: str, user: str) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "prompt_version": version,
                "system_prompt": system,
                "user_prompt": user,
            }
        ).encode("utf-8")
    ).hexdigest()


class AdaptiveAxisMutationPromptAssembler:
    prompt_version = "adaptive-axis-mutation-prompt-v1"

    def __init__(
        self,
        *,
        original: Any,
        decision: AdaptiveControllerDecision,
        allowed_premise_ids: Sequence[str],
        attempt_history: Sequence[Mapping[str, Any]],
    ) -> None:
        self.original = original
        self.decision = decision
        self.allowed_premise_ids = list(
            dict.fromkeys(map(str, allowed_premise_ids))
        )
        self.attempt_history = list(attempt_history)[-8:]

    def build(self, context: HypothesisContext) -> HypothesisPrompt:
        base = HypothesisPromptAssembler(max_hypotheses=1).build(context)

        system = base.system_prompt + """

ADAPTIVE DISCOVERY — AXIS MUTATION
==================================
The previous bounded search attempts have reached an axis-level failure.

Generate exactly ONE scientifically different, falsifiable hypothesis within
the SAME research question and SAME grounded HypothesisContext, or abstain.

This is stronger than wording repair and stronger than evidence re-axis.
Change the conceptual relation/backbone itself.

Rules:
1. Positive scientific premises must come ONLY from ALLOWED GROUNDED PREMISE IDS.
2. External prior art and attempt history are negative search-boundary
   information only. Never use them as positive premises.
3. Do not reproduce a previously rejected known relation with cosmetic
   qualifiers.
4. The new candidate should alter the relation topology, dependency structure,
   mechanism arrangement, boundary/regime logic, competing explanation, or
   another scientifically meaningful axis.
5. Stay DIRECT or SUBORDINATE to the original task.
6. Do not claim literature-wide novelty or absence.
7. Return exactly ONE hypothesis or abstain.
8. Falsifiers must test the mutated axis itself.
"""
        user = base.user_prompt + "\n\n" + "\n".join(
            [
                "AXIS-MUTATION SEARCH STATE",
                "==========================",
                f"original_hypothesis_id: {self.original.hypothesis_id}",
                f"original_title: {self.original.title}",
                f"original_statement: {self.original.hypothesis_statement}",
                "",
                "ALREADY-KNOWN / EXCLUDED REGION",
                "================================",
                *(
                    [
                        f"- {x}"
                        for x in self.decision.already_known_boundary
                    ]
                    or ["- NONE"]
                ),
                "",
                "UNRESOLVED REGION",
                "=================",
                *(
                    [
                        f"- {x}"
                        for x in self.decision.unresolved_boundary
                    ]
                    or ["- NONE"]
                ),
                "",
                "PRIOR ATTEMPT HISTORY",
                "=====================",
                json.dumps(
                    self.attempt_history,
                    ensure_ascii=False,
                    indent=2,
                ),
                "",
                "ALLOWED GROUNDED PREMISE IDS",
                "============================",
                *[
                    f"- {x}"
                    for x in self.allowed_premise_ids
                ],
                "",
                "TASK",
                "====",
                (
                    "Generate one different relation/backbone axis that remains "
                    "grounded in the supplied context and task, or abstain."
                ),
                (
                    "External prior art is boundary-only and must not appear as "
                    "positive premise_statement_ids."
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


def _compile(
    *,
    context: HypothesisContext,
    draft: HypothesisPortfolioDraft,
) -> tuple[HypothesisPortfolio | None, list[str]]:
    try:
        portfolio = HypothesisCompiler().compile(context, draft)
    except HypothesisCompileError as exc:
        return None, [
            f"{x.code}:{x.location}:{x.message}"
            for x in exc.issues
        ]
    except Exception as exc:
        return None, [f"{type(exc).__name__}:{exc}"]

    validation = HypothesisValidator().validate(
        context,
        portfolio,
    )
    if not validation.passes:
        return None, [
            f"{x.code}:{x.location}:{x.message}"
            for x in validation.issues
            if x.severity == "error"
        ]
    return portfolio, []


def generate_axis_mutation(
    *,
    context: HypothesisContext,
    original: Any,
    decision: AdaptiveControllerDecision,
    attempt_history: Sequence[Mapping[str, Any]],
    model: str,
    critic_model: str,
    api_key_env: str,
    base_url: str | None,
    output_prefix: str,
) -> tuple[Any | None, dict[str, Any]]:
    allowed = safe_premise_ids(context)
    assembler = AdaptiveAxisMutationPromptAssembler(
        original=original,
        decision=decision,
        allowed_premise_ids=allowed,
        attempt_history=attempt_history,
    )
    prompt = assembler.build(context)

    backend = InstructorOpenAICompatibleHypothesisBackend(
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
        parse_retries=3,
        telemetry_path=output_prefix + ".telemetry.jsonl",
        telemetry_context={
            "pipeline": "adaptive_discovery_controller_v1",
            "stage": "axis_mutation",
            "source_hypothesis_id": str(original.hypothesis_id),
        },
    )
    generation = backend.generate(prompt)
    draft = generation.draft

    record: dict[str, Any] = {
        "source_hypothesis_id": str(original.hypothesis_id),
        "route": "AXIS_MUTATION",
        "decision": None,
        "generated_hypothesis_id": None,
        "reason_codes": [],
    }

    if not draft.hypotheses:
        record["decision"] = "ABSTAINED"
        record["reason_codes"] = ["model_abstained"]
        return None, record

    if len(draft.hypotheses) != 1:
        record["decision"] = "REJECTED_CARDINALITY"
        record["reason_codes"] = [
            "axis_mutation_requires_exactly_one_hypothesis"
        ]
        return None, record

    compiled, issues = _compile(
        context=context,
        draft=draft,
    )
    if compiled is None:
        feedback = "\n".join(
            [
                "ADAPTIVE AXIS-MUTATION REPAIR",
                "Repair the single candidate without returning to a prior known axis.",
                "Use only allowed grounded premise IDs.",
                "External prior art is exclusion-boundary information only.",
                "Issues:",
                *[f"- {x}" for x in issues],
                "Return exactly one corrected hypothesis or abstain.",
            ]
        )
        repaired = backend.repair(
            prompt,
            draft,
            feedback,
        ).draft
        compiled, issues = _compile(
            context=context,
            draft=repaired,
        )

    if compiled is None:
        record["decision"] = "COMPILE_OR_VALIDATION_REJECTED"
        record["reason_codes"] = issues[:12]
        return None, record

    candidate = compiled.hypotheses[0]
    candidate_premises = set(map(str, candidate.premise_statement_ids))
    if not candidate_premises.issubset(set(allowed)):
        record["decision"] = "GROUNDING_DRIFT_REJECTED"
        record["generated_hypothesis_id"] = str(candidate.hypothesis_id)
        record["reason_codes"] = [
            "axis_mutation_used_noneligible_positive_premise"
        ]
        return None, record

    if (
        _norm(candidate.hypothesis_statement)
        == _norm(original.hypothesis_statement)
    ):
        record["decision"] = "UNCHANGED_AXIS_REJECTED"
        record["generated_hypothesis_id"] = str(candidate.hypothesis_id)
        record["reason_codes"] = [
            "axis_mutation_reproduced_original_statement"
        ]
        return None, record

    task_backend = OpenRouterQuestionAxisResponsivenessBackend(
        model=critic_model,
        temperature=0.0,
        reasoning_effort="medium",
        telemetry_path=output_prefix + ".task.telemetry.jsonl",
        telemetry_context={
            "pipeline": "adaptive_discovery_controller_v1",
            "stage": "axis_mutation_task_preservation",
        },
    )
    task, stability = evaluate_hypothesis_task_preservation(
        question=context.question,
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
            "axis_mutation_lost_original_task"
        ]
        return None, record

    record.update(
        {
            "decision": "ACCEPTED_GENERATION_SHADOW",
            "generated_hypothesis_id": str(candidate.hypothesis_id),
            "generated_title": str(candidate.title),
            "task_preservation": task.task_class,
            "task_decision_stable": task.decision_stable,
            "task_source_decision_stable": task.source_decision_stable,
            "task_stability": stability.model_dump(mode="json"),
            "reason_codes": [
                "axis_mutation_compiled_and_validated",
                "task_preservation_passed",
                "external_prior_art_used_as_boundary_only",
            ],
        }
    )
    return candidate, record
