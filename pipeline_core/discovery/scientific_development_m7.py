"""M7 opt-in research-program development. No scientific/novelty authority.

Domain-general drafting, adversarial feedback, and bounded scientific evolution.
Domain-specific *illustrations* reside in m7_sers_null_models and are never
positive evidence for a new HypothesisCard.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Any

ROLES = ("REPAIR_PREDICTION", "DESIGN_DECISIVE_EXPERIMENT", "OPEN_ALTERNATE_BRANCH")


def canonical(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def digest(x: Any) -> str:
    return sha256(canonical(x).encode("utf-8")).hexdigest()


@dataclass
class ScientificProgramDraft:
    """Unverified research program; deliberately not a HypothesisCard."""
    idea_label: str
    parent_idea_id: str
    source_role: str
    research_question: str
    mechanism_model: str
    competing_null_model: str
    formal_prediction: str
    predicted_contrast: str
    manipulated_variables: list[str]
    independent_observables: list[str]
    decisive_experiment: str
    identifying_assumptions: list[str]
    nuisance_processes: list[str]
    falsification_scope: str
    predicted_failure_modes: list[str]
    alternative_branch_if_failed: str
    scientific_value_even_if_known: str
    model_equations: list[str] = field(default_factory=list)
    measurement_feasibility_caveats: list[str] = field(default_factory=list)
    source_of_draft: str = "MODEL_SPECULATIVE"
    novelty_certified: bool = False
    scientific_truth_authority: bool = False
    evidence_claim_authority: bool = False

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "ScientificProgramDraft":
        if not isinstance(raw, dict):
            raise ValueError("program must be object")
        # Strictly prevent hallucinated authority flags and unsupported schema keys.
        fields = set(cls.__dataclass_fields__)
        unknown = set(raw) - fields
        if unknown:
            raise ValueError(f"unknown scientific program keys: {sorted(unknown)}")
        program = cls(**raw)
        if (program.novelty_certified or program.scientific_truth_authority
                or program.evidence_claim_authority):
            raise ValueError("scientific authority cannot be granted by generated output")
        for key in ("idea_label", "parent_idea_id", "source_role", "research_question",
                    "mechanism_model", "competing_null_model", "formal_prediction",
                    "predicted_contrast", "decisive_experiment", "falsification_scope",
                    "alternative_branch_if_failed"):
            if not isinstance(getattr(program, key), str) or not getattr(program, key).strip():
                raise ValueError(f"missing nonempty {key}")
        if program.source_role not in ROLES:
            raise ValueError("unknown source role")
        for key in ("manipulated_variables", "independent_observables",
                    "identifying_assumptions", "nuisance_processes",
                    "predicted_failure_modes", "model_equations", "measurement_feasibility_caveats"):
            values = getattr(program, key)
            if not isinstance(values, list) or any(not isinstance(x, str) or not x.strip() for x in values):
                raise ValueError(f"invalid list {key}")
        if not program.manipulated_variables or not program.independent_observables:
            raise ValueError("intervention and observable required")
        return program

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_development_tasks(m64: dict, m67: dict, *, limit: int = 6) -> dict:
    """Create an intentionally reserve-inclusive cohort, frozen from actual M6.4 drafts.

    This is a selected *experiment*, not the official SIS scheduler.
    """
    if m64.get("candidate_count") != 6 or m64.get("completed_valid_responses") != 6:
        raise ValueError("expected the six actual M6.4 completed role drafts")
    if m67.get("status") != "M67_EXPLICIT_CURATED_NATIVE_G5_SHADOW_REPLAY_NO_PROVIDER_CALLS":
        raise ValueError("expected a successful nonofficial M6.7 shadow")
    if m67.get("native_raw_children") != 4 or m67.get("native_semantic_dispositions") != {"GENUINE_CHILD": 4}:
        raise ValueError("M6.7 native replay not successful")
    if m67.get("official_SIS_scheduler_executed") is not False or m67.get("new_provider_api_calls") != 0:
        raise ValueError("official authority/provider call mismatch")
    if len(m64.get("rows", [])) != 6:
        raise ValueError("M6.4 cohort count mismatch")
    selected = {r["label"] for r in m67["selected_existing_m64_candidates"]}
    reserved = {r["label"] for r in m67["unselected_reserved_m64_candidates"]}
    if selected & reserved or len(selected) != 4 or len(reserved) != 2:
        raise ValueError("selected/reserve partition mismatch")
    tasks = []
    for row in m64["rows"]:
        label, role, parent = row["label"], row["role"], row["parent_idea_id"]
        if label not in selected | reserved or role not in ROLES:
            raise ValueError("unexpected M6.4 label/role")
        if row.get("candidate_count") != 1 or len(row.get("candidates", [])) != 1:
            raise ValueError("missing completed M6.4 candidate")
        cand = row["candidates"][0]
        tasks.append({
            "label": label, "parent_idea_id": parent, "role": role,
            "population_lane": "G5_SHADOW_CHILD" if label in selected else "RESERVE_RESEARCH_IDEA",
            "kernel_sha": digest(cand["kernel"]),
            "original_scientific_question": cand["kernel"].get("question_commitment"),
            "original_conceptual_change": cand["conceptual_change_summary"],
            "original_differential_prediction": cand["differential_prediction"],
            "original_discriminating_observation": cand["discriminating_observation"],
            "original_falsification_condition": cand["falsification_condition"],
            "original_kernel": cand["kernel"],
        })
    if limit < 1 or limit > 6:
        raise ValueError("limit must be 1..6")
    # Favor different scientific tasks; explicitly keep P2 Repair reserve.
    priority = ["P2_REPAIR_PREDICTION", "P2_DESIGN_DECISIVE_EXPERIMENT",
                "P2_OPEN_ALTERNATE_BRANCH", "P1_REPAIR_PREDICTION",
                "P1_DESIGN_DECISIVE_EXPERIMENT", "P1_OPEN_ALTERNATE_BRANCH"]
    task_by_label = {t["label"]: t for t in tasks}
    if set(task_by_label) != set(priority):
        raise ValueError("missing frozen roles")
    ordered = [task_by_label[label] for label in priority[:limit]]
    return {
        "schema_version": "m7-scientific-development-plan-v1",
        "status": "FROZEN_RESEARCH_PROGRAM_DEVELOPMENT_PLAN",
        "tasks": ordered,
        "source_m64_canonical_sha": digest(m64),
        "source_m67_canonical_sha": digest(m67),
        "official_scheduler_executed": False,
        "grounding_controls_claims_not_imagination": True,
        "novelty_certified": False,
        "new_model_calls": 0,
    }


def create_model_prompt(task: dict[str, Any], prior_feedback: dict | None = None) -> list[dict[str, str]]:
    system = """You are a scientific research-program development agent, NOT a scientific-truth certifier.
Develop a real mechanistic and quantitative research program from a ResearchIdea. Compare an explicit competing null model and propose an independent discriminating observable/intervention.
Keep conceptual imagination open, including mechanisms beyond any KG. Scientific claims, citations, novelty and evidence provenance must not be invented.
Distinguish mathematical predictions that FOLLOW under stated assumptions from hypotheses that are merely testable.
Never pretend a new quantitative measurement has occurred. Never silently assume independent control of coupled variables.
When a proposed observable is confounded, identify an independent measurement and how it could fail.
Your answer must be a single JSON object following the schema and keys shown in the user prompt. No markdown."""
    # Ordinary JSON object schema; validates strictly at output boundary.
    example = ScientificProgramDraft(
        idea_label=task["label"], parent_idea_id=task["parent_idea_id"], source_role=task["role"],
        research_question="...", mechanism_model="...", competing_null_model="...",
        formal_prediction="...", predicted_contrast="...", manipulated_variables=["..."],
        independent_observables=["..."], decisive_experiment="...", identifying_assumptions=["..."],
        nuisance_processes=["..."], falsification_scope="...", predicted_failure_modes=["..."],
        alternative_branch_if_failed="...", scientific_value_even_if_known="...",
        model_equations=["..."], measurement_feasibility_caveats=["..."],
    ).to_dict()
    user = {
        "source_research_idea_search_context_only": task,
        "prior_scientific_feedback_search_context_only": prior_feedback,
        "task": "Formulate a specific, bounded, potentially falsifiable research program. Offer equations or falsifiable direction-of-effect claims with explicit assumptions; if quantitative parameters unknown, give symbolic relations and identify what must be measured. Contrast against at least one substantive known-physics null and one nuisance alternative. Include a plan to distinguish coupled optical/chemical/kinetic confounders when applicable. Explain scientific value even if novelty remains unverified. Preserve original meaningful differentiator unless explicitly state why replaced.",
        "return_exact_json_keys_and_types": example,
        "authority_fields_must_be_false": ["novelty_certified", "scientific_truth_authority", "evidence_claim_authority"],
    }
    return [{"role":"system","content":system},{"role":"user","content":json.dumps(user,ensure_ascii=False,indent=2)}]


def challenge_program(program: ScientificProgramDraft) -> dict[str, Any]:
    """Non-authoritative but actionable development feedback (not science certification)."""
    reasons: list[str] = []
    if not program.model_equations:
        reasons.append("MISSING_FORMAL_EQUATIONS_SPECIFY_SYMBOLIC_NULL_AND_ALTERNATIVE")
    if len(program.identifying_assumptions) < 2:
        reasons.append("UNDEREXPLICIT_IDENTIFICATION_ASSUMPTIONS")
    if len(program.nuisance_processes) < 2:
        reasons.append("UNDEREXPLICIT_CONFOUNDING_OR_INSTRUMENTAL_NUISANCES")
    if not program.measurement_feasibility_caveats:
        reasons.append("MISSING_MEASUREMENT_FEASIBILITY_BOUNDARY")
    if len(program.manipulated_variables) < 2:
        reasons.append("ONE_DIMENSION_INTERVENTION_MAY_NOT_SEPARATE_CAUSAL_EXPLANATIONS")
    if len(program.independent_observables) < 2:
        reasons.append("SINGLE_OBSERVABLE_MAY_NOT_IDENTIFY_LATENT_MECHANISM")
    if not program.predicted_failure_modes:
        reasons.append("NO_KNOWN_NULL_OR_ABLATION_FAILURE_CASE")
    return {
        "label": program.idea_label,
        "status": "DEVELOPED_SPECULATIVE_PROGRAM_NOT_SCIENCE_CERTIFIED",
        "review_required_codes": reasons,
        "generation_feedback_for_next_iteration": [
            "REPAIR: explicitly target " + code for code in reasons
        ] or ["EXPLORE: choose an independent model family, hold out a measurable domain boundary"],
        "next_generation_research_prompt": (
            "Using the following unverified research program and known failure risks, propose a scientifically distinct child or decisive measurement. "
            "No novelty or truth claims. Program=" + canonical(program.to_dict()) + " Issues=" + canonical(reasons)
        ),
        "claim_authority": False,
    }


def retention_brief(program: ScientificProgramDraft, task: dict) -> dict[str, Any]:
    """Reminder for strict realization. Never used as positive evidence or an instruction to break compiler rules."""
    return {
        "label": program.idea_label,
        "parent_idea_id": program.parent_idea_id,
        "speculative_differentiator": program.predicted_contrast,
        "original_prediction_search_only": task["original_differential_prediction"],
        "independent_observables_search_only": program.independent_observables,
        "grounded_realization_policy": "Preserve distinguishing science only where supported by eligible premises; otherwise abstain, or retain as separate speculative ResearchIdea. Never invent evidence, numbers or protocol and never downgrade silently.",
        "not_positive_evidence": True,
        "not_claim_authority": True,
    }
