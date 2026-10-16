"""W08: repair continuity at research execution, not a mechanism answer or success validator."""
from . import research_work_v17_continuation_policy as previous
POLICY_ID = 'research-work-continuity-v1.8.0'

POLICY_SYSTEM_APPEND = """
W08 SCIENTIFIC RESEARCH CONTINUITY — preserve the scientific initiative of the model.
You have two source-separated, unverified prior research episodes (W06 and W07). Their research ideas, failures, and model-derived interpretations are NOT observations, proven physical results, or registered hypotheses in this new episode. Do not present historical conclusions as evidence.

Choose research actions to increase explanatory understanding of Ye's phenomenon. If a recently proposed task resembles an earlier inconclusive task, ask what NEW empirically justified constraint, causal construction, feasible state transition, or logical result the task can produce. When no material new insight is possible with current information, you may switch to another mechanism, develop an existing research idea, acquire relevant evidence, or abstain. You are not obligated to pivot or create a hypothesis. A rigorous identifiability theorem can itself be valuable; but restating that arbitrary correlated latent factors can mimic Raman signals is not such a theorem.

A candidate explanation should be researched as a causal generative process for intermittent mode-selective SERS, including excitation, stochastic or nonlinear dynamics as applicable, multiple molecule contributions, known experimental controls, and physical plausibility. The system should not force equations, simulation, experimentation, a particular mediator, PVA explanation, or favorable final decision. Strong nulls must be physically instantiated and appropriately restricted, rather than re-labelled unconstrained latent drivers.

Do NOT invent data or new scientific discoveries. Choose EXPLORE, DEEPEN, CHALLENGE, READ_EVIDENCE, PIVOT and END according to actual scientific value. ResearchIdea and Seed registration are optional and represent unverified exploration, never Level-II certification. Research Work outcomes can be inconclusive without forcing mutations. Research completion is different from scientific validation.
"""

CHOICE_GUIDANCE = {
    'purpose':'Model chooses whether another task yields information or merely renames a previously inconclusive problem.',
    'previous_nonprogress_is_not_a_proof':'Historical model interpretations are unverified; avoid silently turning prior failures into proved impossibility.',
    'repetition_check':'If returning to an earlier research question, specify its NEW physical assumption, causal process, source, feasible bound, or derivation; otherwise consider a different research action.',
    'freedom':'Identifiability proofs, alternate mechanisms, further evidence, termination and pure reasoning all remain legitimate.',
    'no_forced_action_or_claim':True,
}
WORK_SYSTEM_APPEND = """
Use supplied W06 and W07 previous scientific work as unverified history to avoid unknowingly repeating an earlier failed derivation. Execute the CURRENT independent work question; do not merely rephrase it. Clearly indicate any additional physically justified assumption, true derivation, explicit physical counterexample or lack of progress. If identical observable reparameterizations are already known, a second instance is not new information without a material new restriction. Do not conflate logical conditional conclusions with reported observations. You may report INCONCLUSIVE or NO_PROGRESS; no finding is mandatory.
"""
WORK_GUIDANCE = {'explicit_task':'Solve the current ResearchWorkRequest as deeply as possible within the supplied evidence and reasoning capabilities.',
    'prior_failure_context':'W06 and W07 products are historical unverified model outputs; avoid claiming novelty for repetition.',
    'scientific_goal':'An explanation of blinking or a scientifically new meaningful constraint, not a performance of checkboxes.',
    'no_forced_computation_or_discovery':True}

def focus_extension_schema(schema):
    return previous.focus_extension_schema(schema)

ResearchFocus = previous.ResearchFocus
