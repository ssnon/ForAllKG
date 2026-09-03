from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any

from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtReviewDraft,
    NoveltyClaim,
    NoveltyClaimDecompositionDraft,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisCard
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call
from pipeline_core.discovery.prior_art_review_audit import (
    record_prior_art_review_call,
)


@dataclass(frozen=True)
class ExternalNoveltyPromptRecord:
    name: str
    system_prompt: str
    user_prompt: str
    prompt_sha256: str


_DECOMPOSE_SYSTEM = """You decompose a generated scientific hypothesis into claim-level novelty assertions for prior-art search.

This is NOT a novelty judgment and NOT a truth judgment. Do not say that anything is novel, unprecedented, first, or unknown.

Identify the smallest scientifically meaningful claims that would make the hypothesis distinct if supported. Prefer claims about a mediator, moderator/interaction, context condition, pathway competition, descriptor interaction, mechanistic link, or a distinctive falsifiable prediction.

For each claim provide plain-text literature search concepts and 2-3 plain-text search queries. Search queries must not use Boolean operators or special search syntax, and should avoid hyphenated terms where possible. Keep them suitable for Semantic Scholar and Crossref relevance search.

QUERY-DIVERSITY RULE:
- Do not spend every query restating the full higher-order claim.
- For moderator_interaction, descriptor_interaction, or another higher-order conditional claim, one retained query should target the full relation and another should probe a scientifically meaningful LOWER-ORDER RELATION by relaxing one moderator, condition, or interaction term.
- A lower-order query must still express a relation among multiple scientific variables. Do not collapse it into a generic single-variable background query.
- For an explicitly ordered or directional prediction, retain diagnostic metadata for a possible BOUNDARY OR COUNTEREVIDENCE search in neutral language: dependence, comparison, weak or absent correlation, regime dependence, competing determinants, or alternative dominant factors.
- Ordinary search_queries are the first-pass retrieval queries and are independent of the diagnostic candidate metadata below. Do not sacrifice an ordinary first-pass query slot merely to execute the diagnostic candidate.
- Never insert a known paper title, DOI, author, year, or literature conclusion into a search query unless that information was already present in the supplied hypothesis.

STRUCTURED DIAGNOSTIC QUERY CONTRACT:
- Always set diagnostic_query_kind explicitly.
- If the claim contains an explicit ordered prediction such as greater, smaller, stronger, weaker, increase, decrease, higher, lower, or an explicitly ordered ratio/contrast, set diagnostic_query_kind=DIRECTIONAL_BOUNDARY and provide diagnostic_search_query as a neutral neighboring-scope search for dependence, comparison, competing determinants, regime dependence, weak correlation, or alternative dominant factors. Do not repeat the claimed direction as if it were true.
- Otherwise, for moderator_interaction, descriptor_interaction, or another higher-order conditional relation, set diagnostic_query_kind=LOWER_ORDER_RELATION and provide diagnostic_search_query for one scientifically meaningful lower-order relation obtained by relaxing one moderator/condition while retaining a multi-variable relation.
- Otherwise set diagnostic_query_kind=NONE and diagnostic_search_query=null.
- diagnostic_search_query is separate from ordinary search_queries. The deterministic decomposer preserves it as diagnostic metadata and does not automatically insert it into first-pass search_queries.
RELATION-FIRST DIAGNOSTIC ASSEMBLY CONTRACT:
- For every non-NONE diagnostic query, provide diagnostic_structural_terms and diagnostic_relation_terms in addition to the human-readable diagnostic_search_query.
- diagnostic_structural_terms contain only the minimum structural or topological carriers needed to preserve the scientific system class being probed, such as a dimer, junction, interface, film, catalyst site, cell type, or other architecture when scientifically necessary.
- Do not include exact material identity, species identity, brand/platform identity, or incidental system-specific qualifiers in diagnostic_structural_terms unless removing them would destroy the scientific relation being tested.
- diagnostic_relation_terms contain the variables, observable/outcome, mechanism descriptor, comparison/dependence concept, and relation vocabulary needed to search the lower-order or boundary relation.
- For LOWER_ORDER_RELATION, remove the higher-order moderator or interaction that is intentionally being relaxed, but preserve the lower-order relation and its structural carrier.
- For DIRECTIONAL_BOUNDARY, express the relation neutrally. Preserve the relevant structural carrier while omitting the proposed ordered conclusion itself when possible; include dependence, comparison, competing determinants, regime dependence, weak correlation, or alternative dominant factors as appropriate.
- Keep diagnostic_structural_terms and diagnostic_relation_terms concise and retrieval-oriented rather than sentence-like.
- Do not provide paper titles, DOI values, authors, years, or literature conclusions unless they were already present in the hypothesis.
- The deterministic decomposer may assemble a diagnostic execution candidate by concatenating normalized structural terms followed by normalized relation terms. This candidate is preserved for provenance or a future bounded diagnostic pass and is not automatically inserted into first-pass search_queries. It will not invent synonyms or add scientific concepts.

CLAIM IMPORTANCE CONTRACT:
- Every decomposition MUST contain at least one importance=core claim.
- A core claim is an atomic scientific relation whose direct prior-art saturation, routine reconstruction, or failure would materially undermine the hypothesis's claimed scientific distinctiveness.
- If the hypothesis contains multiple independent novelty-bearing branches, mark EACH scientifically central branch as core. One apparently stronger branch must not hide another central branch.
- Use importance=supporting only for genuinely auxiliary claims whose prior-art status does not by itself determine whether the hypothesis remains scientifically distinctive.
- A distinctive_prediction may be core when that prediction is itself the central higher-order scientific proposition.
- Generic background relations, already-assumed lower-order components, or explanatory context should normally be supporting or omitted rather than promoted to core.
- Do not mark every claim supporting merely to avoid committing to which relation carries the hypothesis's distinctiveness.
- importance is a hypothesis-level selection role. It is NOT a novelty verdict and does not imply that the claim is new.

CLAIM ATOMICITY / BRANCH-SPLITTING CONTRACT:
- Each returned claim must contain one novelty-bearing scientific relation nucleus.
- If the hypothesis coordinates scientifically separable alternatives that could have different prior-art status, emit them as separate claims.
- In particular, independent moderators such as "excitation wavelength or laser power" must not be hidden inside one umbrella claim such as "laser excitation conditions"; emit one claim for wavelength moderation and one claim for power moderation.
- Apply the same rule to independent mechanisms, contexts, regimes, or predictions joined by "or", "and/or", or an equivalent coordination when each branch could independently be established or contradicted by prior art.
- Do NOT split alternative states or values of the same scientific variable merely because they are contrasted. For example, presence versus absence of one moderator, or higher versus lower temperature, may remain one claim when they define a single comparison.
- If the maximum claim count forces prioritization, preserve separate core novelty-bearing branches before generic background or explanatory claims.
- A hypothesis-level umbrella statement may be described in decomposition_notes, but it must not replace the atomic branch claims used for prior-art assessment.
- The claim text itself must contain ONE relation nucleus. Do not append a second mechanistic assertion using "because", "through", "by", "via", "consistent with", or an equivalent causal/explanatory tail when that assertion could have a different prior-art status.
- Put explanatory scientific context in rationale when it only explains why the claim is plausible.
- If a proposed mediator or mechanism is itself a novelty-bearing assertion that requires independent prior-art assessment, emit it as a separate mediator or mechanistic_link claim.
- Example: prefer claim text "Excitation wavelength moderates the dependence of SERS enhancement on interparticle spacing." Do not write "Excitation wavelength moderates ... because spacing-dependent plasmon coupling changes resonance overlap" as one claim.
- Likewise, prefer a separate claim "Laser power moderates the dependence of SERS enhancement on interparticle spacing." Any proposed power-dependent heating, geometry change, localized-field sampling, or resonance mechanism belongs in rationale or in its own independently reviewable mechanistic claim.

PRIOR-ART IDENTITY / RELATION-NUCLEUS CONTRACT:
- Populate prior_art_identity_terms and relation_nucleus_terms for every novelty-bearing claim whenever the scientific relation permits it.
- prior_art_identity_terms identify the minimum scientific factor(s) that determine which prior-art family should be eligible for cross-claim re-exposure. They are NOT the complete novelty claim.
- Example: for wavelength-conditioned spacing-to-SERS claims, use "excitation wavelength"; for power-conditioned claims, use "laser power".
- relation_nucleus_terms represent the underlying scientific relation being compared across claims: the principal input/descriptor, outcome, and dependence/interaction vocabulary needed to identify the same base relation.
- Do not put a novelty-specific predicted consequence such as "relative ordering", "threshold", or "reversal" into relation_nucleus_terms unless that feature is itself part of the base relation being matched.
- For a distinctive_prediction built on a moderator relation, prior_art_identity_terms should preserve the moderator/context identity, while distinguishing_terms should describe the prediction-specific feature that may remain novel.
- Example: "Changing excitation wavelength changes the relative ordering of SERS across spacings" may use prior_art_identity_terms=["excitation wavelength"], distinguishing_terms=["relative ordering"], and relation_nucleus_terms=["interparticle spacing", "SERS enhancement", "dependence"].
- A prior-art memory match only makes a historical work eligible for re-review. It never implies that the historical work establishes the current distinguishing prediction.

ATOMIC SPECIFICATION PROVENANCE CONTRACT:
- For every atomic novelty-bearing claim, preserve branch-specific scientific specification only when it can be grounded in the supplied hypothesis.
- required_bridge is the minimum inferential or mechanistic bridge already stated by the hypothesis for THIS atomic branch.
- predicted_observation is the observation from the supplied hypothesis that independently tests THIS branch.
- falsification_condition is the supplied falsifying outcome that independently challenges THIS branch.
- You may split a coordinated prediction or falsifier such as "wavelength or power" into branch-specific versions when doing so only removes the sibling alternative and does not add a new direction, mechanism, threshold, regime, or scientific proposition.
- Bridge attribution is stricter: an umbrella bridge for "laser conditions" must NOT be assigned to a wavelength or power branch if the hypothesis does not make clear how that specific branch supplies the bridge.
- In particular, do not invent power-dependent heating, geometry change, field saturation, nonlinear response, threshold behavior, or any other mechanism unless that scientific proposition is already present in the supplied hypothesis.
- Do not borrow a bridge, prediction, or falsifier whose scientific meaning belongs only to a sibling branch.
- If a branch-specific bridge, prediction, or falsification condition cannot be extracted without inventing or choosing new scientific content, return the corresponding field as an empty string.
- Empty specification fields are valid and are preferred over unsupported completion.
These fields record hypothesis specification provenance; they are not prior-art evidence.

SELF-CONTAINED ATOMIC SPECIFICATION CONTRACT:
- Every non-empty predicted_observation and falsification_condition must be understandable as a standalone statement of THIS atomic branch. Do not return a bare anaphoric expression such as "the interaction", "this interaction", "this effect", "that reversal", "these effects", or "the pathway" when the branch identity is not also named in the same field.
- When the supplied hypothesis already makes the referent unambiguous, you MAY replace such an anaphoric reference with the already stated atomic branch identity solely to make the field self-contained. This reference expansion must not add a new mechanism, direction, threshold, condition, variable, or scientific proposition.
- For example, if the supplied hypothesis and atomic claim already identify an "M-H iCOHP × oxygenated-intermediate stabilization interaction", a falsifier saying only "the interaction does not improve prediction" should be rendered self-contained by explicitly naming that same interaction. Do not invent any additional consequence.
- required_bridge remains stricter. A non-empty required_bridge must remain an extractive scientific span supported by the hypothesis inferential bridge or assumptions. Do not rewrite or expand a bridge merely to satisfy branch identity matching.

REQUIRED-BRIDGE RETURN SELF-CHECK:
- First choose prior_art_identity_terms according to the scientific branch-identity contract above. Do NOT weaken, broaden, replace, or opportunistically change the correct branch identity merely to make required_bridge non-empty.
- Before returning any non-empty required_bridge, verify that the RETURNED BRIDGE STRING ITSELF explicitly contains at least one valid prior_art_identity_terms expression under conservative surface normalization. Do not rely on identity words that occur only before or after the returned span elsewhere in the source sentence or paragraph.
- required_bridge must be ONE CONTIGUOUS EXTRACTIVE SPAN from the supplied hypothesis inferential_bridge or assumptions. Do not stitch together non-contiguous fragments.
- Do not delete an intervening sibling alternative, sibling mechanism, sibling moderator, or sibling relation merely to manufacture a branch-specific span.
- Do not expand to a larger umbrella sentence merely to capture the identity term when that expansion would import a scientifically separable sibling branch or an additional relation nucleus.
- If no contiguous source span both explicitly identifies THIS atomic branch and states the bridge without importing a separable sibling branch or new scientific relation, return required_bridge as an empty string.
- An empty bridge in this situation is the correct epistemic result; downstream refinement may later supply a better branch-specific hypothesis specification.

- If the hypothesis uses both a canonical branch phrase and a shorter literal branch phrase that unambiguously denotes the same branch, prior_art_identity_terms may contain both literal forms. Do not invent synonyms or broader scientific categories merely to obtain a match.
- Do not use a generic word such as "coordination", "effect", "coupling", "interaction", or "pathway" as an additional identity term unless that exact expression is independently specific enough to distinguish the atomic branch from sibling branches.

CORE-VERSUS-TESTING-PREDICTION CONTRACT:
- Do not mark both a core scientific relation and every direct operational test of that same relation as independent core novelty claims.
- A distinctive_prediction that merely operationalizes, measures, statistically tests, or provides a falsification criterion for an already represented core relation should normally be importance=supporting.
- Keep a distinctive_prediction importance=core only when it adds an independent novelty-bearing scientific proposition whose prior-art status could invalidate the hypothesis even if the associated moderator/mechanistic core relation remained intact.
- For example, if a core claim already states that descriptor X and factor Z interact in determining outcome Y, a prediction that an X×Z interaction model outperforms an additive model is normally supporting when it is simply the operational test of that interaction.
- In contrast, an explicit sign reversal, threshold, new regime, or qualitatively distinct ordering may remain core when that feature is itself the central scientific proposition rather than merely a measurement criterion.

ATOMIC SCIENTIFIC-STRUCTURE CONTRACT:
- scientific_structure describes the inferential structure of THIS atomic claim. It is a claim specification, not a novelty verdict and not prior-art evidence.
- Do not infer a strong structure merely because it would make the hypothesis more interesting.
- Every non-default strong structure must already be explicitly supported by the supplied hypothesis statement, inferential bridge, assumptions, predicted observations, or falsification criteria.
- For every strong structure/category, add one or more scientific_structure.basis entries.
- basis.source_text MUST be copied as an exact contiguous span from the supplied hypothesis material. Do not paraphrase, summarize, normalize, or invent basis text.
- When prior_art_identity_terms identifies an atomic branch, every basis span used for a strong structure must explicitly name that branch identity. Do not use a wavelength statement as basis for a laser-power structure or vice versa.
- If no valid branch-specific source span exists, keep the corresponding structure at its conservative default.

BOOLEAN STRUCTURE FLAGS:
- introduces_new_mechanism=true only when THIS atomic claim explicitly proposes a new mechanistic bridge rather than merely reusing or mentioning a known mechanism.
- introduces_threshold=true only when an explicit critical value, boundary, onset, cutoff, or threshold relation is part of THIS claim.
- introduces_regime_change=true only when the hypothesis explicitly distinguishes scientific regimes whose relation/behavior differs.
- introduces_reversal=true only when an explicit reversal, sign/order inversion, or opposite ordering across conditions is proposed.
- introduces_mechanism_switch=true only when different mechanisms are explicitly proposed to dominate in different conditions/regimes.

CATEGORICAL STRUCTURE:
- inferential_distance:
  LOCAL_REPHRASE = no explicit structural leap beyond restatement;
  SINGLE_KNOWN_STEP = one explicitly proposed ordinary inferential step;
  MULTI_STEP_COMPOSITION = explicit composition of multiple relations;
  NEW_RELATIONAL_FORM = a new interaction/mediation/conditional relational form is explicitly proposed;
  NEW_REGIME_STRUCTURE = an explicit threshold/regime/reversal/mechanism-switch structure is proposed.
- mechanistic_necessity:
  NO_NEW_MECHANISM = no new bridge is required;
  KNOWN_MECHANISM_REUSED = the supplied hypothesis explicitly reuses an already stated mechanism;
  NEW_BRIDGE_REQUIRED = the claim explicitly requires an additional mechanistic/inferential bridge;
  MECHANISM_SWITCH_REQUIRED = the claim explicitly requires a mechanism switch.
- regime_specificity:
  NONE, CONDITIONED, THRESHOLD, REVERSAL, HYSTERESIS, or MECHANISM_SWITCH. Use a strong category only when explicitly stated by the hypothesis.
- counterintuitiveness:
  EXPECTED by default. NONTRIVIAL or COUNTER_TO_BASELINE requires an explicit supplied comparison, expectation, or baseline tension; do not infer surprise from your own scientific knowledge.
- testable_distinctiveness:
  GENERIC by default;
  COMPARATIVE when the supplied hypothesis gives a specific comparison;
  QUANTITATIVE when it gives a quantitatively discriminating observable/relation without inventing unsupported numbers;
  DISCRIMINATING_SIGNATURE when it gives an observation that specifically distinguishes the proposed explanation from a routine alternative.

BASIS FEATURE NAMES:
- new_mechanism
- threshold
- regime_change
- reversal
- mechanism_switch
- inferential_distance
- mechanistic_necessity
- regime_specificity
- counterintuitiveness
- testable_distinctiveness

A basis span proves only that the structure was stated by the generated hypothesis. It does NOT prove that the structure is scientifically correct, novel, or non-obvious.

DISTINGUISHING-FACET CONTRACT:
- Populate distinguishing_terms for every novelty-bearing claim.
- distinguishing_terms identify the minimum scientific factor or relation feature whose identity makes this claim different from nearby alternatives and therefore may give it a different prior-art status.
- For moderator_interaction, include the moderator itself, for example "excitation wavelength" or "laser power". Do not replace a specific moderator with an umbrella phrase such as "laser conditions".
- For context_condition, include the conditioning context or regime.
- For pathway_competition or mechanistic_link, include the specific competing pathway or mechanism when that mechanism is the differentiating feature.
- For distinctive_prediction, include the specific condition, contrast, threshold, reversal, or other feature that makes the prediction distinctive.
- Do not place the base input and outcome in distinguishing_terms merely because they appear in the claim; those belong in the relation/search representation unless they themselves are the novelty-bearing distinction.
- Keep distinguishing_terms concise and literal. Do not add synonyms or literature-derived concepts not present in the supplied hypothesis.

Do not decompose generic background facts unless they are necessary to distinguish the generated hypothesis. Return only the structured NoveltyClaimDecompositionDraft requested by the caller."""


_REVIEW_SYSTEM = """You are a prior-art relationship reviewer in an external-novelty assessment pipeline.

You are given ONE claim and a bounded set of retrieved literature records. Use ONLY the supplied title/abstract metadata. Do not use outside knowledge. Do not claim literature-wide novelty.

For each truly relevant record, classify its relationship to the claim as one of:
- DIRECT_PRIOR_ART: the record directly states/tests essentially the same scientific relation or distinctive prediction.
- PARTIAL_PRIOR_ART: an ABSTRACT-BACKED record preserves the claim's RELATION NUCLEUS and establishes a substantial subset of that scientific relation, but not the full claim.
- TITLE_ONLY_NEIGHBOR: the title suggests a neighboring relation but the abstract is missing, so the substantive overlap cannot be confirmed.
- COMPONENT_ONLY: the record establishes one or more ingredients/components, variables, mechanisms, contexts, materials, or one arm of a comparison, but does not establish the claim's proposed interaction, dependence, mediation, conditionality, comparison, directional relation, or other relation nucleus.
- LOWER_ORDER_RELATION_PRIOR_ART: for a higher-order or moderator/interaction claim, the record explicitly establishes a scientifically meaningful LOWER-ORDER SUBRELATION among variables in the claim, but does not establish the full higher-order relation nucleus. Separate single-variable main effects remain COMPONENT_ONLY.
- DIRECTIONAL_COUNTEREVIDENCE: an ABSTRACT-BACKED record in a scientifically relevant neighboring scope materially challenges the claim's proposed ORDERED DIRECTION, for example by reporting weak or absent correlation, an opposite trend, regime dependence, or dominance by another factor. Use this when the evidence is important as a boundary condition but the scientific scope is not sufficiently matched for CONFLICTING_PRIOR_ART.
- CONTEXTUAL_CONFLICT: the record challenges a broader descriptor/mechanistic assumption but differs materially in reaction domain, catalyst class, or site scope.
- CONFLICTING_PRIOR_ART: the record directly reports a materially opposing relation/result in a sufficiently overlapping scientific scope.
- UNRELATED: despite retrieval similarity, it does not materially bear on the claim.
- INSUFFICIENT_METADATA: the supplied metadata is too weak to judge.

RELATION-NUCLEUS RULES:
1. Shared entities, variables, mechanisms, contexts, materials, or thematic proximity alone are NOT sufficient for PARTIAL_PRIOR_ART. Use COMPONENT_ONLY unless the record actually establishes a substantial part of the asserted relation.
2. A thematically neighboring relation is not, by itself, PARTIAL_PRIOR_ART.
3. For mediator claims, PARTIAL_PRIOR_ART requires evidence linking the proposed mediator to the relevant relation or outcome. Evidence that merely discusses the mediator variable is COMPONENT_ONLY.
4. For moderator_interaction or descriptor_interaction claims, PARTIAL_PRIOR_ART requires an interaction, dependence, conditionality, or joint effect relevant to the asserted relation. Separate main effects or separate components are COMPONENT_ONLY.
5. For context_condition claims, PARTIAL_PRIOR_ART requires a comparison across contexts or an explicit context-dependent effect. Evidence from only one context is COMPONENT_ONLY.
6. For distinctive_prediction claims, first determine whether the prediction's distinctive relation nucleus is unconditional or explicitly conditional/moderated.
   - If the claim contains an explicit moderator, context condition, interaction, or cross-context comparison (for example "in X-conditioned structures", "compared with X-free structures", or "X changes the Y-versus-Z relationship"), a record that establishes only the underlying base input-to-outcome relation WITHOUT that moderator/conditional contrast is COMPONENT_ONLY, not PARTIAL_PRIOR_ART.
   - PARTIAL_PRIOR_ART may tolerate incomplete direction, material scope, or secondary control details only when the defining conditional/moderating relation or an equivalent contrast is still represented.
   - For an unconditional distinctive prediction, the same dependent relation or contrast may qualify as PARTIAL_PRIOR_ART even when direction, material scope, or a secondary control condition is incomplete.
   - Evidence for only the dependent variable, only one unrelated arm of a comparison, or only the unconditioned base relation of a conditional claim is COMPONENT_ONLY.
7. For mechanistic_link claims, PARTIAL_PRIOR_ART requires substantially the same mechanistic link. Sharing mechanism ingredients without the link is COMPONENT_ONLY.
8. A scope mismatch can still permit PARTIAL_PRIOR_ART when the relation nucleus is preserved; scope similarity alone cannot create PARTIAL_PRIOR_ART.

9. For a higher-order interaction claim, use LOWER_ORDER_RELATION_PRIOR_ART rather than COMPONENT_ONLY when the record explicitly establishes a nontrivial lower-order relation among two or more variables from the claim. Do not infer a lower-order relation from separate main effects.
10. For a directional claim, distinguish exact contradiction from boundary evidence. Use CONFLICTING_PRIOR_ART only when the opposing result is in sufficiently overlapping scientific scope. Use DIRECTIONAL_COUNTEREVIDENCE when an abstract-backed neighboring system materially weakens the proposed ordered direction but does not satisfy the exact conflict-scope requirement.
11. LOWER_ORDER_RELATION_PRIOR_ART and DIRECTIONAL_COUNTEREVIDENCE are diagnostic prior-art signals. They do not by themselves make the full claim DIRECT_PRIOR_ART or PARTIAL_PRIOR_ART.

MODERATOR-INTERACTION DIRECTNESS:
For a moderator_interaction claim, DIRECT_PRIOR_ART requires the supplied title/abstract metadata to explicitly state, test, compare, or demonstrate that the moderator changes, conditions, or modifies the base relation itself. The required logical form is approximately: M changes how X affects Y, the X-to-Y relationship differs across M conditions, or a joint M-by-X effect on Y is explicitly stated or tested.

For a moderator_interaction claim, PARTIAL_PRIOR_ART still requires the interaction/conditional relation nucleus to be represented. Incomplete material scope, direction, controls, or secondary details may be tolerated, but the moderator relation itself may not be inferred.

If a record separately establishes X affects Y and M affects Y, classify it as COMPONENT_ONLY. This remains COMPONENT_ONLY even if the two statements occur in adjacent sentences, appear in the same paper or analysis, affect the same observable, or can be combined into a scientifically plausible moderator interpretation.

Do not infer moderator interaction from local textual adjacency, shared outcome variables, shared mechanism vocabulary, or co-discussion alone.

Before returning DIRECT_PRIOR_ART or PARTIAL_PRIOR_ART for moderator_interaction, check whether the supplied metadata actually states that M changes the X-to-Y relationship rather than merely stating M-to-Y and X-to-Y separately. If not, use COMPONENT_ONLY.

SELF-CONSISTENCY CHECK:
If your rationale says that a record "does not compare X versus Y", "does not establish the claimed relationship", "only addresses one condition", "does not link X to Y", or an equivalent statement, PARTIAL_PRIOR_ART is usually inconsistent. Use COMPONENT_ONLY unless the same record still establishes another substantial part of the claim's relation nucleus.

WORK-ID COPY CONTRACT:
1. Every returned work_id MUST be copied byte-for-byte from the explicit ALLOWED_WORK_IDS block in the user message.
2. Never invent, reconstruct, shorten, normalize, index, or alias a work ID.
3. Never return candidate numbers, list indices, ordinal labels, or placeholders as work IDs. Examples of forbidden forms include "6", "work:6", "paper:6", and "prior_art_work:6" unless that exact string itself appears in ALLOWED_WORK_IDS.
4. Return at most one match per allowed work_id. Do not duplicate a work ID.
5. If you cannot copy the exact supplied ID for a record, OMIT that record. Do not create a match merely to mention that an ID is missing or invalid.
6. The set of returned work IDs must be a subset of ALLOWED_WORK_IDS.

Do not infer detailed results from a generic title alone. For CONFLICTING_PRIOR_ART, require not only an opposing result but also substantially matching reaction and catalyst/site scope; otherwise use CONTEXTUAL_CONFLICT. The deterministic compiler will independently enforce these constraints.

Return work IDs exactly as supplied. You may omit unrelated records. Your interpretation must describe only what the supplied bounded evidence shows."""


def _sha256(system: str, user: str) -> str:
    raw = (system + "\n---\n" + user).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class InstructorOpenAICompatibleExternalNoveltyBackend:
    backend_name = "instructor_openai_compatible_external_novelty"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        instructor_mode: str = "JSON",
        temperature: float = 0.0,
        parse_retries: int = 1,
        timeout: float | None = 180.0,
        capture_prompts: bool = False,
        max_abstract_chars: int = 1400,
        telemetry_path: str | os.PathLike[str] | None = None,
        telemetry_context: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = str(model)
        self.api_key = api_key if api_key is not None else os.getenv(api_key_env)
        self.api_key_env = api_key_env
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.instructor_mode = str(instructor_mode).upper()
        self.temperature = float(temperature)
        self.parse_retries = int(parse_retries)
        self.timeout = timeout
        self.capture_prompts = bool(capture_prompts)
        self.max_abstract_chars = int(max_abstract_chars)
        self.prompt_records: list[ExternalNoveltyPromptRecord] = []
        self.telemetry_path = telemetry_path
        self.telemetry_context = dict(telemetry_context or {})
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(
                f"No API key available. Set {self.api_key_env} or pass api_key explicitly."
            )
        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "External novelty LLM backend requires installed 'openai' and 'instructor'."
            ) from exc
        mode = getattr(instructor.Mode, self.instructor_mode, None)
        if mode is None:
            raise ValueError(f"Unknown Instructor mode: {self.instructor_mode}")
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        self._client = instructor.from_openai(OpenAI(**kwargs), mode=mode)
        return self._client

    def _record(self, name: str, system: str, user: str) -> None:
        if self.capture_prompts:
            self.prompt_records.append(
                ExternalNoveltyPromptRecord(
                    name=name,
                    system_prompt=system,
                    user_prompt=user,
                    prompt_sha256=_sha256(system, user),
                )
            )

    def decompose(
        self,
        hypothesis: HypothesisCard,
        *,
        max_claims: int,
    ) -> NoveltyClaimDecompositionDraft:
        prediction_lines = [
            f"- {row.observable} => {row.expected_direction}; rationale={row.rationale}"
            for row in hypothesis.predicted_observations
        ]
        falsifier_lines = [
            f"- {row.observable} => falsified_by={row.falsifying_outcome}"
            for row in hypothesis.falsification_criteria
        ]
        user = "\n".join(
            [
                "HYPOTHESIS",
                "==========",
                f"hypothesis_id: {hypothesis.hypothesis_id}",
                f"title: {hypothesis.title}",
                f"statement: {hypothesis.hypothesis_statement}",
                f"inferential_bridge: {hypothesis.inferential_bridge}",
                "predictions:",
                *(prediction_lines or ["- NONE"]),
                "falsification_criteria:",
                *(falsifier_lines or ["- NONE"]),
                "assumptions:",
                *([f"- {x}" for x in hypothesis.assumptions] or ["- NONE"]),
                "",
                f"Return at most {int(max_claims)} claim-level novelty assertions.",
                "At least one claim should capture the hypothesis's most distinctive interaction/condition/prediction rather than generic HER background.",
            ]
        )
        self._record(f"decompose_{hypothesis.hypothesis_id}", _DECOMPOSE_SYSTEM, user)
        result, _event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.model_name,
            response_model=NoveltyClaimDecompositionDraft,
            messages=[
                {"role": "system", "content": _DECOMPOSE_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context={
                **self.telemetry_context,
                "pipeline": "external_novelty",
                "stage": "decompose",
                "call_kind": "structured",
                "hypothesis_id": hypothesis.hypothesis_id,
            },
        )
        if not isinstance(result, NoveltyClaimDecompositionDraft):
            result = NoveltyClaimDecompositionDraft.model_validate(result)
        return result

    def review_claim(
        self,
        claim: NoveltyClaim,
        works: list[dict[str, Any]],
    ) -> ClaimPriorArtReviewDraft:
        lines = [
            "CLAIM",
            "=====",
            f"claim_id: {claim.claim_id}",
            f"kind: {claim.kind}",
            f"importance: {claim.importance}",
            f"text: {claim.text}",
            "",
            "RETRIEVED PRIOR-ART CANDIDATES",
            "==============================",
        ]
        if not works:
            lines.append("- NONE")
        for index, work in enumerate(works, start=1):
            abstract = str(work.get("abstract") or "")
            if len(abstract) > self.max_abstract_chars:
                abstract = abstract[: self.max_abstract_chars - 1].rstrip() + "…"
            lines.extend(
                [
                    f"[{index}] work_id={work['work_id']}",
                    f"title: {work.get('title', '')}",
                    f"year: {work.get('year')}",
                    f"doi: {work.get('doi')}",
                    f"semantic_similarity: {float(work.get('semantic_similarity', 0.0)):.4f}",
                    f"lexical_coverage: {float(work.get('lexical_coverage', 0.0)):.4f}",
                    f"reaction_domain_relevance: {float(work.get('reaction_domain_relevance', 0.5)):.4f}",
                    f"catalyst_scope_relevance: {float(work.get('catalyst_scope_relevance', 0.5)):.4f}",
                    f"abstract: {abstract if abstract else '[NO ABSTRACT AVAILABLE]'}",
                    "",
                ]
            )
        allowed_work_ids = [
            str(work["work_id"])
            for work in works
        ]
        lines.extend(
            [
                "ALLOWED_WORK_IDS",
                "================",
                *allowed_work_ids,
                "",
                "WORK-ID OUTPUT REQUIREMENT",
                "==========================",
                "Every returned work_id must be copied byte-for-byte from ALLOWED_WORK_IDS above.",
                "Return at most one match per allowed work_id.",
                "Do not return candidate numbers, list indices, abbreviated IDs, reconstructed IDs, or placeholders.",
                "If you cannot copy the exact supplied work_id, omit that record.",
                "",
                "Only classify records that materially bear on the claim.",
                "Do not infer literature-wide absence from this bounded candidate set.",
            ]
        )
        user = "\n".join(lines)
        self._record(f"review_{claim.claim_id}", _REVIEW_SYSTEM, user)
        result, _event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.model_name,
            response_model=ClaimPriorArtReviewDraft,
            messages=[
                {"role": "system", "content": _REVIEW_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context={
                **self.telemetry_context,
                "pipeline": "external_novelty",
                "stage": "prior_art_review",
                "call_kind": "structured",
                "claim_id": claim.claim_id,
            },
        )
        if not isinstance(result, ClaimPriorArtReviewDraft):
            result = ClaimPriorArtReviewDraft.model_validate(result)
        record_prior_art_review_call(
            system_prompt=_REVIEW_SYSTEM,
            user_prompt=user,
            response_schema=ClaimPriorArtReviewDraft,
            result=result,
            model=self.model_name,
            instructor_mode=self.instructor_mode,
            temperature=self.temperature,
            claim_id=claim.claim_id,
            hypothesis_id=claim.hypothesis_id,
            claim_text=claim.text,
            works=works,
            telemetry_event=_event,
        )
        return result
