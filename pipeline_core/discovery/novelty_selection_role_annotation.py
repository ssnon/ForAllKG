from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaim,
    NoveltySelectionRole,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
)
from pipeline_core.llm.llm_telemetry import (
    run_instructor_structured_call,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoveltySelectionRoleAssignmentDraft(_StrictModel):
    claim_id: str = Field(min_length=1)
    novelty_selection_role: NoveltySelectionRole | None
    rationale: str = Field(min_length=1)


class NoveltySelectionRoleAnnotationDraft(_StrictModel):
    assignments: list[
        NoveltySelectionRoleAssignmentDraft
    ] = Field(default_factory=list)


@dataclass(frozen=True)
class NoveltySelectionRolePromptRecord:
    hypothesis_id: str
    system_prompt: str
    user_prompt: str
    prompt_sha256: str


_ROLE_SYSTEM = """You assign hypothesis-level scientific selection roles to an EXISTING set of atomic scientific claims.

This is NOT a novelty judgment, prior-art judgment, truth judgment, or non-obviousness judgment.

You are not deciding whether any claim is known, unknown, routine, novel, surprising, correct, or supported by literature.

You must reason ONLY from:
1. the supplied generated hypothesis structure, and
2. the supplied existing atomic claims.

No literature results, retrieval results, prior-art statuses, N9 outcomes, search coverage, or external novelty labels are available to you.

ROLE VOCABULARY

NOVELTY_BEARING
The hypothesis presents this atomic scientific relation as part of the distinct scientific proposition whose loss would remove one claimed source of hypothesis-level distinctiveness.

REQUIRED_ENABLING_RELATION
The relation is a premise, lower-order dependency, or enabling relation needed for the higher-order hypothesis to make scientific sense, but it may already be established without by itself eliminating the higher-order distinctiveness.

TESTING_PREDICTION
The atomic claim primarily operationalizes, discriminates, measures, or falsifies a separately represented novelty-bearing scientific relation rather than constituting an independent source of distinctiveness.

AUXILIARY
The atomic claim supplies explanatory, contextual, or secondary scientific content that is not itself part of the hypothesis's novelty-bearing structure.

NULL
Use null only when the supplied hypothesis and atomic claim do not support a defensible role assignment without inventing, importing, or assuming additional scientific content.

OUTCOME-BLIND CONTRACT

- Do not use your memory of scientific literature.
- Do not infer that a familiar-looking relation is REQUIRED_ENABLING_RELATION because you suspect it is already known.
- Do not infer that an unusual-looking relation is NOVELTY_BEARING because it sounds novel.
- Role depends on the claim's FUNCTION IN THIS HYPOTHESIS, not on expected literature status.
- An interaction, threshold, reversal, mechanism, or mediator may be novelty-bearing only when the supplied hypothesis actually presents it as part of the distinct proposition.
- A lower-order relation may be REQUIRED_ENABLING_RELATION only when the supplied hypothesis structurally uses it to support or enable a higher-order proposition.
- A prediction is TESTING_PREDICTION only when it tests a separately represented relation. If the prediction itself adds an independent threshold, reversal, regime, ordering, or other scientific proposition central to the hypothesis, it may instead be NOVELTY_BEARING.
- Do not use claim importance. It is intentionally unavailable.
- Do not create, merge, split, rewrite, delete, or renumber claims.
- Return every supplied ALLOWED_CLAIM_ID exactly once.
- Return no claim ID outside ALLOWED_CLAIM_IDS.

The rationale must explain only the role of the claim in the supplied hypothesis. It must not make a literature-wide novelty statement.
"""


def _prompt_sha256(
    system_prompt: str,
    user_prompt: str,
) -> str:
    raw = (
        system_prompt
        + "\n---\n"
        + user_prompt
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


def build_role_annotation_prompt(
    hypothesis: HypothesisCard,
    claims: list[NoveltyClaim],
) -> tuple[str, str]:
    """Build an outcome-blind role prompt.

    Intentionally excluded:
    - claim importance
    - search queries
    - prior-art metadata
    - external novelty status
    - N9 intake/shadow state
    - N9 full adjudication
    """

    if not claims:
        raise ValueError(
            "role annotation requires at least one claim"
        )

    for claim in claims:
        if claim.hypothesis_id != hypothesis.hypothesis_id:
            raise ValueError(
                "claim/hypothesis mismatch: "
                + claim.claim_id
            )

    prediction_lines = [
        (
            "- "
            + row.observable
            + " => "
            + row.expected_direction
            + "; rationale="
            + row.rationale
        )
        for row in hypothesis.predicted_observations
    ]

    falsifier_lines = [
        (
            "- "
            + row.observable
            + " => falsified_by="
            + row.falsifying_outcome
        )
        for row in hypothesis.falsification_criteria
    ]

    lines = [
        "HYPOTHESIS",
        "==========",
        f"hypothesis_id: {hypothesis.hypothesis_id}",
        f"title: {hypothesis.title}",
        (
            "statement: "
            + hypothesis.hypothesis_statement
        ),
        (
            "inferential_bridge: "
            + hypothesis.inferential_bridge
        ),
        "predictions:",
        *(prediction_lines or ["- NONE"]),
        "falsification_criteria:",
        *(falsifier_lines or ["- NONE"]),
        "assumptions:",
        *(
            [
                "- " + value
                for value in hypothesis.assumptions
            ]
            or ["- NONE"]
        ),
        "",
        "EXISTING ATOMIC CLAIMS",
        "======================",
    ]

    for claim in claims:
        lines.extend(
            [
                f"claim_id: {claim.claim_id}",
                f"kind: {claim.kind}",
                f"text: {claim.text}",
                f"rationale: {claim.rationale}",
                "",
            ]
        )

    allowed_ids = [
        claim.claim_id
        for claim in claims
    ]

    lines.extend(
        [
            "ALLOWED_CLAIM_IDS",
            "=================",
            *allowed_ids,
            "",
            (
                "Return every ALLOWED_CLAIM_ID "
                "exactly once."
            ),
            (
                "Do not return any other claim ID."
            ),
        ]
    )

    return (
        _ROLE_SYSTEM,
        "\n".join(lines),
    )


def compile_role_annotation(
    *,
    hypothesis: HypothesisCard,
    claims: list[NoveltyClaim],
    draft: NoveltySelectionRoleAnnotationDraft,
) -> list[dict[str, Any]]:
    """Fail closed on any claim-ID coverage mismatch."""

    allowed_ids = [
        claim.claim_id
        for claim in claims
    ]

    if len(
        allowed_ids
    ) != len(
        set(allowed_ids)
    ):
        raise ValueError(
            "duplicate canonical claim_id in role input"
        )

    returned_ids = [
        row.claim_id
        for row in draft.assignments
    ]

    if len(
        returned_ids
    ) != len(
        set(returned_ids)
    ):
        raise ValueError(
            "duplicate claim_id in role annotation"
        )

    unknown = sorted(
        set(returned_ids)
        - set(allowed_ids)
    )

    if unknown:
        raise ValueError(
            "unknown claim_id in role annotation: "
            + ", ".join(unknown)
        )

    missing = [
        claim_id
        for claim_id in allowed_ids
        if claim_id not in set(returned_ids)
    ]

    if missing:
        raise ValueError(
            "missing claim_id in role annotation: "
            + ", ".join(missing)
        )

    by_id = {
        row.claim_id: row
        for row in draft.assignments
    }

    compiled: list[dict[str, Any]] = []

    for claim in claims:
        row = by_id[claim.claim_id]

        compiled.append(
            {
                "claim_id": claim.claim_id,
                "hypothesis_id":
                    hypothesis.hypothesis_id,
                "claim_kind": claim.kind,
                "claim_text": claim.text,
                "novelty_selection_role":
                    row.novelty_selection_role,
                "rationale": row.rationale,
                "role_assignment_source":
                    "hypothesis_structure_only",
                "outcome_blind": True,
            }
        )

    return compiled


class InstructorOpenAICompatibleSelectionRoleBackend:
    backend_name = (
        "instructor_openai_compatible_"
        "selection_role"
    )

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
        telemetry_path: str | os.PathLike[str] | None = None,
        telemetry_context: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = str(model)
        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv(api_key_env)
        )
        self.api_key_env = api_key_env
        self.base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or None
        )
        self.instructor_mode = (
            str(instructor_mode).upper()
        )
        self.temperature = float(temperature)
        self.parse_retries = int(parse_retries)
        self.timeout = timeout
        self.telemetry_path = telemetry_path
        self.telemetry_context = dict(
            telemetry_context or {}
        )
        self._client: Any | None = None
        self.prompt_records: list[
            NoveltySelectionRolePromptRecord
        ] = []

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise RuntimeError(
                "No API key available. Set "
                + self.api_key_env
                + " or pass api_key explicitly."
            )

        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Selection-role backend requires "
                "installed openai and instructor."
            ) from exc

        mode = getattr(
            instructor.Mode,
            self.instructor_mode,
            None,
        )

        if mode is None:
            raise ValueError(
                "Unknown Instructor mode: "
                + self.instructor_mode
            )

        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
        }

        if self.base_url:
            kwargs["base_url"] = self.base_url

        if self.timeout is not None:
            kwargs["timeout"] = self.timeout

        self._client = instructor.from_openai(
            OpenAI(**kwargs),
            mode=mode,
        )

        return self._client

    def annotate(
        self,
        hypothesis: HypothesisCard,
        claims: list[NoveltyClaim],
    ) -> NoveltySelectionRoleAnnotationDraft:
        system, user = build_role_annotation_prompt(
            hypothesis,
            claims,
        )

        self.prompt_records.append(
            NoveltySelectionRolePromptRecord(
                hypothesis_id=hypothesis.hypothesis_id,
                system_prompt=system,
                user_prompt=user,
                prompt_sha256=_prompt_sha256(
                    system,
                    user,
                ),
            )
        )

        result, _event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.model_name,
            response_model=(
                NoveltySelectionRoleAnnotationDraft
            ),
            messages=[
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": user,
                },
            ],
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context={
                **self.telemetry_context,
                "pipeline":
                    "novelty_selection_role_shadow",
                "stage":
                    "outcome_blind_role_annotation",
                "call_kind": "structured",
                "hypothesis_id":
                    hypothesis.hypothesis_id,
            },
        )

        if not isinstance(
            result,
            NoveltySelectionRoleAnnotationDraft,
        ):
            result = (
                NoveltySelectionRoleAnnotationDraft
                .model_validate(result)
            )

        return result
