"""Diagnostic-only N10 specification-repair context.

This artifact carries deterministic N9 specification failures into a
possible bounded Alpha6 repair step.

It is NOT scientific evidence, does NOT grant novelty/non-obviousness
authority, and may never turn prior-art absence into positive evidence.

Only novelty-bearing claims that the deterministic N9 specification gate
classifies as NEEDS_REFINEMENT are eligible.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
)


REPAIR_ACTION = (
    "REFINE_NOVELTY_BEARING_SPECIFICATION"
)

POST_GENERATION_SCOPE = (
    "alpha6_post_generation_candidate"
)

ALLOWED_MISSING_FIELDS = frozenset(
    {
        "required_bridge",
        "predicted_observation",
        "falsification_condition",
    }
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )


class N10SpecificationRepairClaimDiagnostic(
    StrictModel
):
    claim_id: str = Field(
        min_length=1
    )

    claim_text: str = Field(
        min_length=1
    )

    novelty_selection_role: Literal[
        "NOVELTY_BEARING"
    ] = "NOVELTY_BEARING"

    specification_status: Literal[
        "NEEDS_REFINEMENT"
    ] = "NEEDS_REFINEMENT"

    missing_fields: list[
        Literal[
            "required_bridge",
            "predicted_observation",
            "falsification_condition",
        ]
    ] = Field(
        min_length=1
    )

    reason_codes: list[str] = Field(
        min_length=1
    )


class N10SpecificationRepairContext(
    StrictModel
):
    schema_version: Literal[
        "n10-specification-repair-context-v1"
    ] = (
        "n10-specification-repair-context-v1"
    )

    context_id: str
    context_sha256: str

    source_hypothesis_id: str

    source_query_plan_id: str
    source_query_plan_sha256: str

    source_external_report_id: str
    source_external_report_sha256: str

    source_intake_sha256: str
    source_n10_gate_sha256: str

    source_n10_gate_schema: Literal[
        "scientific-novelty-fallback-gate-v2"
    ] = (
        "scientific-novelty-fallback-gate-v2"
    )

    source_n10_authority_scope: Literal[
        "alpha6_post_generation_candidate"
    ] = (
        "alpha6_post_generation_candidate"
    )

    selection_class: Literal[
        "CONDITIONAL"
    ] = "CONDITIONAL"

    repair_action: Literal[
        "REFINE_NOVELTY_BEARING_SPECIFICATION"
    ] = (
        "REFINE_NOVELTY_BEARING_SPECIFICATION"
    )

    claim_diagnostics: list[
        N10SpecificationRepairClaimDiagnostic
    ] = Field(
        min_length=1
    )

    diagnostic_only: Literal[
        True
    ] = True

    production_authority: Literal[
        False
    ] = False

    scientific_evidence_authority: Literal[
        False
    ] = False

    external_prior_art_can_be_positive_premise: Literal[
        False
    ] = False

    absence_is_novelty: Literal[
        False
    ] = False


def _canonical_json(
    value: Any,
) -> str:
    if hasattr(
        value,
        "model_dump",
    ):
        value = value.model_dump(
            mode="json"
        )

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(
    value: Any,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            value
        ).encode(
            "utf-8"
        )
    ).hexdigest()


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(
        str(part)
        for part in parts
    ).encode(
        "utf-8"
    )

    return (
        f"{prefix}:"
        f"{hashlib.sha256(raw).hexdigest()[:length]}"
    )


def _require_bool(
    value: object,
    *,
    expected: bool,
    message: str,
) -> None:
    if value is not expected:
        raise ValueError(
            message
        )


def build_n10_specification_repair_context(
    *,
    source_hypothesis_id: str,
    query_plan: LiteratureQueryPlan,
    intake_shadow: Mapping[str, Any],
    post_generation_gate: Mapping[str, Any],
) -> N10SpecificationRepairContext:
    """Build one diagnostic-only bounded repair context.

    Inputs are already-produced N9/N10 artifacts. This builder never
    performs retrieval, never creates scientific content, and never
    changes any N9/N10 decision.
    """

    source_hypothesis_id = str(
        source_hypothesis_id
        or ""
    ).strip()

    if not source_hypothesis_id:
        raise ValueError(
            "source_hypothesis_id is required"
        )

    # -------------------------------------------------------------
    # N9 intake provenance / epistemic state.
    # -------------------------------------------------------------

    if (
        intake_shadow.get(
            "schema_version"
        )
        != "nonobviousness-shadow-v1"
    ):
        raise ValueError(
            "unexpected N9 intake schema"
        )

    _require_bool(
        intake_shadow.get(
            "shadow_only"
        ),
        expected=True,
        message=(
            "N9 specification repair context "
            "requires shadow-only intake"
        ),
    )

    _require_bool(
        intake_shadow.get(
            "scientific_selection_changed"
        ),
        expected=False,
        message=(
            "N9 intake must not have changed "
            "scientific selection"
        ),
    )

    if (
        intake_shadow.get(
            "source_query_plan_id"
        )
        != query_plan.plan_id
    ):
        raise ValueError(
            "N9 intake/query-plan ID mismatch"
        )

    if (
        intake_shadow.get(
            "source_query_plan_sha256"
        )
        != query_plan.plan_sha256
    ):
        raise ValueError(
            "N9 intake/query-plan SHA mismatch"
        )

    # -------------------------------------------------------------
    # N10 post-generation gate: exact frozen repair condition.
    # -------------------------------------------------------------

    if (
        post_generation_gate.get(
            "schema_version"
        )
        != "scientific-novelty-fallback-gate-v2"
    ):
        raise ValueError(
            "specification repair requires N10-v2 gate"
        )

    _require_bool(
        post_generation_gate.get(
            "production_authority"
        ),
        expected=True,
        message=(
            "post-generation N10 gate lacks "
            "production authority"
        ),
    )

    if (
        post_generation_gate.get(
            "authority_scope"
        )
        != POST_GENERATION_SCOPE
    ):
        raise ValueError(
            "unexpected N10 authority scope"
        )

    _require_bool(
        post_generation_gate.get(
            "conditional_is_positive"
        ),
        expected=False,
        message=(
            "CONDITIONAL must remain non-positive"
        ),
    )

    _require_bool(
        post_generation_gate.get(
            "absence_is_novelty"
        ),
        expected=False,
        message=(
            "absence must not become novelty"
        ),
    )

    _require_bool(
        post_generation_gate.get(
            "candidate_semantics_preserved"
        ),
        expected=True,
        message=(
            "N10 gate must preserve candidate semantics"
        ),
    )

    gate_rows = [
        row
        for row in post_generation_gate.get(
            "gates",
            []
        )
        if (
            isinstance(
                row,
                Mapping,
            )
            and str(
                row.get(
                    "hypothesis_id"
                )
                or ""
            )
            == source_hypothesis_id
        )
    ]

    if len(gate_rows) != 1:
        raise ValueError(
            "expected exactly one N10 gate row "
            "for repair hypothesis"
        )

    gate_row = gate_rows[0]

    if (
        gate_row.get(
            "selection_class"
        )
        != "CONDITIONAL"
    ):
        raise ValueError(
            "specification repair requires CONDITIONAL"
        )

    _require_bool(
        gate_row.get(
            "positive_nonobviousness_authority"
        ),
        expected=False,
        message=(
            "specification repair cannot start "
            "from positive authority"
        ),
    )

    _require_bool(
        gate_row.get(
            "fallback_allowed"
        ),
        expected=False,
        message=(
            "specification repair requires "
            "fallback_allowed=False"
        ),
    )

    action = (
        gate_row.get(
            "base_aggregation_action"
        )
        or gate_row.get(
            "action"
        )
    )

    if action != REPAIR_ACTION:
        raise ValueError(
            "N10 action is not the frozen "
            "specification-repair action"
        )

    # -------------------------------------------------------------
    # Canonical query-plan claims.
    # -------------------------------------------------------------

    claim_by_id = {}

    for group in query_plan.claims:

        if (
            group.hypothesis_id
            != source_hypothesis_id
        ):
            continue

        for claim in group.claims:

            if (
                claim.hypothesis_id
                != source_hypothesis_id
            ):
                raise ValueError(
                    "query-plan claim hypothesis drift"
                )

            if claim.claim_id in claim_by_id:
                raise ValueError(
                    "duplicate query-plan claim ID"
                )

            claim_by_id[
                claim.claim_id
            ] = claim

    if not claim_by_id:
        raise ValueError(
            "query plan contains no claims "
            "for source hypothesis"
        )

    # -------------------------------------------------------------
    # Exact N9 hypothesis row.
    # -------------------------------------------------------------

    intake_rows = [
        row
        for row in intake_shadow.get(
            "hypotheses",
            []
        )
        if (
            isinstance(
                row,
                Mapping,
            )
            and str(
                row.get(
                    "hypothesis_id"
                )
                or ""
            )
            == source_hypothesis_id
        )
    ]

    if len(intake_rows) != 1:
        raise ValueError(
            "expected exactly one N9 intake "
            "hypothesis row"
        )

    intake_row = intake_rows[0]

    diagnostics = []

    seen_intake_claim_ids = set()

    for decision in intake_row.get(
        "claims",
        []
    ):

        if not isinstance(
            decision,
            Mapping,
        ):
            raise ValueError(
                "N9 intake claim decision must be object"
            )

        claim_payload = decision.get(
            "claim"
        )

        specification = decision.get(
            "specification"
        )

        if not isinstance(
            claim_payload,
            Mapping,
        ):
            raise ValueError(
                "N9 intake claim payload missing"
            )

        if not isinstance(
            specification,
            Mapping,
        ):
            raise ValueError(
                "N9 intake specification payload missing"
            )

        claim_id = str(
            claim_payload.get(
                "claim_id"
            )
            or ""
        ).strip()

        if not claim_id:
            raise ValueError(
                "N9 intake claim lacks claim_id"
            )

        if claim_id in seen_intake_claim_ids:
            raise ValueError(
                "duplicate N9 intake claim ID"
            )

        seen_intake_claim_ids.add(
            claim_id
        )

        canonical = claim_by_id.get(
            claim_id
        )

        if canonical is None:
            raise ValueError(
                "N9 intake claim missing from "
                "canonical query plan"
            )

        intake_claim_text = str(
            claim_payload.get(
                "claim_text"
            )
            or ""
        ).strip()

        if (
            intake_claim_text
            != canonical.text.strip()
        ):
            raise ValueError(
                "N9 intake claim text drift"
            )

        if (
            decision.get(
                "shadow_state"
            )
            != "NEEDS_REFINEMENT"
            or specification.get(
                "status"
            )
            != "NEEDS_REFINEMENT"
        ):
            continue

        # The frozen action is specifically novelty-bearing
        # specification repair. Testing/enabling/auxiliary branches
        # remain visible in N9 but must not leak into this context.
        if (
            canonical.novelty_selection_role
            != "NOVELTY_BEARING"
        ):
            continue

        if (
            decision.get(
                "next_action"
            )
            != "REFINE_HYPOTHESIS_SPECIFICATION"
        ):
            raise ValueError(
                "N9 NEEDS_REFINEMENT has "
                "unexpected next action"
            )

        missing = [
            str(value)
            for value in (
                specification.get(
                    "missing_fields",
                    []
                )
                or []
            )
        ]

        if not missing:
            raise ValueError(
                "N9 NEEDS_REFINEMENT lacks "
                "missing_fields"
            )

        if len(missing) != len(
            set(missing)
        ):
            raise ValueError(
                "duplicate missing specification field"
            )

        unknown_missing = (
            set(missing)
            - ALLOWED_MISSING_FIELDS
        )

        if unknown_missing:
            raise ValueError(
                "unsupported missing specification fields: "
                + repr(
                    sorted(
                        unknown_missing
                    )
                )
            )

        reasons = [
            str(value)
            for value in (
                specification.get(
                    "reason_codes",
                    []
                )
                or []
            )
            if str(value).strip()
        ]

        if not reasons:
            raise ValueError(
                "N9 NEEDS_REFINEMENT lacks "
                "reason codes"
            )

        if (
            "atomic_residue_under_specified"
            not in reasons
        ):
            raise ValueError(
                "N9 specification diagnosis lacks "
                "under-specification reason"
            )

        for field in missing:

            expected_reason = (
                "missing_"
                + field
            )

            if expected_reason not in reasons:
                raise ValueError(
                    "N9 missing field lacks matching "
                    f"reason code: {field}"
                )

        diagnostics.append(
            (
                canonical.claim_rank,
                canonical.claim_id,
                N10SpecificationRepairClaimDiagnostic(
                    claim_id=canonical.claim_id,
                    claim_text=canonical.text,
                    missing_fields=missing,
                    reason_codes=reasons,
                ),
            )
        )

    if not diagnostics:
        raise ValueError(
            "N10 requested novelty-bearing "
            "specification repair but no matching "
            "N9 diagnostic was found"
        )

    diagnostics.sort(
        key=lambda row: (
            row[0],
            row[1],
        )
    )

    claim_diagnostics = [
        row[2]
        for row in diagnostics
    ]

    source_external_report_id = str(
        intake_shadow.get(
            "source_external_report_id"
        )
        or ""
    ).strip()

    source_external_report_sha256 = str(
        intake_shadow.get(
            "source_external_report_sha256"
        )
        or ""
    ).strip()

    if not source_external_report_id:
        raise ValueError(
            "N9 intake lacks external report ID"
        )

    if not source_external_report_sha256:
        raise ValueError(
            "N9 intake lacks external report SHA"
        )

    source_intake_sha256 = (
        _sha256_json(
            intake_shadow
        )
    )

    source_n10_gate_sha256 = (
        _sha256_json(
            post_generation_gate
        )
    )

    diagnostics_payload = [
        row.model_dump(
            mode="json"
        )
        for row in claim_diagnostics
    ]

    context_id = _stable_id(
        "n10_specification_repair_context",
        source_hypothesis_id,
        query_plan.plan_id,
        query_plan.plan_sha256,
        source_external_report_id,
        source_external_report_sha256,
        source_intake_sha256,
        source_n10_gate_sha256,
        _canonical_json(
            diagnostics_payload
        ),
    )

    body = {
        "schema_version":
            "n10-specification-repair-context-v1",

        "context_id":
            context_id,

        "source_hypothesis_id":
            source_hypothesis_id,

        "source_query_plan_id":
            query_plan.plan_id,

        "source_query_plan_sha256":
            query_plan.plan_sha256,

        "source_external_report_id":
            source_external_report_id,

        "source_external_report_sha256":
            source_external_report_sha256,

        "source_intake_sha256":
            source_intake_sha256,

        "source_n10_gate_sha256":
            source_n10_gate_sha256,

        "source_n10_gate_schema":
            "scientific-novelty-fallback-gate-v2",

        "source_n10_authority_scope":
            POST_GENERATION_SCOPE,

        "selection_class":
            "CONDITIONAL",

        "repair_action":
            REPAIR_ACTION,

        "claim_diagnostics":
            diagnostics_payload,

        "diagnostic_only":
            True,

        "production_authority":
            False,

        "scientific_evidence_authority":
            False,

        "external_prior_art_can_be_positive_premise":
            False,

        "absence_is_novelty":
            False,
    }

    return (
        N10SpecificationRepairContext(
            **body,
            context_sha256=(
                _sha256_json(
                    body
                )
            ),
        )
    )
