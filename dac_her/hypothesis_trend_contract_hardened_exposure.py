from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import Field, model_validator

from dac_her.hypothesis_trend_contracts import StrictModel
from dac_her.hypothesis_trend_directional_exposure import (
    DirectionalTrendMakerView,
    build_directional_trend_maker_exposure,
    verify_directional_trend_maker_exposure,
)
from dac_her.hypothesis_trend_input import TrendAwareHypothesisInput


HYPOTHESIS_TREND_HARDENED_EXPOSURE_SEMANTICS_ID = (
    "hypothesis_trend_contract_hardened_exposure_v1_alpha4c5i"
)


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


class ContractHardenedAuthorityPolicy(StrictModel):
    source_directional_exposure_immutable: Literal[True] = True
    llm_selects_evidence_ids_only: Literal[True] = True
    llm_authored_trend_direction_allowed: Literal[False] = False
    llm_authored_trend_observable_allowed: Literal[False] = False
    llm_authored_new_paper_absence_claim_allowed: Literal[False] = False
    trend_direction_derived_by_compiler: Literal[True] = True
    trend_observable_derived_by_compiler: Literal[True] = True
    trend_language_rendered_deterministically: Literal[True] = True
    falsifier_observable_bound_to_prediction: Literal[True] = True
    causal_evidence_authorization: Literal[False] = False
    universal_evidence_authorization: Literal[False] = False
    majority_direction_vote_allowed: Literal[False] = False
    unknown_context_fill_allowed: Literal[False] = False
    count_thresholds_used: Literal[False] = False


class ContractHardenedTrendMakerExposure(StrictModel):
    schema_version: Literal[
        "contract-hardened-trend-maker-exposure-v1"
    ] = "contract-hardened-trend-maker-exposure-v1"

    exposure_id: str
    exposure_sha256: str
    semantics_id: str

    source_directional_exposure_id: str
    source_directional_exposure_sha256: str
    source_input_id: str
    source_input_sha256: str
    domain_profile_id: str
    corpus_id: str

    views: list[DirectionalTrendMakerView] = Field(default_factory=list)

    eligible_gap_statement_ids: list[str] = Field(default_factory=list)
    partial_blocked_gap_statement_ids: list[str] = Field(
        default_factory=list
    )
    partial_absence_blocked_paper_ids: list[str] = Field(
        default_factory=list
    )

    policy: ContractHardenedAuthorityPolicy = Field(
        default_factory=ContractHardenedAuthorityPolicy
    )

    @model_validator(mode="after")
    def _consistency(
        self,
    ) -> "ContractHardenedTrendMakerExposure":
        if self.semantics_id != (
            HYPOTHESIS_TREND_HARDENED_EXPOSURE_SEMANTICS_ID
        ):
            raise ValueError("5i exposure semantics mismatch")

        ids = [row.source_view.view_id for row in self.views]
        if ids != sorted(set(ids)):
            raise ValueError("5i views must be sorted and unique")

        for name in (
            "eligible_gap_statement_ids",
            "partial_blocked_gap_statement_ids",
            "partial_absence_blocked_paper_ids",
        ):
            values = getattr(self, name)
            if values != sorted(set(values)):
                raise ValueError(f"{name} must be sorted and unique")

        expected_id = _stable_id(
            "contract_hardened_trend_maker_exposure",
            self.semantics_id,
            self.source_directional_exposure_sha256,
            self.source_input_sha256,
        )
        if self.exposure_id != expected_id:
            raise ValueError("5i exposure_id is not stable")

        payload = self.model_dump(mode="json")
        observed = str(payload.pop("exposure_sha256", ""))
        expected = _sha256_json(payload)
        if observed != expected:
            raise ValueError("5i exposure SHA mismatch")
        return self


def build_contract_hardened_trend_maker_exposure(
    source: TrendAwareHypothesisInput,
) -> ContractHardenedTrendMakerExposure:
    directional = build_directional_trend_maker_exposure(source)
    verify_directional_trend_maker_exposure(source, directional)

    blocked_papers = sorted(
        set(source.grounded_context.partial_absence_blocked_paper_ids)
    )
    blocked_set = set(blocked_papers)

    gaps = sorted(
        row.statement_id
        for row in source.grounded_context.evidence_statements
        if row.eligible_as_gap
    )
    partial_blocked_gaps = sorted(
        row.statement_id
        for row in source.grounded_context.evidence_statements
        if row.eligible_as_gap and set(row.paper_ids) & blocked_set
    )

    views = sorted(
        directional.views,
        key=lambda row: row.source_view.view_id,
    )

    exposure_id = _stable_id(
        "contract_hardened_trend_maker_exposure",
        HYPOTHESIS_TREND_HARDENED_EXPOSURE_SEMANTICS_ID,
        directional.exposure_sha256,
        source.input_sha256,
    )
    payload = {
        "schema_version":
            "contract-hardened-trend-maker-exposure-v1",
        "exposure_id": exposure_id,
        "semantics_id":
            HYPOTHESIS_TREND_HARDENED_EXPOSURE_SEMANTICS_ID,
        "source_directional_exposure_id": directional.exposure_id,
        "source_directional_exposure_sha256":
            directional.exposure_sha256,
        "source_input_id": source.input_id,
        "source_input_sha256": source.input_sha256,
        "domain_profile_id": source.domain_profile_id,
        "corpus_id": source.corpus_id,
        "views": [row.model_dump(mode="json") for row in views],
        "eligible_gap_statement_ids": gaps,
        "partial_blocked_gap_statement_ids": partial_blocked_gaps,
        "partial_absence_blocked_paper_ids": blocked_papers,
        "policy": ContractHardenedAuthorityPolicy().model_dump(
            mode="json"
        ),
    }
    payload["exposure_sha256"] = _sha256_json(payload)
    return ContractHardenedTrendMakerExposure.model_validate(payload)


def verify_contract_hardened_trend_maker_exposure(
    source: TrendAwareHypothesisInput,
    exposure: ContractHardenedTrendMakerExposure,
) -> None:
    expected = build_contract_hardened_trend_maker_exposure(source)
    if exposure.model_dump(mode="json") != expected.model_dump(
        mode="json"
    ):
        raise ValueError(
            "5i exposure is not the exact deterministic projection "
            "of the supplied frozen Trend input"
        )
