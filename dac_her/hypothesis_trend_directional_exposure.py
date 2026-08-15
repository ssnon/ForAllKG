from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dac_her.hypothesis_trend_directional_contracts import (
    TrendDependentChange,
    canonical_dependent_change,
)
from dac_her.hypothesis_trend_input import TrendAwareHypothesisInput
from dac_her.hypothesis_trend_maker_exposure import (
    TrendMakerExposedView,
    TrendMakerExposure,
    build_trend_maker_exposure,
    verify_trend_maker_exposure,
)


HYPOTHESIS_TREND_DIRECTIONAL_EXPOSURE_SEMANTICS_ID = (
    "hypothesis_trend_directional_exposure_v1_alpha4c5d1"
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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


class DirectionalTrendMakerPolicy(StrictModel):
    source_5d_exposure_immutable: Literal[True] = True
    canonical_independent_increase_frame_required: Literal[True] = True
    every_positive_trend_support_requires_prediction_binding: Literal[
        True
    ] = True
    dependent_change_derived_from_source_direction: Literal[True] = True
    ambiguous_direction_maps_to_unspecified: Literal[True] = True
    decrease_reframing_in_generated_directional_text_allowed: Literal[
        False
    ] = False
    source_5b_views_mutated: Literal[False] = False
    source_5c_contracts_mutated: Literal[False] = False
    source_5d_runtime_mutated: Literal[False] = False
    trend_causal_authorization: Literal[False] = False
    trend_universal_authorization: Literal[False] = False


class DirectionalTrendMakerView(StrictModel):
    source_view: TrendMakerExposedView
    canonical_independent_change: Literal["increase"] = "increase"
    expected_dependent_change: TrendDependentChange
    direction_binding_required_if_positive: bool


class DirectionalTrendMakerExposure(StrictModel):
    schema_version: Literal[
        "directional-trend-maker-exposure-v1"
    ] = "directional-trend-maker-exposure-v1"

    exposure_id: str
    exposure_sha256: str
    semantics_id: str

    source_5d_exposure_id: str
    source_5d_exposure_sha256: str
    source_input_id: str
    source_input_sha256: str
    domain_profile_id: str
    corpus_id: str

    views: list[DirectionalTrendMakerView] = Field(
        default_factory=list
    )
    policy: DirectionalTrendMakerPolicy = Field(
        default_factory=DirectionalTrendMakerPolicy
    )

    @model_validator(mode="after")
    def _consistency(self) -> "DirectionalTrendMakerExposure":
        if self.semantics_id != (
            HYPOTHESIS_TREND_DIRECTIONAL_EXPOSURE_SEMANTICS_ID
        ):
            raise ValueError(
                "Directional Trend Maker exposure semantics mismatch."
            )
        expected_id = _stable_id(
            "directional_trend_maker_exposure",
            self.semantics_id,
            self.source_5d_exposure_sha256,
        )
        if self.exposure_id != expected_id:
            raise ValueError(
                "Directional Trend Maker exposure_id is not stable."
            )
        payload = self.model_dump(mode="json")
        observed = str(payload.pop("exposure_sha256", ""))
        expected = _sha256_json(payload)
        if observed != expected:
            raise ValueError(
                "Directional Trend Maker exposure SHA mismatch."
            )
        return self


def build_directional_trend_maker_exposure(
    source: TrendAwareHypothesisInput,
    *,
    source_exposure: TrendMakerExposure | None = None,
) -> DirectionalTrendMakerExposure:
    source_exposure = (
        source_exposure or build_trend_maker_exposure(source)
    )
    verify_trend_maker_exposure(source, source_exposure)

    views: list[DirectionalTrendMakerView] = []
    for row in source_exposure.views:
        views.append(
            DirectionalTrendMakerView(
                source_view=row,
                canonical_independent_change="increase",
                expected_dependent_change=
                    canonical_dependent_change(row.directions),
                direction_binding_required_if_positive=(
                    row.allowed_use_role
                    in {
                        "positive_empirical_support",
                        "cross_paper_empirical_support",
                    }
                ),
            )
        )

    exposure_id = _stable_id(
        "directional_trend_maker_exposure",
        HYPOTHESIS_TREND_DIRECTIONAL_EXPOSURE_SEMANTICS_ID,
        source_exposure.exposure_sha256,
    )
    payload = {
        "schema_version":
            "directional-trend-maker-exposure-v1",
        "exposure_id": exposure_id,
        "semantics_id":
            HYPOTHESIS_TREND_DIRECTIONAL_EXPOSURE_SEMANTICS_ID,
        "source_5d_exposure_id": source_exposure.exposure_id,
        "source_5d_exposure_sha256":
            source_exposure.exposure_sha256,
        "source_input_id": source.input_id,
        "source_input_sha256": source.input_sha256,
        "domain_profile_id": source.domain_profile_id,
        "corpus_id": source.corpus_id,
        "views": [
            row.model_dump(mode="json") for row in views
        ],
        "policy": DirectionalTrendMakerPolicy().model_dump(
            mode="json"
        ),
    }
    payload["exposure_sha256"] = _sha256_json(payload)
    return DirectionalTrendMakerExposure.model_validate(payload)


def verify_directional_trend_maker_exposure(
    source: TrendAwareHypothesisInput,
    exposure: DirectionalTrendMakerExposure,
) -> None:
    base = build_trend_maker_exposure(source)
    verify_trend_maker_exposure(source, base)
    if exposure.source_5d_exposure_id != base.exposure_id:
        raise ValueError(
            "Directional exposure source 5d exposure ID mismatch."
        )
    if exposure.source_5d_exposure_sha256 != base.exposure_sha256:
        raise ValueError(
            "Directional exposure source 5d exposure SHA mismatch."
        )
    rebuilt = build_directional_trend_maker_exposure(
        source,
        source_exposure=base,
    )
    if rebuilt.model_dump(mode="json") != exposure.model_dump(
        mode="json"
    ):
        raise ValueError(
            "Directional Trend Maker exposure is not deterministic."
        )
