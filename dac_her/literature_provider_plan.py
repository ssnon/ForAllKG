from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.prior_art_retrieval import (
    CrossrefProvider,
    LiteratureSearchProvider,
    OpenAlexProvider,
    SemanticScholarProvider,
)


PROVIDER_PLAN_SEMANTICS_ID = "literature_provider_plan_v1"

ProviderName = Literal[
    "openalex",
    "crossref",
    "semantic_scholar",
]
ProviderMode = Literal[
    "FULL_3_PROVIDER",
    "STANDARD_2_PROVIDER",
    "DEGRADED_PROVIDER_SET",
    "NO_PROVIDER",
    "EXPLICIT_PROVIDER_SET",
]

AUTO_PROVIDER_ORDER: tuple[ProviderName, ...] = (
    "openalex",
    "crossref",
    "semantic_scholar",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderBinding(StrictModel):
    provider: ProviderName
    active: bool
    reason: str = Field(min_length=1)
    authentication_mode: str = Field(min_length=1)


class LiteratureProviderPlan(StrictModel):
    schema_version: Literal[
        "literature-provider-plan-v1"
    ] = "literature-provider-plan-v1"
    semantics_id: Literal[
        "literature_provider_plan_v1"
    ] = PROVIDER_PLAN_SEMANTICS_ID
    plan_id: str
    plan_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    requested_mode: Literal["auto", "explicit"]
    mode: ProviderMode
    active_providers: list[ProviderName]
    bindings: list[ProviderBinding]
    openalex_api_key_configured: bool
    crossref_mailto_configured: bool
    semantic_scholar_api_key_configured: bool
    provider_set_frozen_for_run: Literal[True] = True
    runtime_failure_changes_provider_set: Literal[False] = False
    query_rewrite_authorized: Literal[False] = False
    ranking_policy_change_authorized: Literal[False] = False
    novelty_policy_change_authorized: Literal[False] = False
    scientific_equivalence_to_full_3_provider_established: Literal[
        False
    ] = False
    secret_values_persisted: Literal[False] = False

    @model_validator(mode="after")
    def _validate_plan(self) -> "LiteratureProviderPlan":
        active_from_bindings = [
            row.provider
            for row in self.bindings
            if row.active
        ]
        if active_from_bindings != self.active_providers:
            raise ValueError(
                "active_providers does not match active bindings"
            )
        if len(self.active_providers) != len(
            set(self.active_providers)
        ):
            raise ValueError(
                "duplicate active provider"
            )

        body = self.model_dump(mode="json")
        observed_id = body.pop("plan_id")
        observed_sha = body.pop("plan_sha256")
        expected_sha = _sha256_json(body)
        expected_id = (
            "literature_provider_plan:"
            + expected_sha[:20]
        )
        if observed_sha != expected_sha:
            raise ValueError(
                "provider plan SHA mismatch"
            )
        if observed_id != expected_id:
            raise ValueError(
                "provider plan ID mismatch"
            )
        return self


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _env(
    env: Mapping[str, str] | None,
) -> Mapping[str, str]:
    return os.environ if env is None else env


def _configured(
    env: Mapping[str, str],
    key: str,
) -> bool:
    return bool(str(env.get(key, "")).strip())


def _binding(
    provider: ProviderName,
    *,
    active: bool,
    reason: str,
    authentication_mode: str,
) -> ProviderBinding:
    return ProviderBinding(
        provider=provider,
        active=active,
        reason=reason,
        authentication_mode=authentication_mode,
    )


def _mode_for_auto(
    active: Sequence[ProviderName],
) -> ProviderMode:
    values = list(active)
    if values == [
        "openalex",
        "crossref",
        "semantic_scholar",
    ]:
        return "FULL_3_PROVIDER"
    if values == [
        "openalex",
        "crossref",
    ]:
        return "STANDARD_2_PROVIDER"
    if not values:
        return "NO_PROVIDER"
    return "DEGRADED_PROVIDER_SET"


def resolve_literature_provider_plan(
    *,
    env: Mapping[str, str] | None = None,
    requested: Sequence[str] | None = None,
) -> LiteratureProviderPlan:
    values = _env(env)
    openalex_key = _configured(
        values,
        "OPENALEX_API_KEY",
    )
    s2_key = _configured(
        values,
        "SEMANTIC_SCHOLAR_API_KEY",
    )
    crossref_mailto = _configured(
        values,
        "CROSSREF_MAILTO",
    )

    if requested is None:
        requested_mode = "auto"
        bindings = [
            _binding(
                "openalex",
                active=openalex_key,
                reason=(
                    "API_KEY_CONFIGURED"
                    if openalex_key
                    else "API_KEY_ABSENT"
                ),
                authentication_mode=(
                    "api_key"
                    if openalex_key
                    else "unavailable"
                ),
            ),
            _binding(
                "crossref",
                active=True,
                reason=(
                    "PUBLIC_API_MAILTO_CONFIGURED"
                    if crossref_mailto
                    else "PUBLIC_API_NO_MAILTO"
                ),
                authentication_mode="public",
            ),
            _binding(
                "semantic_scholar",
                active=s2_key,
                reason=(
                    "API_KEY_CONFIGURED"
                    if s2_key
                    else "API_KEY_ABSENT_OPTIONAL_PROVIDER"
                ),
                authentication_mode=(
                    "api_key"
                    if s2_key
                    else "unavailable"
                ),
            ),
        ]
        active = [
            row.provider
            for row in bindings
            if row.active
        ]
        mode = _mode_for_auto(active)
    else:
        requested_mode = "explicit"
        normalized: list[ProviderName] = []
        for raw in requested:
            name = str(raw).strip().lower()
            if name not in AUTO_PROVIDER_ORDER:
                raise ValueError(
                    f"Unknown literature provider: {raw!r}"
                )
            typed = name  # type: ignore[assignment]
            if typed not in normalized:
                normalized.append(typed)

        if not normalized:
            raise ValueError(
                "Explicit provider set cannot be empty"
            )
        if (
            "openalex" in normalized
            and not openalex_key
        ):
            raise ValueError(
                "Explicit OpenAlex request requires "
                "OPENALEX_API_KEY."
            )
        if (
            "semantic_scholar" in normalized
            and not s2_key
        ):
            raise ValueError(
                "Explicit Semantic Scholar request requires "
                "SEMANTIC_SCHOLAR_API_KEY. Use --providers auto "
                "to omit Semantic Scholar when its key is absent."
            )

        bindings = []
        for provider in AUTO_PROVIDER_ORDER:
            active_now = provider in normalized
            if provider == "openalex":
                auth = (
                    "api_key"
                    if active_now
                    else "not_requested"
                )
            elif provider == "semantic_scholar":
                auth = (
                    "api_key"
                    if active_now
                    else "not_requested"
                )
            else:
                auth = (
                    "public"
                    if active_now
                    else "not_requested"
                )
            bindings.append(
                _binding(
                    provider,
                    active=active_now,
                    reason=(
                        "EXPLICITLY_REQUESTED"
                        if active_now
                        else "NOT_REQUESTED"
                    ),
                    authentication_mode=auth,
                )
            )
        active = [
            provider
            for provider in AUTO_PROVIDER_ORDER
            if provider in normalized
        ]
        mode = "EXPLICIT_PROVIDER_SET"

    body: dict[str, Any] = {
        "schema_version":
            "literature-provider-plan-v1",
        "semantics_id":
            PROVIDER_PLAN_SEMANTICS_ID,
        "requested_mode":
            requested_mode,
        "mode":
            mode,
        "active_providers":
            active,
        "bindings": [
            row.model_dump(mode="json")
            for row in bindings
        ],
        "openalex_api_key_configured":
            openalex_key,
        "crossref_mailto_configured":
            crossref_mailto,
        "semantic_scholar_api_key_configured":
            s2_key,
        "provider_set_frozen_for_run":
            True,
        "runtime_failure_changes_provider_set":
            False,
        "query_rewrite_authorized":
            False,
        "ranking_policy_change_authorized":
            False,
        "novelty_policy_change_authorized":
            False,
        "scientific_equivalence_to_full_3_provider_established":
            False,
        "secret_values_persisted":
            False,
    }
    digest = _sha256_json(body)
    return LiteratureProviderPlan(
        **body,
        plan_sha256=digest,
        plan_id=(
            "literature_provider_plan:"
            + digest[:20]
        ),
    )


def _assert_environment_matches_plan(
    plan: LiteratureProviderPlan,
    env: Mapping[str, str],
) -> None:
    observed = {
        "openalex":
            _configured(
                env,
                "OPENALEX_API_KEY",
            ),
        "semantic_scholar":
            _configured(
                env,
                "SEMANTIC_SCHOLAR_API_KEY",
            ),
    }
    if (
        observed["openalex"]
        != plan.openalex_api_key_configured
    ):
        raise RuntimeError(
            "OPENALEX_API_KEY configuration changed after "
            "provider plan freeze."
        )
    if (
        observed["semantic_scholar"]
        != plan.semantic_scholar_api_key_configured
    ):
        raise RuntimeError(
            "SEMANTIC_SCHOLAR_API_KEY configuration changed "
            "after provider plan freeze."
        )


def build_literature_providers(
    plan: LiteratureProviderPlan,
    *,
    env: Mapping[str, str] | None = None,
) -> list[LiteratureSearchProvider]:
    values = _env(env)
    _assert_environment_matches_plan(
        plan,
        values,
    )

    providers: list[LiteratureSearchProvider] = []
    for name in plan.active_providers:
        if name == "openalex":
            providers.append(
                OpenAlexProvider(
                    api_key=str(
                        values.get(
                            "OPENALEX_API_KEY",
                            "",
                        )
                    ).strip()
                )
            )
        elif name == "crossref":
            providers.append(
                CrossrefProvider(
                    mailto=(
                        str(
                            values.get(
                                "CROSSREF_MAILTO",
                                "",
                            )
                        ).strip()
                        or None
                    )
                )
            )
        elif name == "semantic_scholar":
            providers.append(
                SemanticScholarProvider(
                    api_key=str(
                        values.get(
                            "SEMANTIC_SCHOLAR_API_KEY",
                            "",
                        )
                    ).strip()
                )
            )
        else:
            raise AssertionError(
                f"Unhandled provider: {name}"
            )

    if not providers:
        raise RuntimeError(
            "No literature provider is available. Configure "
            "OPENALEX_API_KEY or explicitly configure a supported "
            "provider before retrieval."
        )
    return providers



def load_literature_provider_plan(
    path: str | Path,
) -> LiteratureProviderPlan:
    return LiteratureProviderPlan.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def require_standard_or_full_auto_plan(
    plan: LiteratureProviderPlan,
) -> None:
    if (
        plan.requested_mode == "auto"
        and plan.mode
        not in {
            "STANDARD_2_PROVIDER",
            "FULL_3_PROVIDER",
        }
    ):
        raise RuntimeError(
            "Automatic literature-provider resolution did not produce "
            "STANDARD_2_PROVIDER or FULL_3_PROVIDER. "
            "Configure OPENALEX_API_KEY; Crossref alone is not accepted "
            "as the default scientific retrieval mode."
        )
