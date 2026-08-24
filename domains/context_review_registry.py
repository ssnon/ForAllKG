from __future__ import annotations

from typing import Any

from domains.registry import get_domain_profile
from domains.sers.context_review_adapter import (
    SERS_AU_AG_CONTEXT_REVIEW_ADAPTER,
)
from pipeline_core.domain.domain_profile import (
    ScientificDomainProfile,
)


_ADAPTERS: dict[str, Any] = {
    SERS_AU_AG_CONTEXT_REVIEW_ADAPTER.adapter_id:
        SERS_AU_AG_CONTEXT_REVIEW_ADAPTER,
}


# Context review is an optional discovery capability, not part of the
# historical ScientificDomainProfile identity contract.
_PROFILE_CAPABILITIES: dict[str, str] = {
    "sers_au_ag":
        SERS_AU_AG_CONTEXT_REVIEW_ADAPTER.adapter_id,
}


def register_context_review_adapter(
    adapter: Any,
    *,
    replace: bool = False,
) -> None:
    adapter_id = str(
        adapter.adapter_id
    ).strip().lower()

    if not adapter_id:
        raise ValueError(
            "context-review adapter_id must not be empty"
        )

    if (
        adapter_id in _ADAPTERS
        and not replace
    ):
        raise ValueError(
            "context-review adapter already registered: "
            f"{adapter_id}"
        )

    _ADAPTERS[
        adapter_id
    ] = adapter


def register_context_review_capability(
    *,
    profile_id: str,
    adapter_id: str,
    replace: bool = False,
) -> None:
    profile_key = str(
        profile_id
    ).strip().lower()

    adapter_key = str(
        adapter_id
    ).strip().lower()

    if not profile_key:
        raise ValueError(
            "context-review profile_id must not be empty"
        )

    if not adapter_key:
        raise ValueError(
            "context-review capability adapter_id "
            "must not be empty"
        )

    if adapter_key not in _ADAPTERS:
        raise ValueError(
            "cannot register context-review capability "
            f"for unknown adapter {adapter_key!r}"
        )

    if (
        profile_key in _PROFILE_CAPABILITIES
        and not replace
    ):
        raise ValueError(
            "context-review capability already registered "
            f"for profile {profile_key!r}"
        )

    _PROFILE_CAPABILITIES[
        profile_key
    ] = adapter_key


def resolve_context_review_adapter(
    profile: ScientificDomainProfile,
) -> Any:
    profile_id = str(
        profile.profile_id
    ).strip().lower()

    try:
        adapter_id = (
            _PROFILE_CAPABILITIES[
                profile_id
            ]
        )
    except KeyError as exc:
        raise ValueError(
            "Scientific domain profile "
            f"{profile.profile_id!r} has no "
            "context-review capability. Refusing "
            "to apply another domain's scientific "
            "context rules."
        ) from exc

    try:
        adapter = _ADAPTERS[
            adapter_id
        ]
    except KeyError as exc:
        available = ", ".join(
            sorted(_ADAPTERS)
        )

        raise ValueError(
            "Unknown context-review adapter "
            f"{adapter_id!r} for domain profile "
            f"{profile.profile_id!r}; "
            f"available adapters: {available}"
        ) from exc

    if (
        str(
            adapter.domain_profile_id
        ).strip().lower()
        != profile_id
    ):
        raise ValueError(
            "Context-review adapter/domain mismatch: "
            f"profile={profile.profile_id!r}, "
            f"adapter={adapter.adapter_id!r}, "
            f"adapter_domain="
            f"{adapter.domain_profile_id!r}"
        )

    return adapter


def get_context_review_adapter(
    profile_id: str,
) -> Any:
    if (
        profile_id is None
        or not str(
            profile_id
        ).strip()
    ):
        raise ValueError(
            "context-review domain profile_id "
            "must be explicit"
        )

    return resolve_context_review_adapter(
        get_domain_profile(
            profile_id
        )
    )


def available_context_review_adapters() -> tuple[str, ...]:
    return tuple(
        sorted(_ADAPTERS)
    )


def available_context_review_profiles() -> tuple[str, ...]:
    return tuple(
        sorted(
            _PROFILE_CAPABILITIES
        )
    )
