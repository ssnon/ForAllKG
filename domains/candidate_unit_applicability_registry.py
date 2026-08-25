from __future__ import annotations

from typing import Any

from domains.registry import get_domain_profile
from domains.sers.candidate_unit_applicability import (
    SERS_AU_AG_CANDIDATE_UNIT_APPLICABILITY,
)
from pipeline_core.domain.domain_profile import ScientificDomainProfile


_ADAPTERS: dict[str, Any] = {
    SERS_AU_AG_CANDIDATE_UNIT_APPLICABILITY.adapter_id:
        SERS_AU_AG_CANDIDATE_UNIT_APPLICABILITY,
}

_PROFILE_CAPABILITIES: dict[str, str] = {
    "sers_au_ag":
        SERS_AU_AG_CANDIDATE_UNIT_APPLICABILITY.adapter_id,
}


def resolve_candidate_unit_applicability_adapter(
    profile: ScientificDomainProfile,
) -> Any:
    profile_id = str(profile.profile_id).strip().lower()

    try:
        adapter_id = _PROFILE_CAPABILITIES[profile_id]
    except KeyError as exc:
        raise ValueError(
            "Scientific domain profile "
            f"{profile.profile_id!r} has no candidate-unit "
            "applicability capability."
        ) from exc

    try:
        adapter = _ADAPTERS[adapter_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown candidate-unit applicability adapter "
            f"{adapter_id!r}"
        ) from exc

    if (
        str(adapter.domain_profile_id).strip().lower()
        != profile_id
    ):
        raise ValueError(
            "candidate-unit applicability adapter/domain mismatch"
        )

    return adapter


def get_candidate_unit_applicability_adapter(
    profile_id: str,
) -> Any:
    if not str(profile_id or "").strip():
        raise ValueError(
            "candidate-unit applicability profile_id must be explicit"
        )

    return resolve_candidate_unit_applicability_adapter(
        get_domain_profile(profile_id)
    )


def available_candidate_unit_applicability_profiles(
) -> tuple[str, ...]:
    return tuple(sorted(_PROFILE_CAPABILITIES))
