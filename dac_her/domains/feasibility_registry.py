from __future__ import annotations

from pipeline_core.domain_profile import ScientificDomainProfile
from domains.dac_her.feasibility import DAC_HER_FEASIBILITY_ADAPTER
from domains.registry import get_domain_profile
from pipeline_core.feasibility_domain import FeasibilityDomainAdapter


_ADAPTERS: dict[str, FeasibilityDomainAdapter] = {
    DAC_HER_FEASIBILITY_ADAPTER.adapter_id: DAC_HER_FEASIBILITY_ADAPTER,
}


def register_feasibility_adapter(
    adapter: FeasibilityDomainAdapter,
    *,
    replace: bool = False,
) -> None:
    key = adapter.adapter_id.strip().lower()
    if not key:
        raise ValueError("feasibility adapter_id must not be empty")
    if key in _ADAPTERS and not replace:
        raise ValueError(f"feasibility adapter already registered: {key}")
    _ADAPTERS[key] = adapter


def resolve_feasibility_adapter(
    profile: ScientificDomainProfile,
) -> FeasibilityDomainAdapter:
    adapter_id = (profile.feasibility_adapter_id or "").strip().lower()
    if not adapter_id:
        raise ValueError(
            f"Scientific domain profile {profile.profile_id!r} has no feasibility "
            "adapter. Refusing to apply another domain's scientific rules."
        )
    try:
        adapter = _ADAPTERS[adapter_id]
    except KeyError as exc:
        available = ", ".join(sorted(_ADAPTERS))
        raise ValueError(
            f"Unknown feasibility adapter {adapter_id!r} for domain profile "
            f"{profile.profile_id!r}; available adapters: {available}"
        ) from exc

    if adapter.domain_profile_id != profile.profile_id:
        raise ValueError(
            "Feasibility adapter/domain mismatch: "
            f"profile={profile.profile_id!r}, "
            f"adapter={adapter.adapter_id!r}, "
            f"adapter_domain={adapter.domain_profile_id!r}"
        )
    return adapter


def get_feasibility_adapter(
    profile_id: str | None = None,
) -> FeasibilityDomainAdapter:
    return resolve_feasibility_adapter(
        get_domain_profile(profile_id)
    )


def available_feasibility_adapters() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))
