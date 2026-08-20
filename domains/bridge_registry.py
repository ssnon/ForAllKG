from __future__ import annotations

from pipeline_core.bridge_domain import BridgeDomainAdapter
from pipeline_core.domain_profile import ScientificDomainProfile
from domains.dac_her.bridge import DAC_HER_BRIDGE_ADAPTER
from domains.sers.bridge import SERS_AU_AG_BRIDGE_ADAPTER
from domains.registry import get_domain_profile


_ADAPTERS: dict[str, BridgeDomainAdapter] = {
    DAC_HER_BRIDGE_ADAPTER.adapter_id: DAC_HER_BRIDGE_ADAPTER,
    SERS_AU_AG_BRIDGE_ADAPTER.adapter_id: SERS_AU_AG_BRIDGE_ADAPTER,
}


def register_bridge_adapter(
    adapter: BridgeDomainAdapter,
    *,
    replace: bool = False,
) -> None:
    key = adapter.adapter_id.strip().lower()
    if not key:
        raise ValueError("Bridge adapter_id must not be empty.")
    if key in _ADAPTERS and not replace:
        raise ValueError(f"Bridge adapter already registered: {key}")
    _ADAPTERS[key] = adapter


def resolve_bridge_adapter(
    profile: ScientificDomainProfile,
) -> BridgeDomainAdapter:
    adapter_id = (profile.bridge_adapter_id or "").strip().lower()
    if not adapter_id:
        raise ValueError(
            f"Scientific domain profile {profile.profile_id!r} has no Bridge "
            "adapter. Refusing to use another domain's Bridge semantics."
        )

    try:
        adapter = _ADAPTERS[adapter_id]
    except KeyError as exc:
        available = ", ".join(sorted(_ADAPTERS))
        raise ValueError(
            f"Unknown Bridge adapter {adapter_id!r} for profile "
            f"{profile.profile_id!r}; available: {available}"
        ) from exc

    if adapter.domain_profile_id != profile.profile_id:
        raise ValueError(
            "Bridge adapter/domain mismatch: "
            f"profile={profile.profile_id!r}, "
            f"adapter={adapter.adapter_id!r}, "
            f"adapter_domain={adapter.domain_profile_id!r}"
        )
    return adapter


def get_bridge_adapter(
    profile_id: str | None = None,
) -> BridgeDomainAdapter:
    return resolve_bridge_adapter(get_domain_profile(profile_id))


def available_bridge_adapters() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))
