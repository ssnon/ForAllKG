from __future__ import annotations

from dac_her.cross_context_trend import CrossContextTrendAdapter
from dac_her.domain_profile import ScientificDomainProfile
from dac_her.domains.registry import get_domain_profile
from domains.sers.cross_context_trend import (
    SERS_AU_AG_CROSS_CONTEXT_TREND_ADAPTER,
)


class CrossContextTrendAdapterUnavailableError(RuntimeError):
    pass


_ADAPTERS: dict[str, CrossContextTrendAdapter] = {
    SERS_AU_AG_CROSS_CONTEXT_TREND_ADAPTER.adapter_id:
        SERS_AU_AG_CROSS_CONTEXT_TREND_ADAPTER,
}


def available_cross_context_trend_adapters() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


def register_cross_context_trend_adapter(
    adapter: CrossContextTrendAdapter,
    *,
    replace: bool = False,
) -> None:
    if adapter.adapter_id in _ADAPTERS and not replace:
        raise ValueError(
            "Cross-context trend adapter already registered: "
            f"{adapter.adapter_id!r}."
        )
    _ADAPTERS[adapter.adapter_id] = adapter


def get_cross_context_trend_adapter(
    profile: ScientificDomainProfile | str,
) -> CrossContextTrendAdapter:
    resolved = (
        profile
        if isinstance(profile, ScientificDomainProfile)
        else get_domain_profile(str(profile))
    )

    # Reuse the frozen trend-adapter identity. No new domain-profile field is
    # introduced solely for the 4c.3 context layer.
    adapter_id = resolved.trend_adapter_id
    if not adapter_id:
        raise CrossContextTrendAdapterUnavailableError(
            "No trend adapter is active for domain profile "
            f"{resolved.profile_id!r}."
        )
    try:
        adapter = _ADAPTERS[adapter_id]
    except KeyError as exc:
        raise CrossContextTrendAdapterUnavailableError(
            "No cross-context trend adapter is registered for "
            f"{adapter_id!r}."
        ) from exc

    if adapter.domain_profile_id != resolved.profile_id:
        raise CrossContextTrendAdapterUnavailableError(
            "Cross-context trend adapter/domain mismatch: "
            f"{adapter.domain_profile_id!r} != "
            f"{resolved.profile_id!r}."
        )
    return adapter
