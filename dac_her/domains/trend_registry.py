from __future__ import annotations

from dac_her.domain_profile import ScientificDomainProfile
from dac_her.domains.registry import get_domain_profile
from dac_her.domains.sers_au_ag_trend_alpha4c2121 import SERS_AU_AG_TREND_ADAPTER
from dac_her.trend_domain import TrendDomainAdapter


class TrendAdapterUnavailableError(RuntimeError):
    pass


_ADAPTERS: dict[str, TrendDomainAdapter] = {
    SERS_AU_AG_TREND_ADAPTER.adapter_id: SERS_AU_AG_TREND_ADAPTER,
}


def available_trend_adapters() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


def register_trend_adapter(
    adapter: TrendDomainAdapter,
    *,
    replace: bool = False,
) -> None:
    if adapter.adapter_id in _ADAPTERS and not replace:
        raise ValueError(
            f"Trend adapter already registered: {adapter.adapter_id!r}."
        )
    _ADAPTERS[adapter.adapter_id] = adapter


def get_trend_adapter(
    profile: ScientificDomainProfile | str,
) -> TrendDomainAdapter:
    resolved = (
        profile
        if isinstance(profile, ScientificDomainProfile)
        else get_domain_profile(str(profile))
    )
    adapter_id = resolved.trend_adapter_id
    if not adapter_id:
        raise TrendAdapterUnavailableError(
            "No trend adapter is registered for domain profile "
            f"{resolved.profile_id!r}."
        )
    try:
        adapter = _ADAPTERS[adapter_id]
    except KeyError as exc:
        raise TrendAdapterUnavailableError(
            f"Unknown trend adapter {adapter_id!r} for domain profile "
            f"{resolved.profile_id!r}."
        ) from exc
    if adapter.domain_profile_id != resolved.profile_id:
        raise TrendAdapterUnavailableError(
            "Trend adapter/domain mismatch: "
            f"{adapter.domain_profile_id!r} != {resolved.profile_id!r}."
        )
    return adapter
