from __future__ import annotations

from dac_her.domain_profile import ScientificDomainProfile
from dac_her.domains.registry import get_domain_profile
from domains.sers.trend_precision_alpha4c21211 import (
    SERS_AU_AG_TREND_PRECISION_ADAPTER,
)
from dac_her.trend_precision import TrendPrecisionAdapter


class TrendPrecisionAdapterUnavailableError(RuntimeError):
    pass


_ADAPTERS: dict[str, TrendPrecisionAdapter] = {
    SERS_AU_AG_TREND_PRECISION_ADAPTER.adapter_id:
        SERS_AU_AG_TREND_PRECISION_ADAPTER,
}


def available_trend_precision_adapters() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


def get_trend_precision_adapter(
    profile: ScientificDomainProfile | str,
) -> TrendPrecisionAdapter:
    resolved = (
        profile
        if isinstance(profile, ScientificDomainProfile)
        else get_domain_profile(str(profile))
    )
    adapter_id = resolved.trend_adapter_id
    if not adapter_id:
        raise TrendPrecisionAdapterUnavailableError(
            "No trend adapter is active for domain profile "
            f"{resolved.profile_id!r}."
        )
    try:
        adapter = _ADAPTERS[adapter_id]
    except KeyError as exc:
        raise TrendPrecisionAdapterUnavailableError(
            f"No trend precision adapter is registered for {adapter_id!r}."
        ) from exc
    if adapter.domain_profile_id != resolved.profile_id:
        raise TrendPrecisionAdapterUnavailableError(
            "Trend precision adapter/domain mismatch: "
            f"{adapter.domain_profile_id!r} != {resolved.profile_id!r}."
        )
    return adapter
