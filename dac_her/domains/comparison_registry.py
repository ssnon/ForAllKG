from __future__ import annotations

from dac_her.comparison_domain import ComparisonDomainAdapter
from dac_her.domain_profile import ScientificDomainProfile
from dac_her.domains.registry import get_domain_profile
from dac_her.domains.sers_au_ag_comparison import (
    SERS_AU_AG_COMPARISON_ADAPTER,
)


class ComparisonAdapterUnavailableError(RuntimeError):
    pass


_ADAPTERS: dict[str, ComparisonDomainAdapter] = {
    SERS_AU_AG_COMPARISON_ADAPTER.adapter_id: (
        SERS_AU_AG_COMPARISON_ADAPTER
    ),
}


def available_comparison_adapters() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


def get_comparison_adapter(
    profile: ScientificDomainProfile | str,
) -> ComparisonDomainAdapter:
    resolved = (
        profile
        if isinstance(profile, ScientificDomainProfile)
        else get_domain_profile(str(profile))
    )
    adapter_id = resolved.comparison_adapter_id
    if not adapter_id:
        raise ComparisonAdapterUnavailableError(
            "No comparison adapter is registered for domain profile "
            f"{resolved.profile_id!r}."
        )
    try:
        adapter = _ADAPTERS[adapter_id]
    except KeyError as exc:
        raise ComparisonAdapterUnavailableError(
            "Unknown comparison adapter "
            f"{adapter_id!r} for domain profile {resolved.profile_id!r}."
        ) from exc
    if adapter.domain_profile_id != resolved.profile_id:
        raise ComparisonAdapterUnavailableError(
            "Comparison adapter/domain mismatch: "
            f"{adapter.domain_profile_id!r} != {resolved.profile_id!r}."
        )
    return adapter
