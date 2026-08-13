from __future__ import annotations

from dac_her.domain_profile import ScientificDomainProfile
from dac_her.domains.registry import get_domain_profile
from dac_her.domains.sers_au_ag_reproducibility import (
    SERS_AU_AG_REPRODUCIBILITY_ADAPTER,
)
from dac_her.reproducibility_domain import ReproducibilityDomainAdapter


class ReproducibilityAdapterUnavailableError(RuntimeError):
    pass


_ADAPTERS: dict[str, ReproducibilityDomainAdapter] = {
    SERS_AU_AG_REPRODUCIBILITY_ADAPTER.adapter_id: (
        SERS_AU_AG_REPRODUCIBILITY_ADAPTER
    ),
}


def available_reproducibility_adapters() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


def get_reproducibility_adapter(
    profile: ScientificDomainProfile | str,
) -> ReproducibilityDomainAdapter:
    resolved = (
        profile
        if isinstance(profile, ScientificDomainProfile)
        else get_domain_profile(str(profile))
    )
    adapter_id = resolved.reproducibility_adapter_id
    if not adapter_id:
        raise ReproducibilityAdapterUnavailableError(
            "No reproducibility adapter is registered for domain profile "
            f"{resolved.profile_id!r}."
        )
    try:
        adapter = _ADAPTERS[adapter_id]
    except KeyError as exc:
        raise ReproducibilityAdapterUnavailableError(
            "Unknown reproducibility adapter "
            f"{adapter_id!r} for domain profile {resolved.profile_id!r}."
        ) from exc
    if adapter.domain_profile_id != resolved.profile_id:
        raise ReproducibilityAdapterUnavailableError(
            "Reproducibility adapter/domain mismatch: "
            f"{adapter.domain_profile_id!r} != {resolved.profile_id!r}."
        )
    return adapter
