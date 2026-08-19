from __future__ import annotations

from dac_her.domain_profile import ScientificDomainProfile
from dac_her.domains.registry import get_domain_profile
from domains.sers.metric_definition import (
    SERS_AU_AG_METRIC_DEFINITION_ADAPTER,
)
from dac_her.metric_definition_domain import MetricDefinitionDomainAdapter


class MetricDefinitionAdapterUnavailableError(RuntimeError):
    pass


_ADAPTERS: dict[str, MetricDefinitionDomainAdapter] = {
    SERS_AU_AG_METRIC_DEFINITION_ADAPTER.adapter_id: (
        SERS_AU_AG_METRIC_DEFINITION_ADAPTER
    ),
}


def available_metric_definition_adapters() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


def get_metric_definition_adapter(
    profile: ScientificDomainProfile | str,
) -> MetricDefinitionDomainAdapter:
    resolved = (
        profile
        if isinstance(profile, ScientificDomainProfile)
        else get_domain_profile(str(profile))
    )
    adapter_id = resolved.metric_definition_adapter_id
    if not adapter_id:
        raise MetricDefinitionAdapterUnavailableError(
            "No metric-definition adapter is registered for domain profile "
            f"{resolved.profile_id!r}."
        )
    try:
        adapter = _ADAPTERS[adapter_id]
    except KeyError as exc:
        raise MetricDefinitionAdapterUnavailableError(
            "Unknown metric-definition adapter "
            f"{adapter_id!r} for domain profile {resolved.profile_id!r}."
        ) from exc
    if adapter.domain_profile_id != resolved.profile_id:
        raise MetricDefinitionAdapterUnavailableError(
            "Metric-definition adapter/domain mismatch: "
            f"{adapter.domain_profile_id!r} != {resolved.profile_id!r}."
        )
    return adapter
