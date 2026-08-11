from __future__ import annotations

from dac_her.domain_profile import ScientificDomainProfile
from dac_her.domains.dac_her_graph import DAC_HER_GRAPH_ADAPTER
from dac_her.domains.registry import get_domain_profile
from dac_her.domains.sers_au_ag_graph import SERS_AU_AG_GRAPH_ADAPTER
from dac_her.graph_domain import GraphDomainAdapter


_ADAPTERS: dict[str, GraphDomainAdapter] = {
    DAC_HER_GRAPH_ADAPTER.adapter_id: DAC_HER_GRAPH_ADAPTER,
    SERS_AU_AG_GRAPH_ADAPTER.adapter_id: SERS_AU_AG_GRAPH_ADAPTER,
}


def register_graph_adapter(
    adapter: GraphDomainAdapter,
    *,
    replace: bool = False,
) -> None:
    key = adapter.adapter_id.strip().lower()
    if not key:
        raise ValueError("graph adapter_id must not be empty")
    if key in _ADAPTERS and not replace:
        raise ValueError(f"graph adapter already registered: {key}")
    _ADAPTERS[key] = adapter


def resolve_graph_adapter(
    profile: ScientificDomainProfile,
) -> GraphDomainAdapter:
    adapter_id = (profile.graph_adapter_id or "").strip().lower()
    if not adapter_id:
        raise ValueError(
            f"Scientific domain profile {profile.profile_id!r} has no graph "
            "adapter. Refusing to use another domain's graph semantics."
        )
    try:
        adapter = _ADAPTERS[adapter_id]
    except KeyError as exc:
        available = ", ".join(sorted(_ADAPTERS))
        raise ValueError(
            f"Unknown graph adapter {adapter_id!r} for profile "
            f"{profile.profile_id!r}; available: {available}"
        ) from exc
    if adapter.domain_profile_id != profile.profile_id:
        raise ValueError(
            "Graph adapter/domain mismatch: "
            f"profile={profile.profile_id!r}, "
            f"adapter={adapter.adapter_id!r}, "
            f"adapter_domain={adapter.domain_profile_id!r}"
        )
    return adapter


def get_graph_adapter(
    profile_id: str | None = None,
) -> GraphDomainAdapter:
    return resolve_graph_adapter(get_domain_profile(profile_id))


def available_graph_adapters() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))
