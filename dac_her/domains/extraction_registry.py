from __future__ import annotations

from dac_her.domain_profile import ScientificDomainProfile
from dac_her.domains.catalysis_mechanism_extraction import (
    CATALYSIS_MECHANISM_EXTRACTION_ADAPTER,
)
from domains.dac_her.extraction import DAC_HER_EXTRACTION_ADAPTER
from domains.sers.extraction import SERS_AU_AG_EXTRACTION_ADAPTER
from dac_her.domains.registry import get_domain_profile
from pipeline_core.extraction_domain import ExtractionDomainAdapter


_ADAPTERS: dict[str, ExtractionDomainAdapter] = {
    CATALYSIS_MECHANISM_EXTRACTION_ADAPTER.adapter_id: (
        CATALYSIS_MECHANISM_EXTRACTION_ADAPTER
    ),
    DAC_HER_EXTRACTION_ADAPTER.adapter_id: DAC_HER_EXTRACTION_ADAPTER,
    SERS_AU_AG_EXTRACTION_ADAPTER.adapter_id: SERS_AU_AG_EXTRACTION_ADAPTER,
}


def register_extraction_adapter(
    adapter: ExtractionDomainAdapter,
    *,
    replace: bool = False,
) -> None:
    key = adapter.adapter_id.strip().lower()
    if not key:
        raise ValueError("extraction adapter_id must not be empty")
    if key in _ADAPTERS and not replace:
        raise ValueError(f"extraction adapter already registered: {key}")
    _ADAPTERS[key] = adapter


def resolve_extraction_adapter(
    profile: ScientificDomainProfile,
) -> ExtractionDomainAdapter:
    adapter_id = (profile.extraction_adapter_id or "").strip().lower()
    if not adapter_id:
        raise ValueError(
            f"Scientific domain profile {profile.profile_id!r} has no extraction "
            "adapter. Refusing to use another domain's extraction semantics."
        )
    try:
        adapter = _ADAPTERS[adapter_id]
    except KeyError as exc:
        available = ", ".join(sorted(_ADAPTERS))
        raise ValueError(
            f"Unknown extraction adapter {adapter_id!r} for profile "
            f"{profile.profile_id!r}; available: {available}"
        ) from exc
    if adapter.domain_profile_id != profile.profile_id:
        raise ValueError(
            "Extraction adapter/domain mismatch: "
            f"profile={profile.profile_id!r}, "
            f"adapter={adapter.adapter_id!r}, "
            f"adapter_domain={adapter.domain_profile_id!r}"
        )
    return adapter


def get_extraction_adapter(
    profile_id: str | None = None,
) -> ExtractionDomainAdapter:
    return resolve_extraction_adapter(get_domain_profile(profile_id))


def available_extraction_adapters() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))
