from __future__ import annotations

from pipeline_core.domain_profile import ScientificDomainProfile
from domains.catalysis_mechanism.profile import CATALYSIS_MECHANISM_PROFILE
from domains.dac_her.profile import DAC_HER_PROFILE
from domains.sers.profile import SERS_AU_AG_PROFILE


_PROFILES: dict[str, ScientificDomainProfile] = {
    CATALYSIS_MECHANISM_PROFILE.profile_id: CATALYSIS_MECHANISM_PROFILE,
    DAC_HER_PROFILE.profile_id: DAC_HER_PROFILE,
    SERS_AU_AG_PROFILE.profile_id: SERS_AU_AG_PROFILE,
}
_ALIASES = {
    'default': 'dac_her',
    'her': 'dac_her',
    'dac-her': 'dac_her',
    'broad': 'catalysis_mechanism',
    'broad-catalysis': 'catalysis_mechanism',
    'catalysis-mechanism': 'catalysis_mechanism',
    'sers': 'sers_au_ag',
    'au-ag-sers': 'sers_au_ag',
    'sers-au-ag': 'sers_au_ag',
}


def register_domain_profile(
    profile: ScientificDomainProfile,
    *,
    replace: bool = False,
) -> None:
    key = profile.profile_id.strip().lower()
    if not key:
        raise ValueError('domain profile_id must not be empty')
    if key in _PROFILES and not replace:
        raise ValueError(f'domain profile already registered: {key}')
    _PROFILES[key] = profile


def get_domain_profile(profile_id: str | None = None) -> ScientificDomainProfile:
    raw = (profile_id or 'default').strip().lower()
    key = _ALIASES.get(raw, raw)
    try:
        return _PROFILES[key]
    except KeyError as exc:
        available = ', '.join(sorted(_PROFILES))
        raise ValueError(
            f'unknown scientific domain profile {profile_id!r}; available: {available}'
        ) from exc


def available_domain_profiles() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES))
