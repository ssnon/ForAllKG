import pytest

from dac_her.domains.reproducibility_registry import (
    ReproducibilityAdapterUnavailableError,
    available_reproducibility_adapters,
    get_reproducibility_adapter,
)


def test_only_sers_reproducibility_adapter_is_registered():
    assert available_reproducibility_adapters() == ("sers_au_ag",)
    adapter = get_reproducibility_adapter("sers_au_ag")
    assert adapter.domain_profile_id == "sers_au_ag"
    assert adapter.semantics_id == "sers_au_ag_reproducibility_v2_alpha4b3b4a1"


def test_domains_without_reproducibility_semantics_fail_closed():
    with pytest.raises(ReproducibilityAdapterUnavailableError):
        get_reproducibility_adapter("dac_her")
