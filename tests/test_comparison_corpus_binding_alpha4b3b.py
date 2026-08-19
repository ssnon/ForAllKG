from __future__ import annotations

from domains.sers.profile import SERS_AU_AG_PROFILE


def test_alpha4b3b_sers_profile_declares_comparison_adapter():
    assert SERS_AU_AG_PROFILE.comparison_adapter_id == "sers_au_ag"
    assert SERS_AU_AG_PROFILE.corpus is not None
    assert (
        SERS_AU_AG_PROFILE.corpus.semantics_id
        == "sers_au_ag_corpus_v1_alpha4b3a"
    )
