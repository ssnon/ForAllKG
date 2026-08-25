from __future__ import annotations

from domains.registry import get_domain_profile


def _scope(text: str) -> set[str]:
    return get_domain_profile(
        "sers_au_ag"
    ).novelty.scope_features(text)


def test_unicode_dash_au_ag_is_recognized() -> None:
    assert "au_ag" in _scope(
        "Hybridization of localized surface plasmon "
        "resonance-based Au–Ag nanoparticles"
    )

    assert "au_ag" in _scope(
        "Ag–Au nanoparticle heterodimers"
    )


def test_silver_film_gold_particle_architecture_is_not_au_ag_alias() -> None:
    # This is the false-positive diagnostic case. Silver and gold occur
    # in one optical architecture, but the text does not assert an Au-Ag
    # bimetallic particle/dimer identity.
    text = (
        "The extended plasmon wave propagates on the silver film "
        "surface and couples with the gold nanoparticles dispersed "
        "on top."
    )

    assert "au_ag" not in _scope(text)


def test_interparticle_spacing_is_nanogap_family() -> None:
    assert "nanogap" in _scope(
        "A new interparticle-spacing-dependent coupling "
        "model for heterodimers is required."
    )


def test_interparticle_separation_is_nanogap_family() -> None:
    assert "nanogap" in _scope(
        "The effect of interparticle separation on the "
        "near field enhancement is analyzed."
    )


def test_generic_separation_does_not_become_nanogap() -> None:
    assert "nanogap" not in _scope(
        "The sample separation between two measurement "
        "groups was varied."
    )


def test_collective_plasmon_mode_is_lspr_family_but_generic_coupling_is_not() -> None:
    assert "lspr" in _scope(
        "Symmetry breaking occurs in collective plasmon modes."
    )

    assert "lspr" in _scope(
        "The coupled plasmon mode shifts with geometry."
    )

    # Deliberately do not make all plasmonic coupling an LSPR feature.
    assert "lspr" not in _scope(
        "Direct plasmonic coupling between the particles is observed."
    )


def test_electromagnetic_field_enhancement_is_recognized() -> None:
    assert "electromagnetic_enhancement" in _scope(
        "The structures exhibit strong electromagnetic field "
        "enhancement and surface-enhanced Raman scattering."
    )
