from __future__ import annotations

from domains.registry import get_domain_profile


def _novelty():
    return get_domain_profile(
        "sers_au_ag"
    ).novelty


def test_domain_neutral_optical_claim_does_not_require_sers_document() -> None:
    novelty = _novelty()

    claim = (
        "Varying interparticle separation across otherwise "
        "comparable Au Ag dimers produces a measurable change "
        "in near field enhancement."
    )

    document = (
        "Near Field Enhancement in Ag Au Nanospheres "
        "Heterodimers. The optical response and near field "
        "enhancement are evaluated as interparticle separation "
        "is varied."
    )

    compatible, domain, scope, reasons = (
        novelty.strong_scope_compatibility(
            claim,
            document,
            min_domain=0.75,
            min_scope=0.75,
        )
    )

    assert novelty.domains(claim) == set()

    # Neutral domain score is diagnostic, not a hard failure.
    assert domain == 0.5
    assert scope == 1.0

    assert compatible is True
    assert (
        novelty.domain_mismatch_reason
        not in reasons
    )


def test_explicit_sers_claim_still_requires_sers_domain_match() -> None:
    novelty = _novelty()

    claim = (
        "In Au Ag dimers, interparticle nanogap variation "
        "changes SERS performance through changes in the "
        "coupled plasmon near field."
    )

    document = (
        "Near Field Enhancement in Ag Au Nanospheres "
        "Heterodimers. Interparticle separation controls "
        "the optical near field."
    )

    compatible, domain, _scope, reasons = (
        novelty.strong_scope_compatibility(
            claim,
            document,
            min_domain=0.75,
            min_scope=0.50,
        )
    )

    assert novelty.domains(claim) == {"SERS"}

    assert domain < 0.75
    assert compatible is False

    assert (
        novelty.domain_mismatch_reason
        in reasons
    )


def test_domain_neutral_claim_still_rejects_explicit_incompatible_context() -> None:
    novelty = _novelty()

    claim = (
        "In Au Ag plasmonic dimers, interparticle separation "
        "changes plasmon near field enhancement."
    )

    document = (
        "Electrocatalytic hydrogen evolution on Au Ag "
        "nanostructures is examined together with local "
        "near field behavior."
    )

    compatible, domain, scope, reasons = (
        novelty.strong_scope_compatibility(
            claim,
            document,
            min_domain=0.75,
            min_scope=0.75,
        )
    )

    assert novelty.domains(claim) == set()

    # The explicit electrocatalysis/HER mismatch remains a
    # scientific domain barrier even for a domain-neutral claim.
    assert domain < 0.75
    assert scope == 1.0

    assert compatible is False
    assert (
        novelty.domain_mismatch_reason
        in reasons
    )
