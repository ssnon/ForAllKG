import numpy as np

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaim,
)
from pipeline_core.discovery.prior_art_memory import (
    PriorArtMemoryEntry,
    PriorArtMemoryMatcher,
)


class FlatEncoder:
    def encode_query(self, text: str):
        return np.array(
            [1.0, 0.0],
            dtype=np.float32,
        )

    def encode_documents(
        self,
        texts,
        *,
        batch_size=32,
    ):
        return np.array(
            [
                [1.0, 0.0]
                for _ in texts
            ],
            dtype=np.float32,
        )


def claim(
    claim_id,
    text,
    distinguishing,
):
    return NoveltyClaim(
        claim_id=claim_id,
        hypothesis_id="h1",
        claim_rank=1,
        kind="moderator_interaction",
        importance="core",
        text=text,
        rationale="test",
        search_concepts=[
            "interparticle spacing",
            "SERS enhancement",
            *distinguishing,
        ],
        search_queries=["test query"],
        distinguishing_terms=(
            distinguishing
        ),
        diagnostic_query_kind=(
            "LOWER_ORDER_RELATION"
        ),
        diagnostic_search_query=(
            "spacing SERS relationship"
        ),
        diagnostic_execution_query=(
            "spacing SERS relationship"
        ),
        diagnostic_structural_terms=[
            "plasmonic assembly"
        ],
        diagnostic_relation_terms=[
            "interparticle spacing",
            "SERS enhancement",
            "spacing dependence",
        ],
    )


def test_memory_requires_matching_distinguishing_facet():
    old = claim(
        "old",
        (
            "Changing excitation wavelength "
            "changes spacing dependent SERS."
        ),
        ["excitation wavelength"],
    )

    wavelength = claim(
        "new-wavelength",
        (
            "Spacing dependent SERS changes "
            "across excitation wavelengths."
        ),
        ["excitation wavelength"],
    )

    power = claim(
        "new-power",
        (
            "Spacing dependent SERS changes "
            "across laser powers."
        ),
        ["laser power"],
    )

    memory = [
        PriorArtMemoryEntry(
            claim=old,
            work_ids=("work:pre2000",),
        )
    ]

    matcher = PriorArtMemoryMatcher(
        FlatEncoder()
    )

    assert len(
        matcher.match(
            wavelength,
            memory,
        )
    ) == 1

    assert (
        matcher.match(
            power,
            memory,
        )
        == []
    )


def test_memory_does_not_copy_prior_art_status():
    old = claim(
        "old",
        "Excitation wavelength changes spacing SERS.",
        ["excitation wavelength"],
    )

    memory = [
        PriorArtMemoryEntry(
            claim=old,
            work_ids=("work:1",),
        )
    ]

    match = PriorArtMemoryMatcher(
        FlatEncoder()
    ).match(
        claim(
            "new",
            (
                "Spacing SERS changes across "
                "excitation wavelength."
            ),
            ["excitation wavelength"],
        ),
        memory,
    )[0]

    assert match.work_ids == ("work:1",)
    assert not hasattr(
        match,
        "prior_art_status",
    )


def structured_claim(
    claim_id,
    *,
    kind,
    text,
    identity,
    distinguishing,
    relation,
):
    return NoveltyClaim(
        claim_id=claim_id,
        hypothesis_id="h-structured",
        claim_rank=1,
        kind=kind,
        importance="core",
        text=text,
        rationale="test",
        search_concepts=[
            *identity,
            *distinguishing,
            *relation,
        ],
        search_queries=["test query"],
        distinguishing_terms=distinguishing,
        prior_art_identity_terms=identity,
        relation_nucleus_terms=relation,
    )


def test_memory_cross_kind_uses_identity_plus_relation_not_prediction_facet():
    old = structured_claim(
        "old-wavelength-moderator",
        kind="moderator_interaction",
        text=(
            "Excitation wavelength moderates the dependence "
            "of SERS enhancement on interparticle spacing."
        ),
        identity=["excitation wavelength"],
        distinguishing=["excitation wavelength"],
        relation=[
            "interparticle spacing",
            "SERS enhancement",
            "dependence",
        ],
    )

    prediction = structured_claim(
        "new-wavelength-prediction",
        kind="distinctive_prediction",
        text=(
            "Changing excitation wavelength changes the relative "
            "ordering of SERS enhancement across assemblies with "
            "different interparticle spacings."
        ),
        identity=["excitation wavelength"],
        distinguishing=["relative ordering"],
        relation=[
            "interparticle spacing",
            "SERS enhancement",
            "dependence",
        ],
    )

    memory = [
        PriorArtMemoryEntry(
            claim=old,
            work_ids=("work:pre2000",),
        )
    ]

    matches = PriorArtMemoryMatcher(
        FlatEncoder()
    ).match(
        prediction,
        memory,
    )

    assert len(matches) == 1
    assert matches[0].work_ids == ("work:pre2000",)
    assert not hasattr(matches[0], "prior_art_status")


def test_memory_blocks_different_identity_even_with_same_relation():
    old = structured_claim(
        "old-wavelength",
        kind="moderator_interaction",
        text="Wavelength moderates spacing-dependent SERS.",
        identity=["excitation wavelength"],
        distinguishing=["excitation wavelength"],
        relation=[
            "interparticle spacing",
            "SERS enhancement",
            "dependence",
        ],
    )

    power = structured_claim(
        "new-power",
        kind="distinctive_prediction",
        text="Laser power changes spacing-dependent SERS.",
        identity=["laser power"],
        distinguishing=["relative ordering"],
        relation=[
            "interparticle spacing",
            "SERS enhancement",
            "dependence",
        ],
    )

    assert (
        PriorArtMemoryMatcher(
            FlatEncoder()
        ).match(
            power,
            [
                PriorArtMemoryEntry(
                    claim=old,
                    work_ids=("work:pre2000",),
                )
            ],
        )
        == []
    )


def test_memory_blocks_same_identity_with_unrelated_relation():
    old = structured_claim(
        "old-spacing-sers",
        kind="moderator_interaction",
        text="Wavelength moderates spacing-dependent SERS.",
        identity=["excitation wavelength"],
        distinguishing=["excitation wavelength"],
        relation=[
            "interparticle spacing",
            "SERS enhancement",
            "dependence",
        ],
    )

    unrelated = structured_claim(
        "new-size-lspr",
        kind="distinctive_prediction",
        text=(
            "Excitation wavelength changes nanoparticle-size "
            "dependent LSPR ordering."
        ),
        identity=["excitation wavelength"],
        distinguishing=["relative ordering"],
        relation=[
            "nanoparticle size",
            "LSPR peak",
        ],
    )

    assert (
        PriorArtMemoryMatcher(
            FlatEncoder()
        ).match(
            unrelated,
            [
                PriorArtMemoryEntry(
                    claim=old,
                    work_ids=("work:pre2000",),
                )
            ],
        )
        == []
    )
