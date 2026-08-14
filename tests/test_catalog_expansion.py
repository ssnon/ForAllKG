from dac_her.corpus_acquisition.catalog_expansion import append_catalog_expansion
from dac_her.literature_catalog_contracts import (
    CatalogQuery,
    CatalogQueryExecution,
    CatalogWork,
    LiteratureCatalogPacket,
)


def _packet(catalog_id: str, works: list[CatalogWork], provider: str):
    query = CatalogQuery(
        query_id=f"q:{provider}",
        profile_id="profile",
        axis_id="axis",
        query_text="sers au ag",
    )
    return LiteratureCatalogPacket(
        catalog_id=catalog_id,
        catalog_sha256=f"sha:{catalog_id}",
        acquisition_profile_id="profile",
        searched_at_utc="2026-08-14T00:00:00+00:00",
        providers_requested=[provider],
        queries=[query],
        works=works,
        executions=[
            CatalogQueryExecution(
                query_id=query.query_id,
                axis_id="axis",
                provider=provider,
                success=True,
                result_count=len(works),
            )
        ],
        raw_work_count=len(works),
        canonical_work_count=len(works),
        deduplicated_work_count=0,
        supplementary_records_collapsed=0,
    )


def test_append_expansion_freezes_base_rows_and_appends_only_new_works():
    base_work = CatalogWork(
        work_id="catalog_work:base",
        title="Gold silver nanogap SERS substrate mechanism study",
        doi="10.1000/base",
        abstract="base abstract",
        providers=["semantic_scholar"],
    )
    overlapping = CatalogWork(
        work_id="catalog_work:incoming-overlap",
        title="Gold silver nanogap SERS substrate mechanism study",
        doi="https://doi.org/10.1000/base",
        abstract="much richer incoming metadata that must not rewrite the base row",
        providers=["openalex"],
    )
    new_work = CatalogWork(
        work_id="catalog_work:new",
        title="Composition controlled Au Ag alloy SERS field enhancement",
        doi="10.1000/new",
        providers=["openalex"],
    )

    result = append_catalog_expansion(
        base=_packet("base", [base_work], "semantic_scholar"),
        incoming=_packet("incoming", [overlapping, new_work], "openalex"),
        expansion_id="exp1",
    )

    assert [row.work_id for row in result.packet.works] == [
        "catalog_work:base",
        "catalog_work:new",
    ]
    assert result.packet.works[0].model_dump() == base_work.model_dump()
    assert result.report["overlap_work_count"] == 1
    assert result.report["new_work_count"] == 1
    assert result.report["base_prefix_preserved"] is True
    assert result.report["base_metadata_preserved"] is True


def test_append_expansion_rejects_profile_mismatch():
    base = _packet("base", [], "semantic_scholar")
    incoming = _packet("incoming", [], "openalex").model_copy(
        update={"acquisition_profile_id": "other"}
    )
    try:
        append_catalog_expansion(base=base, incoming=incoming, expansion_id="exp")
    except ValueError as exc:
        assert "profile mismatch" in str(exc).casefold()
    else:
        raise AssertionError("profile mismatch must fail")
