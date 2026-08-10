from pathlib import Path

from dac_her.ingestion.contracts import PaperRegistryEntry
from dac_her.ingestion.corpus_manifest import build_corpus_manifest
from dac_her.ingestion.registry import PaperRegistry


def test_registry_roundtrip_and_manifest(tmp_path: Path):
    registry_path = tmp_path / "registry.json"
    md = tmp_path / "main.md"
    md.write_text("# paper", encoding="utf-8")
    registry = PaperRegistry(registry_path)
    registry.put(
        PaperRegistryEntry(
            paper_id="Kiwook_1",
            title="Paper",
            annotator="홍기욱",
            source_file_name="홍기욱_1.pdf",
            main_drive_file={"file_id": "m", "md5_checksum": "abc", "fingerprint": "md5:abc"},
            marker_version="1.0",
            main_markdown=str(md),
            qc_status="passed",
        )
    )
    registry.save()
    loaded = PaperRegistry(registry_path)
    assert loaded.get("Kiwook_1") is not None
    payload = build_corpus_manifest(loaded, tmp_path / "manifest.json", "test")
    assert payload["document_count"] == 1
    assert payload["documents"][0]["paper_id"] == "Kiwook_1"
