import os
from pathlib import Path

from dac_her.ingestion.contracts import DriveFile
from dac_her.ingestion.runtime import SyncConfig, run_sync


class FakeWorkspace:
    def __init__(self, _credentials):
        pass

    def read_sheet(self, _spreadsheet_id, _range):
        return [
            ["Title", "Reason", "Annotator", "Date", "Redundancy", "Flag", "File_Name", "SIExistance"],
            ["Test Paper", "DAC", "홍기욱", "2026-08-09", "FALSE", "1", "홍기욱_1.pdf", "1"],
        ]

    def list_recursive(self, _folder_id):
        return [
            DriveFile("main-id", "홍기욱_1.pdf", "application/pdf", md5_checksum="aaa"),
            DriveFile("si-id", "홍기욱_1_SI1.pdf", "application/pdf", md5_checksum="bbb"),
        ]

    def download_file(self, file_id, output):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("%PDF-" + file_id).encode())
        return path


def test_full_runtime_with_fake_drive_and_marker(tmp_path: Path, monkeypatch):
    from dac_her.ingestion import runtime

    monkeypatch.setattr(runtime, "GoogleWorkspaceReader", FakeWorkspace)
    fake = tmp_path / "marker_single"
    fake.write_text(
        '''#!/usr/bin/env python3\n'''
        '''import pathlib, sys\n'''
        '''if "--help" in sys.argv:\n'''
        '''    print("--output_dir --output_format --paginate_output")\n'''
        '''    raise SystemExit(0)\n'''
        '''inp=pathlib.Path(sys.argv[1])\n'''
        '''out=pathlib.Path(sys.argv[sys.argv.index("--output_dir")+1]) / inp.stem\n'''
        '''out.mkdir(parents=True, exist_ok=True)\n'''
        '''title="# Test Paper\\n" if inp.name=="main.pdf" else "# SI\\n"\n'''
        '''(out/(inp.stem+".md")).write_text(title + "body "*1000, encoding="utf-8")\n''',
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    alias = tmp_path / "aliases.json"
    alias.write_text('{"홍기욱":"Kiwook"}', encoding="utf-8")
    data_root = tmp_path / "data"
    report = run_sync(
        SyncConfig(
            credentials_path=tmp_path / "dummy.json",
            drive_folder_id="folder",
            spreadsheet_id="sheet",
            data_root=data_root,
            registry_path=data_root / "registry" / "papers.json",
            alias_map_path=alias,
            corpus_id="test_corpus",
            marker_command="marker_single",
        )
    )
    assert report["manifest_document_count"] == 1
    assert report["status_counts"].get("passed", 0) + report["status_counts"].get("passed_with_warnings", 0) == 1
    manifest = data_root / "corpora" / "test_corpus" / "manifest.json"
    assert manifest.exists()
    # Second run should be incremental and skip the same source fingerprints.
    second = run_sync(
        SyncConfig(
            credentials_path=tmp_path / "dummy.json",
            drive_folder_id="folder",
            spreadsheet_id="sheet",
            data_root=data_root,
            registry_path=data_root / "registry" / "papers.json",
            alias_map_path=alias,
            corpus_id="test_corpus",
            marker_command="marker_single",
        )
    )
    assert second["status_counts"] == {"unchanged": 1}
