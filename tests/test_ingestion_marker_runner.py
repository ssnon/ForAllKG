import os
from pathlib import Path

from dac_her.ingestion.marker_runner import MarkerSingleRunner
from dac_her.ingestion.qc import markdown_qc


def test_marker_wrapper_with_fake_cli(tmp_path: Path, monkeypatch):
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
        '''(out/(inp.stem+".md")).write_text("# Test Paper\\n" + "body "*1000, encoding="utf-8")\n''',
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-test")
    runner = MarkerSingleRunner(command="marker_single")
    runner.preflight()
    result = runner.convert(
        pdf,
        tmp_path / "out",
        "P1_main",
        "main",
        {"paper_id": "P1", "document_role": "main"},
    )
    assert result.succeeded
    normalized = Path(result.normalized_markdown)
    text = normalized.read_text(encoding="utf-8")
    assert text.startswith("---\npaper_id:")
    assert not any(issue.severity == "error" for issue in markdown_qc(result, "Test Paper"))
