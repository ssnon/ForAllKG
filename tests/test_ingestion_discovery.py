from dac_her.ingestion.article_list import parse_article_rows
from dac_her.ingestion.contracts import DriveFile
from dac_her.ingestion.discovery import match_articles_to_drive


def _drive(name, file_id):
    return DriveFile(file_id=file_id, name=name, mime_type="application/pdf", md5_checksum=file_id)


def test_drive_sheet_join_and_si_variants():
    values = [
        ["Title", "Reason", "Annotator", "Date", "Redundancy", "Flag", "File_Name", "SIExistance"],
        ["Paper", "DAC", "홍기욱", "2026-01-01", "FALSE", "1", "홍기욱_1.pdf", "2"],
    ]
    rows = parse_article_rows(values)
    papers, global_issues = match_articles_to_drive(
        rows,
        [
            _drive("홍기욱_1.pdf", "m"),
            _drive("홍기욱_1_SI1.pdf", "s1"),
            _drive("홍기욱_1_SI_2.pdf", "s2"),
        ],
        {"홍기욱": "Kiwook"},
    )
    assert not global_issues
    assert len(papers) == 1
    assert papers[0].paper_id == "Kiwook_1"
    assert papers[0].ready_for_download
    assert [item.name for item in papers[0].si_files] == ["홍기욱_1_SI1.pdf", "홍기욱_1_SI_2.pdf"]


def test_missing_si_blocks_when_sheet_requires_more():
    values = [
        ["Title", "Reason", "Annotator", "Date", "Redundancy", "Flag", "File_Name", "SIExistance"],
        ["Paper", "DAC", "A", "2026", "FALSE", "1", "A_1.pdf", "1"],
    ]
    rows = parse_article_rows(values)
    papers, _ = match_articles_to_drive(rows, [_drive("A_1.pdf", "m")])
    assert not papers[0].ready_for_download
    assert any(issue.code == "si_count_mismatch" and issue.severity == "error" for issue in papers[0].issues)


def test_duplicate_article_rows_are_blocking():
    values = [
        ["Title", "Reason", "Annotator", "Date", "Redundancy", "Flag", "File_Name", "SIExistance"],
        ["Paper", "DAC", "A", "2026", "FALSE", "1", "A_1.pdf", "None"],
        ["Paper duplicate", "DAC", "A", "2026", "FALSE", "1", "A_1.pdf", "None"],
    ]
    papers, issues = match_articles_to_drive(parse_article_rows(values), [_drive("A_1.pdf", "m")])
    assert any(issue.code == "duplicate_article_row" for issue in issues)
    assert all(not paper.ready_for_download for paper in papers)
