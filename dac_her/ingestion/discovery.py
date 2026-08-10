from __future__ import annotations

from collections import Counter, defaultdict
import unicodedata

from .contracts import ArticleRow, DiscoveredPaper, DriveFile, IngestionIssue
from .naming import paper_id_for, parse_pdf_name


def _name_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def match_articles_to_drive(
    article_rows: list[ArticleRow],
    drive_files: list[DriveFile],
    aliases: dict[str, str] | None = None,
) -> tuple[list[DiscoveredPaper], list[IngestionIssue]]:
    pdfs = [item for item in drive_files if item.name.lower().endswith(".pdf")]
    exact: dict[str, list[DriveFile]] = defaultdict(list)
    si_by_main: dict[str, list[tuple[int, DriveFile]]] = defaultdict(list)
    for item in pdfs:
        exact[_name_key(item.name)].append(item)
        parsed = parse_pdf_name(item.name)
        if parsed and parsed.role == "supporting_information":
            si_by_main[_name_key(parsed.main_filename)].append((parsed.si_index or 0, item))

    global_issues: list[IngestionIssue] = []
    counts = Counter(_name_key(row.file_name) for row in article_rows)
    for file_key, count in counts.items():
        if count > 1:
            global_issues.append(
                IngestionIssue(
                    code="duplicate_article_row",
                    message=f"Article list contains {count} rows for the same File_Name key {file_key!r}.",
                    severity="error",
                )
            )

    papers: list[DiscoveredPaper] = []
    registered_names = set()
    for article in article_rows:
        file_key = _name_key(article.file_name)
        registered_names.add(file_key)
        pid = paper_id_for(article.file_name, article.annotator, aliases)
        issues: list[IngestionIssue] = []
        if counts[file_key] > 1:
            issues.append(
                IngestionIssue(
                    code="duplicate_article_row",
                    message=f"Article list contains multiple rows for {article.file_name!r}.",
                    severity="error",
                    paper_id=pid,
                    file_name=article.file_name,
                )
            )
        mains = exact.get(file_key, [])
        main = mains[0] if len(mains) == 1 else None
        if not mains:
            issues.append(
                IngestionIssue(
                    code="missing_main_file",
                    message=f"No Drive PDF matches Article_lists File_Name={article.file_name!r}.",
                    severity="error",
                    paper_id=pid,
                    file_name=article.file_name,
                )
            )
        elif len(mains) > 1:
            issues.append(
                IngestionIssue(
                    code="duplicate_main_file",
                    message=f"Drive contains {len(mains)} files named {article.file_name!r}.",
                    severity="error",
                    paper_id=pid,
                    file_name=article.file_name,
                )
            )
        sis = [item for _, item in sorted(si_by_main.get(file_key, []))]
        expected = article.si_existence or 0
        if len(sis) != expected:
            severity = "error" if len(sis) < expected else "warning"
            issues.append(
                IngestionIssue(
                    code="si_count_mismatch",
                    message=(
                        f"Article list expects {expected} SI file(s) for {article.file_name}, "
                        f"but Drive discovery found {len(sis)}."
                    ),
                    severity=severity,
                    paper_id=pid,
                    file_name=article.file_name,
                )
            )
        papers.append(
            DiscoveredPaper(
                paper_id=pid,
                article=article,
                main_file=main,
                si_files=sis,
                issues=issues,
            )
        )

    # PDFs that do not participate in any registered main/SI family.
    registered_family = set(registered_names)
    for item in pdfs:
        parsed = parse_pdf_name(item.name)
        family = _name_key(parsed.main_filename if parsed else item.name)
        if family not in registered_family:
            global_issues.append(
                IngestionIssue(
                    code="unregistered_pdf",
                    message=f"Drive PDF {item.name!r} has no matching Article_lists File_Name family.",
                    severity="warning",
                    file_name=item.name,
                )
            )
    return papers, global_issues
