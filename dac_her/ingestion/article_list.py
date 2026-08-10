from __future__ import annotations

from .contracts import ArticleRow


_EXPECTED = [
    "Title",
    "Reason",
    "Annotator",
    "Date",
    "Redundancy",
    "Flag",
    "File_Name",
    "SIExistance",
]


def _norm(value: str) -> str:
    return value.strip().replace(" ", "")


def parse_si_count(value: object) -> int | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    # ".None", "None.", " none " 등의 표기 변형 허용
    sentinel = text.strip(" .,_;:-/\\").lower()
    if sentinel in {"none", "null", "n/a", "na", "nan"}:
        return None

    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid SIExistance value {value!r}; "
            "expected an integer count or None/null/N/A."
        ) from exc

    if number < 0 or not number.is_integer():
        raise ValueError(
            f"Invalid SIExistance value {value!r}; "
            "expected a non-negative integer."
        )

    return int(number)


def parse_article_rows(values: list[list[object]]) -> list[ArticleRow]:
    if not values:
        return []
    headers = [str(item).strip() for item in values[0]]
    normalized = {_norm(name): idx for idx, name in enumerate(headers)}
    missing = [name for name in _EXPECTED if _norm(name) not in normalized]
    if missing:
        raise ValueError(f"Article list is missing columns: {missing}")

    def get(row: list[object], name: str) -> str:
        idx = normalized[_norm(name)]
        return str(row[idx]).strip() if idx < len(row) and row[idx] is not None else ""

    result: list[ArticleRow] = []
    for row in values[1:]:
        file_name = get(row, "File_Name")
        if not file_name:
            continue
        result.append(
            ArticleRow(
                title=get(row, "Title"),
                reason=get(row, "Reason"),
                annotator=get(row, "Annotator"),
                date=get(row, "Date"),
                redundancy=get(row, "Redundancy"),
                flag=get(row, "Flag"),
                file_name=file_name,
                si_existence=parse_si_count(get(row, "SIExistance")),
            )
        )
    return result
