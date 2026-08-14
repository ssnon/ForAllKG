from __future__ import annotations


def progress_prefix(
    label: str,
    current: int,
    total: int,
) -> str:
    width = max(1, len(str(max(total, 1))))
    return (
        f"[{label} "
        f"{int(current):0{width}d}/{int(total):0{width}d}]"
    )


def compact_text(
    value: str,
    *,
    max_length: int = 88,
) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_length:
        return text
    return text[: max(0, max_length - 1)] + "…"
