from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show the latest GraphAgentsDAC ingestion progress checkpoint."
    )
    parser.add_argument("--data-root", default="data_dac/ingestion")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    return parser.parse_args()


def _render(payload: dict) -> str:
    completed = int(payload.get("completed", 0) or 0)
    total = int(payload.get("total", 0) or 0)
    percent = (100.0 * completed / total) if total else 0.0
    counts = payload.get("status_counts") or {}
    count_text = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "-"
    lines = [
        "GraphAgentsDAC ingestion progress",
        "=" * 34,
        f"phase:    {payload.get('phase', '-')}",
        f"progress: {completed}/{total}" + (f" ({percent:.1f}%)" if total else ""),
        f"current:  {payload.get('current_paper_id') or '-'}",
        f"elapsed:  {float(payload.get('elapsed_seconds', 0.0) or 0.0):.1f}s",
        f"statuses: {count_text}",
    ]
    if payload.get("detail"):
        lines.append(f"detail:   {payload['detail']}")
    lines.append(f"updated:  {payload.get('updated_at', '-')}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    path = Path(args.data_root) / "runs" / "latest_progress.json"
    while True:
        if not path.exists():
            text = f"No progress checkpoint yet: {path}"
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            text = _render(payload)
        if args.watch:
            os.system("clear" if os.name != "nt" else "cls")
        print(text, flush=True)
        if not args.watch:
            break
        try:
            time.sleep(max(0.2, args.interval))
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
