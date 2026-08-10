from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

from .contracts import MarkerResult


def marker_version() -> str:
    for distribution in ("marker-pdf", "marker_pdf"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    return "unknown"


def _yaml_value(value: object) -> str:
    # JSON string/scalar syntax is valid YAML 1.2 and avoids custom escaping.
    return json.dumps(value, ensure_ascii=False)


def build_frontmatter(metadata: dict[str, object]) -> str:
    lines = ["---"]
    for key, value in metadata.items():
        if isinstance(value, (list, dict)):
            lines.append(f"{key}: {_yaml_value(value)}")
        else:
            lines.append(f"{key}: {_yaml_value(value)}")
    lines.extend(["---", ""])
    return "\n".join(lines)


class MarkerSingleRunner:
    def __init__(
        self,
        command: str = "marker_single",
        paginate_output: bool = True,
        extra_args: list[str] | None = None,
    ):
        self.command = command
        self.paginate_output = paginate_output
        self.extra_args = list(extra_args or [])
        self.version = marker_version()

    def preflight(self) -> None:
        if shutil.which(self.command) is None:
            raise RuntimeError(f"{self.command!r} was not found on PATH.")
        check = subprocess.run(
            [self.command, "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        help_text = (check.stdout or "") + (check.stderr or "")
        for flag in ("--output_dir", "--output_format"):
            if flag not in help_text:
                raise RuntimeError(
                    f"Installed marker_single does not expose required flag {flag}. "
                    "Check the marker-pdf installation/version."
                )
        if self.paginate_output and "--paginate_output" not in help_text:
            raise RuntimeError(
                "--paginate_output is requested but unsupported by this marker_single."
            )

    def convert(
        self,
        input_pdf: str | Path,
        output_dir: str | Path,
        document_id: str,
        role: str,
        metadata: dict[str, object],
        force: bool = False,
        progress: Callable[[str], None] | None = None,
        heartbeat_seconds: float = 30.0,
    ) -> MarkerResult:
        input_path = Path(input_pdf)
        output_path = Path(output_dir)
        if force and output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        stdout_path = output_path / "marker.stdout.log"
        stderr_path = output_path / "marker.stderr.log"
        command = [
            self.command,
            str(input_path),
            "--output_format",
            "markdown",
            "--output_dir",
            str(output_path),
        ]
        if self.paginate_output:
            command.append("--paginate_output")
        command.extend(self.extra_args)

        started = time.monotonic()
        if progress is not None:
            progress(f"marker start: {input_path.name}")

        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            process = subprocess.Popen(
                command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            next_heartbeat = started + max(1.0, heartbeat_seconds)
            while process.poll() is None:
                now = time.monotonic()
                if progress is not None and heartbeat_seconds > 0 and now >= next_heartbeat:
                    progress(
                        f"marker still running: {input_path.name} "
                        f"(elapsed {int(now - started)}s)"
                    )
                    next_heartbeat = now + heartbeat_seconds
                time.sleep(0.5)
            return_code = process.returncode

        elapsed = time.monotonic() - started
        if progress is not None:
            progress(
                f"marker finished: {input_path.name} "
                f"(exit={return_code}, elapsed={elapsed:.1f}s)"
            )

        if return_code != 0:
            return MarkerResult(
                document_id=document_id,
                document_role=role,  # type: ignore[arg-type]
                input_pdf=str(input_path),
                output_dir=str(output_path),
                raw_markdown=None,
                normalized_markdown=None,
                marker_version=self.version,
                return_code=return_code,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                error="marker_single returned a non-zero exit code",
            )

        candidates = [
            item for item in output_path.rglob("*.md")
            if item.name != "normalized.md"
        ]
        if not candidates:
            return MarkerResult(
                document_id=document_id,
                document_role=role,  # type: ignore[arg-type]
                input_pdf=str(input_path),
                output_dir=str(output_path),
                raw_markdown=None,
                normalized_markdown=None,
                marker_version=self.version,
                return_code=return_code,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                error="marker_single completed but no Markdown file was found",
            )
        stem = input_path.stem.casefold()
        preferred = [item for item in candidates if item.stem.casefold() == stem]
        raw_md = sorted(preferred or candidates, key=lambda p: (len(p.parts), str(p)))[0]
        normalized = raw_md.parent / "normalized.md"
        raw_text = raw_md.read_text(encoding="utf-8", errors="replace")
        normalized.write_text(
            build_frontmatter(metadata) + raw_text.lstrip("\ufeff"),
            encoding="utf-8",
        )
        return MarkerResult(
            document_id=document_id,
            document_role=role,  # type: ignore[arg-type]
            input_pdf=str(input_path),
            output_dir=str(output_path),
            raw_markdown=str(raw_md),
            normalized_markdown=str(normalized),
            marker_version=self.version,
            return_code=return_code,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )
