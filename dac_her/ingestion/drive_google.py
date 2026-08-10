from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from .contracts import DriveFile


_DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
_SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets.readonly"
_FOLDER_MIME = "application/vnd.google-apps.folder"


def _imports():
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Google ingestion dependencies are missing. Install "
            "requirements-ingestion.txt."
        ) from exc
    return service_account, build, MediaIoBaseDownload


class GoogleWorkspaceReader:
    def __init__(self, credentials_path: str | Path):
        service_account, build, media = _imports()
        credentials = service_account.Credentials.from_service_account_file(
            str(credentials_path),
            scopes=[_DRIVE_SCOPE, _SHEETS_SCOPE],
        )
        self._drive = build("drive", "v3", credentials=credentials, cache_discovery=False)
        self._sheets = build("sheets", "v4", credentials=credentials, cache_discovery=False)
        self._media_cls = media

    def list_recursive(self, root_folder_id: str) -> list[DriveFile]:
        result: list[DriveFile] = []
        stack: list[tuple[str, str]] = [(root_folder_id, "")]
        while stack:
            folder_id, folder_path = stack.pop()
            page_token = None
            while True:
                response = (
                    self._drive.files()
                    .list(
                        q=f"'{folder_id}' in parents and trashed = false",
                        fields=(
                            "nextPageToken,files(id,name,mimeType,modifiedTime,size,"
                            "md5Checksum,parents)"
                        ),
                        pageSize=1000,
                        pageToken=page_token,
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                    )
                    .execute()
                )
                for raw in response.get("files", []):
                    name = raw["name"]
                    child_path = "/".join(part for part in [folder_path, name] if part)
                    if raw.get("mimeType") == _FOLDER_MIME:
                        stack.append((raw["id"], child_path))
                        continue
                    size = raw.get("size")
                    result.append(
                        DriveFile(
                            file_id=raw["id"],
                            name=name,
                            mime_type=raw.get("mimeType", ""),
                            modified_time=raw.get("modifiedTime"),
                            size=int(size) if size is not None else None,
                            md5_checksum=raw.get("md5Checksum"),
                            parent_id=(raw.get("parents") or [folder_id])[0],
                            folder_path=folder_path,
                        )
                    )
                page_token = response.get("nextPageToken")
                if not page_token:
                    break
        return result

    def read_sheet(self, spreadsheet_id: str, range_a1: str) -> list[list[Any]]:
        response = (
            self._sheets.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=range_a1)
            .execute()
        )
        return response.get("values", [])

    def download_file(self, file_id: str, output: str | Path) -> Path:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        request = self._drive.files().get_media(fileId=file_id, supportsAllDrives=True)
        buffer = io.FileIO(output_path, "wb")
        try:
            downloader = self._media_cls(buffer, request, chunksize=8 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        finally:
            buffer.close()
        return output_path
