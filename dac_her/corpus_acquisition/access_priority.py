from __future__ import annotations

from dac_her.corpus_acquisition.access_contracts import AccessLocation


def access_location_priority(row: AccessLocation) -> tuple[int, str]:
    """Stable provider-aware ordering for automatic main-PDF acquisition.

    The ordering is intentionally conservative: existing Unpaywall direct-PDF
    candidates retain first priority, OpenAlex adds recovery candidates ahead
    of the weaker catalog OA URL fallback, and landing-only rows never outrank
    a direct-PDF candidate.
    """

    if row.automatic_download_eligible:
        if row.resolver == "unpaywall" and row.is_best:
            rank = 0
        elif row.resolver == "unpaywall":
            rank = 1
        elif row.resolver == "openalex" and row.is_best:
            rank = 2
        elif (
            row.resolver == "openalex"
            and "primary_location" in row.reason_codes
        ):
            rank = 3
        elif row.resolver == "openalex":
            rank = 4
        elif row.resolver == "catalog_open_access":
            rank = 5
        else:
            rank = 6
    else:
        rank = 10

    return rank, row.location_id
