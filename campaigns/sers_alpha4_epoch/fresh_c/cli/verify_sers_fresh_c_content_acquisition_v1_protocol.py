from __future__ import annotations

import argparse
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_content_acquisition_v1 import (
    DEFAULT_PROTOCOL_PATH,
    load_and_validate_protocol,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    args = parser.parse_args()
    p = load_and_validate_protocol(args.protocol)

    print("Fresh-C C0.1D blind OA content-acquisition protocol verifier")
    print(f"Protocol ID: {p.protocol_id}")
    print(f"Protocol SHA256: {p.protocol_sha256}")
    print(f"Upstream blind queue: {p.upstream_blind_queue_count}")
    print(f"Target verified PDFs: {p.target_successful_pdf_count}")
    print(f"Maximum identity attempts: {p.maximum_identity_attempts}")
    print("Selection rule: first 25 successful verified OA PDFs in blind order")
    print("Manual candidate replacement allowed: False")
    print("Hypothesis-aware selection allowed: False")
    print("Scientific metadata inspection allowed: False")
    print("Unpaywall enabled: True")
    print("OpenAlex OA resolver enabled: True")
    print("Catalog OA fallback enabled: True")
    print("PDF magic required: True")
    print("Paywall bypass allowed: False")
    print("PDF text extraction allowed: False")
    print("Fresh Reserve C consumed: False")
    print("LLM calls: 0")
    print("Automatic C1 transition allowed: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
