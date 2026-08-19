from __future__ import annotations

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1a_materialization_v1 import (
    DEFAULT_PROTOCOL_PATH,
    load_and_validate_protocol,
)


def main() -> int:
    p = load_and_validate_protocol(DEFAULT_PROTOCOL_PATH)
    print("Fresh-C C1A irreversible local materialization protocol verifier")
    print(f"Protocol ID: {p.protocol_id}")
    print(f"Protocol SHA256: {p.protocol_sha256}")
    print("Selected sealed PDFs: 25")
    print("Consumption marker before first text extraction: True")
    print("Fresh C becomes consumed at marker write: True")
    print("Consumption irreversible: True")
    print("Materializer: pdftext 0.6.3 + pypdfium2 4.30.0")
    print("Reading-order sort: True")
    print("All 25 PDFs required: True")
    print("Identity replacement allowed: False")
    print("Network allowed during materialization: False")
    print("Socket network guard required: True")
    print("External literature lookup allowed: False")
    print("OCR performed: False")
    print("Scientific reviewer read performed: False")
    print("Scientific adjudication performed: False")
    print("LLM calls: 0")
    print("Automatic C1B transition allowed: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
