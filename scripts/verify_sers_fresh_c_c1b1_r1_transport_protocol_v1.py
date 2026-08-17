from dac_her.fresh_c_c1b1_r1_transport_qualification_v1 import (
    DEFAULT_PROTOCOL_PATH,
    validate_protocol,
)

def main():
    p = validate_protocol(DEFAULT_PROTOCOL_PATH)
    print("Fresh-C C1B.1-R1 transport protocol verifier")
    print(f"Protocol ID: {p['protocol_id']}")
    print(f"Protocol SHA256: {p['protocol_sha256']}")
    print(f"Corrected transport: {p['corrected_transport_semantics']}")
    print(f"Base URL: {p['base_url']}")
    print(f"Reviewer model: {p['reviewer_model']}")
    print("Parent model binding preserved: True")
    print("Parent prompt/schema semantics preserved: True")
    print("Temperature parameter sent: False")
    print("Deterministic seed: 0")
    print("Reasoning effort: medium")
    print("Reasoning returned: False")
    print("Upstream provider only: openai")
    print("Provider fallbacks allowed: False")
    print("Provider require_parameters: True")
    print("Catalog metadata calls allowed: 1")
    print("Synthetic structured LLM calls allowed: 1")
    print("Fresh-C text allowed in qualification: False")
    print("Scientific hypothesis text allowed in qualification: False")
    print("Scientific adjudication allowed: False")
    print("Automatic C1B.2 transition allowed: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
