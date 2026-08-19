import subprocess
from pathlib import Path
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b2_r1_quote_grounding_recovery_v1 import (
    DEFAULT_PROTOCOL_PATH, validate_parent_failure_state, validate_protocol,
)

def main():
    root = Path(subprocess.check_output(["git","rev-parse","--show-toplevel"], text=True).strip())
    p = validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    validate_parent_failure_state(root)
    print("Fresh-C C1B.2-R1 quote-grounding recovery protocol verifier")
    print(f"Protocol ID: {p['protocol_id']}")
    print(f"Protocol SHA256: {p['protocol_sha256']}")
    print("Parent failure: VERIFIED 0/25 + 1 failed scientific call")
    print("Recovery reason: VERBATIM_QUOTE_GROUNDING_VALIDATION_FAILURE")
    print("Raw reviewer models changed: False")
    print("Scientific prompts/targets changed: False/False")
    print("Relation labels/verdict lattice changed: False/False")
    print("Recovery quote evidence enabled: False")
    print("Recovery paper schema: verbatim_quote required + null-only")
    print("Failed parent response reuse allowed: False")
    print("Recovery order: exact reserve 1..25")
    print("Maximum recovery LLM/network calls: 26/26")
    print("Recovery may claim new Fresh-C reserve: False")
    print("Same recovery-epoch rerun after start: False")
    print("Automatic post-recovery transition: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
