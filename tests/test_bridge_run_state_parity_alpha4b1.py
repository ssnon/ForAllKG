from __future__ import annotations

from pathlib import Path

from dac_her.bridge_policy import BRIDGE_POLICY_VERSION
from dac_her.bridge_prompts import BRIDGE_PROMPT_VERSION
from dac_her.bridge_run_state import (
    compute_bridge_extraction_metadata,
    compute_bridge_policy_run_metadata,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_alpha4b1_explicit_her_adapter_metadata_preserves_extraction_fingerprint(tmp_path):
    strict_run = tmp_path / "run"
    strict_run.mkdir()
    strict = _write(tmp_path / "strict.json", '{"x": 1}')
    source = _write(tmp_path / "source.json", '{"text": "source"}')
    impl = _write(tmp_path / "impl.py", "x = 1\n")
    active = {"run_id": "abc", "run_fingerprint": "def"}

    legacy = compute_bridge_extraction_metadata(
        strict_run_dir=strict_run,
        active_payload=active,
        model="m",
        provider="p",
        strict_chunk_paths=[strict],
        source_chunk_paths=[source],
        implementation_paths=[impl],
        runtime_options={"x": 1},
    )
    explicit = compute_bridge_extraction_metadata(
        strict_run_dir=strict_run,
        active_payload=active,
        model="m",
        provider="p",
        strict_chunk_paths=[strict],
        source_chunk_paths=[source],
        implementation_paths=[impl],
        runtime_options={"x": 1},
        bridge_prompt_version=BRIDGE_PROMPT_VERSION,
        domain_profile_id="dac_her",
        bridge_adapter_id="dac_her",
    )

    assert explicit["bridge_extraction_fingerprint"] == legacy["bridge_extraction_fingerprint"]
    assert explicit["bridge_extraction_id"] == legacy["bridge_extraction_id"]
    assert explicit["domain_profile_id"] == "dac_her"
    assert explicit["bridge_adapter_id"] == "dac_her"


def test_alpha4b1_explicit_her_adapter_metadata_preserves_policy_fingerprint(tmp_path):
    strict_run = tmp_path / "run"
    strict_run.mkdir()
    raw = _write(tmp_path / "raw.json", '{"x": 1}')
    canonical = _write(tmp_path / "graph.graphml", "<graphml/>")
    impl = _write(tmp_path / "policy.py", "x = 1\n")
    extraction = {
        "bridge_extraction_id": "abc",
        "bridge_extraction_fingerprint": "def",
    }

    legacy = compute_bridge_policy_run_metadata(
        strict_run_dir=strict_run,
        extraction_metadata=extraction,
        raw_chunk_paths=[raw],
        canonical_graph_path=canonical,
        implementation_paths=[impl],
    )
    explicit = compute_bridge_policy_run_metadata(
        strict_run_dir=strict_run,
        extraction_metadata=extraction,
        raw_chunk_paths=[raw],
        canonical_graph_path=canonical,
        implementation_paths=[impl],
        bridge_policy_version=BRIDGE_POLICY_VERSION,
        domain_profile_id="dac_her",
        bridge_adapter_id="dac_her",
    )

    assert explicit["bridge_policy_run_fingerprint"] == legacy["bridge_policy_run_fingerprint"]
    assert explicit["bridge_policy_run_id"] == legacy["bridge_policy_run_id"]
    assert explicit["domain_profile_id"] == "dac_her"
    assert explicit["bridge_adapter_id"] == "dac_her"
