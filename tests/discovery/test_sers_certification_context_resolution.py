
import json

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
)
from pipeline_core.discovery.sers_certification_context_resolution import (
    write_grounded_hypothesis_context,
)


def test_dual_wrapper_emits_only_grounded_context(tmp_path):
    grounded = {
        "schema_version": "hypothesis-context-v1",
        "context_id": "c",
        "context_sha256": "sha",
        "domain_profile_id": "sers_au_ag",
        "source_packet_id": "packet",
        "source_packet_sha256": "packetsha",
        "source_report_id": "report",
        "source_report_sha256": "reportsha",
        "task_id": "task",
        "question": "q",
        "corpus_id": "corpus",
        "evidence_statements": [],
    }
    dual = {
        "schema_version": "dual-hypothesis-context-v1",
        "dual_context_id": "dual",
        "dual_context_sha256": "dualsha",
        "domain_profile_id": "sers_au_ag",
        "grounded_context": grounded,
        "discovery_bundle": {},
        "task_lane_inspirations": [],
    }
    portfolio = {
        "schema_version": "hypothesis-portfolio-v1",
        "portfolio_id": "p",
        "domain_profile_id": "sers_au_ag",
        "source_context_id": "c",
        "source_context_sha256": "sha",
        "source_report_id": "report",
        "source_report_sha256": "reportsha",
        "hypotheses": [],
        "abstention_reason": "test",
    }

    source = tmp_path / "dual.json"
    portfolio_path = tmp_path / "portfolio.json"
    output = tmp_path / "context.json"

    source.write_text(
        json.dumps(dual),
        encoding="utf-8",
    )
    portfolio_path.write_text(
        json.dumps(portfolio),
        encoding="utf-8",
    )

    meta = write_grounded_hypothesis_context(
        source_path=source,
        portfolio_path=portfolio_path,
        output_path=output,
    )

    emitted = json.loads(
        output.read_text(encoding="utf-8")
    )
    assert emitted["schema_version"] == "hypothesis-context-v1"
    assert "grounded_context" not in emitted
    assert "dual_context_id" not in emitted
    assert meta["resolution_mode"] == "DUAL_CONTEXT_GROUNDED_CONTEXT"
    HypothesisContext.model_validate(emitted)
