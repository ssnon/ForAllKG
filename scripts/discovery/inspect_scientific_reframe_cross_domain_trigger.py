from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.reframing.cross_domain_diagnostics import (
    build_cross_domain_task_trigger_diagnostic,
    build_cross_domain_trigger_diagnostic_pack,
)
from pipeline_core.discovery.reframing.cross_domain_validation import (
    ScientificReframeCrossDomainValidationReport,
)
from pipeline_core.discovery.reframing.generalization_validation import (
    semantics_fingerprint,
)
from pipeline_core.discovery.reframing.evidence_tension import (
    ScientificEvidenceTensionReport,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    ScientificReframeEvidencePacket,
)
from pipeline_core.discovery.reframing.trigger_contracts import (
    ScientificReframeTriggerReport,
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value



def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a zero-LLM evidence-complete trace of why a frozen cross-domain "
            "reframing trigger did or did not fire. No trigger semantics are changed."
        )
    )
    parser.add_argument("--validation-audit", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    validation = ScientificReframeCrossDomainValidationReport.model_validate(
        _load(args.validation_audit)
    )
    freeze = _load(args.freeze)
    frozen = str(freeze.get("semantics_fingerprint") or "").strip()
    current, _ = semantics_fingerprint(args.repository_root)
    if not frozen or frozen != validation.frozen_semantics_fingerprint:
        raise SystemExit("freeze fingerprint does not match cross-domain validation audit")
    if current != frozen:
        raise SystemExit("frozen semantics changed; diagnostic aborted")

    tasks = []
    audit_by_key = {row.task_key: row for row in validation.audit.task_audits}
    for selected in validation.selected_tasks:
        audit_row = audit_by_key.get(selected.task_key)
        if audit_row is None or audit_row.status != "complete":
            raise SystemExit(f"selected task does not have a complete audit row: {selected.task_key}")
        canonical = Path(selected.canonical_dir)
        evidence = ScientificReframeEvidencePacket.model_validate(
            _load(canonical / "scientific_reframing_evidence.json")
        )
        tensions = ScientificEvidenceTensionReport.model_validate(
            _load(canonical / "scientific_reframing_evidence_tensions.json")
        )
        trigger = ScientificReframeTriggerReport.model_validate(
            _load(canonical / "scientific_reframing_triggers.json")
        )
        explorer_report = _load(canonical / "explorer.report.json")
        task = build_cross_domain_task_trigger_diagnostic(
            task_key=selected.task_key,
            case_key=selected.case_key,
            canonical_dir=selected.canonical_dir,
            trigger_pattern=audit_row.trigger_pattern,
            evidence=evidence,
            tensions=tensions,
            trigger=trigger,
            explorer_report=explorer_report,
        )
        tasks.append(task)

    pack = build_cross_domain_trigger_diagnostic_pack(
        source_validation_id=validation.validation_id,
        validation_domain_label=validation.validation_domain_label,
        frozen_semantics_fingerprint=frozen,
        current_semantics_fingerprint=current,
        tasks=tasks,
    )

    output = args.output
    if output is None:
        output = args.validation_audit.parent / (
            f"scientific_reframing_cross_domain_{validation.validation_domain_label}_trigger_diagnostics.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(pack.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print("Scientific reframing cross-domain trigger diagnostics complete")
    print("LLM calls: 0")
    print("Validation domain:", pack.validation_domain_label)
    print("Tasks:", pack.task_count)
    for task in pack.tasks:
        summary = task.summary
        print(
            f"{task.task_key}: trigger={task.trigger_pattern}; "
            f"premise_response={summary.premise_response_cue_match_count}/{summary.premise_count}; "
            f"explorer_tensions={summary.explorer_raw_tension_count}; "
            f"grounded_tensions={summary.grounded_evidence_tension_witness_count}; "
            f"direct_signals={summary.actual_direct_trigger_signal_count}"
        )
        if summary.diagnostic_flags:
            print("  flags:", ", ".join(summary.diagnostic_flags))
    print("Trigger failure cause determined: false")
    print("Source-marker causality evaluated: false")
    print("No trigger, prompt, critic, portfolio, or threshold semantics were modified.")
    print("Output:", output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
