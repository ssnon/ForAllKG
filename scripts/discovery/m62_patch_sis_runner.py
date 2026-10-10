"""Apply a small fail-closed M6.1 opt-in hook to existing SIS v3.4 runner.

Dry-run by default. Does not run E2E, import LLM SDK, or call providers.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
from pathlib import Path

TARGET = Path("scripts/discovery/run_research_idea_novelty_aware_cycle_v3_4.py")
MARKER = "--m61-feedback-hints"


def patch_source(source: str) -> str:
    if MARKER in source:
        raise ValueError("M62_ALREADY_PATCHED: refusing second patch")
    replacements = [
        (
            "from pipeline_core.discovery.research_idea_offspring_execution import (",
            "from pipeline_core.discovery.research_idea_m61_feedback_bridge import (\n"
            "    M61FeedbackPromptAdapter,\n"
            "    load_m61_feedback_bundle,\n"
            ")\nfrom pipeline_core.discovery.research_idea_offspring_execution import (",
        ),
        (
            '    p.add_argument("--generate-next", action="store_true")',
            '    p.add_argument("--generate-next", action="store_true")\n'
            '    p.add_argument("--m61-feedback-hints", type=Path, default=None,\n'
            '                   help="Opt-in M6.1 speculative search hints; requires --m61-feedback-population")\n'
            '    p.add_argument("--m61-feedback-population", type=Path, default=None,\n'
            '                   help="Exact linked M6.1 population used to validate hint provenance")',
        ),
        (
            '    feedback_cache = args.feedback_cache_dir.expanduser().resolve() if args.feedback_cache_dir else None',
            '    feedback_cache = args.feedback_cache_dir.expanduser().resolve() if args.feedback_cache_dir else None\n'
            '    if (args.m61_feedback_hints is None) != (args.m61_feedback_population is None):\n'
            '        raise ValueError("Both --m61-feedback-hints and --m61-feedback-population are required together")\n'
            '    m61_bundle = (\n'
            '        load_m61_feedback_bundle(args.m61_feedback_hints.expanduser().resolve(),\n'
            '                                 args.m61_feedback_population.expanduser().resolve())\n'
            '        if args.m61_feedback_hints is not None else None\n'
            '    )',
        ),
        (
            '    _write(sp / f"sis_v3_4.g{generation + 1}_generation_plan.json", next_plan)',
            '    _write(sp / f"sis_v3_4.g{generation + 1}_generation_plan.json", next_plan)\n'
            '    if m61_bundle is not None:\n'
            '        _write(sp / f"{tag}_m61_feedback_optin_consumption.json",\n'
            '               m61_bundle.selection_report(next_plan.selected_parent_idea_ids))',
        ),
        (
            '        raw_next, prompts = execute_epistemic_generation(',
            '        prompt_capture_adapter = adapter\n'
            '        if m61_bundle is not None:\n'
            '            adapter = M61FeedbackPromptAdapter(\n'
            '                backend=adapter,\n'
            '                bundle=m61_bundle,\n'
            '                task_to_idea={\n'
            '                    task.task_id: task.primary_parent_idea_id for task in next_plan.tasks\n'
            '                },\n'
            '            )\n'
            '        raw_next, prompts = execute_epistemic_generation(',
        ),
        (
            '            _write_augmented_prompts(work / "augmented_generation_prompts", adapter)',
            '            _write_augmented_prompts(work / "augmented_generation_prompts", prompt_capture_adapter)',
        ),
    ]
    patched = source
    for old, new in replacements:
        if patched.count(old) != 1:
            raise ValueError(f"M62_ANCHOR_MISMATCH ({patched.count(old)}): {old[:85]}")
        patched = patched.replace(old, new, 1)
    ast.parse(patched)
    return patched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Explicitly patch exact target and save external backup")
    parser.add_argument("--backup-dir", type=Path)
    args = parser.parse_args()
    target = args.repo / TARGET
    source = target.read_text(encoding="utf-8")
    patched = patch_source(source)
    before = hashlib.sha256(source.encode()).hexdigest()
    after = hashlib.sha256(patched.encode()).hexdigest()
    if not args.apply:
        diff = list(difflib.unified_diff(source.splitlines(), patched.splitlines(), fromfile=str(TARGET), tofile=str(TARGET)+" (M6.2)", lineterm=""))
        print("M62_PATCH_DRY_RUN_PASS", "lines", len(diff), "old_sha256", before, "new_sha256", after)
        print("\n".join(diff[:150]))
        return
    if args.backup_dir is None:
        raise ValueError("--apply requires --backup-dir outside the repository")
    backup = args.backup_dir.resolve()
    repo = args.repo.resolve()
    if backup == repo or repo in backup.parents:
        raise ValueError("backup must be outside repository")
    backup.mkdir(parents=True, exist_ok=True)
    destination = backup / f"sis_v3_4_runner_before_m62_{before[:16]}.py"
    if destination.exists() and destination.read_text(encoding="utf-8") != source:
        raise FileExistsError("backup filename collision")
    if not destination.exists():
        destination.write_text(source, encoding="utf-8")
    # Re-read to protect against concurrent edits between dry run and write.
    if target.read_text(encoding="utf-8") != source:
        raise RuntimeError("runner changed during patch; refusing")
    target.write_text(patched, encoding="utf-8")
    print("M62_PATCH_APPLIED", "old_sha256", before, "new_sha256", after, "backup", destination)

if __name__ == "__main__":
    main()
