from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from dac_her.draft_schema import KnowledgeGraphDraft
from dac_her.extraction_policy import ExtractionPolicy
from dac_her.llm_openrouter import OpenRouterLLM
from dac_her.lossless_normalization import (
    normalize_knowledge_graph_payload,
)
from dac_her.semantic_patch import (
    PatchRejected,
    apply_semantic_patch,
)
from dac_her.semantic_patch_prompts import (
    PATCH_SYSTEM_PROMPT,
    build_semantic_patch_prompt,
)
from dac_her.semantic_patch_schema import (
    KnowledgeGraphPatch,
)
from dac_her.strict_validation import (
    validate_draft,
)


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one frozen post-micro draft through "
            "one semantic patch."
        )
    )

    parser.add_argument(
        "--paper-id",
        required=True,
    )

    parser.add_argument(
        "--run-id",
        required=True,
    )

    parser.add_argument(
        "--chunk-id",
        required=True,
    )

    parser.add_argument(
        "--micro-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--model",
        default=None,
    )

    parser.add_argument(
        "--provider",
        default=None,
    )

    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def write_json(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def authoritative_source_payload(
    raw: dict,
) -> dict:
    # 현재 source_chunks/*.json이 top-level chunk
    # record라면 그대로 사용한다.
    #
    # 혹시 {"chunk": {...}} 형태로 바뀌어도
    # 동작하도록 fallback을 둔다.
    chunk = raw.get("chunk")

    if isinstance(chunk, dict):
        return chunk

    return raw


def enforce_metadata(
    draft: KnowledgeGraphDraft,
    source: dict,
) -> KnowledgeGraphDraft:
    payload = draft.model_dump()

    # LLM이 생성하는 scientific content와 달리
    # 이 값들은 pipeline이 이미 알고 있는
    # control-plane metadata다.
    for key in (
        "paper_id",
        "chunk_id",
        "section",
        "document_id",
        "document_role",
        "page_ids",
        "asset_ids",
    ):
        if key in source:
            payload[key] = source[key]

    return KnowledgeGraphDraft.model_validate(
        payload
    )


def main() -> None:
    args = parse_args()

    model = (
        args.model
        or os.getenv("OPENROUTER_EXTRACT_MODEL")
    )

    provider = (
        args.provider
        or os.getenv("OPENROUTER_PROVIDER")
        or None
    )

    if not model:
        raise RuntimeError(
            "OPENROUTER_EXTRACT_MODEL is not defined."
        )

    run_dir = (
        PROJECT_ROOT
        / "data_dac"
        / "extracted"
        / args.paper_id
        / "runs"
        / args.run_id
    )

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}"
        )

    safe_chunk_id = args.chunk_id.replace(
        ":",
        "__",
    )

    source_path = (
        run_dir
        / "source_chunks"
        / f"{safe_chunk_id}.json"
    )

    micro_path = (
        run_dir
        / "candidates"
        / (
            f"{safe_chunk_id}"
            f"__micro_reextract_"
            f"{args.micro_index}.json"
        )
    )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source chunk not found: {source_path}"
        )

    if not micro_path.exists():
        raise FileNotFoundError(
            f"Micro candidate not found: {micro_path}"
        )

    source_raw = read_json(source_path)
    source = authoritative_source_payload(
        source_raw
    )

    draft = KnowledgeGraphDraft.model_validate(
        read_json(micro_path)
    )

    draft = enforce_metadata(
        draft,
        source,
    )

    # --------------------------------------------------
    # Recompute validation from the frozen micro draft.
    # Do not blindly trust the old validation JSON.
    # --------------------------------------------------

    before_report = validate_draft(draft)

    print("=" * 80)
    print("POST-MICRO REPLAY")
    print("=" * 80)
    print("Paper:", args.paper_id)
    print("Run:", args.run_id)
    print("Chunk:", args.chunk_id)
    print("Micro candidate:", micro_path)
    print()
    print(
        "Before patch valid:",
        before_report.valid,
    )
    print(
        "Before patch issues:",
        before_report.code_counts(),
    )

    if before_report.valid:
        print(
            "\nFrozen micro draft is already valid; "
            "no post-micro patch is necessary."
        )
        return

    core_text = source.get("core_text")

    if not isinstance(core_text, str):
        raise RuntimeError(
            "source chunk JSON has no string "
            "'core_text' field.\n"
            f"Available keys: {sorted(source.keys())}"
        )

    policy = ExtractionPolicy()

    llm = OpenRouterLLM(
        model=model,
        provider=provider,
        reproducible=False,
        zdr=True,
    )

    replay_dir = (
        run_dir
        / "replay"
        / safe_chunk_id
    )

    replay_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    debug_path = (
        replay_dir
        / "post_micro_patch_debug.json"
    )

    # --------------------------------------------------
    # ONE post-micro semantic patch
    # --------------------------------------------------

    patch = llm.generate_structured(
        system_prompt=PATCH_SYSTEM_PROMPT,
        prompt=build_semantic_patch_prompt(
            paper_id=source.get(
                "paper_id",
                args.paper_id,
            ),
            chunk_id=source.get(
                "chunk_id",
                args.chunk_id,
            ),
            document_id=source.get(
                "document_id",
                "main",
            ),
            document_role=source.get(
                "document_role",
                "main",
            ),
            page_ids=source.get(
                "page_ids",
                [],
            ),
            asset_ids=source.get(
                "asset_ids",
                [],
            ),
            core_text=core_text,
            asset_context=source.get(
                "asset_context",
                "",
            ),
            graph_payload=draft.model_dump(),
            report=before_report,
            previous_patch_feedback=(
                "This is a single post-micro patch. "
                "The graph has already been fully "
                "re-extracted from this small source "
                "leaf. Correct only the remaining "
                "strict-validation residuals. Do not "
                "broaden the graph or invent evidence."
            ),
        ),
        response_model=KnowledgeGraphPatch,
        temperature=0.0,
        max_tokens=policy.patch_completion_tokens,
        debug_path=debug_path,
    )

    write_json(
        replay_dir / "post_micro_patch.json",
        patch.model_dump(mode="json"),
    )

    print()
    print(
        "Patch operations:",
        len(patch.operations),
    )

    for op in patch.operations:
        print(
            " -",
            op.op,
            "issue=",
            getattr(op, "issue_id", None),
        )

    # --------------------------------------------------
    # Apply patch with exactly the same executor used
    # by strict_recovery.
    # --------------------------------------------------

    try:
        applied = apply_semantic_patch(
            draft=draft,
            patch=patch,
            report=before_report,
            max_operations=(
                policy.max_patch_operations
            ),
            allow_destructive=(
                policy.allow_destructive_patches
            ),
        )
    except PatchRejected as error:
        print()
        print("PATCH REJECTED")
        print(error)
        raise SystemExit(2) from error

    patched_draft = applied.draft

    # Same lossless normalization used in recovery.
    normalization = (
        normalize_knowledge_graph_payload(
            patched_draft.model_dump(),
            issues=before_report.issues,
        )
    )

    if normalization.operations:
        patched_draft = (
            KnowledgeGraphDraft.model_validate(
                normalization.payload
            )
        )

    patched_draft = enforce_metadata(
        patched_draft,
        source,
    )

    after_report = validate_draft(
        patched_draft
    )

    write_json(
        replay_dir / "patched_draft.json",
        patched_draft.model_dump(),
    )

    write_json(
        replay_dir / "after_post_micro_patch.json",
        after_report.model_dump(
            mode="json"
        ),
    )

    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)

    print(
        "Before:",
        before_report.code_counts(),
    )

    print(
        "After:",
        after_report.code_counts(),
    )

    print(
        "Draft strict-valid:",
        after_report.valid,
    )

    if after_report.valid:
        print()
        print(
            "SUCCESS: post-micro patch removed "
            "all draft-level strict validation issues."
        )
        raise SystemExit(0)

    print()
    print("Remaining issues:")

    for issue in after_report.issues:
        print(
            " -",
            issue.code.value,
            "|",
            issue.message,
        )

    raise SystemExit(2)


if __name__ == "__main__":
    main()