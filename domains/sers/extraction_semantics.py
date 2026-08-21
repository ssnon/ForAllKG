from __future__ import annotations

import re

from pipeline_core.corpus.extraction.draft_schema import (
    KnowledgeGraphDraft,
)
from pipeline_core.runtime.validation_issues import (
    IssueCode,
    IssueStage,
    ValidationIssue,
    issue,
)


SERS_STRICT_SEMANTIC_CONTRACT_ID = (
    "sers_au_ag_precursor_role_v2"
)

SERS_STRICT_SEMANTIC_CONTRACT_RULES = (
    "Precursor must not be explicitly source-grounded as a reducing agent or reductant.",
    "Precursor must not be explicitly source-grounded as a stabilizer.",
    "Precursor must not be explicitly source-grounded as a surfactant.",
    "Precursor must not be explicitly source-grounded as a capping agent.",
    "Precursor must not be explicitly source-grounded as a structure-directing agent.",
    "Precursor must not be explicitly source-grounded as serving or acting as a solvent.",
    (
        "Precursor must not be an explicit physical target or foil used as "
        "feedstock in laser ablation, sputtering, or evaporation."
    ),
)


_PHYSICAL_TARGET_OBJECT_PATTERN = re.compile(
    r"\b(?:targets?|foils?)\b",
    re.I,
)

_PHYSICAL_TARGET_PROCESS_PATTERN = re.compile(
    r"\b(?:"
    r"laser\s+ablation|"
    r"pulsed\s+laser|"
    r"sputter(?:ing|ed)?|"
    r"thermal\s+evaporation|"
    r"electron[- ]beam\s+evaporation|"
    r"e[- ]beam\s+evaporation|"
    r"evaporat(?:e|ed|ion)|"
    r"physical\s+vapor\s+deposition|"
    r"\bPVD\b"
    r")\b",
    re.I,
)


_DESCRIPTION_PATTERNS = {
    "reducing_agent": re.compile(
        r"(?:^|\b(?:is|was|acts?|acting|serves?|serving)\s+as\s+)"
        r"(?:an?\s+|aqueous\s+)?"
        r"(?:reducing agent|reducing reagent|reductant)\b",
        re.I,
    ),
    "stabilizer": re.compile(
        r"(?:^|\b(?:is|was|acts?|acting|serves?|serving)\s+as\s+)"
        r"(?:an?\s+)?stabilizer\b",
        re.I,
    ),
    "surfactant": re.compile(
        r"(?:^|\b(?:is|was|acts?|acting|serves?|serving)\s+as\s+)"
        r"(?:an?\s+)?surfactant\b",
        re.I,
    ),
    "capping_agent": re.compile(
        r"(?:^|\b(?:is|was|acts?|acting|serves?|serving)\s+as\s+)"
        r"(?:an?\s+)?capping agent\b",
        re.I,
    ),
    "structure_directing_agent": re.compile(
        r"(?:^|\b(?:is|was|acts?|acting|serves?|serving)\s+as\s+)"
        r"(?:an?\s+)?structure[- ]directing agent\b",
        re.I,
    ),
    "solvent": re.compile(
        r"\b(?:used|serves?|served|serving|acts?|acting)\s+as\s+"
        r"(?:an?\s+)?solvent\b",
        re.I,
    ),
}


_AS_ROLE_PATTERN = re.compile(
    r"\bas\s+(?:an?\s+|the\s+)?"
    r"(?P<role>"
    r"reducing agent|reducing reagent|reductant|"
    r"stabilizer|surfactant|capping agent|"
    r"structure[- ]directing agent|solvent"
    r")\b",
    re.I,
)


def _aliases(label: str) -> list[str]:
    label = str(label or "").strip()
    values = {label.casefold()} if label else set()

    for value in re.findall(
        r"\(([^()]+)\)",
        label,
    ):
        value = value.strip()
        if len(value) >= 2:
            values.add(value.casefold())

    for token in re.findall(
        r"[A-Za-z0-9][A-Za-z0-9+·().\-]*",
        label,
    ):
        token = token.strip("().,")
        if len(token) >= 3:
            values.add(token.casefold())

    values -= {
        "solution",
        "precursor",
        "material",
        "acid",
        "metal",
        "aqueous",
    }

    return sorted(
        values,
        key=len,
        reverse=True,
    )


def _description_cues(
    description: str,
) -> list[str]:
    return sorted(
        name
        for name, pattern in _DESCRIPTION_PATTERNS.items()
        if pattern.search(str(description or ""))
    )


def _anchored_evidence_cues(
    *,
    label: str,
    evidence_text: str,
) -> list[str]:
    evidence = str(evidence_text or "")
    evidence_cf = evidence.casefold()
    aliases = _aliases(label)

    cues = []

    for match in _AS_ROLE_PATTERN.finditer(evidence):
        prefix = evidence_cf[
            max(0, match.start() - 180):
            match.start()
        ]

        if not any(
            alias in prefix
            for alias in aliases
        ):
            continue

        role = (
            match.group("role")
            .casefold()
            .replace("-", "_")
            .replace(" ", "_")
        )
        cues.append(role)

    return sorted(set(cues))


def _physical_target_cue(
    *,
    target_label: str,
    target_description: str,
    source_label: str,
    source_description: str,
    evidence_text: str,
) -> str | None:
    """Return a cue only for explicit physical-process feedstock targets.

    Deliberately does NOT classify generic powders, reagents, seeds,
    source materials, or intermediates. Those remain source-dependent.
    """
    target_text = " ".join([
        str(target_label or ""),
        str(target_description or ""),
    ])

    if not _PHYSICAL_TARGET_OBJECT_PATTERN.search(
        target_text
    ):
        return None

    process_text = " ".join([
        str(source_label or ""),
        str(source_description or ""),
        str(evidence_text or ""),
        target_text,
    ])

    if not _PHYSICAL_TARGET_PROCESS_PATTERN.search(
        process_text
    ):
        return None

    return "physical_process_target"


def collect_sers_strict_semantic_issues(
    draft: KnowledgeGraphDraft,
) -> list[ValidationIssue]:
    """Reject only explicit high-confidence Precursor role contradictions.

    Generic reagent/feedstock/seed semantics are intentionally NOT hard-gated.
    Those remain review-only until a stronger ontology contract is justified.
    """
    entities = {
        node.id: node
        for node in draft.entities
    }

    issues: list[ValidationIssue] = []

    for node in draft.entities:
        if str(node.type) != "Precursor":
            continue

        label = str(node.label or "")
        description = str(
            node.description or ""
        )

        cues = [
            f"description:{name}"
            for name in _description_cues(
                description
            )
        ]

        implicated_edges: list[
            tuple[int, object]
        ] = []

        for edge_index, edge in enumerate(
            draft.edges
        ):
            if edge.target != node.id:
                continue

            source = entities.get(
                edge.source
            )

            if source is None:
                continue

            if str(source.type) != "SynthesisMethod":
                continue

            if edge.relation not in {
                "USES_PRECURSOR",
                "USES_MATERIAL",
            }:
                continue

            edge_cues = _anchored_evidence_cues(
                label=label,
                evidence_text=edge.evidence_text,
            )

            physical_target_cue = _physical_target_cue(
                target_label=label,
                target_description=description,
                source_label=str(
                    source.label or ""
                ),
                source_description=str(
                    source.description or ""
                ),
                evidence_text=edge.evidence_text,
            )

            if physical_target_cue is not None:
                edge_cues.append(
                    physical_target_cue
                )

            edge_cues = sorted(
                set(edge_cues)
            )

            if edge_cues:
                implicated_edges.append(
                    (edge_index, edge)
                )

            cues.extend(
                f"evidence_as_role:{name}"
                for name in edge_cues
            )

        cues = sorted(set(cues))

        if not cues:
            continue

        edge_index = None
        source_id = None
        relation = None

        if implicated_edges:
            edge_index, edge = (
                implicated_edges[0]
            )
            source_id = edge.source
            relation = edge.relation

        issues.append(
            issue(
                code=(
                    IssueCode
                    .DOMAIN_SEMANTIC_ROLE_CONTRADICTION
                ),
                stage=IssueStage.RELATION,
                message=(
                    f"Precursor {node.id!r} ({label!r}) is "
                    "explicitly source-grounded as a non-precursor "
                    f"synthesis role: {', '.join(cues)}. "
                    "Under the SERS extraction contract, explicit "
                    "reducing agents/stabilizers/surfactants/"
                    "capping agents/structure-directing agents/"
                    "solvents and explicit physical ablation/"
                    "sputtering/evaporation targets or foils must "
                    "not be typed as Precursor merely to satisfy "
                    "USES_PRECURSOR endpoints."
                ),
                node_id=node.id,
                node_collection="entities",
                edge_index=edge_index,
                source_id=source_id,
                target_id=node.id,
                relation=relation,
                expected={
                    "semantic_role": (
                        "actual precursor, or Material for an "
                        "explicit non-precursor synthesis input"
                    ),
                },
                actual={
                    "type": "Precursor",
                    "label": label,
                    "explicit_nonprecursor_cues": cues,
                },
            )
        )

    return issues
