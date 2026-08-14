from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal


DEFAULT_STRONG_CAUSAL_TEXT_PATTERNS: tuple[str, ...] = (
    r"\bcauses?\b",
    r"\bcaused by\b",
    r"\bdrives?\b",
    r"\bleads? to\b",
    r"\bresults? in\b",
    r"\bpromotes?\b",
    r"\benhances?\b",
    r"\bfacilitates?\b",
    r"\benables?\b",
    r"\bimproves?\b",
    r"\baccelerates?\b",
    r"\blowers?\b",
    r"\breduces?\b",
    r"\bincreases?\b",
    r"\bdecreases?\b",
    r"\bmodulates?\b",
    r"\bregulates?\b",
    r"\bcontrols?\b",
    r"\bstabiliz(?:e|es|ed|ing)\b",
    r"\binduc(?:e|es|ed|ing)\b",
)

DEFAULT_STRONG_CAUSAL_RELATION_MARKERS: tuple[str, ...] = (
    "CAUSE", "DRIVE", "LEAD", "RESULT", "PROMOT", "ENHANC",
    "FACILITAT", "ENABLE", "IMPROV", "ACCELERAT", "LOWER",
    "REDUC", "INCREAS", "DECREAS", "MODULAT", "REGULAT",
    "CONTROL", "STABIL", "INDUC",
)

PatternRows = tuple[tuple[str, tuple[str, ...]], ...]


def _compiled(rows: PatternRows) -> dict[str, tuple[re.Pattern[str], ...]]:
    return {
        name: tuple(re.compile(pattern, re.I) for pattern in patterns)
        for name, patterns in rows
    }


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in patterns)


@dataclass(frozen=True)
class ResolutionSemantics:
    resolvable_node_types: frozenset[str]
    auto_merge_types: frozenset[str]
    text_replacements: tuple[tuple[str, str], ...]
    reaction_aliases: tuple[tuple[str, str], ...]
    dual_atom_terms: tuple[str, ...] = ('dual atom', 'dimer')
    single_atom_terms: tuple[str, ...] = ('single atom',)
    nanoparticle_terms: tuple[str, ...] = ('nanoparticle',)
    support_signature_tokens: frozenset[str] = frozenset()
    high_priority_review_types: frozenset[str] = frozenset()

    def catalyst_nuclearity(self, tokens: frozenset[str]) -> str:
        joined = ' '.join(sorted(tokens))
        if {'dual', 'atom'} <= tokens or 'dimer' in tokens or any(
            term in joined for term in self.dual_atom_terms
        ):
            return 'dual_atom'
        if {'single', 'atom'} <= tokens or any(
            term in joined for term in self.single_atom_terms
        ):
            return 'single_atom'
        if 'nanoparticle' in tokens or any(
            term in joined for term in self.nanoparticle_terms
        ):
            return 'nanoparticle'
        return 'unspecified'


@dataclass(frozen=True)
class DiscoverySemantics:
    generic_entity_types: frozenset[str]
    mechanism_node_markers: tuple[str, ...]
    mechanism_relation_markers: tuple[str, ...]
    scaffold_relations: frozenset[str]
    context_node_types: frozenset[str]
    shared_entity_types: frozenset[str] = frozenset()
    legacy_mechanism_id_prefixes: tuple[str, ...] = ("mech_",)
    strong_causal_text_patterns: tuple[str, ...] = (
        DEFAULT_STRONG_CAUSAL_TEXT_PATTERNS
    )
    strong_causal_relation_markers: tuple[str, ...] = (
        DEFAULT_STRONG_CAUSAL_RELATION_MARKERS
    )

    def normalized_context_node_types(self) -> frozenset[str]:
        return frozenset(
            ''.join(ch for ch in value.upper() if ch.isalnum())
            for value in self.context_node_types
        )


@dataclass(frozen=True)
class NoveltySemantics:
    domain_patterns: PatternRows
    scope_patterns: PatternRows
    critical_scope_features: frozenset[str]
    claim_context_patterns: tuple[str, ...] = ()
    document_mismatch_patterns: tuple[str, ...] = ()
    document_compatible_patterns: tuple[str, ...] = ()
    mismatch_multiplier: float = 1.0
    domain_mismatch_reason: str = 'domain_mismatch'
    low_scope_reason: str = 'low_system_scope_overlap'

    def compiled_domain_patterns(self) -> dict[str, tuple[re.Pattern[str], ...]]:
        return _compiled(self.domain_patterns)

    def compiled_scope_patterns(self) -> dict[str, tuple[re.Pattern[str], ...]]:
        return _compiled(self.scope_patterns)

    def domains(self, text: str) -> set[str]:
        return {
            name
            for name, patterns in self.compiled_domain_patterns().items()
            if any(pattern.search(text) for pattern in patterns)
        }

    def scope_features(self, text: str) -> set[str]:
        return {
            name
            for name, patterns in self.compiled_scope_patterns().items()
            if any(pattern.search(text) for pattern in patterns)
        }

    def domain_relevance(self, claim_text: str, document: str) -> float:
        claim_domains = self.domains(claim_text)
        doc_domains = self.domains(document)
        if not claim_domains:
            score = 0.5
        elif claim_domains & doc_domains:
            score = 1.0
        elif doc_domains:
            score = 0.05
        else:
            score = 0.35

        claim_context = _matches_any(claim_text, self.claim_context_patterns)
        mismatch = _matches_any(document, self.document_mismatch_patterns)
        compatible = _matches_any(document, self.document_compatible_patterns)
        if claim_context and mismatch and not compatible:
            score *= self.mismatch_multiplier
        return max(0.0, min(1.0, score))

    def scope_relevance(self, claim_text: str, document: str) -> float:
        claim_features = self.scope_features(claim_text)
        if not claim_features:
            return 0.5
        doc_features = self.scope_features(document)
        return len(claim_features & doc_features) / len(claim_features)

    def strong_scope_compatibility(
        self,
        claim_text: str,
        document: str,
        *,
        min_domain: float,
        min_scope: float,
    ) -> tuple[bool, float, float, list[str]]:
        domain = self.domain_relevance(claim_text, document)
        scope = self.scope_relevance(claim_text, document)
        claim_features = self.scope_features(claim_text)
        doc_features = self.scope_features(document)
        reasons: list[str] = []
        if domain < min_domain:
            reasons.append(self.domain_mismatch_reason)
        for critical in self.critical_scope_features:
            if critical in claim_features and critical not in doc_features:
                reasons.append(f'missing_critical_scope:{critical}')
        if scope < min_scope:
            reasons.append(self.low_scope_reason)
        return (not reasons, domain, scope, reasons)


CorpusPatternAlignmentMode = Literal["disabled", "confirmed_exact"]


@dataclass(frozen=True)
class CorpusSemantics:
    semantics_id: str
    review_candidate_types: frozenset[str]
    pattern_alignment_mode: CorpusPatternAlignmentMode = "confirmed_exact"
    high_priority_review_types_override: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if not self.semantics_id.strip():
            raise ValueError("Corpus semantics_id must not be empty.")
        if any(not value.strip() for value in self.review_candidate_types):
            raise ValueError("Corpus review candidate types must not be empty.")
        if self.pattern_alignment_mode not in {"disabled", "confirmed_exact"}:
            raise ValueError(
                "Corpus pattern_alignment_mode must be 'disabled' "
                "or 'confirmed_exact'."
            )
        if self.high_priority_review_types_override is not None:
            unknown = (
                self.high_priority_review_types_override
                - self.review_candidate_types
            )
            if unknown:
                raise ValueError(
                    "Corpus high-priority review override must be a subset "
                    "of review_candidate_types: "
                    f"{sorted(unknown)!r}"
                )

    def effective_high_priority_review_types(
        self,
        resolution: ResolutionSemantics,
    ) -> frozenset[str]:
        if self.high_priority_review_types_override is not None:
            return self.high_priority_review_types_override
        return (
            resolution.high_priority_review_types
            & self.review_candidate_types
        )


ProjectionBacktraceDirection = Literal["incoming", "outgoing"]


@dataclass(frozen=True)
class ProjectionBacktraceRule:
    relation: str
    direction: ProjectionBacktraceDirection

    def __post_init__(self) -> None:
        if not self.relation.strip():
            raise ValueError("Projection backtrace relation must not be empty.")


@dataclass(frozen=True)
class ProjectionSemantics:
    semantics_id: str
    mechanism_node_types: frozenset[str]
    origin_node_types: frozenset[str]
    backtrace_rules: tuple[ProjectionBacktraceRule, ...]
    max_backtrace_depth: int = 3

    def __post_init__(self) -> None:
        if not self.semantics_id.strip():
            raise ValueError("Projection semantics_id must not be empty.")
        if self.max_backtrace_depth < 1:
            raise ValueError("Projection max_backtrace_depth must be >= 1.")
        signatures = [
            (rule.relation, rule.direction)
            for rule in self.backtrace_rules
        ]
        if len(signatures) != len(set(signatures)):
            raise ValueError("Projection backtrace rules must be unique.")


@dataclass(frozen=True)
class ScientificDomainProfile:
    profile_id: str
    description: str
    resolution: ResolutionSemantics
    discovery: DiscoverySemantics
    novelty: NoveltySemantics
    projection: ProjectionSemantics | None = None
    corpus: CorpusSemantics | None = None
    comparison_adapter_id: str | None = None
    extraction_adapter_id: str | None = None
    graph_adapter_id: str | None = None
    bridge_adapter_id: str | None = None
    feasibility_adapter_id: str | None = None
    reproducibility_adapter_id: str | None = None
    metric_definition_adapter_id: str | None = None
    trend_adapter_id: str | None = None
