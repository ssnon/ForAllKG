from __future__ import annotations

import re
from typing import Any, Iterable

from pipeline_core.bridge_policy_runtime import (
    PluginBridgePolicyIssue,
    PluginBridgePolicyPartition,
    dedupe_policy_issues,
    partition_with_policy,
)
from dac_her.bridge_schemas import BridgeChunkGraph, BridgeConcept, BridgeLink
from domains.sers.bridge_signatures import normalize_sers_bridge_text


SERS_BRIDGE_POLICY_VERSION = 'sers-au-ag-bridge-policy-v1-alpha4b2b2'

_GENERIC_LABELS = {
    'high performance',
    'excellent performance',
    'good performance',
    'important effect',
    'strong enhancement',
    'this behavior',
    'experimental result',
    'calculation result',
    'optimized structure',
    'stable structure',
    'favorable property',
}

_METRIC_TERMS = re.compile(
    r'\b(?:enhancement factor|\baef\b|\bef\b|limit of detection|\blod\b|'
    r'sers intensity|raman intensity|raman peak|raman shift|lspr wavelength|'
    r'plasmon resonance wavelength|extinction peak|absorption peak|'
    r'local[- ]field enhancement|field enhancement|nanogap width|gap size|'
    r'shell thickness|particle size|aspect ratio|au\s*[:/]\s*ag|'
    r'composition ratio|concentration|\brsd\b|relative standard deviation)\b',
    re.I,
)

_SCALAR_FRONTIER_TERMS = re.compile(
    r'^(?:enhancement factor|aef|ef|limit of detection|lod|sers intensity|'
    r'raman intensity|raman peak|raman shift|lspr wavelength|'
    r'plasmon resonance wavelength|nanogap width|gap size|shell thickness|'
    r'particle size|aspect ratio|concentration|rsd|relative standard deviation)$',
    re.I,
)

_NUMERIC_OR_UNIT = re.compile(
    r'(?:\b\d+(?:\.\d+)?(?:\s*[x×]\s*10\^?[-+]?\d+)?\b|'
    r'\bnm\b|\bcm\s*[-^]?1\b|\bmol(?:ar)?\b|\bmw\b|\bms\b|%|'
    r'\bm\b)',
    re.I,
)

# Relation lexical support is intentionally high precision. alpha4b.2b2 expands
# only constructions that appeared as clear false-candidate cases during the
# SERS_1/5/8 calibration. Ambiguous captions ("with different particle sizes"),
# theoretical agreement ("agrees with Mie theory"), and broad "however"
# contrasts remain review candidates.
_RELATION_EVIDENCE_CUES: dict[str, re.Pattern[str]] = {
    'CORRELATES_WITH': re.compile(
        r'(?:\bcorrelat\w*\b|\bassociat\w*\b|\brelationship\b|'
        r'\brelated to\b|\bcorrespond\w* to\b)',
        re.I,
    ),
    'VARIES_WITH': re.compile(
        r'(?:'
        r'\bvar(?:y|ies|ied|iation)\b|'
        r'\bdepend\w*\s+(?:on|upon)\b|'
        r'\bchange\w*\s+with\b|'
        r'\bincreas(?:e|es|ed|ing)\s+(?:with|as|along with)\b|'
        r'\bdecreas(?:e|es|ed|ing)\s+(?:with|as)\b|'
        r'\bred[- ]?shift\w*(?:\s+as|\s+with|\s+compared to)\b|'
        r'\bblue[- ]?shift\w*(?:\s+as|\s+with|\s+compared to)\b|'
        r'\bproportional to\b|'
        r'\blinear (?:correlation|dependence) between\b|'
        r'\bas a function of\b|'
        r'\bwith varying\b|'
        r'\bhas a marked effect on\b|'
        r'\bprimarily depends upon\b|'
        r'\bincreases along with\b|'
        r'\bcloser\b.{0,120}\bhigher\b|'
        r'\bwhen\b.{0,120}\bincreas\w*\b.{0,120}\bbecame '
        r'(?:larger|smaller|higher|lower)\b|'
        r'\bchanged from\b.{0,120}\bas\b.{0,120}\bincreas\w*\b|'
        r'\badjust\w*\b.{0,120}\b(?:control\w*|tun\w*)\b|'
        r'\b(?:control\w*|tun\w*)\b.{0,120}\badjust\w*\b|'
        r'\bcan be reliably controlled\b|'
        r'\bcan be significantly enhanced\b'
        r')',
        re.I,
    ),
    'COMPETES_WITH': re.compile(r'\b(?:compet\w*|competitive)\b', re.I),
    'COMPETES_FOR': re.compile(r'\b(?:compet\w*|competitive)\b', re.I),
    'SELECTS': re.compile(
        r'\b(?:select\w*|prefer\w*|favou?r\w*|preferential\w*)\b', re.I
    ),
    'CONTRASTS_WITH': re.compile(
        r'(?:\bcontrast\w*\b|\bwhereas\b|\bcompared (?:with|to)\b|'
        r'\bin contrast\b|'
        r'\b(?:higher|lower|greater|smaller|stronger|weaker|better|worse)'
        r'(?:\s+[\w-]+){0,6}\s+than\b|'
        r'\bwhere\b)',
        re.I,
    ),
    'MODULATES': re.compile(
        r'(?:\bmodulat\w*\b|\baffect\w*\b|\balter\w*\b|'
        r'\btun(?:e|es|ed|ing)\b|\bshift\w*\b|\bcontrol\w*\b|'
        r'\bgovern\w*\b|\bplay\w* an important role in\b)',
        re.I,
    ),
    'MEDIATES': re.compile(r'\bmediat\w*\b', re.I),
    'PROMOTES': re.compile(
        r'(?:\bpromot\w*\b|\bfacilitat\w*\b|\bboost\w*\b|'
        r'\bimprov\w*\b|\benhanc(?:e|es|ed|ing)\b|'
        r'\blead(?:s|ing)? to\b|\bresult\w* in\b|\bensure\b|'
        r'\benabl\w*\b|\bto achieve\b|\bbecause of\b|'
        r'\bgenerated\b.{0,100}\bby\b|'
        r'\bvia (?:the )?[\w-]+(?:\s+[\w-]+){0,3}\s+process\b)',
        re.I,
    ),
    'SUPPRESSES': re.compile(
        r'(?:\bsuppress\w*\b|\binhibit\w*\b|\bretard\w*\b|'
        r'\bprevent\w*\b|\bprotect\w*\b|\breduc(?:e|es|ed|ing)\b|'
        r'\bresist\w*\b|\bavoid\w*\b|\brestrict\w*\b|'
        r'\bdeteriorat\w*\b|\blimit\w*\b)',
        re.I,
    ),
    'SUGGESTS_DESIGN_RULE': re.compile(
        r'(?:\bsuggest\w*\b|\bdesign\b|\bstrategy\b|\bprinciple\b|'
        r'\boptimization\b|\boptimisation\b|'
        r'\bcan be exploited for use as\b|'
        r'\bcan be controlled by adjusting\b|'
        r'\bcan be improved by\b|\bcan be tuned by\b)',
        re.I,
    ),
    'IMPOSES_TRADEOFF': re.compile(
        r'\b(?:trade[- ]?off|at the expense of|balance between|compromise)\b',
        re.I,
    ),
    'IDENTIFIES_FAILURE_MODE': re.compile(
        r'\b(?:degrad\w*|oxid\w*|aggregat\w*|instabil\w*|dissolv\w*|'
        r'corrod\w*|reshap\w*|collaps\w*|detach\w*|leach\w*)\b',
        re.I,
    ),
}

_AXIS_TERMS = re.compile(
    r'\b(?:nanogap|gap size|shell thickness|particle size|aspect ratio|'
    r'composition|metal identity|architecture|core[- ]?shell|alloy|'
    r'au\s*[:/]\s*ag|metal ratio|gold content|au content|precursor amount|'
    r'concentration|excitation wavelength|laser power|integration time|'
    r'geometry|morphology)\b',
    re.I,
)
_OUTCOME_TERMS = re.compile(
    r'\b(?:sers activity|sers response|sers signal|sers intensity|'
    r'raman signal|raman intensity|enhancement|enhancement factor|\bef\b|\baef\b|'
    r'\blod\b|detection limit|local field|field enhancement|plasmon resonance|'
    r'lspr|peak position|peak count|bandwidth|stability|reproducibility|'
    r'charge transfer)\b',
    re.I,
)
_ANALYTICAL_SIGNAL = re.compile(
    r'\b(?:sers|raman)\b.{0,40}\b(?:signal|intensity|peak|response)\b|'
    r'\b(?:signal|intensity|peak)\b.{0,40}\b(?:sers|raman)\b',
    re.I,
)
_ANALYTE_CONTEXT = re.compile(
    r'\b(?:analyte|reporter|target|dna|atp|r6g|methylene|'
    r'crystal violet|rhodamine|molecule)\b',
    re.I,
)
_FRONTIER_RELATIONAL_PREDICATE = re.compile(
    r'(?:\bproduces?\b|\bpromotes?\b|\bsuppresses?\b|\bvaries with\b|'
    r'\bdepends on\b|\bleads? to\b|\bresults? in\b|\bcauses?\b|'
    r'\bcaused by\b|\bassociated with\b|'
    r'\bdealloying[- ]induced\b.{0,80}\bformation\b)',
    re.I,
)
_PASSIVE_CAUSE_MARKER = re.compile(
    r'\b(?:by|because of|owing to|due to)\b',
    re.I,
)
_FAILURE_EVENT = _RELATION_EVIDENCE_CUES['IDENTIFIES_FAILURE_MODE']
_TABLE_ROW = re.compile(r'^\s*\|.*\|\s*$', re.S)
_CANDIDATE_ONLY_CODES = frozenset({
    'RELATION_CUE_MISMATCH',
    'RELATION_ARGUMENT_SCOPE_AMBIGUOUS',
    'CAUSAL_ARGUMENT_SCOPE_AMBIGUOUS',
    'CAUSAL_ARGUMENT_DIRECTION',
    'TABLE_DERIVED_RELATION_REQUIRES_CONTEXT',
})


def _issue(
    code: str,
    field: str,
    detail: str,
    *,
    repairable: bool = False,
) -> PluginBridgePolicyIssue:
    return PluginBridgePolicyIssue(
        code=code,
        field=field,
        detail=detail,
        repairable=repairable,
    )


def _strict_labels(strict_nodes: Iterable[dict[str, Any]]) -> set[str]:
    labels: set[str] = set()
    for node in strict_nodes:
        for key in ('label', 'metric_id'):
            value = normalize_sers_bridge_text(node.get(key, ''))
            if value:
                labels.add(value)
    return labels


def _relation_supported(concept: BridgeConcept) -> bool:
    relation = str(concept.pattern_relation or '')
    cue = _RELATION_EVIDENCE_CUES.get(relation)
    if cue is None:
        return False
    fields = [
        concept.relation_evidence_phrase or '',
        concept.source_phrase,
        *concept.supporting_phrases,
    ]
    return any(cue.search(field) for field in fields if field)


def _has_any_relation_cue(text: str) -> bool:
    return any(cue.search(text) for cue in _RELATION_EVIDENCE_CUES.values())


def _analytical_calibration_pattern(
    concept: BridgeConcept,
) -> bool:
    if str(concept.pattern_relation or '') != 'VARIES_WITH':
        return False
    subject = normalize_sers_bridge_text(concept.pattern_subject or '')
    object_ = normalize_sers_bridge_text(concept.pattern_object or '')
    return bool(
        _ANALYTICAL_SIGNAL.search(subject)
        and 'concentration' in object_
        and _ANALYTE_CONTEXT.search(object_)
    )


def _contrast_argument_scope_ambiguous(
    concept: BridgeConcept,
) -> bool:
    if str(concept.pattern_relation or '') != 'CONTRASTS_WITH':
        return False

    subject = normalize_sers_bridge_text(concept.pattern_subject or '')
    object_ = normalize_sers_bridge_text(concept.pattern_object or '')

    subject_outcome = bool(_OUTCOME_TERMS.search(subject))
    object_outcome = bool(_OUTCOME_TERMS.search(object_))
    subject_axis = bool(_AXIS_TERMS.search(subject))
    object_axis = bool(_AXIS_TERMS.search(object_))

    # CONTRASTS_WITH should compare peers. A property vs category/axis shape is
    # better represented as VARIES_WITH or as two peer property expressions.
    return bool(
        (subject_outcome and object_axis and not object_outcome)
        or (object_outcome and subject_axis and not subject_outcome)
    )


def _passive_causal_direction_reversed(
    concept: BridgeConcept,
) -> bool:
    if str(concept.pattern_relation or '') not in {
        'MODULATES',
        'MEDIATES',
        'PROMOTES',
        'SUPPRESSES',
    }:
        return False

    relation_phrase = normalize_sers_bridge_text(
        concept.relation_evidence_phrase or ''
    )
    if not relation_phrase or not _PASSIVE_CAUSE_MARKER.search(relation_phrase):
        return False

    source = normalize_sers_bridge_text(concept.source_phrase)
    subject_phrase = normalize_sers_bridge_text(
        concept.subject_evidence_phrase or concept.pattern_subject or ''
    )
    object_phrase = normalize_sers_bridge_text(
        concept.object_evidence_phrase or concept.pattern_object or ''
    )
    if not source or not subject_phrase or not object_phrase:
        return False

    subject_index = source.find(subject_phrase)
    object_index = source.find(object_phrase)
    relation_index = source.find(relation_phrase)

    if min(subject_index, object_index, relation_index) < 0:
        return False

    # "effect ... restricted/generated ... by cause" means the raw pattern has
    # effect as subject and cause as object, which is reversed for directional
    # cause -> effect relations such as PROMOTES/SUPPRESSES.
    return subject_index < relation_index < object_index


def concept_policy_issues(
    concept: BridgeConcept,
    *,
    strict_nodes: Iterable[dict[str, Any]],
    core_text: str | None = None,
    linked_links: list[BridgeLink] | None = None,
) -> list[PluginBridgePolicyIssue]:
    del core_text, linked_links
    issues: list[PluginBridgePolicyIssue] = []
    normalized_label = normalize_sers_bridge_text(concept.label)
    normalized_source = normalize_sers_bridge_text(concept.source_phrase)

    if normalized_label in _GENERIC_LABELS:
        issues.append(_issue(
            'GENERIC_LANGUAGE', 'label',
            'The label is too generic to be a reusable SERS Bridge concept.'
        ))

    if concept.retention_lane == 'accepted_pattern':
        subject = normalize_sers_bridge_text(concept.pattern_subject or '')
        object_ = normalize_sers_bridge_text(concept.pattern_object or '')
        relation = str(concept.pattern_relation or '')

        if not subject or not object_ or subject == object_:
            issues.append(_issue(
                'RELATION_MISSING', 'pattern_subject/pattern_object',
                'Accepted patterns require two distinct non-empty arguments.'
            ))

        if relation not in _RELATION_EVIDENCE_CUES:
            issues.append(_issue(
                'UNSUPPORTED_RELATION', 'pattern_relation',
                f'No deterministic SERS cue policy exists for {relation!r}.'
            ))
        elif not _relation_supported(concept):
            issues.append(_issue(
                'RELATION_CUE_MISMATCH', 'relation_evidence_phrase',
                f'The supplied source wording does not lexically support {relation}.',
                repairable=True,
            ))

        if _analytical_calibration_pattern(concept):
            issues.append(_issue(
                'ANALYTICAL_CALIBRATION_PATTERN',
                'pattern_subject/pattern_object',
                'Direct SERS/Raman signal versus analyte/reporter concentration '
                'is an analytical calibration relation already represented by '
                'strict measurements and is not retained as a discovery Bridge.'
            ))

        if _contrast_argument_scope_ambiguous(concept):
            issues.append(_issue(
                'RELATION_ARGUMENT_SCOPE_AMBIGUOUS',
                'pattern_subject/pattern_object',
                'CONTRASTS_WITH should compare peer scientific expressions; '
                'property-versus-axis/category shapes require review.',
                repairable=True,
            ))

        if _passive_causal_direction_reversed(concept):
            issues.append(_issue(
                'CAUSAL_ARGUMENT_DIRECTION',
                'pattern_subject/pattern_object',
                'The source uses a passive cause marker (by/because of/owing to/'
                'due to) with the apparent cause after the relation phrase, so '
                'the directional Bridge arguments appear reversed.',
                repairable=True,
            ))

        if (
            _METRIC_TERMS.search(subject)
            and _METRIC_TERMS.search(object_)
            and not _has_any_relation_cue(concept.source_phrase)
            and concept.pattern_support_mode != 'derived_multi_span'
        ):
            issues.append(_issue(
                'UNSUPPORTED_RELATION', 'source_phrase',
                'Two metric-like arguments require explicit relation evidence.'
            ))

        if relation == 'VARIES_WITH':
            if _AXIS_TERMS.search(subject) and _OUTCOME_TERMS.search(object_):
                issues.append(_issue(
                    'RELATION_ARGUMENT_DIRECTION', 'pattern_subject',
                    'VARIES_WITH should orient the varying SERS outcome/property '
                    'as subject and the condition/axis as object.'
                ))

        if relation == 'IDENTIFIES_FAILURE_MODE':
            evidence = ' '.join(filter(None, (
                concept.source_phrase,
                concept.relation_evidence_phrase or '',
            )))
            if not _FAILURE_EVENT.search(evidence):
                issues.append(_issue(
                    'FAILURE_MODE_WITHOUT_FAILURE_EVENT', 'pattern_relation',
                    'Failure-mode patterns require explicit degradation, oxidation, '
                    'aggregation, instability, dissolution, corrosion, or reshaping.'
                ))

        if concept.pattern_support_mode == 'derived_multi_span':
            pairs = {
                (
                    normalize_sers_bridge_text(item.subject_value),
                    normalize_sers_bridge_text(item.object_value),
                )
                for item in concept.comparison_items
            }
            if len(pairs) < 2:
                issues.append(_issue(
                    'INSUFFICIENT_COMPARISON_EVIDENCE', 'comparison_items',
                    'At least two distinct explicit comparison items are required.'
                ))
            phrases = [
                item.source_phrase for item in concept.comparison_items
                if item.source_phrase
            ]
            if phrases and all(_TABLE_ROW.fullmatch(phrase) for phrase in phrases):
                issues.append(_issue(
                    'TABLE_DERIVED_RELATION_REQUIRES_CONTEXT', 'comparison_items',
                    'Bare table rows require header/column semantic context before '
                    'confirmed Bridge acceptance.',
                    repairable=True,
                ))

        if relation in {'MODULATES', 'MEDIATES', 'PROMOTES', 'SUPPRESSES'}:
            source = normalize_sers_bridge_text(concept.source_phrase)
            if re.search(
                r'\bsuggest\w*\b.{0,160}\band\b.{0,80}'
                r'\b(?:enhanc|promot|suppress|modulat)',
                source,
            ):
                issues.append(_issue(
                    'CAUSAL_ARGUMENT_SCOPE_AMBIGUOUS', 'pattern_subject',
                    'Coordinated author interpretations make the causal actor '
                    'ambiguous.',
                    repairable=True,
                ))

        return dedupe_policy_issues(issues)

    # paper_local_frontier
    if _SCALAR_FRONTIER_TERMS.fullmatch(normalized_label):
        issues.append(_issue(
            'SCALAR_METRIC', 'label',
            'A bare scalar/measurement field belongs in the strict evidence graph.'
        ))

    if _FRONTIER_RELATIONAL_PREDICATE.search(normalized_label):
        issues.append(_issue(
            'RELATIONAL_FRONTIER', 'label',
            'The frontier label encodes a subject-relation-object claim. '
            'Frontier concepts must be atomic source-explicit concepts; '
            'relations belong in RelationPattern or remain outside the Bridge.'
        ))

    strict_labels = _strict_labels(strict_nodes)
    if normalized_label in strict_labels or normalized_source in strict_labels:
        issues.append(_issue(
            'STRICT_DUPLICATE', 'label/source_phrase',
            'The candidate duplicates content already represented in the strict graph.'
        ))

    if (
        _NUMERIC_OR_UNIT.search(concept.source_phrase)
        and _METRIC_TERMS.search(concept.source_phrase)
    ):
        issues.append(_issue(
            'INSTANCE_ONLY', 'source_phrase',
            'The candidate is a paper-specific numeric/condition instance rather '
            'than a reusable frontier concept.'
        ))

    return dedupe_policy_issues(issues)


def partition_sers_bridge_result(
    result: BridgeChunkGraph,
    *,
    strict_nodes: list[dict[str, Any]],
    core_text: str | None = None,
) -> PluginBridgePolicyPartition:
    return partition_with_policy(
        result,
        strict_nodes=strict_nodes,
        core_text=core_text,
        issue_builder=concept_policy_issues,
        normalizer=normalize_sers_bridge_text,
        candidate_only_codes=_CANDIDATE_ONLY_CODES,
    )
