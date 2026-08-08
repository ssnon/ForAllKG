from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import networkx as nx
import numpy as np

DEFAULT_EMBED_MODEL = os.getenv("GRAPHAGENTS_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_number}.")
            rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_match_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = re.sub(r"[\s_\-]+", " ", text)
    text = re.sub(r"[^\w\s*+./Δδ]", "", text)
    return " ".join(text.split())


def _as_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes"}


def _node_is_candidate(record: dict[str, Any]) -> bool:
    return (
        _as_bool(record.get("requires_verification", False))
        or str(record.get("evidence_status", "")) == "semantic_candidate"
        or str(record.get("graph_layer", "")) == "bridge_candidate"
    )


def _model_prefix(model_name: str, *, query: bool) -> str:
    if "nomic" in model_name.lower():
        return "search_query: " if query else "search_document: "
    return ""


class QueryEncoder(Protocol):
    def encode_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEncoder:
    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, *, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if device:
            kwargs["device"] = device
        try:
            self.model = SentenceTransformer(model_name, **kwargs)
        except TypeError:
            kwargs.pop("trust_remote_code", None)
            self.model = SentenceTransformer(model_name, **kwargs)

    def encode_documents(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        prefix = _model_prefix(self.model_name, query=False)
        encoded = self.model.encode(
            [prefix + text for text in texts],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(encoded, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        prefix = _model_prefix(self.model_name, query=True)
        encoded = self.model.encode(
            [prefix + text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(encoded[0], dtype=np.float32)


@dataclass(frozen=True)
class QueryConcept:
    text: str
    allowed_node_types: tuple[str, ...] = ()
    required_terms: tuple[str, ...] = ()
    allow_candidates: bool = False
    allow_alignment_hubs: bool = False
    top_k: int = 10
    min_similarity: float = -1.0


@dataclass(frozen=True)
class NodeMatch:
    node_id: str
    node_type: str
    label: str
    semantic_similarity: float
    ranking_score: float
    exact_label_match: bool
    label_contains_query: bool
    requires_verification: bool
    graph_layer: str
    evidence_status: str
    source_paper_id: str
    source_paper_ids_json: str
    corpus_node_kind: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NodeEmbeddingIndex:
    records: list[dict[str, Any]]
    embeddings: np.ndarray
    manifest: dict[str, Any]

    @property
    def model_name(self) -> str:
        return str(self.manifest["model_name"])


def build_node_embedding_index(
    *,
    navigation_graph_path: str | Path,
    node_text_path: str | Path,
    output_dir: str | Path,
    model_name: str = DEFAULT_EMBED_MODEL,
    device: str | None = None,
    batch_size: int = 32,
    include_alignment_hubs: bool = False,
) -> dict[str, Any]:
    navigation_graph_path = Path(navigation_graph_path)
    node_text_path = Path(node_text_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = nx.read_graphml(navigation_graph_path)
    rows = _read_jsonl(node_text_path)
    by_id = {str(row.get("node_id", "")): row for row in rows if str(row.get("node_id", "")).strip()}

    records: list[dict[str, Any]] = []
    texts: list[str] = []
    for node_id in sorted(map(str, graph.nodes)):
        attrs = dict(graph.nodes[node_id])
        corpus_kind = str(attrs.get("corpus_node_kind", ""))
        if corpus_kind == "alignment_hub" and not include_alignment_hubs:
            continue
        row = dict(by_id.get(node_id, {}))
        label = str(row.get("label") or attrs.get("label") or attrs.get("statement") or attrs.get("name") or node_id)
        text = str(row.get("node_text") or attrs.get("node_text") or f"type: {attrs.get('type', 'Unknown')}\nlabel: {label}")
        record = {
            "node_id": node_id,
            "type": str(row.get("type") or attrs.get("type", "Unknown")),
            "label": label,
            "node_text": text,
            "graph_layer": str(row.get("graph_layer") or attrs.get("graph_layer", "")),
            "evidence_status": str(row.get("evidence_status") or attrs.get("evidence_status", "")),
            "requires_verification": _as_bool(row.get("requires_verification", attrs.get("requires_verification", False))),
            "source_paper_id": str(row.get("source_paper_id") or attrs.get("source_paper_id", "")),
            "source_paper_ids_json": str(row.get("source_paper_ids_json") or attrs.get("source_paper_ids_json", "[]")),
            "corpus_node_kind": corpus_kind,
        }
        records.append(record)
        texts.append(text)

    if not records:
        raise RuntimeError("No navigation nodes were eligible for the embedding index.")

    encoder = SentenceTransformerEncoder(model_name, device=device)
    embeddings = encoder.encode_documents(texts, batch_size=batch_size)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
        raise RuntimeError(f"Unexpected embedding matrix shape: {embeddings.shape!r}")

    embeddings_path = output_dir / "embeddings.npy"
    records_path = output_dir / "records.jsonl"
    manifest_path = output_dir / "manifest.json"
    np.save(embeddings_path, embeddings)
    _write_jsonl(records_path, records)

    manifest = {
        "schema_version": 1,
        "model_name": model_name,
        "normalized_embeddings": True,
        "query_prefix": _model_prefix(model_name, query=True),
        "document_prefix": _model_prefix(model_name, query=False),
        "node_count": len(records),
        "embedding_dimension": int(embeddings.shape[1]),
        "include_alignment_hubs": bool(include_alignment_hubs),
        "navigation_graph": str(navigation_graph_path),
        "node_text": str(node_text_path),
        "navigation_graph_sha256": _sha256_file(navigation_graph_path),
        "node_text_sha256": _sha256_file(node_text_path),
        "embeddings": str(embeddings_path),
        "records": str(records_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_node_embedding_index(index_dir: str | Path) -> NodeEmbeddingIndex:
    index_dir = Path(index_dir)
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    records = _read_jsonl(index_dir / "records.jsonl")
    embeddings = np.load(index_dir / "embeddings.npy")
    if embeddings.shape[0] != len(records):
        raise RuntimeError(f"Embedding/record count mismatch: {embeddings.shape[0]} != {len(records)}")
    return NodeEmbeddingIndex(records=records, embeddings=np.asarray(embeddings, dtype=np.float32), manifest=manifest)


class NodeMapper:
    def __init__(self, index: NodeEmbeddingIndex, *, encoder: QueryEncoder | None = None, device: str | None = None) -> None:
        self.index = index
        self.encoder = encoder if encoder is not None else SentenceTransformerEncoder(index.model_name, device=device)

    @classmethod
    def from_directory(cls, index_dir: str | Path, *, device: str | None = None) -> "NodeMapper":
        return cls(load_node_embedding_index(index_dir), device=device)

    def map(self, concept: QueryConcept) -> list[NodeMatch]:
        if concept.top_k <= 0:
            return []
        query_vector = np.asarray(self.encoder.encode_query(concept.text), dtype=np.float32)
        if query_vector.ndim != 1 or query_vector.shape[0] != self.index.embeddings.shape[1]:
            raise ValueError("Query/index embedding dimension mismatch.")
        norm = float(np.linalg.norm(query_vector))
        if norm == 0.0:
            raise ValueError("Query embedding has zero norm.")
        query_vector = query_vector / norm
        similarities = self.index.embeddings @ query_vector

        allowed_types = set(map(str, concept.allowed_node_types))
        required_terms = [normalize_match_text(x) for x in concept.required_terms if normalize_match_text(x)]
        normalized_query = normalize_match_text(concept.text)
        matches: list[NodeMatch] = []

        for index, record in enumerate(self.index.records):
            node_type = str(record.get("type", "Unknown"))
            if allowed_types and node_type not in allowed_types:
                continue
            if not concept.allow_candidates and _node_is_candidate(record):
                continue
            if not concept.allow_alignment_hubs and str(record.get("corpus_node_kind", "")) == "alignment_hub":
                continue
            normalized_text = normalize_match_text(record.get("node_text", ""))
            if any(term not in normalized_text for term in required_terms):
                continue

            similarity = float(similarities[index])
            if similarity < concept.min_similarity:
                continue
            label = str(record.get("label", record.get("node_id", "")))
            normalized_label = normalize_match_text(label)
            exact = bool(normalized_query) and normalized_label == normalized_query
            contains = bool(normalized_query) and normalized_query in normalized_label
            ranking = similarity + (0.10 if exact else 0.02 if contains else 0.0)

            matches.append(NodeMatch(
                node_id=str(record["node_id"]),
                node_type=node_type,
                label=label,
                semantic_similarity=similarity,
                ranking_score=float(ranking),
                exact_label_match=exact,
                label_contains_query=contains,
                requires_verification=_node_is_candidate(record),
                graph_layer=str(record.get("graph_layer", "")),
                evidence_status=str(record.get("evidence_status", "")),
                source_paper_id=str(record.get("source_paper_id", "")),
                source_paper_ids_json=str(record.get("source_paper_ids_json", "[]")),
                corpus_node_kind=str(record.get("corpus_node_kind", "")),
            ))

        matches.sort(key=lambda x: (-x.ranking_score, -x.semantic_similarity, x.node_id))
        return matches[: concept.top_k]
