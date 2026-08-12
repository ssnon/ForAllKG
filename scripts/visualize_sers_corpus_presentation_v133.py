#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


CATEGORY_COLORS = {
    "composition_material": "#e97b8d",
    "structure_morphology": "#3b97c8",
    "optics_mechanism": "#4daf62",
    "sers_performance": "#8e69cf",
    "analyte_reporter": "#d8a031",
    "other": "#9aa3ad",
}

CATEGORY_LABELS = {
    "composition_material": "Composition / material",
    "structure_morphology": "Structure / morphology",
    "optics_mechanism": "Optics / mechanism",
    "sers_performance": "SERS performance",
    "analyte_reporter": "Analyte / reporter",
    "other": "Other",
}

IGNORE_TYPES = {
    "Paper", "Figure", "Table", "Section", "Asset", "Author", "Journal",
    "Institution", "Reference", "Citation"
}

THEME_TYPES = {
    "PlasmonicSubstrate", "Nanostructure", "Morphology", "StructuralMotif",
    "OpticalCondition", "RamanReporter", "Analyte", "Material", "Metal",
    "Support", "Measurement", "MechanismClaim", "ObservationClaim",
    "Property", "PerformanceMetric", "Experiment", "Calculation",
}

GENERIC_LABELS = {
    "paper", "figure", "table", "section", "supporting information",
    "supplementary information", "main text", "result", "discussion",
}

CANONICAL_PATTERNS = [
    (r"\benhancement factor\b|\bsers enhancement factor\b|\bef\b", "Enhancement factor"),
    (r"\bdetection limit\b|\blod\b", "Detection limit"),
    (r"\bparticle size\b|\bsize\b", "Particle size"),
    (r"\bshell thickness\b", "Shell thickness"),
    (r"\braman intensity\b", "Raman intensity"),
    (r"\braman peak position\b|\bpeak position\b", "Raman peak position"),
    (r"\batomic fraction\b|\bcomposition ratio\b|\bau:ag\b|\bag:au\b", "Atomic fraction"),
    (r"\brelative standard deviation\b|\brsd\b|\breproducibility\b", "Relative standard deviation / RSD"),
    (r"\bhot ?spot\b", "Hotspot"),
    (r"\bnanogap\b|\bgap size\b", "Nanogap"),
    (r"\bcore[- ]shell\b", "Core-shell arrangement"),
    (r"\bnanorod\b", "Nanorod"),
    (r"\bnanocube\b", "Nanocube"),
    (r"\bnanobox\b|\bhollow\b", "Nanobox / hollow"),
    (r"\b3d substrate\b|\bthree-dimensional substrate\b", "3D substrate"),
    (r"\bplasmon resonance tuning\b", "Plasmon resonance tuning"),
    (r"\blspr\b|\bplasmon resonance\b", "LSPR"),
    (r"\bem enhancement\b|\belectromagnetic enhancement\b", "EM enhancement"),
    (r"\bchemical enhancement\b", "Chemical enhancement"),
    (r"\bcharge transfer\b", "Charge transfer"),
    (r"\blocal field\b", "Local field"),
    (r"\bstability\b", "Stability"),
    (r"\bexcitation wavelength\b|\babsorption-band wavelength\b|\bwavelength\b", "Absorption-band wavelength"),
    (r"\b4-atp\b|\b4 atp\b", "4-ATP"),
    (r"\br6g\b|\brhodamine 6g\b", "R6G"),
    (r"\bcrystal violet\b", "Crystal violet"),
    (r"\bmethylene blue\b", "Methylene blue"),
    (r"\bau-ag\b|\bag-au\b|\bbimetallic\b|\balloy\b", "Au-Ag bimetallic"),
    (r"\bau\b", "Au"),
    (r"\bag\b", "Ag"),
    (r"\bsensitivity\b", "Sensitivity"),
    (r"\bsingle[- ]molecule sers\b|\bsingle molecule\b", "Single-molecule SERS"),
]

COMPOSITION_HINTS = ["au", "ag", "alloy", "bimetallic", "atomic fraction", "composition"]
STRUCTURE_HINTS = [
    "particle size", "shell thickness", "core-shell", "nanorod", "nanocube",
    "nanobox", "3d substrate", "nanogap", "morphology", "gap size",
]
OPTICS_HINTS = [
    "lspr", "plasmon", "charge transfer", "local field", "em enhancement",
    "chemical enhancement", "hotspot", "wavelength",
]
PERFORMANCE_HINTS = [
    "enhancement factor", "detection limit", "sensitivity",
    "relative standard deviation", "rsd", "raman intensity",
    "raman peak position", "stability", "single-molecule",
]
ANALYTE_HINTS = ["4-atp", "r6g", "crystal violet", "methylene blue"]


def _clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("_", " ").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def normalize_theme_label(label: str) -> str | None:
    raw = _clean_text(label)
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in GENERIC_LABELS:
        return None
    for pattern, repl in CANONICAL_PATTERNS:
        if re.search(pattern, lowered):
            return repl
    if len(raw) > 42:
        raw = raw[:39].rstrip() + "..."
    return raw if raw.isupper() else raw[0].upper() + raw[1:]


def infer_category(node_type: str, label: str) -> str:
    t = (node_type or "").strip()
    l = (label or "").lower()
    if t in {"RamanReporter", "Analyte"}:
        return "analyte_reporter"
    if t in {"Morphology", "StructuralMotif", "Nanostructure", "Support"}:
        return "structure_morphology"
    if t == "PlasmonicSubstrate":
        return "composition_material" if any(h in l for h in COMPOSITION_HINTS) else "structure_morphology"
    if t in {"Metal", "Material"}:
        return "analyte_reporter" if any(h in l for h in ANALYTE_HINTS) else "composition_material"
    if t in {"Measurement", "PerformanceMetric"}:
        if any(h in l for h in ANALYTE_HINTS):
            return "analyte_reporter"
        if any(h in l for h in OPTICS_HINTS):
            return "optics_mechanism"
        return "sers_performance"
    if t in {"MechanismClaim", "ObservationClaim", "OpticalCondition", "Calculation"}:
        return "sers_performance" if any(h in l for h in PERFORMANCE_HINTS) else "optics_mechanism"
    if any(h in l for h in ANALYTE_HINTS):
        return "analyte_reporter"
    if any(h in l for h in PERFORMANCE_HINTS):
        return "sers_performance"
    if any(h in l for h in OPTICS_HINTS):
        return "optics_mechanism"
    if any(h in l for h in STRUCTURE_HINTS):
        return "structure_morphology"
    if any(h in l for h in COMPOSITION_HINTS):
        return "composition_material"
    return "other"


def is_theme_candidate(node_type: str, label: str) -> bool:
    if node_type in IGNORE_TYPES:
        return False
    if node_type in THEME_TYPES:
        return True
    l = (label or "").lower()
    return any(k in l for k in (
        "sers", "raman", "enhancement", "detection limit", "lspr",
        "particle size", "shell thickness", "nanogap", "hotspot",
        "core-shell", "alloy", "bimetallic", "wavelength", "stability",
        "r6g", "4-atp", "methylene blue", "crystal violet"
    ))


def collect_graph_paths(data_root: Path) -> list[Path]:
    return [p for p in sorted(data_root.glob("extracted/*/*.graphml")) if p.is_file()]


def extract_paper_themes(graph_path: Path) -> dict[str, str]:
    try:
        G = nx.read_graphml(graph_path)
    except Exception as exc:
        print(f"[WARN] failed to read {graph_path}: {exc}")
        return {}

    theme_to_category: dict[str, str] = {}
    for node_id, attrs in G.nodes(data=True):
        node_type = attrs.get("type", "")
        label = attrs.get("label") or attrs.get("name") or str(node_id)
        if not is_theme_candidate(node_type, label):
            continue
        norm = normalize_theme_label(label)
        if not norm or len(norm) < 3:
            continue
        category = infer_category(node_type, norm)
        previous = theme_to_category.get(norm)
        if previous is None or previous == "other":
            theme_to_category[norm] = category
    return theme_to_category


def build_theme_graph(graph_paths: list[Path]) -> tuple[nx.Graph, int]:
    theme_counts = Counter()
    pair_counts = Counter()
    category_votes: dict[str, Counter] = defaultdict(Counter)
    paper_count = 0

    for graph_path in graph_paths:
        themes = extract_paper_themes(graph_path)
        if not themes:
            continue
        paper_count += 1
        unique_themes = sorted(set(themes))
        for theme in unique_themes:
            theme_counts[theme] += 1
            category_votes[theme][themes[theme]] += 1
        for a, b in combinations(unique_themes, 2):
            pair_counts[(a, b)] += 1

    G = nx.Graph()
    for theme, count in theme_counts.items():
        category = category_votes[theme].most_common(1)[0][0]
        G.add_node(theme, count=count, category=category)

    for (a, b), weight in pair_counts.items():
        if a in G and b in G:
            G.add_edge(a, b, weight=weight)

    for node in G.nodes:
        G.nodes[node]["weighted_degree"] = sum(
            data.get("weight", 1.0) for _, _, data in G.edges(node, data=True)
        )
    return G, paper_count


def select_hubs(G: nx.Graph, max_hubs: int) -> list[str]:
    scored = []
    for node, attrs in G.nodes(data=True):
        count = float(attrs.get("count", 0.0))
        wdeg = float(attrs.get("weighted_degree", 0.0))
        score = 3.0 * count + wdeg
        scored.append((score, count, wdeg, node))
    scored.sort(reverse=True)
    return [node for _, _, _, node in scored[:max_hubs]]


def _scale_positions(pos: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    return {
        node: (((x - min_x) / span_x) * 2.0 - 1.0, ((y - min_y) / span_y) * 2.0 - 1.0)
        for node, (x, y) in pos.items()
    }


def compute_base_layout(G: nx.Graph) -> dict[str, tuple[float, float]]:
    k = 1.1 / max(math.sqrt(max(1, G.number_of_nodes())), 1e-6)
    pos = nx.spring_layout(G, seed=42, k=k, iterations=250, weight="weight")
    return _scale_positions(pos)


def compute_center(pos: dict[str, tuple[float, float]], nodes: list[str]) -> tuple[float, float]:
    xs = [pos[n][0] for n in nodes if n in pos]
    ys = [pos[n][1] for n in nodes if n in pos]
    return (sum(xs) / len(xs), sum(ys) / len(ys)) if xs else (0.0, 0.0)


def normalize_vector(dx: float, dy: float) -> tuple[float, float]:
    norm = math.sqrt(dx * dx + dy * dy)
    if norm < 1e-9:
        return (1.0, 0.0)
    return dx / norm, dy / norm


def organic_hub_spread(
    base_pos: dict[str, tuple[float, float]],
    hubs: list[str],
    spread_strength: float = 0.58,
    min_sep: float = 0.34,
    iterations: int = 85,
    anchor_strength: float = 0.055,
) -> dict[str, tuple[float, float]]:
    """Spread hubs with pairwise repulsion while keeping them near the original layout.

    Unlike a ring/ellipse layout, this preserves the original angular/topological
    structure and only pushes hubs apart when they are too close.
    """
    pos = dict(base_pos)
    original = {n: tuple(base_pos[n]) for n in hubs}

    cx, cy = compute_center(base_pos, hubs)

    # Mild first expansion from the original center.
    for n in hubs:
        x, y = pos[n]
        dx, dy = x - cx, y - cy
        pos[n] = (cx + dx * (1.0 + 0.45 * spread_strength),
                  cy + dy * (1.0 + 0.32 * spread_strength))

    for _ in range(iterations):
        delta = {n: [0.0, 0.0] for n in hubs}

        # Pairwise repulsion only where hubs actually collide/crowd.
        for i, a in enumerate(hubs):
            ax, ay = pos[a]
            for b in hubs[i + 1:]:
                bx, by = pos[b]
                dx, dy = ax - bx, ay - by
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < min_sep:
                    ux, uy = normalize_vector(dx, dy)
                    overlap = min_sep - dist
                    push = overlap * 0.23 * spread_strength
                    delta[a][0] += ux * push
                    delta[a][1] += uy * push
                    delta[b][0] -= ux * push
                    delta[b][1] -= uy * push

        # Anchor force: keep organic relation to the original spring layout.
        for n in hubs:
            x, y = pos[n]
            ox, oy = original[n]
            delta[n][0] += (ox - x) * anchor_strength
            delta[n][1] += (oy - y) * anchor_strength

        # Weak outward pressure to avoid central collapse.
        for n in hubs:
            x, y = pos[n]
            ux, uy = normalize_vector(x - cx, y - cy)
            delta[n][0] += ux * 0.0025 * spread_strength
            delta[n][1] += uy * 0.0020 * spread_strength

        for n in hubs:
            x, y = pos[n]
            pos[n] = (x + delta[n][0], y + delta[n][1])

    return pos


def organic_label_positions(
    hub_pos: dict[str, tuple[float, float]],
    hubs: list[str],
    label_offset: float = 0.14,
    min_sep_y: float = 0.070,
    max_vertical_shift: float = 0.055,
) -> dict[str, tuple[float, float]]:
    """Keep labels close to each hub while applying only bounded de-overlap.

    The previous version could push labels far toward the margins. This version
    places each label just outside its node and allows only a small vertical
    correction when labels on the same side collide.
    """
    cx, cy = compute_center(hub_pos, hubs)
    labels = {}
    base_labels = {}

    for n in hubs:
        x, y = hub_pos[n]
        ux, uy = normalize_vector(x - cx, y - cy)

        # Near-node placement. Horizontal offset is slightly stronger so text
        # starts outside the circle rather than covering it.
        lx = x + ux * label_offset * 1.05
        ly = y + uy * label_offset * 0.78
        labels[n] = (lx, ly)
        base_labels[n] = (lx, ly)

    # Minimal overlap relaxation on each side. Importantly, the shift is capped
    # so labels remain visually attached to their nodes.
    for side in ("left", "right"):
        nodes = [
            n for n in hubs
            if (labels[n][0] < cx and side == "left") or
               (labels[n][0] >= cx and side == "right")
        ]
        nodes.sort(key=lambda n: labels[n][1])

        last_y = None
        for n in nodes:
            x, y = labels[n]
            base_y = base_labels[n][1]

            if last_y is not None and y - last_y < min_sep_y:
                target_y = last_y + min_sep_y
                delta = max(
                    -max_vertical_shift,
                    min(max_vertical_shift, target_y - base_y),
                )
                y = base_y + delta

            labels[n] = (x, y)
            last_y = y

    return labels


def draw_legend(ax):
    handles = [
        Line2D([0], [0], marker="o", color="w", label=label,
               markerfacecolor=CATEGORY_COLORS[key], markersize=10)
        for key, label in CATEGORY_LABELS.items()
        if key != "other"
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.005, 0.005),
              ncol=3, frameon=False, fontsize=11)


def shorten(text: str, max_len: int = 28) -> str:
    return text if len(text) <= max_len else text[: max_len - 3].rstrip() + "..."



def wrap_label_centered(text: str) -> str:
    """Wrap long hub labels into compact two-line labels for node-centered text."""
    manual = {
        "Absorption-band wavelength": "Absorption-band\nwavelength",
        "Core-shell arrangement": "Core-shell\narrangement",
        "Au-Ag bimetallic": "Au-Ag\nbimetallic",
        "Raman peak position": "Raman peak\nposition",
        "Raman intensity": "Raman\nintensity",
        "Enhancement factor": "Enhancement\nfactor",
        "Particle size": "Particle\nsize",
        "Relative standard deviation / RSD": "Relative standard\ndeviation / RSD",
        "Single-molecule SERS": "Single-molecule\nSERS",
        "Plasmon resonance tuning": "Plasmon resonance\ntuning",
        "Chemical enhancement": "Chemical\nenhancement",
        "EM enhancement": "EM\nenhancement",
        "Charge transfer": "Charge\ntransfer",
    }
    if text in manual:
        return manual[text]

    # General fallback: if the label is long, split near the middle at a word
    # boundary. At most two lines are produced.
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= 16 or " " not in cleaned:
        return cleaned

    words = cleaned.split()
    best_i = None
    best_delta = None
    total_len = len(cleaned)

    for i in range(1, len(words)):
        left = " ".join(words[:i])
        right = " ".join(words[i:])
        delta = abs(len(left) - len(right))
        # Avoid very short one-word fragments unless unavoidable.
        penalty = 5 if min(len(left), len(right)) < 5 else 0
        score = delta + penalty
        if best_delta is None or score < best_delta:
            best_delta = score
            best_i = i

    if best_i is None:
        return cleaned

    return " ".join(words[:best_i]) + "\n" + " ".join(words[best_i:])



def draw_network(
    G: nx.Graph,
    paper_count: int,
    base_pos: dict[str, tuple[float, float]],
    hubs: list[str],
    out_path: Path,
    spread_strength: float,
    min_sep: float,
    label_offset: float,
):
    pos = organic_hub_spread(
        base_pos,
        hubs,
        spread_strength=spread_strength,
        min_sep=min_sep,
    )

    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    bg = "#f4f4f4"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    hub_set = set(hubs)

    background_edges = []
    hub_edges = []
    for u, v, data in G.edges(data=True):
        if u in hub_set and v in hub_set:
            hub_edges.append((u, v, data))
        else:
            background_edges.append((u, v, data))

    if background_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v) for u, v, _ in background_edges],
            width=[0.20 + 0.15 * min(data.get("weight", 1), 4) for _, _, data in background_edges],
            edge_color="#9da8b2",
            alpha=0.048,
            ax=ax,
        )

    if hub_edges:
        colors = []
        for u, v, _ in hub_edges:
            cu = CATEGORY_COLORS.get(G.nodes[u].get("category", "other"), "#9aa3ad")
            cv = CATEGORY_COLORS.get(G.nodes[v].get("category", "other"), "#9aa3ad")
            colors.append(cu if cu == cv else "#aeb6bf")
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v) for u, v, _ in hub_edges],
            width=[0.7 + 0.30 * min(data.get("weight", 1), 5) for _, _, data in hub_edges],
            edge_color=colors,
            alpha=0.20,
            ax=ax,
        )

    # Background nodes invisible: only the connection texture remains.
    non_hubs = [n for n in G.nodes if n not in hub_set]
    if non_hubs:
        nx.draw_networkx_nodes(
            G, pos, nodelist=non_hubs,
            node_size=5, node_color="#8fa1b3",
            alpha=0.0, ax=ax,
        )

    counts = [G.nodes[n].get("count", 1) for n in hubs]
    cmin, cmax = min(counts), max(counts)
    span = max(cmax - cmin, 1)

    def hub_size(n: str) -> float:
        count = G.nodes[n].get("count", 1)
        return 1000 + ((count - cmin) / span) * 2700

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=hubs,
        node_size=[hub_size(n) for n in hubs],
        node_color=[CATEGORY_COLORS.get(G.nodes[n].get("category", "other"), "#9aa3ad") for n in hubs],
        edgecolors="white",
        linewidths=1.4,
        alpha=0.92,
        ax=ax,
    )

    max_count = max(counts)
    min_count = min(counts)
    count_span = max(max_count - min_count, 1)

    # Labels are drawn directly at the hub centers.
    # Long labels are wrapped onto two lines to reduce overlap.
    for n in hubs:
        x, y = pos[n]
        count = G.nodes[n].get("count", 1)

        # Keep text readable inside/around the node without dominating it.
        fs = 10.5 + ((count - min_count) / count_span) * 5.5

        ax.text(
            x,
            y,
            wrap_label_centered(n),
            ha="center",
            va="center",
            multialignment="center",
            linespacing=0.92,
            fontsize=fs,
            color="black",
            fontweight="medium",
            zorder=6,
        )

    ax.text(
        0.01, 0.985,
        "Au–Ag SERS — Theme Map",
        transform=ax.transAxes,
        fontsize=25,
        fontweight="bold",
        va="top",
        ha="left",
        color="#111111",
    )
    ax.text(
        0.01, 0.95,
        (
            f"{paper_count} papers · {len(hubs)} main hubs · "
            f"organic spread={spread_strength:.2f} · labels centered on hubs · background nodes hidden"
        ),
        transform=ax.transAxes,
        fontsize=12.5,
        va="top",
        ha="left",
        color="#586069",
    )

    draw_legend(ax)
    ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Organic middle-spread Au-Ag SERS corpus visualization with centered wrapped hub labels"
    )
    parser.add_argument("--data-root", default="data_sers")
    parser.add_argument("--output-dir", default="runs/sers_presentation_v133")
    parser.add_argument("--max-hubs", type=int, default=9)

    # Main tuning knobs.
    parser.add_argument(
        "--spread-strength", type=float, default=0.58,
        help="0 = close to original spring layout; 1 = much more separated. Recommended 0.45-0.70."
    )
    parser.add_argument(
        "--min-hub-separation", type=float, default=0.34,
        help="Minimum hub spacing used by constrained repulsion. Recommended 0.28-0.42."
    )
    parser.add_argument(
        "--label-offset", type=float, default=0.14,
        help="Deprecated in v1.3.3; labels are centered on hub nodes."
    )
    args = parser.parse_args()

    graph_paths = collect_graph_paths(Path(args.data_root))
    if not graph_paths:
        raise SystemExit(
            f"No GraphML files found under {args.data_root}/extracted/*/*.graphml"
        )

    G, paper_count = build_theme_graph(graph_paths)
    if G.number_of_nodes() == 0:
        raise SystemExit("Theme graph is empty after parsing the corpus.")

    hubs = select_hubs(G, args.max_hubs)
    base_pos = compute_base_layout(G)

    draw_network(
        G=G,
        paper_count=paper_count,
        base_pos=base_pos,
        hubs=hubs,
        out_path=Path(args.output_dir) / "au_ag_sers_theme_map_center_labels.png",
        spread_strength=args.spread_strength,
        min_sep=args.min_hub_separation,
        label_offset=args.label_offset,
    )
    print("[done] organic-spread centered-label visualization complete")


if __name__ == "__main__":
    main()
