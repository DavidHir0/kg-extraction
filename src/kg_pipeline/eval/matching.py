"""Node alignment between a predicted and a golden ``ArchitectureGraph``.

The two sides share no node ids (the LLM invents its own, goldens carry
Label Studio region ids), so every metric starts from an assignment problem.
The per-pair feature similarity combines four signals:

- normalized ``raw_text`` cosine over char-3-gram TF-IDF vectors,
- class / sub_type agreement (soft bonus -- a misclassified box still
  matches; the error shows up in the accuracy metrics instead),
- Jaccard overlap of normalized ``key=value`` property pairs.

Structure resolves what text cannot: duplicate labels (a dozen identical
"Conv 3x3" boxes) and implicit ``raw_text: ""`` junction/input nodes. Two
matchers implement that structural term -- ``match_propagation`` (soft
similarity propagation, one Hungarian solve at the end) and ``match_anchor``
(iterative anchor-and-expand) -- so they can be compared on real figures.
``match_propagation`` is the default; ``match_anchor`` stalls on repeated
blocks such as ResNet stages.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from kg_pipeline.eval.preprocess import norm_text, property_pairs
from kg_pipeline.schema.graph import ArchitectureGraph


@dataclass(frozen=True)
class MatchConfig:
    # Feature-signal weights (sum to 1).
    w_text: float = 0.50
    w_class: float = 0.20
    w_sub_type: float = 0.10
    w_properties: float = 0.20
    # Feature-vs-structure blend and iteration budgets. Information travels
    # ~1 hop per iteration, so budgets must cover the longest stretch of
    # identical-text nodes between two unambiguous anchors (GoogLeNet's 9
    # stacked inception modules, ResNet-34's block chain).
    alpha: float = 0.5
    iterations: int = 25
    anchor_max_rounds: int = 64
    # Pairs scoring below this stay unmatched (Hungarian would otherwise
    # happily pair leftover garbage with leftover garbage).
    threshold: float = 0.35
    # match_anchor only: minimum score to freeze a pair as an anchor, and the
    # required gap to the runner-up in its row/column (so duplicate-text
    # pairs are never frozen on an arbitrary permutation).
    anchor_confidence: float = 0.75
    anchor_margin: float = 0.05
    # Post-assignment edge-consistency refinement: among near-tie candidates
    # (score within `edge_refine_eps`, e.g. identical-text duplicates where
    # features carry zero distinguishing signal), swap/move assignments to
    # maximize agreement with the predicted edges. Fixes instance
    # cross-matching on repeated blocks, where a correct pred edge otherwise
    # scores as FP while its gold twin scores as FN.
    edge_refine: bool = True
    edge_refine_eps: float = 0.05
    edge_refine_passes: int = 10


@dataclass(frozen=True)
class MatchedPair:
    pred_id: str
    gold_id: str
    score: float
    signals: dict[str, float] = field(hash=False)


@dataclass(frozen=True)
class MatchResult:
    pairs: list[MatchedPair]
    unmatched_pred: list[str]
    unmatched_gold: list[str]

    @property
    def pred_to_gold(self) -> dict[str, str]:
        return {p.pred_id: p.gold_id for p in self.pairs}


# --- feature similarity ---


def _char_ngrams(text: str, n: int = 3) -> Counter:
    if len(text) < n:
        return Counter([text] if text else [])
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def _tfidf_cosine(pred_texts: list[str], gold_texts: list[str]) -> np.ndarray:
    """Pairwise cosine over char-3-gram TF-IDF; the corpus is both graphs."""
    grams = [_char_ngrams(t) for t in [*pred_texts, *gold_texts]]
    df = Counter(g for doc in grams for g in doc)
    n_docs = len(grams)
    idf = {g: math.log((1 + n_docs) / (1 + c)) + 1 for g, c in df.items()}

    vectors = [{g: tf * idf[g] for g, tf in doc.items()} for doc in grams]
    norms = [math.sqrt(sum(w * w for w in v.values())) or 1.0 for v in vectors]

    sim = np.zeros((len(pred_texts), len(gold_texts)))
    for i, pv in enumerate(vectors[: len(pred_texts)]):
        for j, gv in enumerate(vectors[len(pred_texts) :], start=0):
            if pv and gv:
                dot = sum(w * gv.get(g, 0.0) for g, w in pv.items())
                sim[i, j] = dot / (norms[i] * norms[len(pred_texts) + j])
    return sim


def feature_similarity(
    pred: ArchitectureGraph, gold: ArchitectureGraph, cfg: MatchConfig
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The structure-free similarity matrix plus its per-signal components."""
    p_texts = [norm_text(n.raw_text) for n in pred.nodes]
    g_texts = [norm_text(n.raw_text) for n in gold.nodes]

    text = _tfidf_cosine(p_texts, g_texts)
    # Number-agreement damping (idea from the legacy GraphEvaluator's strict
    # number check, softened): char-ngrams consider "64 570 x 570" and
    # "64 568 x 568" nearly identical, but the digits are exactly what
    # distinguishes duplicate dimension labels. Jaccard over digit tokens
    # scales the text signal instead of zeroing it -- transcription variants
    # legitimately add numbers ("(step 2)" suffixes, "284^2" vs "284 x 284").
    p_nums = [set(re.findall(r"\d+", t)) for t in p_texts]
    g_nums = [set(re.findall(r"\d+", t)) for t in g_texts]
    for i, pt in enumerate(p_texts):
        for j, gt in enumerate(g_texts):
            if not pt and not gt:
                text[i, j] = 0.5  # neutral: implicit nodes are decided by structure
            elif not pt or not gt:
                text[i, j] = 0.0
            elif p_nums[i] and g_nums[j]:
                jac = len(p_nums[i] & g_nums[j]) / len(p_nums[i] | g_nums[j])
                text[i, j] *= 0.4 + 0.6 * jac

    class_ = np.zeros_like(text)
    sub_type = np.zeros_like(text)
    for i, pn in enumerate(pred.nodes):
        for j, gn in enumerate(gold.nodes):
            class_[i, j] = float(pn.class_ == gn.class_)
            sub_type[i, j] = float(pn.sub_type == gn.sub_type)

    p_props = [property_pairs(n) for n in pred.nodes]
    g_props = [property_pairs(n) for n in gold.nodes]
    props = np.zeros_like(text)
    for i, pp in enumerate(p_props):
        for j, gp in enumerate(g_props):
            if not pp and not gp:
                props[i, j] = 0.5  # neutral: most nodes carry no properties
            else:
                props[i, j] = len(pp & gp) / len(pp | gp)

    signals = {"text": text, "class": class_, "sub_type": sub_type, "properties": props}
    s_feat = (
        cfg.w_text * text
        + cfg.w_class * class_
        + cfg.w_sub_type * sub_type
        + cfg.w_properties * props
    )
    return s_feat, signals


# --- structural neighborhoods ---


def _neighbors(graph: ArchitectureGraph) -> tuple[list[list[int]], list[list[int]]]:
    """(out, in) neighbor index lists per node, over ALL edge roles."""
    index = {n.id: i for i, n in enumerate(graph.nodes)}
    out: list[list[int]] = [[] for _ in graph.nodes]
    in_: list[list[int]] = [[] for _ in graph.nodes]
    for edge in graph.edges:
        if edge.source in index and edge.target in index:
            out[index[edge.source]].append(index[edge.target])
            in_[index[edge.target]].append(index[edge.source])
    return out, in_


def _soft_neighborhood(s: np.ndarray, p_nbrs: list[int], g_nbrs: list[int]) -> float:
    """Symmetrized best-match agreement between two neighbor sets under ``s``.

    Averaging both directions makes extra/missing neighbors count against the
    pair; identical neighborhoods under a perfect ``s`` score exactly 1.0.
    """
    if not p_nbrs and not g_nbrs:
        return 1.0
    if not p_nbrs or not g_nbrs:
        return 0.0
    block = s[np.ix_(p_nbrs, g_nbrs)]
    return float(block.max(axis=1).mean() + block.max(axis=0).mean()) / 2


def _soft_structure(
    s: np.ndarray,
    p_adj: tuple[list[list[int]], list[list[int]]],
    g_adj: tuple[list[list[int]], list[list[int]]],
) -> np.ndarray:
    struct = np.zeros_like(s)
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            struct[i, j] = (
                _soft_neighborhood(s, p_adj[0][i], g_adj[0][j])
                + _soft_neighborhood(s, p_adj[1][i], g_adj[1][j])
            ) / 2
    return struct


# --- assignment ---


def _assign(
    pred: ArchitectureGraph,
    gold: ArchitectureGraph,
    s: np.ndarray,
    signals: dict[str, np.ndarray],
    threshold: float,
) -> MatchResult:
    pairs: list[MatchedPair] = []
    matched_p: set[str] = set()
    matched_g: set[str] = set()
    if s.size:
        for i, j in zip(*linear_sum_assignment(-s)):
            if s[i, j] < threshold:
                continue
            pairs.append(
                MatchedPair(
                    pred_id=pred.nodes[i].id,
                    gold_id=gold.nodes[j].id,
                    score=float(s[i, j]),
                    signals={k: float(m[i, j]) for k, m in signals.items()},
                )
            )
            matched_p.add(pred.nodes[i].id)
            matched_g.add(gold.nodes[j].id)
    return MatchResult(
        pairs=pairs,
        unmatched_pred=[n.id for n in pred.nodes if n.id not in matched_p],
        unmatched_gold=[n.id for n in gold.nodes if n.id not in matched_g],
    )


# --- edge-consistency refinement ---


def _refine_by_edges(
    pred: ArchitectureGraph,
    gold: ArchitectureGraph,
    s: np.ndarray,
    signals: dict[str, np.ndarray],
    result: MatchResult,
    cfg: MatchConfig,
) -> MatchResult:
    """Hill-climb the assignment on edge agreement, restricted to near-ties.

    Features cannot distinguish copies of identical nodes (a dozen "conv 3x3"
    boxes score identically against every counterpart), so Hungarian picks an
    arbitrary permutation and correct edges land between wrong instances --
    counted as FP while their gold twins count as FN. Only moves the features
    are indifferent about (within ``edge_refine_eps``) are considered, so
    refinement can permute duplicates but never overrides real text/property
    evidence. Two move types: swapping the gold targets of two matched pred
    nodes, and moving a matched pred node onto an unmatched gold node.
    """
    p_idx = {n.id: i for i, n in enumerate(pred.nodes)}
    g_idx = {n.id: j for j, n in enumerate(gold.nodes)}
    mapping = {p.pred_id: p.gold_id for p in result.pairs}
    gold_edges = {(e.source, e.target) for e in gold.edges}
    # Pred edges incident to each node, for local agreement deltas.
    incident: dict[str, list[tuple[str, str]]] = {}
    for e in pred.edges:
        incident.setdefault(e.source, []).append((e.source, e.target))
        if e.target != e.source:
            incident.setdefault(e.target, []).append((e.source, e.target))

    def local_agreement(node_ids: tuple[str, ...], mp: dict[str, str]) -> int:
        edges = {ed for nid in node_ids for ed in incident.get(nid, [])}
        return sum(1 for a, b in edges if (mp.get(a), mp.get(b)) in gold_edges)

    eps, thr = cfg.edge_refine_eps, cfg.threshold
    for _ in range(cfg.edge_refine_passes):
        improved = False
        matched_pred = list(mapping)
        unmatched_gold = [n.id for n in gold.nodes if n.id not in set(mapping.values())]

        for a_i, p_a in enumerate(matched_pred):
            g_a = mapping.get(p_a)
            if g_a is None:
                continue
            i_a = p_idx[p_a]
            base_pairs: list[tuple[str, str | None]] = []
            # Swap candidates: another matched pred whose gold target is
            # feature-interchangeable with ours.
            for p_b in matched_pred[a_i + 1 :]:
                g_b = mapping.get(p_b)
                if g_b is None:
                    continue
                i_b = p_idx[p_b]
                if (
                    s[i_a, g_idx[g_b]] >= max(thr, s[i_a, g_idx[g_a]] - eps)
                    and s[i_b, g_idx[g_a]] >= max(thr, s[i_b, g_idx[g_b]] - eps)
                ):
                    base_pairs.append((p_b, g_b))
            # Move candidates: an unmatched gold node we score near-equally on.
            for g_free in unmatched_gold:
                if s[i_a, g_idx[g_free]] >= max(thr, s[i_a, g_idx[g_a]] - eps):
                    base_pairs.append((p_a, g_free))  # sentinel: move, not swap

            for p_b, g_b in base_pairs:
                trial = dict(mapping)
                if p_b == p_a:  # move onto unmatched gold
                    trial[p_a] = g_b
                    touched = (p_a,)
                else:  # swap gold targets
                    trial[p_a], trial[p_b] = mapping[p_b], mapping[p_a]
                    touched = (p_a, p_b)
                if local_agreement(touched, trial) > local_agreement(touched, mapping):
                    mapping = trial
                    unmatched_gold = [
                        n.id for n in gold.nodes if n.id not in set(mapping.values())
                    ]
                    improved = True
                    break  # restart candidate scan for p_a with new mapping
        if not improved:
            break

    pairs = [
        MatchedPair(
            pred_id=p,
            gold_id=g,
            score=float(s[p_idx[p], g_idx[g]]),
            signals={k: float(m[p_idx[p], g_idx[g]]) for k, m in signals.items()},
        )
        for p, g in mapping.items()
    ]
    matched_g = set(mapping.values())
    return MatchResult(
        pairs=sorted(pairs, key=lambda p: -p.score),
        unmatched_pred=[n.id for n in pred.nodes if n.id not in mapping],
        unmatched_gold=[n.id for n in gold.nodes if n.id not in matched_g],
    )


# --- matcher 1: similarity propagation ---


def match_propagation(
    pred: ArchitectureGraph, gold: ArchitectureGraph, cfg: MatchConfig | None = None
) -> MatchResult:
    """Soft evidence accumulation: iterate the structural term over the full
    similarity matrix, decide everything in one final Hungarian solve."""
    cfg = cfg or MatchConfig()
    s_feat, signals = feature_similarity(pred, gold, cfg)
    p_adj, g_adj = _neighbors(pred), _neighbors(gold)

    s = s_feat.copy()
    for _ in range(cfg.iterations if s.size else 0):
        s = cfg.alpha * s_feat + (1 - cfg.alpha) * _soft_structure(s, p_adj, g_adj)

    signals = {**signals, "structure": s}
    result = _assign(pred, gold, s, signals, cfg.threshold)
    if cfg.edge_refine:
        result = _refine_by_edges(pred, gold, s, signals, result, cfg)
    return result


# --- matcher 2: iterative anchor-and-expand ---


def _hard_structure(
    mapping: dict[int, int],
    p_adj: tuple[list[list[int]], list[list[int]]],
    g_adj: tuple[list[list[int]], list[list[int]]],
    seed: np.ndarray,
) -> np.ndarray:
    """Neighbor agreement under the current hard anchor mapping.

    Only neighbors that ARE anchored count as evidence. Where no anchored
    neighbor exists yet, fall back to ``seed``: in the bootstrap round that
    is the one-shot soft neighbor-feature similarity, so the process can
    start even on figures with NO unique-text node at all (GoogLeNet Fig2 is
    two twin panels); in later rounds it must be flat neutral 0.5 -- keeping
    the soft seed there holds unevidenced twin pairs at ~1.0, which destroys
    the margin of genuinely evidenced pairs and stalls the expansion.
    """

    def overlap(p_nbrs: list[int], g_nbrs: list[int]) -> float | None:
        if not p_nbrs and not g_nbrs:
            return 1.0
        if not p_nbrs or not g_nbrs:
            return 0.0
        mapped = [mapping[k] for k in p_nbrs if k in mapping]
        if not mapped:
            return None  # no evidence either way
        return sum(1 for m in mapped if m in g_nbrs) / len(mapped)

    struct = np.zeros(seed.shape)
    for i in range(seed.shape[0]):
        for j in range(seed.shape[1]):
            known = [
                v
                for v in (
                    overlap(p_adj[0][i], g_adj[0][j]),
                    overlap(p_adj[1][i], g_adj[1][j]),
                )
                if v is not None
            ]
            struct[i, j] = sum(known) / len(known) if known else seed[i, j]
    return struct


def _confident_anchors(s: np.ndarray, cfg: MatchConfig) -> dict[int, int]:
    """Assignment pairs that score high AND beat their row/column runner-up.

    The margin requirement keeps duplicate-text nodes (identical rows) from
    being frozen on an arbitrary permutation in the first round.
    """
    anchors: dict[int, int] = {}
    for i, j in zip(*linear_sum_assignment(-s)):
        if s[i, j] < cfg.anchor_confidence:
            continue
        row = np.delete(s[i, :], j)
        col = np.delete(s[:, j], i)
        runner_up = max(row.max() if row.size else 0.0, col.max() if col.size else 0.0)
        if s[i, j] - runner_up >= cfg.anchor_margin:
            anchors[i] = j
    return anchors


def match_anchor(
    pred: ArchitectureGraph, gold: ArchitectureGraph, cfg: MatchConfig | None = None
) -> MatchResult:
    """Old-approach reimplementation: freeze confident matches, re-score the
    rest by matched-neighbor overlap, repeat to fixpoint, then assign."""
    cfg = cfg or MatchConfig()
    s_feat, signals = feature_similarity(pred, gold, cfg)
    p_adj, g_adj = _neighbors(pred), _neighbors(gold)
    seed = _soft_structure(s_feat, p_adj, g_adj) if s_feat.size else s_feat

    s = s_feat.copy()
    anchors: dict[int, int] = {}
    for _ in range(cfg.anchor_max_rounds if s.size else 0):
        fallback = seed if not anchors else np.full(s_feat.shape, 0.5)
        s = cfg.alpha * s_feat + (1 - cfg.alpha) * _hard_structure(
            anchors, p_adj, g_adj, fallback
        )
        # Anchors accumulate monotonically; earlier (higher-confidence-round)
        # anchors win any conflict with later candidates.
        anchored_golds = set(anchors.values())
        added = False
        for i, j in _confident_anchors(s, cfg).items():
            if i not in anchors and j not in anchored_golds:
                anchors[i] = j
                anchored_golds.add(j)
                added = True
        if not added:
            break

    signals = {**signals, "structure": s}
    result = _assign(pred, gold, s, signals, cfg.threshold)
    if cfg.edge_refine:
        result = _refine_by_edges(pred, gold, s, signals, result, cfg)
    return result


MATCHERS = {"propagation": match_propagation, "anchor": match_anchor}
