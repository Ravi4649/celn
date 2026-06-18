"""
KnowledgeChannel — Canal de conhecimento factual para GPVE
============================================================

Carrega centróides IDF-ponderados de frases do corpus (pré-computados).
Fornece um vetor de conhecimento factual para o 5º canal do _vsa_scores.

Pipeline:
  query(tokens) → IDF-weighted centroid of prompt tokens
  read(centroid) → top-10% weighted retrieval from corpus centroids
  score(candidates, knowledge_vec) → dot product relevance

Build offline com experiments/build_knowledge_cache.py
Load em < 1s. Query em < 0.05s. Zero dependências externas (só numpy).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from .core import normalize


class KnowledgeChannel:
    """Lightweight factual knowledge injector for GPVE _vsa_scores.

    Stores IDF-weighted sentence centroids from the corpus.
    At query time: IDF-weight the prompt tokens → centroid →
    top-10% cosine retrieval from corpus → weighted average knowledge vector.
    Score candidates by dot(knowledge_vec, candidate_vec).

    Principles: no backprop, no transformer, no fixed lists, no templates.
    Self-calibrating: IDF from empirical corpus distribution.
    """

    def __init__(self, cache_path: str | Path = "sentence_centroids.npz"):
        data = np.load(cache_path, allow_pickle=True)
        self.centroids = data["centroids"].astype(np.float32)
        self._idf = dict(data["idf"].item()) if "idf" in data else {}

    def query_and_read(
        self,
        tokens: list[str],
        vectors: np.ndarray,
        w2i: dict[str, int],
    ) -> np.ndarray:
        """IDF-weighted centroid → top-10% retrieval → knowledge vector.

        Args:
            tokens: Known vocabulary words from the prompt.
            vectors: Word vector matrix (V, D).
            w2i: Word-to-index mapping.

        Returns:
            Knowledge vector (D,), normalized. Encodes factual knowledge
            from the corpus about the topic described by tokens.
        """
        known = [t for t in tokens if t in w2i]
        if not known:
            return np.zeros(vectors.shape[1], dtype=np.float32)

        idx_arr = np.array([w2i[t] for t in known], dtype=np.int32)
        w = np.array([self._idf.get(t, 1.0) for t in known], dtype=np.float32)
        query = normalize((vectors[idx_arr].T @ w) / (w.sum() + 1e-12))

        sims = self.centroids @ query.astype(np.float32)
        k = max(1, len(sims) // 10)
        top_idx = np.argpartition(sims, -k)[-k:]
        top_sims = sims[top_idx]
        weights = np.maximum(top_sims, 0)
        w_sum = weights.sum()
        if w_sum > 1e-12:
            return normalize((self.centroids[top_idx].T @ weights) / w_sum)
        return query

    def score_candidates(
        self,
        candidate_tokens: list[str | None],
        knowledge_vec: np.ndarray,
        vectors: np.ndarray,
        w2i: dict[str, int],
    ) -> list[float]:
        """Score PCFG candidate tokens by factual relevance.

        Args:
            candidate_tokens: First surface tokens of PCFG rule candidates.
            knowledge_vec: Retrieved knowledge vector from query_and_read().
            vectors: Word vector matrix (V, D).
            w2i: Word-to-index mapping.

        Returns:
            Knowledge score for each candidate. Higher = more factually
            relevant to the prompt topic.
        """
        kv = normalize(knowledge_vec)
        scores = []
        for tok in candidate_tokens:
            if tok is not None and tok in w2i:
                scores.append(float(vectors[w2i[tok]] @ kv))
            else:
                scores.append(0.0)
        return scores
