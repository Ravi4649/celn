"""
Content-Aware Phase Lens (CAPL) — IDF-Weighted Alphas
======================================================

Phase-rotates M_pr toward each prompt token's phase signature (single pass).
Alpha = 1.0 / (1.0 + ln(freq)) — IDF-like weighting.

  Palavras RARAS (conteúdo semântico) → freq baixa → ln(freq) baixo → alpha ALTO
  Palavras FREQUENTES (função gramatical) → freq alta → ln(freq) alto → alpha BAIXO

Zero thresholds mágicos — alphas vêm da distribuição empírica de frequências
com decaimento logarítmico natural (IDF).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np

from .core import D, phase_lens, normalize
from .port_adapter import load_word_vectors


@dataclass
class ContentLensPacket:
    m_intent: np.ndarray            # Phase-rotated intent vector
    n_tokens_used: int              # Tokens projetados com alpha > 0
    alphas_used: List[float]        # Alphas aplicados (um por token)
    similarity_shift: float         # L2 delta M_pr → m_intent
    min_alpha: float                # Menor alpha aplicado
    max_alpha: float                # Maior alpha aplicado


class ContentAwareLens:
    """Phase-rotate M_pr toward prompt tokens using frequency-calibrated alphas."""

    def __init__(
        self,
        vectors_path: str | Path,
        corpus_path: str | Path | None = None,
        min_token_len: int = 2,
    ) -> None:
        vectors, word2idx = load_word_vectors(vectors_path)
        self.vectors = vectors.astype(np.float32)
        self.word2idx = word2idx
        self.min_token_len = min_token_len

        # Build token frequency table from corpus
        self._freq: Counter = Counter()
        self._total_tokens: int = 0
        if corpus_path is not None:
            self._freq = self._build_freq_table(corpus_path)

    def _build_freq_table(self, corpus_path: str | Path) -> Counter:
        """Count token frequencies in the corpus."""
        from .train import load_corpus
        sentences = load_corpus(str(corpus_path), max_sentences=None)
        freq: Counter = Counter()
        for sent in sentences:
            for tok in sent:
                if len(tok) >= self.min_token_len:
                    freq[tok] += 1
        self._total_tokens = sum(freq.values())
        return freq

    def _token_alpha(self, token: str) -> float:
        """Compute alpha based on IDF-like weighting.

        alpha = 1.0 / (1.0 + ln(freq))

        Palavras raras (conteúdo) → ln(freq) pequeno → alpha alto (até ~0.7).
        Palavras frequentes (função) → ln(freq) grande → alpha baixo (até ~0.1).
        Tokens OOV → alpha = 0.0
        """
        if token not in self._freq:
            return 0.0

        count = self._freq[token]
        if count <= 0:
            return 0.0
        alpha = 1.0 / (1.0 + np.log(float(count)))
        return float(np.clip(alpha, 0.0, 1.0))

    def project(self, m_pr: np.ndarray, tokens: List[str]) -> ContentLensPacket:
        """Apply content-aware phase lens with per-token auto-calibrated alphas.

        Each token contributes exactly one phase lens rotation,
        with alpha proportional to its corpus frequency percentile.
        Tokens are processed in ascending frequency order (rare → common)
        for stable accumulation.
        """
        m_pr = np.asarray(m_pr, dtype=np.float32)

        state = m_pr.copy()
        token_alphas: List[tuple[str, float, int]] = []

        # Compute alpha for each token
        for tok in tokens:
            if len(tok) < self.min_token_len:
                continue
            idx = self.word2idx.get(tok)
            if idx is None:
                continue
            alpha = self._token_alpha(tok)
            if alpha <= 0.0:
                continue
            token_alphas.append((tok, alpha, idx))

        # Sort by alpha descending (content words first, function words last)
        token_alphas.sort(key=lambda x: -x[1])

        applied_alphas: List[float] = []
        for tok, alpha, idx in token_alphas:
            tok_vec = self.vectors[idx].astype(np.float32)
            tok_vec = normalize(tok_vec)
            state = phase_lens(state, tok_vec, alpha=alpha)
            applied_alphas.append(alpha)

        m_intent = normalize(state)
        similarity_shift = float(np.linalg.norm(m_intent - normalize(m_pr)))

        return ContentLensPacket(
            m_intent=m_intent.astype(np.float32),
            n_tokens_used=len(token_alphas),
            alphas_used=applied_alphas,
            similarity_shift=similarity_shift,
            min_alpha=min(applied_alphas) if applied_alphas else 0.0,
            max_alpha=max(applied_alphas) if applied_alphas else 0.0,
        )
