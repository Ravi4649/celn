"""
Intent Distiller with Auto-Calibrated CAPL
==========================================

Phase-rotates M_pr toward each prompt token with alpha proportional
to that token's frequency percentile in the corpus.

  Token frequente → alpha alto → forte rotação de fase
  Token raro → alpha baixo → rotação sutil
  Tokens OOV → alpha = 0 (sem rotação)

Zero thresholds: alpha = freq_percentile / 100 — puro dado empírico.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import time
import json
from pathlib import Path

from .core import normalize
from .content_lens import ContentAwareLens, ContentLensPacket


@dataclass
class IntentPacket:
    """Result of auto-calibrated CAPL distillation."""
    m_intent: np.ndarray              # Phase-rotated intent vector
    n_tokens_used: int                # Tokens projetados (alpha > 0)
    alphas_used: List[float]          # Alphas aplicados (um por token)
    similarity_shift: float           # L2 delta M_pr → m_intent
    min_alpha: float                  # Menor alpha aplicado
    max_alpha: float                  # Maior alpha aplicado
    confidences: Dict[str, float]     # Trust metrics


class IntentDistiller:
    """Distill M_pr into semantically-tilted m_intent via auto-alpha CAPL.

    Parameters
    ----------
    vectors_path: path to word vectors (npz)
    corpus_path: path to corpus for token frequency calibration
    """

    def __init__(
        self,
        vectors_path: str | None = None,
        corpus_path: str | None = None,
        sample_sentences: int = 200,
        seed: int = 42,
    ) -> None:
        if vectors_path is None:
            raise ValueError("vectors_path is required for IntentDistiller")

        self.vectors_path = vectors_path

        self.lens = ContentAwareLens(
            vectors_path=vectors_path,
            corpus_path=corpus_path,
        )

    def distill(self, m_pr: np.ndarray, tokens: list[str] | None = None) -> IntentPacket:
        """Distill M_pr through auto-calibrated CAPL.

        Each token applies phase_lens once, with alpha = freq_percentile / 100.
        Tokens are processed in ascending frequency order.
        """
        m_pr = np.asarray(m_pr, dtype=np.float32)
        if m_pr.ndim != 1:
            raise ValueError("m_pr must be 1-D")

        tokens = tokens or []

        pkt: ContentLensPacket = self.lens.project(m_pr, tokens)

        confidences = {
            "similarity_shift": pkt.similarity_shift,
            "n_tokens_used": float(pkt.n_tokens_used),
            "min_alpha": pkt.min_alpha,
            "max_alpha": pkt.max_alpha,
        }

        # Log auto-alpha CAPL
        try:
            log_dir = Path('/tmp/opencode')
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / 'intent_distiller_logs.jsonl'
            entry = {
                "timestamp": time.time(),
                "n_tokens": len(tokens),
                "n_tokens_used": pkt.n_tokens_used,
                "alphas_used": pkt.alphas_used,
                "min_alpha": pkt.min_alpha,
                "max_alpha": pkt.max_alpha,
                "similarity_shift": pkt.similarity_shift,
            }
            with log_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception:
            pass

        return IntentPacket(
            m_intent=pkt.m_intent.astype(np.float32),
            n_tokens_used=pkt.n_tokens_used,
            alphas_used=pkt.alphas_used,
            similarity_shift=pkt.similarity_shift,
            min_alpha=pkt.min_alpha,
            max_alpha=pkt.max_alpha,
            confidences=confidences,
        )
