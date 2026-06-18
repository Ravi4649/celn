"""
Multi-Pass Invertible Phase Lens
================================

Aplica phase_lens em 3 etapas com alphas crescentes usando role bins distintos,
per turbando a distribuição de registradores do PortAdapter de forma progressiva.

Princípios:
- Sem backprop, transformers, similaridade em 10k, listas fixas, templates, thresholds mágicos
- Três alphas: 0.2 (leve), 0.5 (médio), 0.8 (agressivo)
- Role bin único por etapa
- Uso exclusivo de operações algébricas (phase_lens + binding)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
from pathlib import Path

from .core import D, phase_lens, bind, normalize, phi
from .port_adapter import load_word_vectors


@dataclass
class InvertiblePacket:
    m_intent: np.ndarray  # Vetor final após multi-pass phase lens
    roles_applied: List[int]  # Número de tokens por etapa
    alphas_used: List[float]  # Alphas aplicados
    projection_norms: List[float]  # Energia média por etapa


class InvertibleLens:
    """
    Multi-pass invertible phase lens: 3 etapas com alphas crescentes.
    """

    def __init__(
        self,
        vectors_path: str | Path,
        min_token_len: int = 2,
        seed: int = 42,
        use_progressive_alphas: bool = True,
    ) -> None:
        """Inicializa com vetores e role bins para 3 etapas."""
        vectors, word2idx = load_word_vectors(vectors_path)
        self.vectors = vectors.astype(np.float32)
        self.word2idx = word2idx
        self.min_token_len = min_token_len
        self.rng = np.random.RandomState(seed)
        
        # Três alphas progressivos
        self.alphas = [0.2, 0.5, 0.8] if use_progressive_alphas else [0.2, 0.2, 0.2]
        
        # Três role bins distintos
        self.role_bins = [
            make_unitary_vector(D, np.random.RandomState(seed + i))
            for i in range(3)
        ]

    def project(self, m_pr: np.ndarray, tokens: List[str]) -> InvertiblePacket:
        """
        Aplica 3 etapas de phase lens com alphas crescentes.
        """
        m_pr = np.asarray(m_pr, dtype=np.float32)
        if m_pr.ndim != 1 or m_pr.shape[0] != D:
            raise ValueError(f"m_pr must have shape ({D},)")

        roles_applied = [0, 0, 0]
        projection_norms = [0.0, 0.0, 0.0]
        
        for step in range(3):
            m_current = m_pr.copy()
            alpha = self.alphas[step]
            role_bin = self.role_bins[step]
            
            for tok in tokens:
                if len(tok) < self.min_token_len or tok not in self.word2idx:
                    continue
                
                # Vetor do token → projeção → binding com role bin
                token_vec = self.vectors[self.word2idx[tok]]
                proj = phi(normalize(token_vec.astype(np.float32)))
                augmented_proj = bind(role_bin, proj)
                
                # Phase lens com alpha da etapa
                m_current = phase_lens(m_current, augmented_proj, alpha=alpha)
                
                projection_norms[step] += np.linalg.norm(proj)
                roles_applied[step] += 1
            
            # Normaliza vetor para próxima etapa
            m_pr = normalize(m_current)
            projection_norms[step] = projection_norms[step] / max(1, roles_applied[step])

        return InvertiblePacket(
            m_intent=normalize(m_pr).astype(np.float32),
            roles_applied=roles_applied,
            alphas_used=self.alphas,
            projection_norms=projection_norms,
        )


def make_unitary_vector(dim: int = D, rng: np.random.RandomState | None = None) -> np.ndarray:
    """Create a real unitary vector with unit-magnitude spectrum."""
    rng = rng or np.random
    spectrum = np.zeros(dim, dtype=np.complex128)
    spectrum[0] = rng.choice([-1.0, 1.0])  # DC component
    half = dim // 2
    for k in range(1, half + 1):
        if dim % 2 == 0 and k == half:
            spectrum[k] = rng.choice([-1.0, 1.0])  # Nyquist frequency
        else:
            phase = rng.uniform(0.0, 2 * np.pi)
            value = np.cos(phase) + 1j * np.sin(phase)
            spectrum[k] = value
            spectrum[-k] = np.conj(value)
    return np.fft.ifft(spectrum).real.astype(np.float32)