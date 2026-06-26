"""
PairGraph — Grafo de transições canônicas para lookahead trajectory scoring
============================================================================

Armazena top-K seguidores por palavra do corpus. Cada aresta A→B é uma
transição atestada pelo corpus. A GPVE consulta o PairGraph para projetar
múltiplos passos à frente antes de escolher um candidato.

lookahead_coherence(start_idx, vectors, M_intent, depth, width):
  Projeta N passos à frente pelo grafo greedy. Cada passo escolhe o
  seguidor com maior ressonância de espectro de magnitude com M_intent
  (a mesma operação do canal ch_mag). O score final é a média dos scores
  de ressonância em cada passo.

  Por que magnitude spectrum resonance em vez de dot product?
  - Em 10k-D, dot product entre dois vetores é inerentemente pequeno
  - O espectro de magnitude captura similaridade estrutural, não angular
  - Já é usado pelo canal ch_mag com resultado comprovado

Princípios: sem backprop, sem transformer, sem similaridade em 10k.
A única operação durante geração é projective_resonance + FFT.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from .core import normalize, projective_resonance as M


class PairGraph:
    """Transition graph for lookahead trajectory scoring.

    Stores top-K followers per word. The graph is built offline from
    corpus transitions. During generation, lookahead_coherence() scores
    how well a candidate's future trajectory aligns with M_intent.

    This is attention without backprop: the transition graph is the
    canonical attention matrix, and walking it projects the generator
    forward in time to evaluate global coherence.
    """

    def __init__(self, cache_path: str | Path = "data/pair_graph.npz"):
        data = np.load(cache_path, allow_pickle=True)

        # follower_map: word_idx → array of follower indices
        self.follower_map: dict[int, np.ndarray] = {}
        sources = data["sources"].astype(np.int32)
        followers = data["followers"].astype(np.int32)
        for i in range(len(sources)):
            self.follower_map[int(sources[i])] = followers[i]

    def has(self, word_idx: int) -> bool:
        """True if this word has known followers."""
        return word_idx in self.follower_map

    def get_followers(self, word_idx: int, top_k: int | None = None) -> list[int]:
        """Return follower indices for a source word (skipping -1)."""
        arr = self.follower_map.get(word_idx)
        if arr is None or len(arr) == 0:
            return []
        result = [int(arr[i]) for i in range(len(arr)) if int(arr[i]) >= 0]
        if top_k is not None:
            return result[:top_k]
        return result

    def _magnitude_resonance(self, state: np.ndarray, target: np.ndarray) -> float:
        """Ressonância de espectro de magnitude (mesmo do ch_mag channel).

        compute(s, t) = dot(|FFT(s)|, |FFT(t)|) após normalizar ambos.
        Retorna em [-1, 1]. Muito mais sensível que dot product em 10k-D.
        """
        sm = np.abs(np.fft.fft(state.astype(np.float32)))
        tm = np.abs(np.fft.fft(target.astype(np.float32)))
        sn = np.linalg.norm(sm)
        tn = np.linalg.norm(tm)
        if sn < 1e-12 or tn < 1e-12:
            return 0.0
        return float(np.dot(sm / sn, tm / tn))

    def lookahead_coherence(
        self,
        start_idx: int,
        vectors: np.ndarray,
        m_intent: np.ndarray,
        depth: int = 2,
        width: int = 2,
    ) -> float:
        """Greedy lookahead: score trajectory coherence with M_intent.

        A cada passo:
          1. state = projective_resonance(state_atual, vec[palavra_atual])
          2. Busca seguidores da palavra atual no PairGraph
          3. Escolhe o seguidor Y com maior magnitude_resonance(M(state, vec[Y]), M_intent)
          4. Avança para Y, repete depth vezes

        O score final é a média dos scores de ressonância.
        Em [-1, 1] — mesma escala dos outros canais.
        """
        if start_idx not in self.follower_map:
            return 0.0

        state = vectors[start_idx].astype(np.float32).copy()
        total_score = 0.0
        current = start_idx

        for step in range(depth):
            followers = self.follower_map.get(current)
            if followers is None or len(followers) == 0:
                break

            candidates = [int(f) for f in followers if int(f) >= 0][:width]
            if not candidates:
                break

            best_score = -1.0
            best_next = -1

            for f_idx in candidates:
                f_state = M(state, vectors[f_idx].astype(np.float32),
                            gamma=1.0, bilateral=True)
                f_score = self._magnitude_resonance(f_state, m_intent)
                if f_score > best_score:
                    best_score = f_score
                    best_next = f_idx

            if best_next < 0:
                break

            state = M(state, vectors[best_next].astype(np.float32),
                      gamma=1.0, bilateral=True)
            total_score += best_score
            current = best_next

        if depth == 0:
            return 0.0
        return total_score / depth
