"""
CELN v3 — Decoder
==================
Extract candidate words from a composed state vector.

Two extraction methods:
  1. Similarity-based: score(w) = cosine_similarity(w, state)
  2. Unbinding-based: recover = unbind(state, context_hint)

The decoder answers: "Given this encoded sentence state, which words
are likely to be part of it, and in what order?"
"""

import numpy as np
from numpy.fft import fft, ifft
from typing import Optional

from .core import (
    D, normalize, batch_normalize,
    bind, unbind,
    projective_resonance,
    resonance_score, resonance_scores_batch,
    similarity,
)


class Decoder:
    """Extract words from composed state vectors.

    Supports two modes:
      - similarity: direct cosine similarity between state and word vectors
      - resonance: magnitude spectrum overlap (via FFT magnitude correlation)
    """

    def __init__(self, word_vectors: np.ndarray):
        self.vectors = word_vectors
        self.vocab_size, self.dim = word_vectors.shape
        # Precompute FFT magnitudes for fast resonance scoring
        self._spectra = np.array([
            np.abs(fft(v)) for v in word_vectors
        ])

    def score_by_similarity(self,
                            state: np.ndarray,
                            exclude_indices: Optional[set[int]] = None
                            ) -> np.ndarray:
        """Score all candidates by cosine similarity to state.

        O(vocab_size * dim) — fast in high dimensions.
        """
        scores = self.vectors @ state  # (vocab,) dot products
        if exclude_indices:
            for idx in exclude_indices:
                scores[idx] = -1.0
        return scores

    def score_by_resonance(self,
                           state: np.ndarray,
                           exclude_indices: Optional[set[int]] = None
                           ) -> np.ndarray:
        """Score all candidates by magnitude spectrum overlap.

        This is the "frequency-domain attention" score:
        measures how well each word's dominant frequencies
        align with the state's dominant frequencies.
        """
        S = np.abs(fft(state))
        S_norm = np.linalg.norm(S) + 1e-12
        scores = self._spectra @ S / (
            S_norm * np.linalg.norm(self._spectra, axis=1) + 1e-12
        )
        if exclude_indices:
            for idx in exclude_indices:
                scores[idx] = -1.0
        return scores

    def score_combined(self,
                       state: np.ndarray,
                       anchor: np.ndarray,
                       anchor_weight: float = 0.4,
                       exclude_indices: Optional[set[int]] = None
                       ) -> np.ndarray:
        """Score by resonance to state AND similarity to anchor.

        This is the primary scoring method for generation:
        - Resonance to state: local coherence (word fits the immediate context)
        - Similarity to anchor: global coherence (word stays on topic)

        Args:
            state: Current encoded state
            anchor: Topic anchor vector (slow-moving average of states)
            anchor_weight: Balance between local and global coherence
                          0 = pure resonance (local only)
                          1 = pure anchor (global only)
        """
        local = self.score_by_resonance(state, exclude_indices)
        global_scores = self.score_by_similarity(anchor, exclude_indices)

        # Normalize to comparable scales
        local = local / (np.abs(local).max() + 1e-12)
        global_scores = global_scores / (np.abs(global_scores).max() + 1e-12)

        return (1 - anchor_weight) * local + anchor_weight * global_scores

    def unbind_context(self,
                       state: np.ndarray,
                       context_hint: np.ndarray) -> np.ndarray:
        """Try to extract a word by unbinding the context from the state.

        If state ≈ bind(context_hint, target_word),
        then unbind(state, context_hint) ≈ target_word.

        The recovered vector can then be matched against the vocabulary
        to find the most likely word.
        """
        recovered = unbind(state, context_hint)
        return normalize(recovered)

    def top_candidates(self,
                       state: np.ndarray,
                       k: int = 10,
                       method: str = 'resonance',
                       exclude_indices: Optional[set[int]] = None,
                       idx2word: Optional[dict[int, str]] = None
                       ) -> list[tuple[str, float]]:
        """Return top-k candidate words for a state.

        Args:
            state: Encoded state vector
            k: Number of candidates
            method: 'similarity' or 'resonance'
            exclude_indices: Indices to exclude
            idx2word: Optional index-to-word mapping

        Returns:
            List of (word, score) tuples
        """
        if method == 'similarity':
            scores = self.score_by_similarity(state, exclude_indices)
        else:
            scores = self.score_by_resonance(state, exclude_indices)

        top_indices = np.argsort(scores)[-k:][::-1]

        if idx2word:
            return [(idx2word[i], float(scores[i])) for i in top_indices]
        return [(str(i), float(scores[i])) for i in top_indices]

    def decode_sequence_iterative(self,
                                  state: np.ndarray,
                                  encoder,  # Encoder instance
                                  max_len: int = 10,
                                  method: str = 'resonance'
                                  ) -> list[int]:
        """Iteratively decode a sequence from a state.

        At each step, finds the word that best explains the state,
        removes its contribution via unbinding, and repeats.

        This is approximate — each unbinding step adds noise.
        """
        from .encoder import Encoder as Enc

        indices = []
        remaining = state.copy()

        for _ in range(max_len):
            scores = self.score_by_resonance(remaining)
            best_idx = int(np.argmax(scores))

            if scores[best_idx] < 0.1:  # weak signal — stop
                break

            indices.append(best_idx)

            # Remove this word's contribution
            remaining = self.unbind_context(
                remaining, self.vectors[best_idx]
            )

        return indices


class BaselineDecoder:
    """Plain cosine similarity decoder (VSA 2.0 baseline).

    No resonance scoring, no unbinding — just direct
    cosine similarity between state and word vectors.
    """

    def __init__(self, word_vectors: np.ndarray):
        self.vectors = word_vectors
        self.vocab_size, self.dim = word_vectors.shape

    def score_by_similarity(self,
                            state: np.ndarray,
                            exclude_indices: Optional[set[int]] = None
                            ) -> np.ndarray:
        scores = self.vectors @ state
        if exclude_indices:
            for idx in exclude_indices:
                scores[idx] = -1.0
        return scores

    def top_candidates(self,
                       state: np.ndarray,
                       k: int = 10,
                       exclude_indices: Optional[set[int]] = None,
                       idx2word: Optional[dict[int, str]] = None
                       ) -> list[tuple[str, float]]:
        scores = self.score_by_similarity(state, exclude_indices)
        top_indices = np.argsort(scores)[-k:][::-1]
        if idx2word:
            return [(idx2word[i], float(scores[i])) for i in top_indices]
        return [(str(i), float(scores[i])) for i in top_indices]
