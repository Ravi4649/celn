"""
CELN v3 — Encoder
==================
Compose word sequences into a single state vector using
Projective Resonance M(x, y) recursively.

  state = M(w1, M(w2, M(w3, ...)))

M(x, y) = IFFT(FFT(x) ⊙ FFT(y) ⊙ φ_weight(FFT(y)))

Properties:
  - Non-commutative: order matters (via bilateral φ)
  - Self-attentive: dominant frequencies amplified during binding
  - O(d log d) via FFT — runs on CPU at microsecond scale

Also provides a plain convolution encoder for VSA 2.0 baseline comparison.
"""

import numpy as np
from numpy.fft import fft, ifft
from typing import Optional

from .core import (
    D, normalize, batch_normalize,
    projective_resonance,
    similarity, make_random_vector,
)


class Encoder:
    """Compose word vectors into sentence vectors via recursive M.

    The encoding scan:
        state_0 = word_0
        state_i = M(state_{i-1}, word_i, gamma, bilateral)

    This preserves word order because M is non-commutative.
    """

    def __init__(self,
                 word_vectors: np.ndarray,
                 gamma: float = 1.0,
                 bilateral: bool = True):
        """
        Args:
            word_vectors: (vocab_size, D) array of normalized word vectors
            gamma: φ amplification exponent (0 = identity, 2 = aggressive)
            bilateral: if True, use bilateral φ for stronger non-commutativity
        """
        self.vectors = word_vectors
        self.vocab_size, self.dim = word_vectors.shape
        self.gamma = gamma
        self.bilateral = bilateral

    def encode_indices(self, indices: list[int]) -> np.ndarray:
        """Encode a sequence of word indices into a state vector.

        Args:
            indices: List of word indices in order

        Returns:
            State vector of shape (dim,) encoding the full sequence
        """
        if not indices:
            return np.zeros(self.dim)

        state = self.vectors[indices[0]].copy()

        for idx in indices[1:]:
            state = projective_resonance(
                state, self.vectors[idx],
                gamma=self.gamma,
                bilateral=self.bilateral
            )

        return state

    def encode_words(self, words: list[str],
                     word2idx: dict[str, int]) -> np.ndarray:
        """Encode a sequence of word strings.

        Args:
            words: List of word strings
            word2idx: Mapping from word to index

        Returns:
            State vector, or zero vector if no valid words
        """
        indices = [word2idx[w] for w in words if w in word2idx]
        return self.encode_indices(indices)

    def encode_all(self, sequences: list[list[int]]) -> np.ndarray:
        """Batch encode multiple sequences.

        Returns:
            Array of shape (n_sequences, dim)
        """
        states = np.zeros((len(sequences), self.dim))
        for i, seq in enumerate(sequences):
            states[i] = self.encode_indices(seq)
        return states


class BaselineEncoder:
    """Plain circular convolution encoder (VSA 2.0 baseline).

    Uses standard bind(x, y) = IFFT(FFT(x) * FFT(y))
    without φ amplification — no frequency-domain attention.
    """

    def __init__(self, word_vectors: np.ndarray):
        self.vectors = word_vectors
        self.vocab_size, self.dim = word_vectors.shape

    def encode_indices(self, indices: list[int]) -> np.ndarray:
        if not indices:
            return np.zeros(self.dim)

        state = self.vectors[indices[0]].copy()

        for idx in indices[1:]:
            X = fft(state)
            Y = fft(self.vectors[idx])
            state = normalize(ifft(X * Y).real)

        return state

    def encode_words(self, words: list[str],
                     word2idx: dict[str, int]) -> np.ndarray:
        indices = [word2idx[w] for w in words if w in word2idx]
        return self.encode_indices(indices)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sequence_similarity(a_state: np.ndarray, b_state: np.ndarray) -> float:
    """Cosine similarity between two encoded sequences."""
    return similarity(a_state, b_state)


def encode_with_progressive_anchors(words: list[int],
                                    word_vectors: np.ndarray,
                                    gamma: float = 1.0,
                                    bilateral: bool = True,
                                    anchor_decay: float = 0.9) -> list[np.ndarray]:
    """Encode with progressive anchor computation.

    Returns the sequence of intermediate states AND anchors.
    Useful for understanding how the encoding evolves.

    Returns:
        (states, anchors) — lists of state vectors at each step
    """
    if not words:
        return [], []

    states = [word_vectors[words[0]].copy()]
    anchors = [states[0].copy()]

    for idx in words[1:]:
        new_state = projective_resonance(
            states[-1], word_vectors[idx],
            gamma=gamma, bilateral=bilateral
        )
        new_anchor = normalize(
            anchor_decay * anchors[-1] + (1 - anchor_decay) * new_state
        )
        states.append(new_state)
        anchors.append(new_anchor)

    return states, anchors
