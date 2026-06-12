"""
CELN v3 — Generator
====================
Generate text using Projective Resonance M(x,y) for state encoding
and a Context Window + Anchor for next-word scoring.

KEY INSIGHT: After binding via M, the state vector is NOT directly
similar to its constituent words. Circular convolution compresses
information into a new region of space. Scoring by similarity to
the bound state returns near-random results.

SOLUTION: Use M(x,y) ONLY for encoding the sequence into a state
(for reasoning, analogy, etc.). Use a CONTEXT WINDOW of recent word
vectors for scoring next-word candidates. The anchor is the
exponential moving average of context window centroids.

This dual design:
  - M(x,y) encodes structure (non-commutative, ordered, attention-weighted)
  - Context Window + Anchor scores candidates (semantically coherent)

Also provides VSA 2.0 baseline generator for direct comparison.
"""

import numpy as np
from typing import Optional

from typing import TYPE_CHECKING

from .encoder import Encoder, BaselineEncoder
from .decoder import Decoder, BaselineDecoder
from .core import (
    D, normalize,
    projective_resonance,
    similarity,
)

if TYPE_CHECKING:
    from .memory import DenseSDM


class ContextWindow:
    """Sliding window of recent word vectors with decay weighting.

    The window provides a "semantic field" for candidate scoring
    that directly reflects the meaning of recent words, unlike
    the bound state which compresses via convolution.
    """

    def __init__(self, max_size: int = 8, decay: float = 0.7):
        self.max_size = max_size
        self.decay = decay
        self.words: list[np.ndarray] = []

    def add(self, vec: np.ndarray):
        self.words.append(vec)
        if len(self.words) > self.max_size:
            self.words.pop(0)

    def centroid(self) -> np.ndarray:
        """Weighted centroid — recent words have higher weight."""
        if not self.words:
            return np.zeros_like(self.words[0]) if self.words else np.zeros(1)
        weights = np.array([
            self.decay ** (len(self.words) - 1 - i)
            for i in range(len(self.words))
        ])
        weights = weights / weights.sum()
        result = np.zeros_like(self.words[0])
        for w, weight in zip(self.words, weights):
            result += weight * w
        return normalize(result)

    def reset(self):
        self.words = []


class ProjectiveGenerator:
    """Generate text: M(x,y) for encoding, Context Window for scoring.

    The anchor tracks the exponential moving average of context window
    centroids, providing topic stability while allowing progression.
    """

    def __init__(self,
                 word_vectors: np.ndarray,
                 gamma: float = 1.0,
                 bilateral: bool = True,
                 window_size: int = 8,
                 window_decay: float = 0.7,
                 anchor_decay: float = 0.9,
                 anchor_weight: float = 0.3,
                 sdm: 'DenseSDM | None' = None,
                 sdm_knowledge_weight: float = 0.2):
        """
        Args:
            word_vectors: (vocab_size, dim) normalized word vectors
            gamma: φ amplification exponent
            bilateral: Use bilateral φ for stronger non-commutativity
            window_size: Max words in context window
            window_decay: Recency weight decay in context window
            anchor_decay: How fast anchor drifts (0.9 = slow, 0.99 = very slow)
            anchor_weight: Balance between context and anchor scores
            sdm: Optional DenseSDM for knowledge-grounded generation.
                 If None, generation is identical to original behavior.
            sdm_knowledge_weight: Max fraction of scoring budget allocated
                 to SDM knowledge (0 = no SDM influence, 1 = all SDM).
                 The effective weight is auto-calibrated per step:
                 effective = sdm_knowledge_weight * (1 - sim(anchor, sdm.read(anchor)))
        """
        self.vectors = word_vectors
        self.vocab_size, self.dim = word_vectors.shape
        self.gamma = gamma
        self.bilateral = bilateral
        self.window_size = window_size
        self.window_decay = window_decay
        self.anchor_decay = anchor_decay
        self.anchor_weight = anchor_weight
        self.sdm = sdm
        self.sdm_knowledge_weight = sdm_knowledge_weight

    def generate(self,
                 prefix_indices: list[int],
                 max_len: int = 15,
                 temperature: float = 0.8,
                 inhibition_window: int = 5,
                 seed: int = None
                 ) -> tuple[list[int], np.ndarray]:
        """Generate word sequence from prefix.

        Returns:
            (generated_indices, final_bound_state)
            The bound state encodes the full sequence via M for reasoning.
        """
        rng = np.random.RandomState(seed)

        if not prefix_indices:
            idx = rng.randint(0, self.vocab_size)
            prefix_indices = [idx]

        # Initialize context window with prefix words
        window = ContextWindow(self.window_size, self.window_decay)
        for idx in prefix_indices:
            window.add(self.vectors[idx])

        # Initial anchor: topic centroid from prefix
        anchor = window.centroid().copy()

        generated = list(prefix_indices)
        recent_indices = list(prefix_indices)

        for step in range(max_len):
            # Exclude recently used words
            excluded = set(recent_indices[-inhibition_window:])

            # 1. Context score: similarity to window centroid
            ctx_centroid = window.centroid()
            ctx_scores = self.vectors @ ctx_centroid  # cosine sim
            for idx in excluded:
                ctx_scores[idx] = -1.0

            # 2. Anchor score: similarity to topic anchor
            anchor_scores = self.vectors @ anchor
            for idx in excluded:
                anchor_scores[idx] = -1.0

            # 3. SDM knowledge score: corpus-grounded candidate boosting
            if self.sdm is not None:
                sdm_result = self.sdm.read(anchor)
                # Self-calibrating weight: SDM influence proportional to
                # how much new information it provides vs the anchor.
                # sim(anchor, sdm_result) ≈ 1.0 → SDM has nothing new → weight ≈ 0
                # sim(anchor, sdm_result) ≈ 0.6 → SDM enriches with context → weight ≈ 0.4 * max
                sdm_novelty = 1.0 - similarity(anchor, sdm_result)
                effective_sdm_weight = self.sdm_knowledge_weight * sdm_novelty

                sdm_scores = self.vectors @ sdm_result
                for idx in excluded:
                    sdm_scores[idx] = -1.0
            else:
                effective_sdm_weight = 0.0
                sdm_scores = np.zeros(self.vocab_size)

            # Auto-calibrate: normalize each channel by its distribution
            ctx_max = np.abs(ctx_scores).max()
            anc_max = np.abs(anchor_scores).max()
            if ctx_max > 1e-12:
                ctx_scores = ctx_scores / ctx_max
            if anc_max > 1e-12:
                anchor_scores = anchor_scores / anc_max
            if effective_sdm_weight > 1e-12:
                sdm_max = np.abs(sdm_scores).max()
                if sdm_max > 1e-12:
                    sdm_scores = sdm_scores / sdm_max

            # Combined score — three independent channels
            # Context always gets at least 5% budget to maintain coherence
            ctx_frac = max(0.05, 1.0 - self.anchor_weight - effective_sdm_weight)
            scores = (ctx_frac * ctx_scores +
                      self.anchor_weight * anchor_scores +
                      effective_sdm_weight * sdm_scores)

            # Temperature sampling
            score_std = np.std(scores)
            effective_temp = temperature * max(score_std, 1e-6)
            scores_centered = scores - scores.max()
            exp_scores = np.exp(scores_centered / effective_temp)
            probs = exp_scores / exp_scores.sum()

            idx = rng.choice(self.vocab_size, p=probs)
            generated.append(idx)
            recent_indices.append(idx)

            # Update context window
            window.add(self.vectors[idx])

            # Update anchor: slow drift toward current context
            new_centroid = window.centroid()
            anchor = normalize(
                self.anchor_decay * anchor +
                (1 - self.anchor_decay) * new_centroid
            )

        # Encode full sequence via M for reasoning/analysis
        bound_state = self._encode_full(generated)

        generated_indices = generated[len(prefix_indices):]
        return generated_indices, bound_state

    def _encode_full(self, indices: list[int]) -> np.ndarray:
        """Encode the full sequence using Projective Resonance M."""
        if not indices:
            return np.zeros(self.dim)
        state = self.vectors[indices[0]].copy()
        for idx in indices[1:]:
            state = projective_resonance(
                state, self.vectors[idx],
                gamma=self.gamma, bilateral=self.bilateral
            )
        return state

    def generate_from_words(self,
                            prefix_words: list[str],
                            word2idx: dict[str, int],
                            idx2word: dict[int, str],
                            max_len: int = 15,
                            temperature: float = 0.8,
                            seed: int = None
                            ) -> tuple[list[str], np.ndarray]:
        """Generate from word strings."""
        prefix_indices = [word2idx[w] for w in prefix_words if w in word2idx]
        gen_indices, state = self.generate(
            prefix_indices, max_len, temperature, seed=seed
        )
        gen_words = [idx2word[i] for i in gen_indices]
        return gen_words, state


class BaselineGenerator:
    """VSA 2.0 "Boca Universal" — plain bind, no anchor, no φ.

    Uses plain circular convolution for encoding and
    cosine similarity to bound state for scoring.
    This IS the original VSA 2.0 approach, warts and all.
    """

    def __init__(self, word_vectors: np.ndarray):
        self.vectors = word_vectors
        self.vocab_size, self.dim = word_vectors.shape

    def generate(self,
                 prefix_indices: list[int],
                 max_len: int = 15,
                 temperature: float = 0.8,
                 seed: int = None
                 ) -> tuple[list[int], np.ndarray]:
        """Generate using VSA 2.0 baseline approach."""
        from numpy.fft import fft, ifft

        rng = np.random.RandomState(seed)

        if not prefix_indices:
            idx = rng.randint(0, self.vocab_size)
            prefix_indices = [idx]

        # Plain bind encoding
        state = self.vectors[prefix_indices[0]].copy()
        for idx in prefix_indices[1:]:
            X = fft(state)
            Y = fft(self.vectors[idx])
            state = normalize(ifft(X * Y).real)

        generated = list(prefix_indices)

        for step in range(max_len):
            # VSA 2.0: cosine similarity to bound state
            scores = self.vectors @ state

            score_std = np.std(scores)
            effective_temp = temperature * max(score_std, 1e-6)
            scores_centered = scores - scores.max()
            exp_scores = np.exp(scores_centered / effective_temp)
            probs = exp_scores / exp_scores.sum()

            idx = rng.choice(self.vocab_size, p=probs)
            generated.append(idx)

            # Update via plain bind
            X = fft(state)
            Y = fft(self.vectors[idx])
            state = normalize(ifft(X * Y).real)

        generated_indices = generated[len(prefix_indices):]
        return generated_indices, state

    def generate_from_words(self,
                            prefix_words: list[str],
                            word2idx: dict[str, int],
                            idx2word: dict[int, str],
                            max_len: int = 15,
                            temperature: float = 0.8,
                            seed: int = None
                            ) -> tuple[list[str], np.ndarray]:
        prefix_indices = [word2idx[w] for w in prefix_words if w in word2idx]
        gen_indices, state = self.generate(
            prefix_indices, max_len, temperature, seed=seed
        )
        gen_words = [idx2word[i] for i in gen_indices]
        return gen_words, state
