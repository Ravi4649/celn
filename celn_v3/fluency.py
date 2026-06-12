"""
CELN v3 — Fluent Generator
===========================
Directional bigram generator with auto-calibrating structure/semantic blend.

Key insight: Current generator favors content words because semantic
scoring (cosine similarity to topic centroid) naturally ranks distinctive
content words above diffuse function words.

Solution: Add a DIRECTIONAL bigram probability channel that captures
asymmetric transition patterns (articles precede nouns, prepositions
follow verbs) from the corpus. The blend weight is auto-calibrated
per step based on how semantic vs structural the current context is.

Principles preserved:
  - ZERO backprop, ZERO templates, ZERO fixed thresholds
  - ZERO grammatical classification of words
  - Auto-calibrating via distribution of actual similarities
  - 100% vector algebra
"""

import numpy as np
from typing import Optional, Tuple

from .core import D, normalize, similarity, auto_threshold


# ---------------------------------------------------------------------------
# Directional bigram model
# ---------------------------------------------------------------------------

def build_directional_bigrams(
    sentences: list[list[str]],
    w2i: dict[str, int],
    vocab_size: int,
    smoothing: float = 0.01
) -> np.ndarray:
    """Build directional bigram probability matrix.

    P(w2 | w1) = count(w1→w2) / sum(count(w1→*))

    Unlike the PPMI matrix (symmetric, captures association), this
    captures ASYMMETRIC transition patterns:
      "o" → nouns (high), nouns → "de" (high), "de" → "o" (low)

    Args:
        sentences: Tokenized sentences (with 1-char words included).
        w2i: Word-to-index mapping.
        vocab_size: Number of words in vocabulary.
        smoothing: Additive smoothing factor (avoids zero probabilities).

    Returns:
        Array of shape (vocab_size, vocab_size), where row i sums to ~1
        (the probability distribution over next words given word i).
    """
    # Count directional bigrams
    bigram_counts = np.zeros((vocab_size, vocab_size), dtype=np.float32)

    for tokens in sentences:
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            if w1 in w2i and w2 in w2i:
                bigram_counts[w2i[w1], w2i[w2]] += 1.0

    # Additive smoothing
    bigram_counts += smoothing

    # Normalize each row to sum to 1 (directional probability)
    row_sums = bigram_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-12] = 1.0
    bigram_prob = bigram_counts / row_sums

    return bigram_prob


# ---------------------------------------------------------------------------
# Context Window (same as generate.py but extracted for independence)
# ---------------------------------------------------------------------------

class ContextWindow:
    """Sliding window of recent words with exponential decay weighting."""

    def __init__(self, max_window: int = 5, decay: float = 0.7):
        self.max_window = max_window
        self.decay = decay
        self.words: list[np.ndarray] = []

    def add(self, word_vec: np.ndarray):
        self.words.append(word_vec)
        if len(self.words) > self.max_window:
            self.words.pop(0)

    def centroid(self) -> np.ndarray:
        """Weighted centroid with exponential decay (recent = higher weight)."""
        if not self.words:
            return np.zeros(D)
        weights = np.array([self.decay ** (len(self.words) - 1 - i)
                           for i in range(len(self.words))])
        weights = weights / weights.sum()
        centroid = np.zeros(D)
        for w_vec, w in zip(self.words, weights):
            centroid += w * w_vec
        return normalize(centroid)

    def reset(self):
        self.words = []


# ---------------------------------------------------------------------------
# Directional Generator
# ---------------------------------------------------------------------------

class DirectionalGenerator:
    """Generate fluent text using auto-calibrated structure/semantic blend.

    Three scoring channels blended per step:
      1. STRUCTURE: Directional bigram probability P(w | last_word)
         → captures natural word order, pulls in articles/prepositions
      2. SEMANTIC: Cosine similarity to context window centroid
         → keeps topic coherence
      3. SDM KNOWLEDGE (optional): Cosine similarity to SDM-retrieved vector
         → injects factual knowledge from corpus memory

    The structure weight is AUTO-CALIBRATED:
      - When the last word is highly similar to the topic centroid
        → it's a content word → increase structure weight (need grammar)
      - When the last word has low similarity to the topic centroid
        → it's a function word → decrease structure weight (need content)

    No word classification needed — purely based on vector similarity.
    """

    def __init__(
        self,
        word_vectors: np.ndarray,
        bigram_prob: np.ndarray,
        w2i: dict[str, int],
        i2w: dict[int, str],
        window_size: int = 5,
        window_decay: float = 0.7,
        base_structure_weight: float = 0.35,
        sdm: Optional['DenseSDM'] = None,  # noqa: F821
        sdm_weight: float = 0.1,
    ):
        """
        Args:
            word_vectors: (vocab_size, D) normalized word vectors.
            bigram_prob: (vocab_size, vocab_size) directional bigram
                         probabilities. Row i = P(next | word i).
            w2i, i2w: Word-index mappings.
            window_size: Context window size for semantic scoring.
            window_decay: Recency weight decay in context window.
            base_structure_weight: Base weight for structural (bigram)
                                   channel. Auto-calibrated per step.
            sdm: Optional DenseSDM for knowledge-grounded generation.
            sdm_weight: Weight for SDM knowledge channel.
        """
        self.vectors = word_vectors.astype(np.float32)
        self.vocab_size, self.dim = word_vectors.shape
        self.bigram_prob = bigram_prob.astype(np.float32)
        self.w2i = w2i
        self.i2w = i2w
        self.window_size = window_size
        self.window_decay = window_decay
        self.base_structure_weight = base_structure_weight
        self.sdm = sdm
        self.sdm_weight = sdm_weight

    def generate(
        self,
        prefix_words: list[str],
        max_len: int = 15,
        temperature: float = 0.8,
        inhibition_window: int = 5,
        seed: Optional[int] = None
    ) -> list[str]:
        """Generate a fluent continuation from prefix words.

        Args:
            prefix_words: Starting words (strings, must be in vocabulary).
            max_len: Maximum number of words to generate.
            temperature: Sampling temperature (lower = more deterministic).
            inhibition_window: How many recent words to inhibit (anti-repetition).
            seed: Random seed for reproducibility.

        Returns:
            List of generated word strings.
        """
        rng = np.random.RandomState(seed)

        # Convert prefix to indices
        prefix_indices = [self.w2i[w] for w in prefix_words if w in self.w2i]
        if not prefix_indices:
            idx = rng.randint(0, self.vocab_size)
            prefix_indices = [idx]
            prefix_words = [self.i2w[idx]]

        # Initialize context window with prefix
        window = ContextWindow(self.window_size, self.window_decay)
        for idx in prefix_indices:
            window.add(self.vectors[idx])

        generated: list[str] = []
        recent_indices: list[int] = list(prefix_indices)

        for step in range(max_len):
            # Inhibit recently used words
            excluded = set(recent_indices[-inhibition_window:]
                          if recent_indices else [])

            last_idx = recent_indices[-1]
            last_vec = self.vectors[last_idx]

            # ── 1. Semantic score: similarity to context window centroid ──
            ctx_centroid = window.centroid()
            semantic_scores = self.vectors @ ctx_centroid
            for idx in excluded:
                semantic_scores[idx] = -1.0

            # ── 2. Structural score: directional bigram probability ──
            raw_probs = self.bigram_prob[last_idx].copy()
            raw_probs[list(excluded)] = 0.0
            structure_scores = raw_probs

            # ── 3. Auto-calibrate structure weight ──
            last_topic_sim = float(np.dot(last_vec, ctx_centroid))
            structure_weight = self.base_structure_weight * (
                0.3 + 0.7 * max(0.0, last_topic_sim)
            )
            structure_weight = min(0.6, max(0.15, structure_weight))

            # ── 4. SDM knowledge (optional) ──
            if self.sdm is not None:
                sdm_result = self.sdm.read(ctx_centroid)
                sdm_scores = self.vectors @ sdm_result
                sdm_max = np.abs(sdm_scores).max()
                if sdm_max > 1e-12:
                    sdm_scores = sdm_scores / sdm_max
            else:
                sdm_scores = np.zeros(self.vocab_size)

            # ── 5. Normalize channels ──
            sem_max = np.abs(semantic_scores).max()
            struct_max = np.abs(structure_scores).max()
            if sem_max > 1e-12:
                semantic_scores = semantic_scores / sem_max
            if struct_max > 1e-12:
                structure_scores = structure_scores / struct_max

            # ── 6. Blend ──
            semantic_weight = 1.0 - structure_weight - self.sdm_weight
            semantic_weight = max(0.2, semantic_weight)

            scores = (
                semantic_weight * semantic_scores +
                structure_weight * structure_scores +
                self.sdm_weight * sdm_scores
            )

            # ── 7. Temperature sampling ──
            score_std = np.std(scores)
            effective_temp = temperature * max(score_std, 1e-6)
            scores_centered = scores - np.max(scores)
            exp_scores = np.exp(scores_centered / effective_temp)
            probs = exp_scores / exp_scores.sum()

            idx = rng.choice(self.vocab_size, p=probs)
            next_word = self.i2w[idx]
            generated.append(next_word)

            # Update state
            window.add(self.vectors[idx])
            recent_indices.append(idx)

        return generated


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def generate_fluent(
    prefix: str,
    vectors: np.ndarray,
    bigram_prob: np.ndarray,
    w2i: dict[str, int],
    i2w: dict[int, str],
    max_len: int = 15,
    temperature: float = 0.8,
    seed: Optional[int] = None,
    sdm: Optional['DenseSDM'] = None,  # noqa: F821
) -> Tuple[list[str], list[str]]:
    """Generate fluent text from a text prefix.

    Args:
        prefix: Text string (e.g., "o cobre é um").
        vectors, bigram_prob, w2i, i2w: Model components.
        max_len, temperature, seed: Generation parameters.
        sdm: Optional SDM for knowledge grounding.

    Returns:
        (prefix_tokens, generated_tokens)
    """
    from .train import tokenize

    prefix_tokens = tokenize(prefix, min_len=1)
    # Filter to known words
    prefix_known = [w for w in prefix_tokens if w in w2i]
    if not prefix_known:
        rng = np.random.RandomState(seed)
        idx = rng.randint(0, len(vectors))
        prefix_known = [i2w[idx]]

    gen = DirectionalGenerator(
        vectors, bigram_prob, w2i, i2w,
        sdm=sdm
    )
    generated = gen.generate(
        prefix_known, max_len, temperature, seed=seed
    )

    return prefix_known, generated
