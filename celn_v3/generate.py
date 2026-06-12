"""
CELN v3 — Text Generation via Projective Resonance
===================================================
Dual approach:
  1. ENCODING: Projective Resonance M(state, word) composes words into a
     bound state vector (non-commutative, self-attentive, order-preserving).
     Used for reasoning tasks (analogy, deduction) and final sequence encoding.

  2. SCORING: Context Window Similarity selects the next word by comparing
     candidate vectors to a weighted centroid of recent context words.
     This avoids the "binding feedback loop" where the bound state is
     spectrally dominated by the most recent word.

Key insight: The same M(x,y) operation can't directly score — because
after binding, the state has high spectral overlap with the last word,
creating a repetition loop. Instead, M encodes structure while a
separate (but related) scoring mechanism handles generation.

This dual design preserves M(x,y) as the unifying operation for
composition while solving the fluency problem pragmatically.
"""

import numpy as np
from typing import Optional

from .core import (
    D, normalize,
    projective_resonance,
    encode_sequence, encode_sequence_plain,
    similarity,
)
from .train import precompute_spectra


# ---------------------------------------------------------------------------
# Context Window Scoring (the generation mechanism)
# ---------------------------------------------------------------------------

class ContextWindow:
    """Sliding window of recent words with exponential decay weighting.

    The context centroid provides a "semantic field" that guides
    generation toward thematically coherent words, without the
    binding feedback loop that causes repetition.
    """

    def __init__(self, max_window: int = 5, decay: float = 0.7):
        self.max_window = max_window
        self.decay = decay
        self.words: list[np.ndarray] = []  # most recent last

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


def context_window_scores(window: ContextWindow,
                          word_vectors: np.ndarray,
                          recent_word_indices: set[int],
                          inhibition_strength: float = 0.01) -> np.ndarray:
    """Score all candidates by similarity to context window.

    Args:
        window: ContextWindow with recent words
        word_vectors: All word vectors, shape (vocab_size, D)
        recent_word_indices: Indices of words to inhibit (recently used)
        inhibition_strength: How strongly to penalize recent words
                            (relative to max score; auto-calibrated)

    Returns:
        Array of scores, shape (vocab_size,)
    """
    centroid = window.centroid()

    # Cosine similarity of each word to context centroid
    # Vectorized: (vocab, D) @ (D,) → (vocab,)
    scores = word_vectors @ centroid  # vectors are already normalized

    # Inhibit recently used words to prevent loops
    if recent_word_indices:
        max_score = scores.max()
        penalty = inhibition_strength * max_score
        for idx in recent_word_indices:
            scores[idx] = min(scores[idx], penalty)

    # Ensure non-negative
    scores = np.maximum(scores, 0.0)

    return scores


# ---------------------------------------------------------------------------
# PMI-Boosted Scoring (sequential probability from corpus statistics)
# ---------------------------------------------------------------------------

def pmi_boosted_scores(base_scores: np.ndarray,
                       last_word_idx: int,
                       ppmi_matrix: np.ndarray,
                       boost_weight: float = 0.3) -> np.ndarray:
    """Boost scores for words that have high PMI with the last word.

    This adds N-gram-like sequential knowledge to the semantic scoring.
    Words that frequently follow the last word in the corpus get boosted.

    Args:
        base_scores: Base scores from context window (shape vocab_size,)
        last_word_idx: Index of the most recent word
        ppmi_matrix: PPMI matrix (vocab_size, vocab_size)
        boost_weight: How much to weight sequential vs semantic fit

    Returns:
        Boosted scores
    """
    if last_word_idx < 0 or last_word_idx >= ppmi_matrix.shape[0]:
        return base_scores

    # PMI transition scores from last word
    transition_scores = ppmi_matrix[last_word_idx]  # (vocab_size,)

    # Normalize transition scores to same scale as base_scores
    t_max = transition_scores.max()
    if t_max > 1e-12:
        transition_scores = transition_scores / t_max

    b_max = base_scores.max()
    if b_max > 1e-12:
        base_norm = base_scores / b_max
    else:
        base_norm = base_scores

    # Weighted combination
    combined = (1 - boost_weight) * base_norm + boost_weight * transition_scores

    return combined


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(window: ContextWindow,
             word_vectors: np.ndarray,
             idx2word: dict[int, str],
             word2idx: dict[str, int],
             ppmi_matrix: np.ndarray,
             max_len: int = 15,
             temperature: float = 0.8,
             gamma: float = 1.0,
             inhibition_window: int = 5,
             boost_weight: float = 0.3,
             seed: int = None) -> tuple[list[str], np.ndarray]:
    """Generate a word sequence using context window scoring.

    This is the PRIMARY generation method. It combines:
      1. Semantic coherence via context window similarity
      2. Sequential probability via PMI boosting
      3. Repetition prevention via recent-word inhibition

    The Projective Resonance M(x,y) is used to ENCODE the final
    generated sequence into a bound state for downstream reasoning.

    Args:
        window: ContextWindow pre-seeded with prefix words
        word_vectors: All word vectors, shape (vocab_size, D)
        idx2word, word2idx: Index-word mappings
        ppmi_matrix: PPMI matrix for transition boosting
        max_len: Maximum words to generate
        temperature: Sampling temperature
        gamma: φ amplification (used for final encoding, not scoring)
        inhibition_window: How many recent words to inhibit
        boost_weight: PMI boost strength (0 = pure semantics, 1 = pure transition)
        seed: Random seed

    Returns:
        (generated_words, final_bound_state)
    """
    rng = np.random.RandomState(seed)
    vocab_size = len(word_vectors)
    generated = []
    recent_indices = []

    for _ in range(max_len):
        # Inhibit recently used words
        inhibited = set(recent_indices[-inhibition_window:]
                       if recent_indices else [])

        # Base scores: semantic fit to context
        scores = context_window_scores(window, word_vectors, inhibited)

        # PMI boost: sequential fit
        last_idx = recent_indices[-1] if recent_indices else -1
        scores = pmi_boosted_scores(scores, last_idx, ppmi_matrix, boost_weight)

        # Temperature scaling (auto-calibrated by score distribution)
        score_std = np.std(scores)
        effective_temp = temperature * max(score_std, 1e-6)

        scores_centered = scores - np.max(scores)
        exp_scores = np.exp(scores_centered / effective_temp)
        probs = exp_scores / exp_scores.sum()

        # Sample next word
        idx = rng.choice(vocab_size, p=probs)
        next_word = idx2word[idx]
        generated.append(next_word)

        # Update context window
        window.add(word_vectors[idx])
        recent_indices.append(idx)

    # Encode the FULL sequence (prefix + generated) via Projective Resonance
    # This is the bound state used for reasoning, NOT for scoring
    all_words = window.words.copy()
    if all_words:
        bound_state = encode_sequence(all_words, gamma=gamma)
    else:
        bound_state = np.zeros(D)

    return generated, bound_state


def generate_baseline(window: ContextWindow,
                      word_vectors: np.ndarray,
                      idx2word: dict[int, str],
                      word2idx: dict[str, int],
                      max_len: int = 15,
                      temperature: float = 0.8,
                      seed: int = None) -> tuple[list[str], np.ndarray]:
    """Generate using plain encoding (no PMI boost, no φ amplification).

    This is the baseline for comparison.
    """
    rng = np.random.RandomState(seed)
    vocab_size = len(word_vectors)
    generated = []
    recent_indices = []

    for _ in range(max_len):
        inhibited = set(recent_indices[-5:] if recent_indices else [])

        scores = context_window_scores(window, word_vectors, inhibited)
        # No PMI boost for baseline

        score_std = np.std(scores)
        effective_temp = temperature * max(score_std, 1e-6)

        scores_centered = scores - np.max(scores)
        exp_scores = np.exp(scores_centered / effective_temp)
        probs = exp_scores / exp_scores.sum()

        idx = rng.choice(vocab_size, p=probs)
        next_word = idx2word[idx]
        generated.append(next_word)

        window.add(word_vectors[idx])
        recent_indices.append(idx)

    # Encode with plain convolution (baseline)
    all_words = window.words.copy()
    if all_words:
        bound_state = encode_sequence_plain(all_words)
    else:
        bound_state = np.zeros(D)

    return generated, bound_state


def generate_from_prefix(prefix: str,
                         word_vectors: np.ndarray,
                         word2idx: dict[str, int],
                         idx2word: dict[int, str],
                         ppmi_matrix: np.ndarray,
                         max_len: int = 15,
                         temperature: float = 0.8,
                         gamma: float = 1.0,
                         boost_weight: float = 0.3,
                         seed: int = None,
                         use_projective: bool = True) -> tuple[list[str], list[str]]:
    """Generate continuation from a text prefix.

    Args:
        prefix: Text string (e.g., "o cobre é um")
        use_projective: If True, use PMI-boosted scoring; else baseline

    Returns:
        (prefix_tokens, generated_tokens)
    """
    from .train import tokenize

    prefix_tokens = tokenize(prefix)
    prefix_indices = [word2idx[w] for w in prefix_tokens if w in word2idx]

    if not prefix_indices:
        rng = np.random.RandomState(seed)
        idx = rng.randint(0, len(word_vectors))
        prefix_indices = [idx]
        prefix_tokens = [idx2word[idx]]

    # Seed context window with prefix words
    window = ContextWindow(max_window=8, decay=0.7)
    for idx in prefix_indices:
        window.add(word_vectors[idx])

    if use_projective:
        generated, state = generate(
            window, word_vectors, idx2word, word2idx, ppmi_matrix,
            max_len, temperature, gamma, boost_weight=boost_weight, seed=seed
        )
    else:
        generated, state = generate_baseline(
            window, word_vectors, idx2word, word2idx,
            max_len, temperature, seed=seed
        )

    return prefix_tokens, generated


# ---------------------------------------------------------------------------
# Beam Search
# ---------------------------------------------------------------------------

def beam_search(prefix_tokens: list[str],
                word_vectors: np.ndarray,
                word2idx: dict[str, int],
                idx2word: dict[int, str],
                ppmi_matrix: np.ndarray,
                beam_width: int = 5,
                max_len: int = 10,
                temperature: float = 0.5) -> list[list[str]]:
    """Beam search generation for higher-quality output.

    Maintains top-k candidate sequences, scored by combined
    semantic coherence and sequential probability.
    """
    vocab_size = len(word_vectors)
    rng = np.random.RandomState(42)

    # Initialize windows with prefix
    class BeamState:
        def __init__(self, window, word_indices, log_prob):
            self.window = window
            self.word_indices = list(word_indices)
            self.log_prob = log_prob

    initial_window = ContextWindow(max_window=8, decay=0.7)
    prefix_indices = []
    for w in prefix_tokens:
        if w in word2idx:
            idx = word2idx[w]
            initial_window.add(word_vectors[idx])
            prefix_indices.append(idx)

    beams = [BeamState(initial_window, prefix_indices, 0.0)]

    for _ in range(max_len):
        candidates = []

        for beam in beams:
            inhibited = set(beam.word_indices[-5:] if beam.word_indices else [])

            scores = context_window_scores(beam.window, word_vectors, inhibited)

            last_idx = beam.word_indices[-1] if beam.word_indices else -1
            scores = pmi_boosted_scores(scores, last_idx, ppmi_matrix, 0.3)

            scores_centered = scores - np.max(scores)
            exp_scores = np.exp(scores_centered / temperature)
            probs = exp_scores / exp_scores.sum()

            top_k = np.argsort(probs)[-beam_width:]

            for idx in top_k:
                new_window = ContextWindow(max_window=8, decay=0.7)
                new_window.words = beam.window.words.copy()
                new_window.add(word_vectors[idx])

                new_indices = beam.word_indices + [idx]
                new_log_prob = beam.log_prob + np.log(probs[idx] + 1e-12)
                candidates.append(BeamState(new_window, new_indices, new_log_prob))

        candidates.sort(key=lambda x: x.log_prob, reverse=True)
        beams = candidates[:beam_width]

    results = []
    for beam in beams:
        gen_indices = beam.word_indices[len(prefix_indices):]
        results.append([idx2word[i] for i in gen_indices])

    return results
