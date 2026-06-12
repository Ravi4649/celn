"""
CELN v3 — Operator Algebra
===========================
Function words as DIRECTIONAL BIAS VECTORS,
content words as VECTORS (points on the unit hypersphere).

Core insight: function words don't carry meaning — they DIRECT the state
toward regions where the next content word lives. "o" doesn't mean
anything; it pushes the state toward masculine nouns.

Mathematical foundation (simplified):
  - Content words: vectors v ∈ R^D on the unit hypersphere
  - Function words (operators): bias vectors b ∈ R^D
    b_op = centroid(words that follow op) - centroid(all words)
  - Operator application: state' = normalize(state + alpha * b_op)
    Cost: O(D)

Learning (purely algebraic, no backprop, no SVD):
  For each operator, collect all words that follow it in the corpus.
  The bias is the difference between the centroid of its followers
  and the global centroid — "where does this operator point?"

Generation:
  Alternates between applying operators (structure) and selecting
  content words (meaning). The decision is auto-calibrated by the
  marginal benefit of each operator.
"""

import numpy as np
from typing import Optional, Tuple, Set

from .core import D, normalize, similarity, projective_resonance


# ---------------------------------------------------------------------------
# Operator Identification
# ---------------------------------------------------------------------------

def identify_operators(
    sentences: list[list[str]],
    w2i: dict[str, int],
    vectors: np.ndarray,
    freq_percentile: float = 95.0,
    min_freq: int = 5
) -> list[str]:
    """Auto-identify which words should be operators.

    Uses TWO self-calibrating criteria:
      1. High frequency: words in the top (100-freq_percentile)% by count.
         Function words appear everywhere; content words are sparse.
      2. High follower entropy: words that precede MANY different words
         are structural (articles, prepositions). Content words have
         few, predictable followers.

    Args:
        sentences: Tokenized sentences (min_len=1).
        w2i: Word-to-index mapping.
        vectors: Word vectors (for content similarity check).
        freq_percentile: Percentile threshold for frequency.
        min_freq: Minimum absolute frequency.

    Returns:
        List of operator words.
    """
    from collections import Counter
    import math

    # Count frequencies
    word_freq = Counter()
    for s in sentences:
        word_freq.update(s)

    # Count follower diversity per word
    follower_sets: dict[str, set] = {}
    for s in sentences:
        for i in range(len(s) - 1):
            w1, w2 = s[i], s[i + 1]
            if w1 not in follower_sets:
                follower_sets[w1] = set()
            follower_sets[w1].add(w2)

    # Compute scores for candidate words
    candidates = []
    for w in w2i:
        freq = word_freq.get(w, 0)
        if freq < min_freq:
            continue

        # Follower entropy: higher = more diverse = more structural
        followers = follower_sets.get(w, set())
        if len(followers) < 2:
            continue

        # Entropy: sum over follower types
        follower_counts = Counter()
        for s in sentences:
            for i in range(len(s) - 1):
                if s[i] == w:
                    follower_counts[s[i + 1]] += 1

        total = follower_counts.total()
        entropy = 0.0
        for count in follower_counts.values():
            p = count / total
            entropy -= p * math.log(p + 1e-12)

        # Composite score: frequency * entropy
        # Both high → structural operator
        candidates.append((w, freq, entropy, freq * entropy))

    if not candidates:
        return []

    # Auto-calibrated threshold: top percentile of composite score
    scores = np.array([c[3] for c in candidates])
    threshold = float(np.percentile(scores, freq_percentile))

    operators = sorted(
        [c for c in candidates if c[3] >= threshold],
        key=lambda c: -c[3]
    )

    return [op[0] for op in operators]


# ---------------------------------------------------------------------------
# Operator Learning (Hebbian outer product + SVD)
# ---------------------------------------------------------------------------

class OperatorMemory:
    """Stores and applies operator directional biases.

    Each operator is a single bias vector:
        b_op = centroid(words that follow op) - centroid(all words)

    This captures the DIRECTION the operator pushes the state toward.
    "o" → masculine nouns, "de" → complements, "é" → attributes.

    Application: state' = normalize(state + alpha * b_op)
    Cost: O(D). Memory: n_operators × D × 4 bytes.
    """

    def __init__(
        self,
        operator_words: list[str],
        w2i: dict[str, int],
        dim: int = D,
        alpha: float = 0.3
    ):
        self.operator_words = list(operator_words)
        self.op_to_idx = {w: i for i, w in enumerate(operator_words)}
        self.n_operators = len(operator_words)
        self.dim = dim
        self.alpha = alpha

        # Bias vectors: (n_operators, D)
        self.biases = np.zeros((self.n_operators, dim), dtype=np.float32)

        # For collecting follower words per operator before computing bias
        self._followers: list[list[np.ndarray]] = [
            [] for _ in range(self.n_operators)
        ]

    def learn_from_corpus(
        self,
        sentences: list[list[str]],
        w2i: dict[str, int],
        vectors: np.ndarray,
        learning_rate: float = 0.01  # unused, kept for API compat
    ):
        """Collect CONTENT-WORD followers for each operator.

        Only collects followers that are NOT themselves operators.
        This isolates the operator's distinctive direction toward
        content words, filtering out function-word noise.

        The bias is computed in finalize() as:
            b_op = centroid(content_followers_of_op)
        """
        for tokens in sentences:
            for i in range(len(tokens) - 1):
                op_word = tokens[i]
                next_word = tokens[i + 1]

                if op_word not in self.op_to_idx:
                    continue
                if next_word not in w2i:
                    continue

                # Only collect CONTENT-word followers
                # (words that are NOT operators themselves)
                if next_word in self.op_to_idx:
                    continue

                idx = self.op_to_idx[op_word]
                self._followers[idx].append(
                    vectors[w2i[next_word]].astype(np.float32)
                )

    def finalize(self):
        """Compute bias vectors from content-word followers.

        b_op = centroid(content_words_that_follow_op)

        These centroids point toward the semantic region where
        the operator's typical CONTENT followers live.
        """
        for idx in range(self.n_operators):
            followers = self._followers[idx]
            if len(followers) < 3:
                continue

            followers_centroid = normalize(
                np.mean(followers, axis=0)
            )
            self.biases[idx] = followers_centroid

        # Free collected data
        del self._followers
        self._followers = []

    def apply(self, op_word: str, state: np.ndarray) -> np.ndarray:
        """Apply operator bias to state.

        state' = normalize(state + alpha * bias_op)

        The bias pushes the state toward the operator's typical
        follower region while preserving the original direction.

        Args:
            op_word: The operator word to apply.
            state: Current state vector (D,).

        Returns:
            Transformed state vector (D,), normalized.
        """
        if op_word not in self.op_to_idx:
            return state
        idx = self.op_to_idx[op_word]
        bias = self.biases[idx]

        if np.linalg.norm(bias) < 1e-12:
            return state

        result = state + self.alpha * bias
        return normalize(result)

    def apply_batch(self, state: np.ndarray) -> np.ndarray:
        """Apply all operators to state, return transformed states.

        Args:
            state: Current state vector (D,).

        Returns:
            Array of shape (n_operators, D) with transformed states.
        """
        # state + alpha * bias for all operators
        results = state[None, :] + self.alpha * self.biases
        # Normalize each row
        norms = np.linalg.norm(results, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return results / norms

    @property
    def stats(self) -> dict:
        """Return operator memory statistics."""
        total_mb = self.biases.nbytes / (1024 * 1024)
        n_followers = [len(f) for f in self._followers] if hasattr(self, '_followers') and self._followers else []
        return {
            'n_operators': self.n_operators,
            'dim': self.dim,
            'alpha': self.alpha,
            'memory_mb': round(total_mb, 2),
            'samples_per_op': n_followers,
        }


# ---------------------------------------------------------------------------
# Operator Generator
# ---------------------------------------------------------------------------

class OperatorGenerator:
    """Generate text alternating between operators and content words.

    At each step:
      1. Try all operators, compute benefit = how much each operator
         improves access to content words.
      2. If best benefit exceeds auto-calibrated threshold:
         apply operator and output it.
      3. Otherwise: select content word by temperature sampling,
         update state via projective_resonance.

    The alternation emerges naturally from the benefit calculation,
    without any grammatical classification of individual words.
    """

    def __init__(
        self,
        content_vectors: np.ndarray,
        operator_memory: OperatorMemory,
        w2i: dict[str, int],
        i2w: dict[int, str],
        operator_words: list[str],
        window_size: int = 8,
        window_decay: float = 0.7,
        benefit_percentile: float = 70.0,
    ):
        """
        Args:
            content_vectors: Word vectors for ALL words (content + operator).
            operator_memory: Trained OperatorMemory with low-rank matrices.
            w2i, i2w: Word-index mappings.
            operator_words: List of words treated as operators.
            window_size: Context window for state tracking.
            window_decay: Recency decay in context window.
            benefit_percentile: Percentile for auto-calibrating the
                               benefit threshold.
        """
        self.vectors = content_vectors.astype(np.float32)
        self.vocab_size, self.dim = content_vectors.shape
        self.op_memory = operator_memory
        self.w2i = w2i
        self.i2w = i2w
        self.operator_words = set(operator_words)
        self.operator_indices = [w2i[w] for w in operator_words if w in w2i]
        self.window_size = window_size
        self.window_decay = window_decay
        self.benefit_percentile = benefit_percentile

    def _context_centroid(self, recent_vectors: list[np.ndarray]) -> np.ndarray:
        """Weighted centroid of recent context vectors."""
        if not recent_vectors:
            return np.zeros(self.dim)
        weights = np.array([
            self.window_decay ** (len(recent_vectors) - 1 - i)
            for i in range(len(recent_vectors))
        ])
        weights = weights / weights.sum()
        centroid = np.zeros(self.dim)
        for w_vec, weight in zip(recent_vectors, weights):
            centroid += weight * w_vec
        return normalize(centroid)

    def generate(
        self,
        prefix_words: list[str],
        max_len: int = 20,
        temperature: float = 0.8,
        inhibition_window: int = 5,
        seed: Optional[int] = None
    ) -> list[str]:
        """Generate text alternating operators and content words.

        Args:
            prefix_words: Starting words (strings).
            max_len: Maximum words to generate.
            temperature: Sampling temperature.
            inhibition_window: Recent word inhibition (anti-repetition).
            seed: Random seed.

        Returns:
            List of generated word strings.
        """
        rng = np.random.RandomState(seed)

        # Initialize state from prefix
        prefix_indices = [self.w2i[w] for w in prefix_words if w in self.w2i]
        if not prefix_indices:
            idx = rng.randint(0, self.vocab_size)
            prefix_indices = [idx]

        recent_vectors: list[np.ndarray] = [
            self.vectors[idx] for idx in prefix_indices
        ]
        state = self._context_centroid(recent_vectors)

        generated: list[str] = []
        recent_indices: list[int] = list(prefix_indices)

        for step in range(max_len):
            excluded = set(recent_indices[-inhibition_window:]
                          if recent_indices else [])

            # ── Phase A: Optionally apply ONE operator before content ──
            # This aims the state before selecting the next content word.
            # At most ONE operator per content word — prevents operator loops.
            if len(self.operator_indices) > 0:
                benefits = np.zeros(len(self.operator_indices), dtype=np.float32)
                sims_from_state = self.vectors @ state.astype(np.float32)
                for idx in excluded:
                    sims_from_state[idx] = -1.0
                best_from_state = sims_from_state.max()

                transformed = self.op_memory.apply_batch(state)
                for j, (op_idx, t_state) in enumerate(
                    zip(self.operator_indices, transformed)
                ):
                    sims = self.vectors @ t_state.astype(np.float32)
                    for idx in excluded:
                        sims[idx] = -1.0
                    benefits[j] = sims.max() - best_from_state

                benefit_threshold = float(np.percentile(
                    benefits, self.benefit_percentile
                ))
                best_j = int(np.argmax(benefits))

                if benefits[best_j] > benefit_threshold and benefits[best_j] > 0.005:
                    op_word = self.i2w[self.operator_indices[best_j]]
                    generated.append(op_word)
                    state = self.op_memory.apply(op_word, state)
                    op_idx = self.operator_indices[best_j]
                    recent_vectors.append(self.vectors[op_idx])
                    recent_indices.append(op_idx)
                    if len(recent_vectors) > self.window_size:
                        recent_vectors.pop(0)

            # ── Phase B: ALWAYS select a content word after ──
            sims_from_state = self.vectors @ state.astype(np.float32)
            for idx in excluded:
                sims_from_state[idx] = -1.0

            score_std = np.std(sims_from_state)
            effective_temp = temperature * max(score_std, 1e-6)
            scores_centered = sims_from_state - sims_from_state.max()
            exp_scores = np.exp(scores_centered / effective_temp)
            probs = exp_scores / exp_scores.sum()

            idx = rng.choice(self.vocab_size, p=probs)
            next_word = self.i2w[idx]
            generated.append(next_word)

            state = projective_resonance(
                state, self.vectors[idx], gamma=1.0, bilateral=True
            )
            recent_vectors.append(self.vectors[idx])
            recent_indices.append(idx)
            if len(recent_vectors) > self.window_size:
                recent_vectors.pop(0)

        return generated


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def generate_with_operators(
    prefix: str,
    vectors: np.ndarray,
    op_memory: OperatorMemory,
    w2i: dict[str, int],
    i2w: dict[int, str],
    operator_words: list[str],
    max_len: int = 20,
    temperature: float = 0.8,
    seed: Optional[int] = None,
) -> tuple[list[str], list[str]]:
    """Generate text from a prefix using operator algebra.

    Args:
        prefix: Text string.
        vectors, op_memory, w2i, i2w, operator_words: Model components.
        max_len, temperature, seed: Generation parameters.

    Returns:
        (prefix_tokens, generated_tokens)
    """
    from .train import tokenize

    prefix_tokens = tokenize(prefix, min_len=1)
    prefix_known = [w for w in prefix_tokens if w in w2i]
    if not prefix_known:
        rng = np.random.RandomState(seed)
        idx = rng.randint(0, len(vectors))
        prefix_known = [i2w[idx]]

    gen = OperatorGenerator(
        vectors, op_memory, w2i, i2w, operator_words,
    )
    generated = gen.generate(
        prefix_known, max_len, temperature, seed=seed
    )
    return prefix_known, generated
