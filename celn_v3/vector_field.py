"""
CELN v3 — Vector Field Generator
=================================
Language as an ASYMMETRIC vector field. Each word generates a local
field that points toward its natural continuations — no frequencies,
no templates, no classification.

Core insight: fluency emerges from FLOW. "o gato" is natural because
"o" generates a field pointing toward the noun region. "gato o" is
unnatural because "gato"'s field points elsewhere (toward verbs,
adjectives — what follows "gato" in the corpus).

Mathematical foundation:
  - Each word w has a position v_w in R^D
  - Each word w generates a FLOW VECTOR f_w = centroid({v_next | w→next}) - v_w
    This points FROM w TOWARD where the language naturally flows after w.
  - Generation: given current word w, compute target = v_w + α · f_w
    Select the word most similar to the target.
  - When f_w is weak (rare word), blend with semantic coherence.

Learning (purely algebraic):
  For each transition w1 → w2 in the corpus:
    accumulate[w1].append(v_w2)
  After scan: f_w = normalize(mean(accumulate[w])) - v_w
  This is O(corpus_size · D) — one pass, no iteration.

Generation:
  Navigate the field by following flow vectors, blending with
  semantic context when the field is weak.
"""

import numpy as np
from typing import Optional

from .core import D, normalize, similarity, projective_resonance


# ---------------------------------------------------------------------------
# Vector Field Learning
# ---------------------------------------------------------------------------

class VectorField:
    """Asymmetric local vector field learned from corpus transitions.

    For each word w, stores:
      - flow: direction from w toward its typical continuations
      - confidence: how many examples support this flow (for auto-calibration)
    """

    def __init__(
        self,
        vectors: np.ndarray,
        w2i: dict[str, int],
        flow_strength: float = 1.0,
        min_confidence: int = 5,
    ):
        """
        Args:
            vectors: Word vectors, shape (V, D), normalized.
            w2i: Word-to-index mapping.
            flow_strength: Base multiplier for flow vector (α).
            min_confidence: Minimum examples before field is trusted.
        """
        self.vectors = vectors.astype(np.float32)
        self.V, self.dim = vectors.shape
        self.w2i = w2i
        self.i2w = {i: w for w, i in w2i.items()}
        self.flow_strength = flow_strength
        self.min_confidence = min_confidence

        # Flow vectors: shape (V, D)
        self.flow = np.zeros((self.V, self.dim), dtype=np.float32)
        # Confidence: how many transitions support each flow
        self.confidence = np.zeros(self.V, dtype=np.int32)

    def learn(self, sentences: list[list[str]]):
        """Learn flow vectors from corpus transitions.

        For each transition w1 → w2, accumulate w2's vector as
        a follower of w1. After all transitions, compute:
            flow[w1] = centroid({v_w2 | w1 → w2}) - v_w1

        This points FROM w1 TOWARD its typical followers.
        """
        # Accumulate follower vectors per word
        accum = [np.zeros(self.dim, dtype=np.float32) for _ in range(self.V)]
        counts = np.zeros(self.V, dtype=np.int32)

        for tokens in sentences:
            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i + 1]
                if w1 not in self.w2i or w2 not in self.w2i:
                    continue
                idx1, idx2 = self.w2i[w1], self.w2i[w2]
                accum[idx1] += self.vectors[idx2]
                counts[idx1] += 1

        # Compute flow vectors
        for i in range(self.V):
            if counts[i] >= self.min_confidence:
                follower_centroid = normalize(accum[i] / counts[i])
                # Flow = direction FROM current word TOWARD followers
                self.flow[i] = follower_centroid - self.vectors[i]
                self.confidence[i] = counts[i]
            else:
                self.flow[i] = np.zeros(self.dim, dtype=np.float32)
                self.confidence[i] = 0

    def get_flow(self, word_idx: int) -> np.ndarray:
        """Get the flow vector for a word.

        Returns:
            Flow vector (D,). Zero if word has insufficient confidence.
        """
        return self.flow[word_idx]

    def get_target(self, word_idx: int) -> np.ndarray:
        """Compute the target point: v_w + α · f_w.

        This is where the field says we should go next.
        Auto-calibrated: weak fields get less influence.

        Returns:
            Target vector (D,), normalized.
        """
        conf = self.confidence[word_idx]
        if conf < self.min_confidence:
            return self.vectors[word_idx]

        # Auto-calibrate flow strength by confidence
        # More examples → stronger flow → more directional generation
        effective_strength = self.flow_strength * min(1.0, conf / 20.0)

        target = self.vectors[word_idx] + effective_strength * self.flow[word_idx]
        return normalize(target)

    @property
    def stats(self) -> dict:
        """Return field statistics."""
        strong = int((self.confidence >= self.min_confidence).sum())
        weak = self.V - strong
        return {
            'vocab_size': self.V,
            'words_with_flow': strong,
            'words_without_flow': weak,
            'mean_confidence': float(self.confidence[self.confidence > 0].mean())
            if (self.confidence > 0).any() else 0.0,
        }


# ---------------------------------------------------------------------------
# Vector Field Generator
# ---------------------------------------------------------------------------

class VectorFieldGenerator:
    """Generate text by navigating a learned vector field.

    At each step:
      1. The CURRENT word generates a flow vector
      2. The flow points toward where language naturally goes next
      3. Select the word closest to the target point
      4. Blend with semantic context when the field is weak
      5. Move to the selected word, repeat
    """

    def __init__(
        self,
        field: VectorField,
        flow_weight: float = 0.5,
        context_weight: float = 0.2,
        bigram_prob: np.ndarray | None = None,
        bigram_weight: float = 0.3,
        window_size: int = 5,
        window_decay: float = 0.7,
    ):
        """
        Args:
            field: Trained VectorField.
            flow_weight: Max weight for field navigation (auto-calibrated by confidence).
            context_weight: Weight for context window semantic coherence.
            bigram_prob: Optional directional bigram probability matrix.
                        Used as fallback when field confidence is low.
            bigram_weight: Weight for bigram fallback channel.
            window_size: Context window for semantic scoring.
            window_decay: Recency decay in context window.
        """
        self.field = field
        self.vectors = field.vectors
        self.w2i = field.w2i
        self.i2w = field.i2w
        self.vocab_size = field.V
        self.flow_weight = flow_weight
        self.context_weight = context_weight
        self.bigram_prob = bigram_prob
        self.bigram_weight = bigram_weight
        self.window_size = window_size
        self.window_decay = window_decay

    def _context_centroid(self, recent: list[np.ndarray]) -> np.ndarray:
        """Weighted centroid of recent word vectors."""
        if not recent:
            return np.zeros(self.field.dim)
        weights = np.array([
            self.window_decay ** (len(recent) - 1 - i)
            for i in range(len(recent))
        ])
        weights = weights / weights.sum()
        result = np.zeros(self.field.dim)
        for v, w in zip(recent, weights):
            result += w * v
        return normalize(result)

    def generate(
        self,
        prefix_words: list[str],
        max_len: int = 15,
        temperature: float = 0.8,
        inhibition_window: int = 5,
        seed: Optional[int] = None,
    ) -> list[str]:
        """Generate text by navigating the vector field.

        Args:
            prefix_words: Starting words.
            max_len: Maximum words to generate.
            temperature: Sampling temperature.
            inhibition_window: Recent word inhibition.
            seed: Random seed.

        Returns:
            List of generated word strings.
        """
        rng = np.random.RandomState(seed)

        # Convert prefix to indices
        prefix_indices = [self.w2i[w] for w in prefix_words if w in self.w2i]
        if not prefix_indices:
            idx = rng.randint(0, self.vocab_size)
            prefix_indices = [idx]

        recent_vectors: list[np.ndarray] = [
            self.vectors[idx] for idx in prefix_indices
        ]
        generated: list[str] = []
        recent_indices: list[int] = list(prefix_indices)

        # State tracking via M for semantic coherence
        state = self._context_centroid(recent_vectors)

        for step in range(max_len):
            excluded = set(recent_indices[-inhibition_window:]
                          if recent_indices else [])

            last_idx = recent_indices[-1]
            last_vec = self.vectors[last_idx]

            # ── 1. Flow score: field-based directional navigation ──
            target = self.field.get_target(last_idx)
            flow_scores = self.vectors @ target.astype(np.float32)
            for idx in excluded:
                flow_scores[idx] = -1.0

            # Auto-calibrate: field is reliable when confidence is high
            conf = self.field.confidence[last_idx]
            field_confidence = min(1.0, conf / 20.0)  # saturates at 20+ examples
            effective_flow_weight = self.flow_weight * field_confidence

            # ── 2. Bigram score: fallback when field is weak ──
            if self.bigram_prob is not None and field_confidence < 0.5:
                bigram_scores = self.bigram_prob[last_idx].copy()
                for idx in excluded:
                    bigram_scores[idx] = 0.0
                effective_bigram_weight = self.bigram_weight * (1.0 - field_confidence)
            else:
                bigram_scores = np.zeros(self.vocab_size, dtype=np.float32)
                effective_bigram_weight = 0.0

            # ── 3. Context score: semantic coherence ──
            ctx_centroid = self._context_centroid(recent_vectors)
            context_scores = self.vectors @ ctx_centroid.astype(np.float32)
            for idx in excluded:
                context_scores[idx] = -1.0

            # ── 4. Normalize and blend ──
            flow_max = np.abs(flow_scores).max()
            ctx_max = np.abs(context_scores).max()
            bigram_max = np.abs(bigram_scores).max()
            if flow_max > 1e-12:
                flow_scores = flow_scores / flow_max
            if ctx_max > 1e-12:
                context_scores = context_scores / ctx_max
            if bigram_max > 1e-12:
                bigram_scores = bigram_scores / bigram_max

            scores = (
                effective_flow_weight * flow_scores +
                effective_bigram_weight * bigram_scores +
                self.context_weight * context_scores
            )

            # ── 4. Temperature sampling ──
            score_std = np.std(scores)
            effective_temp = temperature * max(score_std, 1e-6)
            scores_centered = scores - scores.max()
            exp_scores = np.exp(scores_centered / effective_temp)
            probs = exp_scores / exp_scores.sum()

            idx = rng.choice(self.vocab_size, p=probs)
            next_word = self.i2w[idx]
            generated.append(next_word)

            # Update state
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

def generate_from_field(
    prefix: str,
    field: VectorField,
    max_len: int = 15,
    temperature: float = 0.8,
    flow_weight: float = 0.6,
    context_weight: float = 0.3,
    seed: Optional[int] = None,
) -> tuple[list[str], list[str]]:
    """Generate text from prefix using vector field navigation.

    Args:
        prefix: Text string.
        field: Trained VectorField.
        max_len, temperature, seed: Generation parameters.
        flow_weight: Weight for field navigation.
        context_weight: Weight for semantic coherence.

    Returns:
        (prefix_tokens, generated_tokens)
    """
    from .train import tokenize

    prefix_tokens = tokenize(prefix, min_len=1)
    prefix_known = [w for w in prefix_tokens if w in field.w2i]
    if not prefix_known:
        rng = np.random.RandomState(seed)
        idx = rng.randint(0, field.V)
        prefix_known = [field.i2w[idx]]

    gen = VectorFieldGenerator(
        field, flow_weight=flow_weight, context_weight=context_weight,
    )
    generated = gen.generate(prefix_known, max_len, temperature, seed=seed)
    return prefix_known, generated
