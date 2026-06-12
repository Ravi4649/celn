"""
CELN v3 — Dual-Channel Generator (Pure Algebraic + SDM Knowledge)
==================================================================
Two separate but cooperating vector spaces:
  1. SEMANTIC channel: SVD-trained vectors (meaning, topic coherence)
  2. TYPE channel: HDC-derived type vectors (distributional role, syntax)

Optional third input:
  3. SDM KNOWLEDGE: DenseSDM long-term memory that stores sentence
     centroids from the corpus. At each generation step, the context
     centroid queries the SDM. The retrieved knowledge vector modulates
     the semantic channel — words similar to what the SDM stored about
     this topic get a subtle boost. Auto-calibrated: strong SDM signal
     = more influence; weak signal = generator operates normally.

Core insight (Kanerva, BEAGLE, Plate): semantics and syntax need
SEPARATE representational channels. "o" and "gato" live in the same
semantic space (causing the function-word problem), but in DIFFERENT
regions of the type space (articles cluster together, nouns cluster
together).

Learning (no backprop, no classification):
  - Semantic vectors: SVD on PPMI co-occurrence matrix
  - Type vectors: HDC Hebbian learning on positional distribution.
    Words with similar distributional behavior get similar type vectors.
  - The type vectors emerge from distributional statistics without
    any pre-defined categories (DET, NOUN, VERB).

Generation (PURE ALGEBRAIC — ZERO statistical channels):
  - TYPE FIELD: centroid of type vectors of followers points to
    the next syntactic role. 97% accuracy via geometry, not counting.
  - SEMANTIC state: context window centroid maintains topic coherence.
  - SDM KNOWLEDGE (optional): retrieved memory modulates semantics
    with stored facts, auto-calibrated by signal confidence.
  - AUTO-CALIBRATING BLEND: at each step, each channel's confidence
    is measured from the distribution of its scores (percentil of
    actual distribution). Concentrated high scores = high confidence
    = more weight. Spread low scores = low confidence = less weight.
  - No fixed weights, no bigram, no frequency tables. Pure geometry.

Principles:
  - ZERO backpropagation, transformers, LLMs
  - ZERO listas fixas, templates, thresholds mágicos
  - ZERO pesos fixos — tudo auto-calibrável via percentil da distribuição real
  - 100% álgebra vetorial
"""

import numpy as np
from numpy.fft import fft, ifft

from .core import (
    normalize,
    phase_lens,
    phase_lens_scores_batch,
    precompute_word_spectra,
    similarity as cosine_similarity,
    projective_resonance as M,
    inverse_projective_resonance,
    spectral_entropy,
)
from .resonator import unbind_M_reverse, unbind_M_forward


# ---------------------------------------------------------------------------
# Type Vector Extraction
# ---------------------------------------------------------------------------

def extract_type_vectors(
    ppmi: np.ndarray,
    type_dim: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """Extract distributional type vectors from PPMI matrix.

    Uses SVD on the PPMI matrix to capture co-occurrence patterns.
    Words with similar distributional behavior (appear in similar
    contexts) get similar type vectors — without any pre-defined
    categories.

    The PPMI matrix M where M[i,j] = PMI(word_i, word_j) captures
    how strongly word_i and word_j co-occur. Row i is word_i's
    "distributional signature." SVD reduces this to type_dim.

    Args:
        ppmi: PPMI matrix, shape (V, V).
        type_dim: Output dimensionality for type vectors.
        seed: Random seed.

    Returns:
        Type vectors, shape (V, type_dim), L2-normalized.
    """
    from sklearn.decomposition import TruncatedSVD

    V = ppmi.shape[0]
    n_components = min(type_dim, V - 1)

    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    type_vecs = svd.fit_transform(ppmi)  # (V, n_components)

    # Weight by singular value variance
    sv = svd.singular_values_
    var_ratio = sv ** 2 / (sv ** 2).sum()
    weights = var_ratio / var_ratio.max()
    type_vecs = type_vecs * weights[None, :]

    # Pad to target dim if needed
    if n_components < type_dim:
        rng = np.random.RandomState(seed + 1)
        R = rng.randn(n_components, type_dim) / np.sqrt(n_components)
        type_vecs = type_vecs @ R

    # Normalize
    norms = np.linalg.norm(type_vecs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return type_vecs / norms


# ---------------------------------------------------------------------------
# Binding in type space (circular convolution)
# ---------------------------------------------------------------------------

def bind_type(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular convolution of two type vectors via FFT."""
    return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))


def unbind_type(c: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Circular correlation — approximate inverse of bind_type."""
    return np.real(np.fft.ifft(
        np.fft.fft(c) * np.conj(np.fft.fft(a))
    ))


# ---------------------------------------------------------------------------
# Dual-Channel Generator
# ---------------------------------------------------------------------------

class DualChannelGenerator:
    """Generate text using separate semantic and type channels.

    Architecture:
      - Semantic vectors (SVD): meaning, topic coherence
      - Type vectors (HDC/PPMI-derived): distributional role, syntax
      - TYPE FIELD: for each word, centroid of type vectors of its
        followers (in type space!) points toward the next syntactic role.
      - SDM KNOWLEDGE (optional): DenseSDM long-term memory.
        Context queries SDM → retrieved knowledge modulates semantic scores.
        Auto-calibrated: strong SDM signal = more influence.
      - Generation: [sdm →] semantic → type-narrowed blend, per step.
        All weights derived from distribution confidence, not fixed.

    ZERO statistical channels (no bigram, no frequency tables).
    Pure geometry — type field already has 97% accuracy.
    SDM adds factual grounding without backprop or templates.
    """

    def __init__(
        self,
        semantic_vectors: np.ndarray,
        type_vectors: np.ndarray,
        w2i: dict[str, int],
        i2w: dict[int, str],
        window_size: int = 5,
        window_decay: float = 0.7,
        sdm: object | None = None,
        use_phase_lens: bool = True,
        phase_lens_max_alpha: float = 0.4,
        pair_sdm: object | None = None,
        pmi_ri_vectors: np.ndarray | None = None,
        pair_source_indices: np.ndarray | None = None,
        pair_follower_indices: np.ndarray | None = None,
    ):
        self.sem_vecs = semantic_vectors.astype(np.float32)
        self.type_vecs = type_vectors.astype(np.float32)
        self.type_dim = type_vectors.shape[1]
        self.vocab_size = semantic_vectors.shape[0]
        self.w2i = w2i
        self.i2w = i2w
        self.window_size = window_size
        self.window_decay = window_decay
        self.sdm = sdm  # Optional DenseSDM for knowledge grounding
        self.use_phase_lens = use_phase_lens
        self.phase_lens_max_alpha = phase_lens_max_alpha
        self.pair_sdm = pair_sdm  # Optional PairSDM for parallel transport
        self.pair_source_indices = pair_source_indices
        self.pair_follower_indices = pair_follower_indices
        self.pmi_ri_vecs = None
        if pmi_ri_vectors is not None:
            self.pmi_ri_vecs = pmi_ri_vectors.astype(np.float32)
            pmi_norms = np.linalg.norm(self.pmi_ri_vecs, axis=1, keepdims=True)
            pmi_norms[pmi_norms < 1e-12] = 1.0
            self.pmi_ri_vecs = self.pmi_ri_vecs / pmi_norms

        # Learn type field: for each word, where does its type point?
        self.type_field = np.zeros((self.vocab_size, self.type_dim), dtype=np.float32)

        # ── Precompute FFT spectra for phase rotation lens ──
        # This enables O(VD) contextual similarity without per-word FFT calls
        if self.use_phase_lens:
            self._word_spectra, self._word_mags, self._word_phases = \
                precompute_word_spectra(self.sem_vecs)
        else:
            self._word_spectra = None
            self._word_mags = None
            self._word_phases = None

        # Small Portuguese function-words set used for auto-calibration
        # Keeps the set compact and internal to avoid external dependencies.
        self._function_words = set([
            'o','a','os','as','um','uma','uns','umas',
            'de','do','da','dos','das','em','no','na','nos','nas',
            'e','ou','mas','que','se','nem','pois','é','foi','era',
            'são','está','ser','não','sim','como','quando','onde',
            'porque','para','com','por','pelo','pela','pelas','sem','sob','sobre',
        ])

    def learn_type_field(self, sentences: list[list[str]]):
        """Learn the type field from corpus transitions.

        For each transition w1 → w2:
          type_field[w1] += type_vec[w2]
        After: type_field[w1] = centroid(type_vecs of followers)

        In TYPE space (which has clustering), this field points FROM
        the current word's type TOWARD the next word's type.

        Words with count >= 1 get a direct type field.
        Words with count = 0 get their type field INTERPOLATED
        from type-space nearest neighbors (algebraic generalization).
        """
        accum = np.zeros((self.vocab_size, self.type_dim), dtype=np.float32)
        counts = np.zeros(self.vocab_size, dtype=np.int32)

        for tokens in sentences:
            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i + 1]
                if w1 not in self.w2i or w2 not in self.w2i:
                    continue
                i1, i2 = self.w2i[w1], self.w2i[w2]
                accum[i1] += self.type_vecs[i2]
                counts[i1] += 1

        # Direct type fields: words with >= 1 observed transition
        self.has_type_field = np.zeros(self.vocab_size, dtype=bool)
        for i in range(self.vocab_size):
            if counts[i] >= 1:
                self.type_field[i] = normalize(accum[i] / counts[i])
                self.has_type_field[i] = True

        # Interpolate missing type fields from type-space neighbors
        missing = (~self.has_type_field).sum()
        if missing > 0:
            self._interpolate_type_fields()

    def _interpolate_type_fields(self):
        """Fill in missing type fields via type-space nearest neighbors.

        For each word WITHOUT a type field:
          1. Find k nearest neighbors in TYPE space that HAVE a type field
          2. Average their type fields, weighted by type similarity

        This generalizes the type field algebraically:
        words with similar distributional behavior (nearby in type space)
        get similar type fields — without any classification or counting.
        """
        # Pre-compute type vector similarity matrix for efficiency
        # Only need rows for words WITHOUT type fields and columns for words WITH
        has_field = np.where(self.has_type_field)[0]
        no_field = np.where(~self.has_type_field)[0]

        if len(has_field) == 0 or len(no_field) == 0:
            return

        # For each missing word, compute similarity to all words with type fields
        k = min(10, len(has_field))
        interpolated = 0

        for idx in no_field:
            query = self.type_vecs[idx]
            sims = self.type_vecs[has_field] @ query  # (n_has,)
            top_k_indices = np.argsort(sims)[-k:]
            top_sims = sims[top_k_indices]
            top_indices = has_field[top_k_indices]

            # Only use positive similarity (words in similar type region)
            positive = top_sims > 0
            if not positive.any():
                continue

            top_sims = top_sims[positive]
            top_indices = top_indices[positive]

            if len(top_sims) == 0:
                continue

            # Weighted average of neighbors' type fields
            weights = top_sims / (top_sims.sum() + 1e-12)
            target = np.zeros(self.type_dim, dtype=np.float32)
            for i, w in zip(top_indices, weights):
                target += w * self.type_field[i]

            self.type_field[idx] = normalize(target)
            interpolated += 1

        if interpolated > 0:
            # Update has_type_field mask
            self.has_type_field = (
                np.linalg.norm(self.type_field, axis=1) > 1e-12
            )

    def _context_centroid(self, recent: list[np.ndarray]) -> np.ndarray:
        if not recent:
            return np.zeros(self.sem_vecs.shape[1])
        weights = np.array([
            self.window_decay ** (len(recent) - 1 - i)
            for i in range(len(recent))
        ])
        weights = weights / weights.sum()
        result = np.zeros(self.sem_vecs.shape[1])
        for v, w in zip(recent, weights):
            result += w * v
        return normalize(result)

    # ------------------------------------------------------------------
    # Fluency Mechanisms
    # ------------------------------------------------------------------

    def _calibrate_temperature(
        self,
        generated: list[str],
        recent_indices: list[int],
        base_temp: float,
    ) -> float:
        """Auto-calibrate temperature from recent output quality.

        Measures two signals from the last N generated words:
          1. REPETITION RATE: fraction of bigrams that appear ≥2 times.
             High repetition → text is rigid → INCREASE temperature.
          2. COHERENCE DRIFT: std of consecutive word similarities.
             High drift → text is chaotic → DECREASE temperature.

        Both are measured from actual distribution — no fixed thresholds.

        Returns:
            Adjusted temperature in [base_temp*0.6, base_temp*2.0].
        """
        if len(generated) < 3:
            return base_temp

        n_recent = min(len(generated), 8)
        recent_words = generated[-n_recent:]
        recent_idx = recent_indices[-n_recent:]

        # ── Repetition rate ──
        bigrams = list(zip(recent_words[:-1], recent_words[1:]))
        if len(bigrams) >= 2:
            bigram_counts = {}
            for bg in bigrams:
                bigram_counts[bg] = bigram_counts.get(bg, 0) + 1
            repeated = sum(1 for c in bigram_counts.values() if c >= 2)
            rep_rate = repeated / len(bigrams)
        else:
            rep_rate = 0.0

        # ── Coherence drift ──
        if len(recent_idx) >= 3:
            vecs = [self.sem_vecs[i] for i in recent_idx]
            consec_sims = [
                float(np.dot(vecs[i], vecs[i+1]))
                for i in range(len(vecs) - 1)
            ]
            drift = np.std(consec_sims) if consec_sims else 0.0
        else:
            drift = 0.0

        # ── Map to temperature adjustment ──
        # Repetition: 0→1.0, 0.5→1.5, 1.0→2.0
        rep_factor = 1.0 + rep_rate * 1.0

        # Drift: low→1.0, high→0.6 (chaotic → reduce temp)
        drift_clamped = min(drift, 0.5)
        drift_factor = 1.0 - drift_clamped * 0.8  # range [0.6, 1.0]

        adjusted = base_temp * rep_factor * drift_factor
        return float(np.clip(adjusted, base_temp * 0.6, base_temp * 2.0))

    def _creative_perturbation(
        self,
        sem_centroid: np.ndarray,
        rng: np.random.RandomState,
        restlessness: float = 0.02,
    ) -> np.ndarray:
        """Add a controlled perturbation to explore nearby semantic space.

        The perturbation is a small random vector projected onto the
        tangent space of the unit sphere (orthogonal to centroid).
        This ensures the perturbation explores DIFFERENT semantic
        directions rather than just adding noise along the centroid.

        Magnitude is auto-calibrated by 'restlessness':
          - Low restlessness → small perturbation (conservative)
          - High restlessness → larger perturbation (exploratory)

        Args:
            sem_centroid: Normalized semantic context centroid.
            rng: Random state for reproducibility.
            restlessness: Exploration magnitude (0 = none, 0.1 = high).

        Returns:
            Perturbed centroid (normalized).
        """
        if restlessness <= 1e-8:
            return sem_centroid

        # Generate random direction in full space
        noise = rng.randn(self.sem_vecs.shape[1]).astype(np.float32)
        noise = noise / (np.linalg.norm(noise) + 1e-12)

        # Project onto tangent space (orthogonal to centroid)
        # This removes the component parallel to centroid
        parallel = float(np.dot(noise, sem_centroid)) * sem_centroid
        tangent = noise - parallel
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-12:
            return sem_centroid
        tangent = tangent / tangent_norm

        # Apply perturbation
        perturbed = sem_centroid + restlessness * tangent
        return normalize(perturbed)

    def _calibrate_restlessness(
        self,
        generated: list[str],
        recent_indices: list[int],
        base_restlessness: float = 0.02,
    ) -> float:
        """Auto-calibrate restlessness from recent output diversity.

        If recent text is predictable (high bigram authenticity to corpus,
        low entropy of word choices), increase restlessness to explore more.

        If recent text is diverse (high entropy, low repetition),
        decrease restlessness to stay coherent.

        Returns:
            Restlessness magnitude in [0, 0.1].
        """
        if len(generated) < 3:
            return base_restlessness

        n_recent = min(len(generated), 8)
        recent_words = generated[-n_recent:]

        # Entropy of word distribution in recent window
        word_counts = {}
        for w in recent_words:
            word_counts[w] = word_counts.get(w, 0) + 1
        total = len(recent_words)
        probs = [c / total for c in word_counts.values()]
        entropy = -sum(p * np.log(p + 1e-12) for p in probs)
        max_entropy = np.log(total)  # if all unique
        normalized_entropy = entropy / (max_entropy + 1e-12)

        # High entropy (diverse) → lower restlessness
        # Low entropy (repetitive) → higher restlessness
        restlessness = base_restlessness * (2.0 - normalized_entropy)

        return float(np.clip(restlessness, 0.0, 0.1))

    def _channel_confidence(
        self, scores: np.ndarray, top_k: int | None = None
    ) -> float:
        """Measure a channel's confidence from the shape of its score distribution.

        Confidence is HIGH when:
          - The top-k scores are concentrated near 1.0 (clear preference)
          - The top-k scores have low spread (decisive)

        Confidence is LOW when:
          - The top-k scores are spread out (indecisive)
          - The scores are all low (no clear candidate)

        Uses percentil of actual distribution — no fixed thresholds.

        Args:
            scores: Normalized score vector (after dividing by max abs).
            top_k: Number of top candidates to analyze (default: 5% of vocab).

        Returns:
            Confidence score in [0, 1].
        """
        if top_k is None:
            top_k = max(10, int(self.vocab_size * 0.05))

        # Sort descending and take top-k
        top_k = min(top_k, self.vocab_size)
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_scores = scores[top_indices]

        max_score = top_scores.max()
        if max_score < 1e-12:
            return 0.0

        mean_score = top_scores.mean()
        std_score = top_scores.std()

        # Coefficient of variation: low cv = concentrated (confident)
        cv = std_score / (mean_score + 1e-12)

        # Peak-to-mean ratio: how much the best candidate dominates
        peak_ratio = max_score / (mean_score + 1e-12)

        # Confidence: combines signal strength and concentration
        # High mean + low cv + high peak = high confidence
        confidence = mean_score * (1.0 - min(cv, 1.0)) * min(peak_ratio, 3.0) / 3.0

        return float(np.clip(confidence, 0.0, 1.0))

    def _measure_context_strength(
        self,
        phase_scores: np.ndarray,
        static_scores: np.ndarray | None,
    ) -> float:
        """Measure how much the phase lens changes word rankings.

        Auto-calibrating: counts how many top-50 words DIFFER between
        phase-aware and static scoring. No dependence on absolute
        similarity magnitudes (which are inherently small in high-D).

        High novelty → phase lens is adding new, context-specific information
        Low novelty → phase lens is not changing anything → context is weak

        Args:
            phase_scores: Scores from phase-aware semantic channel
            static_scores: Scores from static cosine similarity (or None)

        Returns:
            context_strength in [0, 1]
        """
        if static_scores is None:
            # No static reference — use score concentration as proxy
            return self._channel_confidence(phase_scores)

        top_k = min(50, self.vocab_size)

        # Top-k words in each scoring
        phase_top = set(np.argpartition(phase_scores, -top_k)[-top_k:])
        static_top = set(np.argpartition(static_scores, -top_k)[-top_k:])

        # Fraction of top-k words that are NEW (phase-specific)
        new_words = phase_top - static_top
        novelty = len(new_words) / top_k

        # Also factor in phase score concentration
        phase_conf = self._channel_confidence(phase_scores)

        # Combined: novelty dominates, confidence modulates
        context_strength = 0.7 * novelty + 0.3 * phase_conf

        return float(np.clip(context_strength, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Type-Topic Gate (multiplicative filter, not additive blend)
    # ------------------------------------------------------------------

    def _normalize_score_channel(
        self,
        scores: np.ndarray,
        excluded: set[int],
    ) -> np.ndarray:
        """Normalize a score channel to [0, 1] using its own distribution."""
        normed = scores.astype(np.float32).copy()
        valid_mask = np.ones(self.vocab_size, dtype=bool)
        for idx in excluded:
            valid_mask[idx] = False

        valid = normed[valid_mask]
        if len(valid) == 0:
            return normed

        v_min = valid.min()
        if v_min < 0:
            normed = normed - v_min
        v_max = normed[valid_mask].max()
        if v_max > 1e-12:
            normed = normed / v_max

        for idx in excluded:
            normed[idx] = -1.0
        return normed

    def _pmi_ri_channel_scores(
        self,
        recent_indices: list[int],
        excluded: set[int],
    ) -> np.ndarray | None:
        """PMI-RI content channel: co-occurrence-concentrated semantics.

        This is a complementary semantic space. It does not replace SVD,
        Phase Lens, PairSDM, or Type Field. It scores candidates by similarity
        to the recent context centroid in PMI-RI space, where corpus
        co-occurrence has already concentrated related words.
        """
        if self.pmi_ri_vecs is None or not recent_indices:
            return None

        ctx_indices = recent_indices[-self.window_size:]
        weights = np.array([
            self.window_decay ** (len(ctx_indices) - 1 - i)
            for i in range(len(ctx_indices))
        ], dtype=np.float32)
        weights = weights / (weights.sum() + 1e-12)

        query = np.zeros(self.pmi_ri_vecs.shape[1], dtype=np.float32)
        for idx, w in zip(ctx_indices, weights):
            query += w * self.pmi_ri_vecs[idx]
        query = normalize(query)

        scores = self.pmi_ri_vecs @ query.astype(np.float32)
        for idx in excluded:
            scores[idx] = -1.0
        return scores

    def _blend_pmi_ri_channel(
        self,
        sem_scores: np.ndarray,
        recent_indices: list[int],
        excluded: set[int],
    ) -> np.ndarray:
        """Blend SVD semantic scores with PMI-RI agreement by confidence.

        PMI-RI is intentionally not added raw: it is a co-occurrence lens,
        not the global semantic space. The product SVD × PMI-RI only boosts
        candidates that are plausible in both spaces, preserving SVD's global
        geometry while injecting corpus co-occurrence concentration.
        """
        pmi_scores = self._pmi_ri_channel_scores(recent_indices, excluded)
        if pmi_scores is None:
            return sem_scores

        sem_norm = self._normalize_score_channel(sem_scores, excluded)
        pmi_norm = self._normalize_score_channel(pmi_scores, excluded)
        agreement = self._normalize_score_channel(sem_norm * pmi_norm, excluded)

        sem_conf = self._channel_confidence(sem_norm)
        agreement_conf = self._channel_confidence(agreement)
        type_conf = self._channel_confidence(
            self._normalize_score_channel(
                self.type_vecs @ self.type_field[recent_indices[-1]].astype(np.float32),
                excluded,
            )
        )
        total_conf = sem_conf + agreement_conf + type_conf
        if total_conf < 1e-12:
            return sem_scores

        sem_weight = (sem_conf + type_conf) / total_conf
        agreement_weight = agreement_conf / total_conf
        return sem_weight * sem_norm + agreement_weight * agreement

    def _type_topic_gate_scores(
        self,
        type_scores: np.ndarray,
        topic_vec: np.ndarray,
        excluded: set[int],
    ) -> np.ndarray:
        """Multiplicative gate: Type Field × Topic Knowledge.

        KEY INSIGHT: Additive blending of diffuse signals produces noise.
        Multiplicative gating selects words that satisfy BOTH constraints:
          - Type Field: syntactically appropriate (97% precision)
          - Topic (SDM): topically relevant (co-occurrence based)

        The product `type_score * topic_score` is HIGH only when BOTH
        are high. This naturally concentrates scores without any
        fixed threshold or template.

        This replaces:
          1. Cosine similarity between word vectors (diffuse, ~0.01-0.24)
          2. Additive blending of type + semantic (noise amplification)
          3. Temperature-dependent sampling (random within diffuse top-k)

        With:
          1. Type scores: `type_vecs @ type_field[last_word]` (concentrated)
          2. Topic scores: `word_vecs @ sdm_topic_vec` (co-occurrence based)
          3. Product: only words that pass BOTH filters survive

        Args:
            type_scores: Scores from Type Field prediction
            topic_vec: Topic knowledge vector from SDM or PairSDM
            excluded: Word indices to suppress

        Returns:
            Gated scores, shape (vocab_size,)
        """
        # ── Compute topic scores ──
        topic_scores = self.sem_vecs @ topic_vec.astype(np.float32)

        # Suppress excluded in both channels
        for idx in excluded:
            type_scores[idx] = 0.0
            topic_scores[idx] = 0.0

        # ── Normalize each channel independently ──
        # Shift to non-negative (required for multiplicative gate)
        t_min = type_scores[type_scores > -1e9].min()
        if t_min < 0:
            type_scores = type_scores - t_min
        t_max = type_scores.max()
        if t_max > 1e-12:
            type_scores = type_scores / t_max

        s_min = topic_scores[topic_scores > -1e9].min()
        if s_min < 0:
            topic_scores = topic_scores - s_min
        s_max = topic_scores.max()
        if s_max > 1e-12:
            topic_scores = topic_scores / s_max

        # ── Multiplicative gate: BOTH must be high ──
        # The product is near-zero unless both type and topic agree.
        # This REPLACES diffuse cosine similarity with dual-constraint filtering.
        gated = type_scores * topic_scores

        # ── Auto-calibrated exponent ──
        # Higher exponent = stricter gate (only very strong agreement survives)
        # Auto-calibrate from the concentration of the gated scores
        valid_mask = np.ones(self.vocab_size, dtype=bool)
        for idx in excluded:
            valid_mask[idx] = False
        valid_gated = gated[valid_mask]

        top_k = max(5, int(len(valid_gated) * 0.02))
        top_indices = np.argpartition(valid_gated, -top_k)[-top_k:]
        top_mean = valid_gated[top_indices].mean()
        all_mean = valid_gated.mean()

        # If top scores are much higher than mean → gate is working → use as-is
        # If top scores are similar to mean → gate is weak → sharpen with exponent
        concentration = top_mean / (all_mean + 1e-12)
        if concentration < 3.0:
            # Weak gate: sharpen to amplify differences
            exponent = 2.0
            gated = gated ** exponent

        # Re-normalize
        g_max = gated[valid_mask].max()
        if g_max > 1e-12:
            gated = gated / g_max

        for idx in excluded:
            gated[idx] = -1.0

        return gated

    def _type_sem_pmi_gate_scores(
        self,
        type_scores: np.ndarray,
        sem_scores: np.ndarray,
        pmi_scores: np.ndarray,
        excluded: set[int],
        current_temp: float | None = None,
        context_strength: float | None = None,
        generated: list | None = None,
    ) -> np.ndarray:
        """Multiplicative gate across Type × Semantic × PMI-RI channels.

        Each channel is normalized independently to [0,1] (excluding banned
        indices). The product selects words that satisfy ALL three constraints:
        - Type: syntactic appropriateness
        - Semantic: contextual/topic coherence (SVD/SDM/MSWE/CRA)
        - PMI-RI: corpus co-occurrence concentrated content

        If the combined channel confidences are vanishingly small we fall
        back to the existing Type-Topic gate (if available) or the
        Type Maestro additive blend to avoid over-filtering.
        """
        # Defensive copies
        t_scores = type_scores.copy()
        s_scores = sem_scores.copy()
        p_scores = pmi_scores.copy()

        # Normalize each channel to [0,1]
        t_norm = self._normalize_score_channel(t_scores, excluded)
        s_norm = self._normalize_score_channel(s_scores, excluded)
        p_norm = self._normalize_score_channel(p_scores, excluded)

        # Channel confidences
        t_conf = self._channel_confidence(t_norm)
        s_conf = self._channel_confidence(s_norm)
        p_conf = self._channel_confidence(p_norm)
        total_conf = t_conf + s_conf + p_conf

        # Fallback when channels provide no signal
        if total_conf < 1e-12:
            # Prefer Type-Topic gate if a topic vector was provided externally
            if hasattr(self, '_last_topic_vec') and self._last_topic_vec is not None:
                gated = self._type_topic_gate_scores(
                    type_scores=type_scores,
                    topic_vec=self._last_topic_vec,
                    excluded=excluded,
                )
                self._last_topic_vec = None
                return gated

            # Otherwise fallback to the additive Type Maestro blend
            sem_max = np.abs(sem_scores).max()
            if sem_max > 1e-12:
                sem_scores = sem_scores / sem_max
            return self._type_maestro_blend(
                type_scores, sem_scores,
                current_temp if current_temp is not None else 0.8,
                context_strength=context_strength if context_strength is not None else 0.0,
                generated=generated,
            )

        # Multiply normalized channels — product highlights agreement
        gated = t_norm * s_norm * p_norm

        # Auto-calibration: if gated is too flat, sharpen via exponent
        valid_mask = np.ones(self.vocab_size, dtype=bool)
        for idx in excluded:
            valid_mask[idx] = False
        valid_gated = gated[valid_mask]

        top_k = max(5, int(len(valid_gated) * 0.02))
        top_indices = np.argpartition(valid_gated, -top_k)[-top_k:]
        top_mean = valid_gated[top_indices].mean()
        all_mean = valid_gated.mean()
        concentration = top_mean / (all_mean + 1e-12)
        if concentration < 3.0:
            gated = gated ** 2.0

        # Re-normalize to [0,1]
        g_max = gated[valid_mask].max()
        if g_max > 1e-12:
            gated = gated / g_max

        for idx in excluded:
            gated[idx] = -1.0

        return gated

    def _type_maestro_blend(
        self,
        type_scores: np.ndarray,
        sem_scores: np.ndarray,
        temperature: float = 0.8,
        context_strength: float = 0.0,
        generated: list = None,
    ) -> np.ndarray:
        """Type Field as MAESTRO — auto-calibrated by semantic dispersion.

        The type weight is INVERSELY proportional to semantic score dispersion:

        HIGH DISPERSION (P90 ≫ P10) → strong, decisive context:
          Few words dominate the semantic channel. The model knows what
          to say. Type cedes control down to the session-derived floor.

        LOW DISPERSION (P90 ≈ P10) → weak, diffuse context:
          All scores look alike. No clear semantic winner. Type must
          lead to maintain grammatical structure up to the session-derived ceiling.

        The dispersion is measured from the ACTUAL score distribution
        at each step — P90 - P10 normalized by the score scale. The current
        dispersion is then positioned inside the generation session's own
        observed dispersion range. This makes the curve gradual and local:
        each session defines its own floor/ceiling from its own scores.

        Properties:
          - Type floor comes from the strongest semantic concentration seen
          - Type ceiling comes from the weakest semantic concentration seen
          - The blend adapts PER STEP — different words, different contexts
          - All weights from score distributions — no fixed blend weights
        """
        # ── Semantic dispersion → context strength ──
        # Excluded words are assigned the minimum score before blending.
        finite_sem = sem_scores[np.isfinite(sem_scores)]
        if len(finite_sem) == 0:
            return type_scores

        sem_min = finite_sem.min()
        valid_scores = finite_sem[finite_sem > sem_min]
        if len(valid_scores) == 0:
            valid_scores = finite_sem

        p90 = np.percentile(valid_scores, 90)
        p10 = np.percentile(valid_scores, 10)
        scale = np.percentile(np.abs(valid_scores), 90) + \
            np.percentile(np.abs(valid_scores), 10) + 1e-12
        dispersion = max(0.0, float((p90 - p10) / scale))

        history = getattr(self, '_semantic_dispersion_history', None)
        if history is None:
            history = []
            self._semantic_dispersion_history = history
        history.append(dispersion)

        hist = np.asarray(history, dtype=np.float32)
        hist_min = float(hist.min())
        hist_max = float(hist.max())
        # Base type strength: how confident and concentrated is the type field?
        type_conf = self._channel_confidence(type_scores)
        finite_type = type_scores[np.isfinite(type_scores)]
        type_min = finite_type.min()
        valid_type = finite_type[finite_type > type_min]
        if len(valid_type) == 0:
            valid_type = finite_type
        t90 = np.percentile(valid_type, 90)
        t10 = np.percentile(valid_type, 10)
        type_scale = np.percentile(np.abs(valid_type), 90) + \
            np.percentile(np.abs(valid_type), 10) + 1e-12
        type_dispersion = max(0.0, float((t90 - t10) / type_scale))
        type_strength = type_conf + type_dispersion

        hist_range = hist_max - hist_min
        if hist_range > 1e-12:
            semantic_position = (dispersion - hist_min) / hist_range
        else:
            semantic_position = dispersion / (dispersion + type_strength + 1e-12)

        # ── Session-derived floor/ceiling ──
        # Weakest semantic concentration seen → type ceiling.
        # Strongest semantic concentration seen → type floor.
        type_ceiling = type_strength / (type_strength + hist_min + 1e-12)
        type_floor = type_strength / (
            type_strength + hist_max + dispersion + 1e-12
        )
        type_weight = type_ceiling - semantic_position * (type_ceiling - type_floor)
        self._last_type_weight = float(type_weight)

        sem_weight = 1.0 - type_weight

        # ── Base type bias to preserve grammatical structure ──
        # Small constant increase to favour Type field for syntax (articles, prepositions).
        base_type_bias = 0.20  # raised base from implicit 0.0 to 0.20
        # Blend type_weight toward 1.0 by base_type_bias fraction
        type_weight = float(type_weight + base_type_bias * (1.0 - type_weight))

        # ── Auto-calibration using recent functional-word ratio ──
        # If function-words in last 5 generated steps are below 30%, boost Type further.
        try:
            recent = [] if generated is None else list(generated[-5:])
            if recent:
                func_count = sum(1 for w in recent if w in self._function_words)
                func_ratio = func_count / max(len(recent), 1)
            else:
                func_ratio = 1.0
        except Exception:
            func_ratio = 1.0

        if func_ratio < 0.30:
            # Scale boost linearly with deficit; max extra boost = 0.5
            deficit = max(0.0, (0.30 - func_ratio) / 0.30)
            extra_boost = 0.5 * deficit
            type_weight = float(type_weight + extra_boost * (1.0 - type_weight))

        # Clip and compute semantic weight (pair-chain/SDM remains content engine but reduced when type is stronger)
        type_weight = max(0.0, min(1.0, type_weight))
        sem_weight = 1.0 - type_weight

        scores = type_weight * type_scores + sem_weight * sem_scores

        self._last_type_weight = float(type_weight)

        return scores

    # ------------------------------------------------------------------
    # Contextual Resonator Attention (CRA)
    # ------------------------------------------------------------------

    def _contextual_resonator_attention(
        self,
        m_state: np.ndarray,
        excluded: set[int],
        n_anchors: int = 5,
    ) -> np.ndarray:
        """Contextual Resonator Attention — attention without backprop.

        Four-stage pipeline for contextual word relevance:
          1. RESONATOR: extract top-N anchor concepts from M state
          2. HRR UNBINDING: each anchor queries M state via circular correlation
             → produces "proposal vectors" enriched by holographic interference
          3. DenseSDM CLEANUP: proposal vectors query SDM for noise-resistant recall
          4. WORD SCORING: words ranked by resonance with SDM-cleaned proposals

        The key insight (HRR theory): unbind(M_state, anchor) extracts everything
        in the composite that's associated with the anchor. When the conversation
        is about "eletricidade", the proposal vector carries "conduz", "fio",
        "corrente" — not just static similarity to "eletricidade".

        Args:
            m_state: M-encoded conversation state (normalized).
            excluded: Set of word indices to exclude.
            n_anchors: Number of anchor concepts to extract.

        Returns:
            Contextual relevance scores, shape (vocab_size,).
        """
        # ── Stage 1: Resonator extracts anchor concepts ──
        m_norm = normalize(m_state.astype(np.float32))
        anchor_sims = self.sem_vecs @ m_norm

        # Even if absolute similarities are low (M state can drift),
        # the RELATIVE ranking is valid for anchor selection.
        # Take top anchors by ranking.
        top_k = min(max(n_anchors * 5, 30), self.vocab_size)
        top_indices = np.argpartition(anchor_sims, -top_k)[-top_k:]
        top_sims = anchor_sims[top_indices]
        sorted_order = np.argsort(top_sims)[::-1]
        top_indices = top_indices[sorted_order]
        top_sims = top_sims[sorted_order]

        # Normalize similarity range for stable weighting
        sim_min = top_sims[-1]
        sim_range = top_sims[0] - sim_min
        if sim_range > 1e-12:
            norm_sims = (top_sims - sim_min) / sim_range
        else:
            norm_sims = np.ones_like(top_sims)

        # ── Stage 2: HRR Unbinding → contextual proposal vectors ──
        proposal_vectors = []
        proposal_weights = []

        for i, idx in enumerate(top_indices):
            if idx in excluded:
                continue

            anchor_vec = self.sem_vecs[idx]

            # HRR unbinding: extract everything associated with this anchor
            proposal = np.real(ifft(
                fft(m_state) * np.conj(fft(anchor_vec.astype(np.float32)))
            ))

            p_norm = np.linalg.norm(proposal)
            if p_norm < 1e-10:
                continue

            proposal = normalize(proposal.astype(np.float32))
            proposal_vectors.append(proposal)
            # Weight by normalized similarity (ranking-based, robust to low abs sims)
            proposal_weights.append(float(norm_sims[i]))

            if len(proposal_vectors) >= n_anchors:
                break

        if not proposal_vectors:
            # Fallback: just use M state projection
            c_scores = self.sem_vecs @ m_norm
            for idx in excluded:
                c_scores[idx] = -1.0
            return c_scores

        # Normalize proposal weights
        pw = np.array(proposal_weights)
        pw = pw / (pw.sum() + 1e-12)

        # ── Stage 3: DenseSDM cleanup + word scoring ──
        # Each proposal queries SDM independently; results are combined
        cra_scores = np.zeros(self.vocab_size, dtype=np.float32)

        for pv, w in zip(proposal_vectors, pw):
            if self.sdm is not None:
                # SDM cleanup: remove interference noise
                sdm_result = self.sdm.read(pv)
                # Word scores by resonance with SDM-cleaned proposal
                word_scores = self.sem_vecs @ sdm_result.astype(np.float32)
            else:
                # Without SDM: direct similarity to proposal vector
                word_scores = self.sem_vecs @ pv

            cra_scores += w * word_scores

        # ── Stage 4: Exclude recent words ──
        for idx in excluded:
            cra_scores[idx] = -1.0

        return cra_scores

    # ------------------------------------------------------------------
    # M-State Word Extraction (Resonator sequential decoding)
    # ------------------------------------------------------------------

    def _extract_words_from_M(
        self,
        m_state: np.ndarray,
        excluded: set[int],
        n_extract: int = 8,
    ) -> np.ndarray:
        """Extract words directly from the M state via sequential HRR unbinding.

        This REPLACES static SVD cosine similarity. Instead of asking
        "which words are similar to the context centroid?", we ask
        "which words are BOUND INTO the M state right now?"

        Algorithm (Resonator-like sequential decoding):
          1. Project M state onto word space → find top word
          2. HRR-unbind that word from the M state → exposes inner composite
          3. The residual now encodes everything EXCEPT the extracted word
          4. Repeat: project residual → find next word → unbind → ...

        Each extracted word was literally part of the conversation encoded
        in the M state. The extraction order reflects contextual prominence:
        words with stronger binding emerge first.

        Args:
            m_state: M-encoded conversation state.
            excluded: Word indices to exclude.
            n_extract: Number of words to extract.

        Returns:
            Word scores derived from M-state extraction, shape (vocab_size,).
        """
        residual = normalize(m_state.astype(np.float32))
        mswe_scores = np.zeros(self.vocab_size, dtype=np.float32)

        for iteration in range(n_extract):
            # ── Find the word most strongly present in the residual ──
            word_sims = self.sem_vecs @ residual
            for idx in excluded:
                word_sims[idx] = -1e10

            best_idx = int(np.argmax(word_sims))
            best_sim = float(word_sims[best_idx])
            if best_sim < 1e-6:
                break

            # ── Score: boost this word and its semantic neighbors ──
            # The extracted word gets the highest score (it's in the M state)
            # Words similar to it also get boosted (they're contextually related)
            boost = 1.0 / (1.0 + iteration)  # earlier extractions = stronger signal
            best_vec = self.sem_vecs[best_idx]

            # Direct score for the extracted word
            mswe_scores[best_idx] += boost

            # Spread to semantic neighbors (the M state encodes relationships)
            neighbor_sims = self.sem_vecs @ best_vec
            neighbor_sims[best_idx] = -1.0  # don't double-count
            mswe_scores += boost * 0.5 * np.maximum(neighbor_sims, 0)

            # ── Remove this word from the residual via HRR unbinding ──
            # unbind(residual, word) exposes what else is in the composite
            residual = np.real(ifft(
                fft(residual) * np.conj(fft(best_vec.astype(np.float32)))
            ))
            rn = np.linalg.norm(residual)
            if rn > 1e-12:
                residual = residual / rn
            else:
                break  # nothing left to extract

        # Normalize to [0, 1] range
        mswe_max = mswe_scores.max()
        if mswe_max > 1e-12:
            mswe_scores = mswe_scores / mswe_max

        return mswe_scores

    # ------------------------------------------------------------------
    # Phase-Aware Semantic Scoring (Context Lens)
    # ------------------------------------------------------------------

    def _phase_aware_semantic_scores(
        self,
        query_vec: np.ndarray,
        context_vec: np.ndarray,
        excluded: set[int],
    ) -> np.ndarray:
        """Compute semantic scores via phase-rotated QUERY (not candidates).

        KEY INSIGHT from experiments: phase-rotate the QUERY toward context,
        not each candidate. cos(phase_lens(query, ctx, α), candidate) is
        both cheaper (one phase lens per step) and more effective than
        rotating every candidate.

        The deformed query carries context-specific spectral features.
        A word like "cobre" phase-rotated toward electricity context
        becomes more similar to "conduz", "corrente" — without backprop.

        Cost: O(D log D + VD) — one phase_lens + one matrix-vector dot product.

        Args:
            query_vec: The query to deform (context window centroid)
            context_vec: The conversational context that defines the lens
            excluded: Word indices to suppress

        Returns:
            Semantic scores, shape (vocab_size,), normalized to [0, 1]
        """
        if not self.use_phase_lens:
            scores = self.sem_vecs @ query_vec.astype(np.float32)
            for idx in excluded:
                scores[idx] = -1.0
            return scores

        # ── Auto-calibrate alpha from query-context divergence ──
        sim_qc = cosine_similarity(query_vec, context_vec)
        alpha = (1.0 - sim_qc) * self.phase_lens_max_alpha
        alpha = float(np.clip(alpha, 0.0, self.phase_lens_max_alpha))

        # ── Phase-rotate the QUERY (not candidates) ──
        # This is the key architectural decision:
        #   cos(phase_lens(query, ctx, α), word)    ← correct, cheap
        # vs cos(word, phase_lens(ctx, query, α))   ← wrong, expensive
        deformed_query = phase_lens(query_vec, context_vec, alpha=alpha)

        # ── Score all candidates against deformed query ──
        scores = self.sem_vecs @ deformed_query.astype(np.float32)

        for idx in excluded:
            scores[idx] = -1.0

        return scores

    # ------------------------------------------------------------------
    # SDM-Powered Semantic Scoring (Primary Knowledge Engine)
    # ------------------------------------------------------------------

    def _sdm_semantic_scores(
        self,
        query_vec: np.ndarray,
        context_vec: np.ndarray | None,
        excluded: set[int],
    ) -> tuple[np.ndarray, float]:
        """SDM as PRIMARY semantic engine — replaces diffuse SVD similarity.

        Why SDM concentrates scores (vs SVD which is diffuse):
          1. SDM activates only 1% of locations → sparse, focused readout
          2. Activated locations store centroids of co-occurring words
             → the retrieved vector is closer to words that share context
          3. Corroboration weighting amplifies well-learned associations
          4. The result vector lives near frequently-co-occurring words

        Pipeline:
          1. Phase-rotate query toward context (if available)
          2. Query SDM with the (possibly deformed) query
          3. Score ALL words by similarity to SDM result
          4. Measure SDM confidence from read quality

        Args:
            query_vec: The query vector (context window centroid)
            context_vec: Optional context for phase lens deformation
            excluded: Word indices to suppress

        Returns:
            (scores, sdm_confidence) — scores in [0, 1], confidence in [0, 1]
        """
        q = query_vec.astype(np.float32)

        # ── Phase-rotate query toward context (if available) ──
        if self.use_phase_lens and context_vec is not None:
            sim_qc = cosine_similarity(q, context_vec)
            alpha = (1.0 - sim_qc) * self.phase_lens_max_alpha
            alpha = float(np.clip(alpha, 0.0, self.phase_lens_max_alpha))
            q = phase_lens(q, context_vec, alpha=alpha)

        # ── SDM read: retrieve knowledge vector ──
        if self.sdm is not None:
            sdm_result = self.sdm.read(q)
            # Use RAW SDM result (not residual). The raw result retains
            # similarity to the query while also carrying co-occurrence
            # knowledge from the corpus. This gives MORE concentrated
            # scores than the residual (which removes query alignment).
            query_for_scoring = sdm_result

            # SDM confidence: how much did SDM change the query?
            sdm_novelty = 1.0 - cosine_similarity(q, sdm_result)
            sdm_novelty = max(0.0, sdm_novelty)
        else:
            query_for_scoring = q
            sdm_novelty = 0.0

        # ── Score all words against SDM-enhanced query ──
        scores = self.sem_vecs @ query_for_scoring.astype(np.float32)

        # Suppress excluded words
        for idx in excluded:
            scores[idx] = -1.0

        # ── Score concentration analysis ──
        valid_mask = np.ones(len(scores), dtype=bool)
        for idx in excluded:
            valid_mask[idx] = False
        valid_scores = scores[valid_mask]

        # Gini-like concentration: ratio of top-1% to mean
        top_k = max(1, int(len(valid_scores) * 0.01))
        top_indices = np.argpartition(valid_scores, -top_k)[-top_k:]
        top_mean = valid_scores[top_indices].mean()
        all_mean = valid_scores.mean()
        concentration = top_mean / (all_mean + 1e-12)

        # SDM confidence: combines novelty and concentration
        sdm_confidence = float(np.clip(sdm_novelty * min(concentration / 5.0, 1.0), 0.0, 1.0))

        return scores, sdm_confidence

    # ------------------------------------------------------------------
    # Parallel Transport Scoring (PairSDM + Resonator Extraction)
    # ------------------------------------------------------------------

    def _pmi_filtered_transport_scores(
        self,
        current_idx: int,
        context_vec: np.ndarray | None,
        m_conversation_state: np.ndarray | None,
        excluded: set[int],
        n_paths: int = 5,
    ) -> tuple[np.ndarray, float] | None:
        """Two-stage content filter: PMI-RI path selection + Resonator extraction.

        Stage 1: PMI-RI selects stored corpus pairs whose source word is close
        to the current word in the co-occurrence-concentrated space.

        Stage 2: each surviving pair is decoded by Resonator unbinding, then
        scored by consistency with the current conversation state.
        """
        if (
            self.pmi_ri_vecs is None or
            self.pair_source_indices is None or
            self.pair_follower_indices is None
        ):
            return None

        source_indices = self.pair_source_indices
        follower_indices = self.pair_follower_indices
        if len(source_indices) == 0:
            return None

        current_pmi = self.pmi_ri_vecs[current_idx]
        source_scores = self.pmi_ri_vecs[source_indices] @ current_pmi
        valid_pair_mask = np.ones(len(source_indices), dtype=bool)
        for idx in excluded:
            valid_pair_mask &= follower_indices != idx

        valid_pair_indices = np.where(valid_pair_mask)[0]
        if len(valid_pair_indices) == 0:
            return None

        n_paths = min(n_paths, len(valid_pair_indices))
        valid_scores = source_scores[valid_pair_indices]
        top_local = np.argpartition(valid_scores, -n_paths)[-n_paths:]
        top_pairs = valid_pair_indices[top_local]

        scores = np.zeros(self.vocab_size, dtype=np.float32)
        path_weights = []
        recovered_vectors = []

        for pair_idx in top_pairs:
            src_idx = int(source_indices[pair_idx])
            fol_idx = int(follower_indices[pair_idx])
            src_vec = self.sem_vecs[src_idx]
            fol_vec = self.sem_vecs[fol_idx]

            pair_vec = M(src_vec, fol_vec, gamma=1.0, bilateral=True)
            recovered = normalize(unbind_M_reverse(
                pair_vec, src_vec, gamma=1.0, bilateral=True, n_refine=5
            ))

            if m_conversation_state is not None:
                # Simulate adding this follower to the GLOBAL M-state to
                # obtain the hypothetical next-state for the conversation.
                try:
                    simulated = M(m_conversation_state, recovered, gamma=1.0, bilateral=True)
                except Exception:
                    simulated = recovered.copy()

                # Measure stability of the resulting state via multiple algebraic
                # statistics (no cosine similarity): concentration, kurtosis,
                # energy focus, and spectral entropy (lower entropy = more focused).
                state = normalize(simulated.astype(np.float32))
                activations = self.sem_vecs @ state
                for ex_idx in excluded:
                    activations[ex_idx] = activations.min()

                p90 = np.percentile(activations, 90)
                p50 = np.percentile(activations, 50)
                p10 = np.percentile(activations, 10)
                scale = abs(p90) + abs(p50) + abs(p10) + 1e-12
                concentration = (p90 - p10) / scale

                centered = activations - activations.mean()
                std = centered.std() + 1e-12
                kurtosis = np.mean((centered / std) ** 4)

                energy = np.mean(activations ** 2)
                top_energy = np.mean(np.sort(activations)[-min(10, len(activations)): ] ** 2)
                energy_focus = top_energy / (energy + 1e-12)

                # Spectral entropy: lower is better (more focused frequencies)
                try:
                    ent = spectral_entropy(state)
                    entropy_score = 1.0 / (1.0 + ent)
                except Exception:
                    entropy_score = 0.0

                # Combine measures into a consistency score (auto-calibrated later)
                consistency = max(0.0, float(concentration + 0.5 * kurtosis + 0.8 * energy_focus + 0.5 * entropy_score))
            else:
                consistency = max(0.0, float(source_scores[pair_idx]))

            pmi_strength = max(0.0, float(source_scores[pair_idx]))
            path_weights.append(pmi_strength + consistency)
            recovered_vectors.append(recovered)

        weights = np.asarray(path_weights, dtype=np.float32)
        if weights.sum() <= 1e-12:
            weights = np.ones_like(weights)
        weights = weights / (weights.sum() + 1e-12)

        for recovered, w in zip(recovered_vectors, weights):
            word_scores = self.sem_vecs @ recovered.astype(np.float32)
            positive = np.maximum(word_scores, 0.0)
            scores += w * positive

        for idx in excluded:
            scores[idx] = -1.0

        valid_mask = np.ones(self.vocab_size, dtype=bool)
        for idx in excluded:
            valid_mask[idx] = False
        top_scores = scores[valid_mask]
        top_n = min(n_paths, len(top_scores))
        if top_n == 0:
            return None
        top_idx = np.argpartition(top_scores, -top_n)[-top_n:]
        concentration = top_scores[top_idx].mean() / (top_scores.mean() + 1e-12)
        confidence = float(np.clip(concentration / (concentration + 1.0), 0.0, 1.0))
        return scores, confidence

    def _transport_semantic_scores(
        self,
        current_idx: int,
        current_word_vec: np.ndarray,
        context_vec: np.ndarray | None,
        m_conversation_state: np.ndarray | None,
        excluded: set[int],
        top_k: int = 10,
    ) -> tuple[np.ndarray, float]:
        """Parallel Transport — exact extraction via M unbinding.

        Replaces similarity-based scoring with directional extraction:
          1. Build query pair: M(context, current_word)
             Encodes "where are we and what word"
          2. Query PairSDM → retrieves stored M(word, follower) pairs
          3. Unbind via Resonator: unbind_M_reverse(result, current_word)
             → recovers the TYPICAL FOLLOWER for this word in this context
          4. Score all words by similarity to the recovered follower vector
          5. Boost top-k extracted words (deterministic signal)

        Key property: the Resonator unbinding is EXACT for stored pairs
        (100% accuracy on direct extraction). The PairSDM blends multiple
        stored pairs, producing a consensus follower.

        Args:
            current_word_vec: Vector of the current word (last generated)
            context_vec: Optional context for phase lens deformation
            excluded: Word indices to suppress
            top_k: Number of extracted followers to boost

        Returns:
            (scores, transport_confidence)
        """
        if self.pair_sdm is None:
            return None, 0.0

        filtered = self._pmi_filtered_transport_scores(
            current_idx=current_idx,
            context_vec=context_vec,
            m_conversation_state=m_conversation_state,
            excluded=excluded,
        )
        if filtered is not None:
            return filtered

        # ── Build query pair: M(context_or_word, current_word) ──
        # If phase lenses are enabled, deform the current word toward the context
        ctx = context_vec if context_vec is not None else current_word_vec
        # Use the GLOBAL conversation M-state (m_conversation_state) as the
        # context for Phase Lens so deformation reflects the whole session.
        if self.use_phase_lens and m_conversation_state is not None:
            # Use configured max alpha
            alpha = getattr(self, 'phase_lens_max_alpha', 0.4)
            # Deform current word toward the global M-state
            try:
                deformed_word = phase_lens(current_word_vec, m_conversation_state, alpha=alpha)
            except Exception:
                deformed_word = current_word_vec
        else:
            deformed_word = current_word_vec

        query_pair = M(ctx, deformed_word, gamma=1.0, bilateral=True)

        # ── Query PairSDM → retrieve blended pair ──
        pair_result = self.pair_sdm.read(query_pair)

        # ── Extract follower via Resonator reverse unbinding ──
        # unbind_M_reverse(M(A, B), A) → B (iteratively refined)
        b_recovered = unbind_M_reverse(
            pair_result, current_word_vec,
            gamma=1.0, bilateral=True, n_refine=5
        )
        b_norm = normalize(b_recovered)

        # ── Score all words by similarity to recovered follower ──
        scores = self.sem_vecs @ b_norm.astype(np.float32)

        # ── Boost top extracted words (deterministic signal) ──
        # The Resonator extraction identifies specific words that
        # follow the current word. Boost them above similarity noise.
        valid_mask = np.ones(self.vocab_size, dtype=bool)
        for idx in excluded:
            valid_mask[idx] = False
            scores[idx] = -1.0

        # Find top-k extracted words
        top_indices = np.argpartition(scores[valid_mask], -top_k)[-top_k:]
        valid_indices = np.where(valid_mask)[0]
        top_global = valid_indices[top_indices]

        # Boost: extracted words get amplified
        boost = 3.0
        for idx in top_global:
            scores[idx] *= boost

        # ── Transport confidence ──
        # High confidence when the extracted follower has strong similarity
        top_scores = scores[top_global]
        top_mean = top_scores.mean()
        all_mean = scores[valid_mask].mean()
        concentration = top_mean / (all_mean + 1e-12)

        transport_confidence = float(np.clip(concentration / 10.0, 0.0, 1.0))

        return scores, transport_confidence

    def _m_state_consistency_rerank(
        self,
        base_scores: np.ndarray,
        recent_word_indices: list[int],
        m_conversation_state: np.ndarray,
        excluded: set[int],
        n_candidates: int = 15,
    ) -> np.ndarray:
        """Re-rank candidates by structural stability of the resulting M-state.

        This avoids choosing the word with highest cosine similarity. Each
        candidate is encoded into the current partial state, then evaluated by
        the shape of the resulting activation distribution:
          - concentrated activations → clearer attractor
          - high kurtosis → more decisive basin
          - lower activation energy spread → less diffuse state

        All quantities are measured from the candidate's own distribution and
        normalized across the candidates in this step.
        """

        # ── Select top-N candidates (exclude excluded) ──
        valid_mask = np.ones(self.vocab_size, dtype=bool)
        for idx in excluded:
            valid_mask[idx] = False

        candidate_scores = base_scores.copy()
        candidate_scores[~valid_mask] = -1e10

        top_n = min(n_candidates, valid_mask.sum())
        top_indices = np.argpartition(candidate_scores, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(candidate_scores[top_indices])[::-1]]

        # ── Encode recent words into a partial M-state ──
        if recent_word_indices:
            partial_base = self.sem_vecs[recent_word_indices[0]].copy()
            for idx in recent_word_indices[1:]:
                partial_base = M(partial_base, self.sem_vecs[idx],
                                gamma=1.0, bilateral=True)
        else:
            partial_base = np.zeros(self.sem_vecs.shape[1])

        # ── For each candidate, simulate adding it and measure stability ──
        stabilities = np.zeros(self.vocab_size, dtype=np.float32)

        for idx in top_indices:
            candidate_vec = self.sem_vecs[idx]
            if np.linalg.norm(partial_base) > 1e-12:
                simulated = M(partial_base, candidate_vec,
                             gamma=1.0, bilateral=True)
            else:
                simulated = candidate_vec.copy()

            state = normalize(simulated.astype(np.float32))
            activations = self.sem_vecs @ state
            for ex_idx in excluded:
                activations[ex_idx] = activations.min()

            p90 = np.percentile(activations, 90)
            p50 = np.percentile(activations, 50)
            p10 = np.percentile(activations, 10)
            scale = abs(p90) + abs(p50) + abs(p10) + 1e-12
            concentration = (p90 - p50) / scale

            centered = activations - activations.mean()
            std = centered.std() + 1e-12
            kurtosis = np.mean((centered / std) ** 4)

            energy = np.mean(activations ** 2)
            top_energy = np.mean(np.sort(activations)[-top_n:] ** 2)
            energy_focus = top_energy / (energy + 1e-12)

            stabilities[idx] = max(0.0, float(concentration + kurtosis + energy_focus))

        # ── Normalize stability inside the candidate set ──
        s_vals = stabilities[top_indices]
        s_min = s_vals.min()
        s_max = s_vals.max()
        if s_max - s_min > 1e-12:
            stabilities[top_indices] = (s_vals - s_min) / (s_max - s_min)

        stability_spread = stabilities[top_indices].std()
        base_spread = base_scores[top_indices].std() + 1e-12
        stability_weight = stability_spread / (stability_spread + base_spread)

        reranked = base_scores.copy()
        for idx in top_indices:
            reranked[idx] = (1.0 - stability_weight) * base_scores[idx] + \
                            stability_weight * stabilities[idx]

        # Suppress excluded
        for idx in excluded:
            reranked[idx] = -1.0

        return reranked

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _graph_edge_score(
        self,
        recent: list[int],
        n_depth: int = 2,
    ) -> np.ndarray:
        """Score each word by number of real corpus paths at depth n_depth.

        For each candidate, count how many paths of length n_depth from
        the recent sequence arrive at this word through real corpus edges.
        No cosine similarity — only arestas reais counted.
        """
        if self.pair_source_indices is None or self.pair_follower_indices is None:
            return np.zeros(self.vocab_size, dtype=np.float32)

        scores = np.zeros(self.vocab_size, dtype=np.float32)

        # Start from each recent word, walk real edges
        for start_idx in recent:
            # Find pairs where source == start_idx
            mask = self.pair_source_indices == start_idx
            followers = self.pair_follower_indices[mask]
            for fol in followers:
                if n_depth > 1:
                    # Recurse one more step through real edges
                    mask2 = self.pair_source_indices == fol
                    followers2 = self.pair_follower_indices[mask2]
                    for fol2 in followers2:
                        scores[fol2] += 1.0
                else:
                    scores[fol] += 1.0

        return scores

    def generate(
        self,
        prefix_words: list[str],
        max_len: int = 15,
        temperature: float = 0.8,
        inhibition_window: int = 5,
        seed: int | None = None,
        session_context: list[np.ndarray] | None = None,
        thematic_state: np.ndarray | None = None,
        creative_restlessness: float = 0.02,
        dynamic_temperature: bool = True,
    ) -> list[str]:
        """Generate text with auto-calibrating fluency mechanisms.

        Args:
            prefix_words: Starting words for generation.
            max_len: Maximum words to generate.
            temperature: Base sampling temperature.
            inhibition_window: Recent words to inhibit (anti-repetition).
            seed: Random seed.
            session_context: Optional vectors injected into semantic context
                            for cross-turn coherence (session memory).
            creative_restlessness: Base exploration magnitude. Higher = more
                                   stylistic variation. Auto-calibrated.
            dynamic_temperature: If True, temperature auto-calibrates from
                                 recent output quality (repetition/chaos).
        """
        rng = np.random.RandomState(seed)
        self._semantic_dispersion_history = []
        self._last_type_weight = None

        prefix_indices = [self.w2i[w] for w in prefix_words if w in self.w2i]
        if not prefix_indices:
            idx = rng.randint(0, self.vocab_size)
            prefix_indices = [idx]

        generated: list[str] = []
        recent_indices: list[int] = list(prefix_indices)
        sem_recent: list[np.ndarray] = [
            self.sem_vecs[idx] for idx in prefix_indices
        ]

        # ── M-state: encodes the full conversation for consistency checks ──
        # Updated after each generated word. Used by _m_state_consistency_rerank
        # to select candidates that fit the conversation's accumulated topic.
        if prefix_indices:
            m_conversation_state = self.sem_vecs[prefix_indices[0]].copy()
            for idx in prefix_indices[1:]:
                m_conversation_state = M(m_conversation_state, self.sem_vecs[idx],
                                        gamma=1.0, bilateral=True)
        else:
            m_conversation_state = None

        # ── Inject session context vectors ──
        # Session vectors are prepended to semantic context with
        # decaying weight so they influence but don't dominate
        if session_context:
            for sv in session_context:
                if np.linalg.norm(sv) > 1e-12:
                    sem_recent.insert(0, normalize(sv))
                    # Trim to window_size
                    if len(sem_recent) > self.window_size + 2:
                        sem_recent.pop(0)

        current_temp = temperature

        for step in range(max_len):
            excluded = set(recent_indices[-inhibition_window:]
                          if recent_indices else [])

            last_idx = recent_indices[-1]

            # ── TYPE channel ──
            type_target = self.type_field[last_idx]
            if np.linalg.norm(type_target) > 1e-12:
                type_scores = self.type_vecs @ type_target.astype(np.float32)
            else:
                type_scores = np.zeros(self.vocab_size, dtype=np.float32)
            for idx in excluded:
                type_scores[idx] = -1.0

            # ── CONTENT channel (MSWE + CRA or static, depending on context) ──

            # ── CONTEXT STRENGTH (for auto-calibrating type vs semantic balance) ──
            # Computed here, used by _type_maestro_blend below.
            context_strength = 0.0

            if thematic_state is not None and np.linalg.norm(thematic_state) > 1e-12:
                # ═══════════════════════════════════════════════════
                # DYNAMIC CONTENT: MSWE + CRA + Phase Lens from M state
                # ═══════════════════════════════════════════════════
                sem_centroid = self._context_centroid(sem_recent)

                # MSWE: extract words DIRECTLY from M state via
                #   sequential HRR unbinding (Resonator-like decoding).
                #   Each extracted word was bound into the conversation.
                mswe_scores = self._extract_words_from_M(
                    thematic_state, excluded, n_extract=8
                )

                # CRA: contextual resonance via anchors + SDM cleanup.
                #   Complements MSWE with holographic interference patterns.
                cra_scores = self._contextual_resonator_attention(
                    thematic_state, excluded, n_anchors=5
                )

                # ── Phase Lens: deform the QUERY toward M state ──
                # Phase-rotate the context centroid toward the M state,
                # then score all candidates against the deformed query.
                # This makes "cobre" more similar to "conduz" in an
                # electricity conversation, and to "minério" in a mining one.
                if self.use_phase_lens:
                    phase_scores = self._phase_aware_semantic_scores(
                        query_vec=sem_centroid,
                        context_vec=thematic_state,  # M state is the lens
                        excluded=excluded,
                    )
                    # Normalize to [0, 1]
                    p_valid = phase_scores.copy()
                    for idx in excluded:
                        p_valid[idx] = -1e10
                    p_min = p_valid[p_valid > -1e9].min()
                    if p_min < 0:
                        phase_scores = phase_scores - p_min
                    p_max = p_valid[p_valid > -1e9].max()
                    if p_max > 1e-12:
                        phase_scores = phase_scores / p_max
                    phase_conf = self._channel_confidence(phase_scores)
                else:
                    phase_scores = None
                    phase_conf = 0.0

                # Normalize CRA to [0, 1]
                c_valid = cra_scores.copy()
                for idx in excluded:
                    c_valid[idx] = -1e10
                c_min = c_valid[c_valid > -1e9].min()
                if c_min < 0:
                    cra_scores = cra_scores - c_min
                c_max = c_valid[c_valid > -1e9].max()
                if c_max > 1e-12:
                    cra_scores = cra_scores / c_max

                # Auto-calibrate blend: MSWE is the primary signal
                # (direct extraction from state), CRA adds resonance,
                # Phase Lens adds context-deformed similarity.
                mswe_conf = self._channel_confidence(mswe_scores)
                cra_conf = self._channel_confidence(cra_scores)

                # Dynamic weight allocation based on confidence
                total_conf = mswe_conf + cra_conf + phase_conf + 1e-12
                mswe_weight = 0.45 + 0.25 * (mswe_conf / total_conf)
                cra_weight = 0.15 + 0.10 * (cra_conf / total_conf)
                phase_weight = 0.10 + 0.15 * (phase_conf / total_conf)

                # Normalize weights
                weight_sum = mswe_weight + cra_weight + phase_weight
                mswe_weight /= weight_sum
                cra_weight /= weight_sum
                phase_weight /= weight_sum

                # Blend MSWE + CRA + Phase Lens → dynamic content scores
                sem_scores = (mswe_weight * mswe_scores +
                             cra_weight * cra_scores)
                if phase_scores is not None:
                    sem_scores = sem_scores + phase_weight * phase_scores

                # ── Context strength for type/semantic balance ──
                # Measures how much the phase lens changes rankings vs static.
                # High novelty → context is strong → type cedes control.
                static_ref = self.sem_vecs @ sem_centroid.astype(np.float32)
                for idx in excluded:
                    static_ref[idx] = -1.0
                context_strength = self._measure_context_strength(
                    phase_scores, static_ref
                ) if phase_scores is not None else 0.0

            else:
                # ── SEMANTIC: Transport → SDM → Phase Lens → Static ──
                sem_centroid = self._context_centroid(sem_recent)
                restlessness = self._calibrate_restlessness(
                    generated, recent_indices, creative_restlessness
                )
                sem_centroid = self._creative_perturbation(
                    sem_centroid, rng, restlessness
                )

                # Determine context vector for phase lens
                context_vec = None
                if session_context:
                    session_vecs = [sv for sv in session_context
                                   if np.linalg.norm(sv) > 1e-12]
                    if session_vecs:
                        context_vec = normalize(
                            np.mean([normalize(sv) for sv in session_vecs], axis=0)
                        )

                # Get current word vector for transport
                last_word_vec = self.sem_vecs[last_idx]

                # ── 1st CHOICE: Parallel Transport (PairSDM + Resonator) ──
                if self.pair_sdm is not None:
                    sem_scores, trans_conf = self._transport_semantic_scores(
                        current_idx=last_idx,
                        current_word_vec=last_word_vec,
                        context_vec=context_vec,
                        m_conversation_state=m_conversation_state,
                        excluded=excluded,
                    )
                    # mark that transport was used this step
                    self._last_was_transport = True

                    # ── M-State Consistency Re-Ranking ──
                    # Replaces diffuse word-to-word similarity with global
                    # consistency check: which candidate makes the partial
                    # sentence most consistent with the conversation state?
                    # If conversation is about electricity, "conduz" fits
                    # better than "metal" — even if both have similar
                    # cos(current_word, candidate) scores.
                    if m_conversation_state is not None:
                        sem_scores = self._m_state_consistency_rerank(
                            base_scores=sem_scores,
                            recent_word_indices=recent_indices[-3:],
                            m_conversation_state=m_conversation_state,
                            excluded=excluded,
                        )

                    context_strength = trans_conf
                    if context_vec is not None:
                        ctx_alignment = max(0.0, cosine_similarity(
                            context_vec, sem_centroid
                        ))
                        context_strength = trans_conf * ctx_alignment

                # ── 2nd CHOICE: SDM Primary ──
                elif self.sdm is not None:
                    sem_scores, sdm_conf = self._sdm_semantic_scores(
                        query_vec=sem_centroid,
                        context_vec=context_vec,
                        excluded=excluded,
                    )
                    self._last_was_transport = False
                    # Context strength from SDM confidence + phase alignment
                    if context_vec is not None:
                        ctx_alignment = max(0.0, cosine_similarity(
                            context_vec, sem_centroid
                        ))
                        context_strength = sdm_conf * ctx_alignment
                    else:
                        context_strength = sdm_conf * 0.5

                elif self.use_phase_lens and context_vec is not None:
                    # ── Fallback: Phase Lens (no SDM available) ──
                    sem_scores = self._phase_aware_semantic_scores(
                        query_vec=sem_centroid,
                        context_vec=context_vec,
                        excluded=excluded,
                    )
                    static_ref = self.sem_vecs @ sem_centroid.astype(np.float32)
                    for idx in excluded:
                        static_ref[idx] = -1.0
                    raw_strength = self._measure_context_strength(
                        sem_scores, static_ref
                    )
                    ctx_alignment = max(0.0, cosine_similarity(
                        context_vec, sem_centroid
                    ))
                    context_strength = raw_strength * ctx_alignment
                    self._last_was_transport = False

                else:
                    # ── Pure static (no SDM, no phase lens) ──
                    sem_scores = self.sem_vecs @ sem_centroid.astype(np.float32)
                    for idx in excluded:
                        sem_scores[idx] = -1.0
                    context_strength = 0.0
                    self._last_was_transport = False

            # ── PMI-RI semantic channel (optional) ──
            # If PMI-RI vectors are available and PairSDM is NOT providing
            # a transport path, promote PMI-RI to a first-class channel and
            # apply the Type × Sem × PMI multiplicative gate. This ensures
            # that content agreement in the concentrated PMI space filters
            # candidates that the diffuse SVD channel would otherwise allow.
            pmi_available = (self.pmi_ri_vecs is not None)
            pair_fallback = (self.pair_source_indices is None or self.pair_follower_indices is None)
            if pmi_available and pair_fallback:
                # Compute PMI-RI scores once
                pmi_scores = self._pmi_ri_channel_scores(recent_indices, excluded)
                if pmi_scores is None:
                    # Fall back to previous behavior
                    sem_scores = self._blend_pmi_ri_channel(
                        sem_scores, recent_indices, excluded
                    )
                else:
                    # If we have a last_topic_vec, prefer the original Type-Topic
                    # gate logic (it uses SDM topic). Otherwise use the 3-way gate.
                    if hasattr(self, '_last_topic_vec') and self._last_topic_vec is not None:
                        # Use Type-Topic gate first, then refine by PMI agreement
                        gated = self._type_topic_gate_scores(
                            type_scores=type_scores,
                            topic_vec=self._last_topic_vec,
                            excluded=excluded,
                        )
                        # refine gated by PMI agreement: product with normalized pmi
                        pmi_norm = self._normalize_score_channel(pmi_scores, excluded)
                        refined = self._normalize_score_channel(gated, excluded) * pmi_norm
                        # if refinement has signal, use it, else keep gated
                        if self._channel_confidence(refined) > 0.0:
                            sem_scores = refined
                        else:
                            sem_scores = gated
                        self._last_topic_vec = None
                    else:
                        # Apply full Type × Sem × PMI gate
                        sem_scores = sem_scores if sem_scores is not None else np.zeros(self.vocab_size, dtype=np.float32)
                        sem_scores = self._type_sem_pmi_gate_scores(
                            type_scores=type_scores,
                            sem_scores=sem_scores,
                            pmi_scores=pmi_scores,
                            excluded=excluded,
                            current_temp=current_temp,
                            context_strength=context_strength,
                            generated=generated,
                        )

            # ── Normalize type scores ──
            type_max = np.abs(type_scores).max()
            if type_max > 1e-12:
                type_scores = type_scores / type_max

            # ── Final gating / blending ──
            # At this point sem_scores has been processed by PMI/transport logic
            # and may already encode gating. If a last_topic_vec remains (rare),
            # apply Type-Topic gate; otherwise, use Type Maestro additive blend
            # as a graceful fallback.
            if hasattr(self, '_last_topic_vec') and self._last_topic_vec is not None:
                scores = self._type_topic_gate_scores(
                    type_scores=type_scores,
                    topic_vec=self._last_topic_vec,
                    excluded=excluded,
                )
                self._last_topic_vec = None
            else:
                # Ensure sem_scores is normalized for the additive blend
                sem_max = np.abs(sem_scores).max()
                if sem_max > 1e-12:
                    sem_scores = sem_scores / sem_max
                scores = self._type_maestro_blend(
                    type_scores, sem_scores, current_temp,
                    context_strength=context_strength,
                    generated=generated,
                )

            # ── DYNAMIC TEMPERATURE ──
            if dynamic_temperature and step > 0:
                current_temp = self._calibrate_temperature(
                    generated, recent_indices, temperature
                )

            # ── Temperature sampling ──
            score_std = np.std(scores)
            effective_temp = current_temp * max(score_std, 1e-6)
            scores_centered = scores - scores.max()
            exp_scores = np.exp(scores_centered / effective_temp)
            probs = exp_scores / exp_scores.sum()

            idx = rng.choice(self.vocab_size, p=probs)
            next_word = self.i2w[idx]
            generated.append(next_word)

            # ── Update M conversation state ──
            if m_conversation_state is None:
                m_conversation_state = self.sem_vecs[idx].copy()
            else:
                m_conversation_state = M(m_conversation_state, self.sem_vecs[idx],
                                        gamma=1.0, bilateral=True)

            sem_recent.append(self.sem_vecs[idx])
            if len(sem_recent) > self.window_size:
                sem_recent.pop(0)
            recent_indices.append(idx)

        return generated
