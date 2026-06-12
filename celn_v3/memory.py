"""
CELN v3 — Dense Sparse Distributed Memory (SDM)
================================================
Long-term associative memory adapted from Kanerva's SDM for
dense real-valued vectors with cosine similarity.

Design decisions (see CLAUDE.md principles):
  - Locations initialized from real corpus centroids (not random)
    to give semantic structure to the address space.
  - Activation via percentile threshold (not fixed k) — self-calibrating.
  - Accumulation via simple addition (not bit counters) — algebraic.
  - Read via raw cosine-similarity weighting (not softmax) — linear.
  - ZERO backprop, ZERO templates, ZERO fixed thresholds.

Mathematical core:
  write:  accumulators[activated] += v; counters[activated] += 1
  read:   centroids = accumulators / counters
          result = Σ (sim_i / Σ sim_j) · centroids_i
"""

import numpy as np
from typing import Optional, Tuple

from .core import normalize, similarity, auto_threshold


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
D = 10_000  # dimensionality (same as core.py)


# ---------------------------------------------------------------------------
# Sentence encoding
# ---------------------------------------------------------------------------

def sentence_to_centroid(
    tokens: list[str],
    vectors: np.ndarray,
    w2i: dict[str, int]
) -> np.ndarray:
    """Encode a sentence as the normalized mean of its word vectors.

    This preserves semantic similarity with individual words, enabling
    topic-based queries against the SDM.

    Args:
        tokens: Tokenized sentence (list of lowercase words)
        vectors: Word vector matrix, shape (vocab_size, D)
        w2i: Word-to-index mapping

    Returns:
        Normalized centroid vector, shape (D,), or zero vector if no
        known words.
    """
    indices = [w2i[w] for w in tokens if w in w2i]
    if not indices:
        return np.zeros(D)
    centroid = vectors[indices].mean(axis=0)
    return normalize(centroid)


# ---------------------------------------------------------------------------
# Dense SDM
# ---------------------------------------------------------------------------

class DenseSDM:
    """Sparse Distributed Memory for dense real-valued vectors.

    Hard locations cover the semantic space via data-derived addresses.
    Writing accumulates vectors at activated locations; reading pools
    their centroids weighted by cosine similarity to the query.

    Parameters
    ----------
    n_locations : int
        Number of hard locations. Default 4096 (~164 MB in float32).
    activation_pct : float
        Fraction of locations activated per access (default 0.01 = 1%).
        Used as a percentile threshold — self-calibrating.
    seed : int or None
        RNG seed for reproducibility when sampling location seeds.
    """

    def __init__(
        self,
        n_locations: int = 4096,
        activation_pct: float = 0.01,
        seed: Optional[int] = None
    ):
        if n_locations < 1:
            raise ValueError("n_locations must be >= 1")
        if not (0 < activation_pct <= 1.0):
            raise ValueError("activation_pct must be in (0, 1]")

        self.n_locations = n_locations
        self.activation_pct = activation_pct
        self.rng = np.random.RandomState(seed)

        # Hard location addresses — initialized later via initialize_addresses()
        self.addresses = np.empty((n_locations, D), dtype=np.float32)

        # Accumulators: sum of all vectors written to each location
        self.accumulators = np.zeros((n_locations, D), dtype=np.float32)

        # Counters: number of writes to each location
        self.counters = np.zeros(n_locations, dtype=np.int32)

        # ── Corroboration tracking ──
        # Per-location confidence weight (1.0 = neutral, >1 = corroborated, <1 = uncertain)
        self.corroboration = np.ones(n_locations, dtype=np.float32)

        # Running consistency score per location (smoothed similarity of recent writes)
        self.consistency = np.zeros(n_locations, dtype=np.float32)

        # Count of corroborating vs contradictory writes per location
        self.corroboration_hits = np.zeros(n_locations, dtype=np.int32)
        self.contradiction_hits = np.zeros(n_locations, dtype=np.int32)

        # ── Competing hypotheses (contradiction isolation) ──
        # When two facts about the same topic make opposite claims,
        # they are stored as SEPARATE hypotheses rather than blended.
        self.alt_accumulators = np.zeros((n_locations, D), dtype=np.float32)
        self.alt_counters = np.zeros(n_locations, dtype=np.int32)
        self.alt_corroboration = np.ones(n_locations, dtype=np.float32)
        self.alt_consistency = np.zeros(n_locations, dtype=np.float32)

        # Track which locations have active conflicts
        self.has_conflict = np.zeros(n_locations, dtype=bool)
        self.conflict_magnitude = np.zeros(n_locations, dtype=np.float32)

        # Global contradiction log
        self.total_conflicts_detected: int = 0

        # Cached activation threshold (computed on each write/read)
        self._last_threshold: float = 0.0
        self._last_n_activated: int = 0

        # Total writes for statistics
        self.total_writes: int = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_addresses(self, seed_vectors: np.ndarray):
        """Initialize hard-location addresses from data-derived vectors.

        Uses a random sample (with replacement if needed) of real sentence
        centroids to give semantic structure to the address space.

        Args:
            seed_vectors: Array of shape (n_samples, D) from which to
                          sample location addresses.
        """
        n_samples = seed_vectors.shape[0]
        if n_samples == 0:
            raise ValueError("seed_vectors must contain at least one vector")

        indices = self.rng.choice(n_samples, size=self.n_locations, replace=True)
        self.addresses = seed_vectors[indices].astype(np.float32)

        # Ensure all addresses are normalized
        norms = np.linalg.norm(self.addresses, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        self.addresses = self.addresses / norms

    def initialize_addresses_random(self):
        """Initialize hard-location addresses from random normalized vectors.

        Fallback when no data-derived vectors are available. In 10k-D,
        random vectors are nearly orthogonal, so activation patterns
        carry less information than data-derived addresses.
        """
        raw = self.rng.randn(self.n_locations, D).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        self.addresses = raw / norms

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def _compute_activation_mask(self, vec: np.ndarray) -> np.ndarray:
        """Compute the activation mask for a vector.

        Returns a boolean array of shape (n_locations,) where True marks
        locations whose cosine similarity to `vec` exceeds the
        self-calibrating percentile threshold.
        """
        # Cosine similarity = dot product (both addresses and vec are normalized)
        sims = self.addresses @ vec.astype(np.float32)

        # Self-calibrating threshold: top (activation_pct) fraction
        k = max(1, int(self.n_locations * self.activation_pct))
        perc = 100.0 * (1.0 - self.activation_pct)
        threshold = float(np.percentile(sims, perc))

        self._last_threshold = threshold
        mask = sims >= threshold
        self._last_n_activated = int(mask.sum())

        # FIX: if mask is empty (can happen with very small activation_pct
        # or adversarial input), fall back to top-1
        if self._last_n_activated == 0:
            top_idx = int(np.argmax(sims))
            mask = np.zeros(self.n_locations, dtype=bool)
            mask[top_idx] = True
            self._last_n_activated = 1

        return mask

    def write(self, vector: np.ndarray) -> int:
        """Write a single vector into the SDM.

        The vector is accumulated at all hard locations whose addresses
        are sufficiently similar (top activation_pct fraction).

        Args:
            vector: Shape (D,), should be normalized.

        Returns:
            Number of locations activated (written to).
        """
        mask = self._compute_activation_mask(vector)
        vec = vector.astype(np.float32)

        self.accumulators[mask] += vec
        self.counters[mask] += 1
        self.total_writes += 1

        return self._last_n_activated

    def write_corroborated(self, vector: np.ndarray) -> dict:
        """Write with automatic corroboration checking.

        Before writing, checks consistency with existing knowledge at
        activated locations. Corroborating writes strengthen confidence;
        contradictory writes are flagged and stored with reduced weight.

        Auto-calibration: consistency threshold is the MEDIAN similarity
        between the new vector and existing centroids at activated locations.
        No fixed threshold — adapts to each location's distribution.

        Args:
            vector: Shape (D,), should be normalized.

        Returns:
            dict with keys: 'activated', 'corroborating', 'contradictory',
            'neutral', 'mean_consistency'.
        """
        mask = self._compute_activation_mask(vector)
        vec = vector.astype(np.float32)
        vec_norm = normalize(vec)

        n_corroborating = 0
        n_contradictory = 0
        n_neutral = 0
        sims_list = []

        active_locs = np.where(mask)[0]

        for loc in active_locs:
            if self.counters[loc] == 0:
                # First write to this location — neutral
                n_neutral += 1
            else:
                # Compare new vector with stored centroid
                stored = normalize(
                    self.accumulators[loc] / self.counters[loc]
                )
                sim = float(np.dot(stored, vec_norm))
                sims_list.append(sim)

        # ── Auto-calibrate consistency threshold via PERCENTILE ──
        if sims_list:
            sims_arr = np.array(sims_list)

            # Percentile-based thresholds — adapt to actual distribution
            p75 = float(np.percentile(sims_arr, 75))  # top quartile = corroborating
            p25 = float(np.percentile(sims_arr, 25))  # bottom quartile = contradictory
            p10 = float(np.percentile(sims_arr, 10))  # very bottom = strong contradiction

            sim_spread = p75 - p25

            for loc in active_locs:
                if self.counters[loc] == 0:
                    continue

                stored = normalize(
                    self.accumulators[loc] / self.counters[loc]
                )
                sim = float(np.dot(stored, vec_norm))

                if sim_spread > 0.05 and sim >= p75:
                    # ═══ CORROBORATION ═══
                    # High similarity → strengthens existing knowledge
                    self.corroboration[loc] = min(
                        5.0, self.corroboration[loc] * 1.15
                    )
                    self.corroboration_hits[loc] += 1
                    n_corroborating += 1
                    # If this location had a conflict, the new evidence
                    # supports the MAIN hypothesis
                    if self.has_conflict[loc]:
                        self.conflict_magnitude[loc] *= 0.8  # reduce conflict
                    self.accumulators[loc] += vec
                    self.counters[loc] += 1

                elif sim_spread > 0.05 and sim <= p10:
                    # ═══ STRONG CONTRADICTION ═══
                    # Very low similarity at same-topic locations
                    # → competing hypothesis detected!
                    # Store in ALT accumulator to preserve BOTH viewpoints
                    self.contradiction_hits[loc] += 1
                    n_contradictory += 1
                    self.total_conflicts_detected += 1

                    if not self.has_conflict[loc]:
                        # First conflict: initialize competing hypothesis
                        self.has_conflict[loc] = True
                        self.alt_accumulators[loc] = vec.copy()
                        self.alt_counters[loc] = 1
                        self.alt_corroboration[loc] = 0.5  # start with low trust
                        self.conflict_magnitude[loc] = 1.0 - sim  # how opposed
                    else:
                        # Reinforce competing hypothesis
                        self.alt_accumulators[loc] += vec
                        self.alt_counters[loc] += 1
                        self.alt_corroboration[loc] = min(
                            5.0, self.alt_corroboration[loc] * 1.15
                        )
                        self.conflict_magnitude[loc] = max(
                            self.conflict_magnitude[loc], 1.0 - sim
                        )

                elif sim_spread > 0.05 and sim <= p25:
                    # ═══ MODERATE CONTRADICTION ═══
                    # Below typical but not extreme → flag but don't isolate yet
                    self.corroboration[loc] = max(
                        0.1, self.corroboration[loc] * 0.85
                    )
                    self.contradiction_hits[loc] += 1
                    n_contradictory += 1
                    self.accumulators[loc] += 0.3 * vec
                    self.counters[loc] += 1

                else:
                    # ═══ NEUTRAL ═══
                    self.accumulators[loc] += vec
                    self.counters[loc] += 1
                    n_neutral += 1

                self.consistency[loc] = (
                    0.8 * self.consistency[loc] + 0.2 * sim
                )

            self.total_writes += 1
        else:
            # No existing knowledge — standard write to all activated
            self.accumulators[mask] += vec
            self.counters[mask] += 1
            self.total_writes += 1
            n_neutral = self._last_n_activated

        return {
            'activated': self._last_n_activated,
            'corroborating': n_corroborating,
            'contradictory': n_contradictory,
            'neutral': n_neutral,
            'mean_consistency': float(np.mean(sims_list)) if sims_list else 0.0,
        }

    def read(self, query: np.ndarray) -> np.ndarray:
        """Read from the SDM using an associative query.

        Activates locations similar to the query, pools their stored
        centroids weighted by cosine similarity AND corroboration.
        Well-corroborated locations dominate the output.

        Args:
            query: Shape (D,), should be normalized.

        Returns:
            Retrieved vector, shape (D,), normalized. If no activated
            location has been written to, returns the query unchanged.
        """
        mask = self._compute_activation_mask(query)

        # Exclude locations that have never been written to
        active_and_written = mask & (self.counters > 0)

        if not active_and_written.any():
            return normalize(query.copy())

        # Compute centroids
        centroids = (
            self.accumulators[active_and_written] /
            self.counters[active_and_written, None].astype(np.float32)
        )

        # Weight by BOTH cosine similarity AND corroboration
        sims = self.addresses @ query.astype(np.float32)
        corr_weights = self.corroboration[active_and_written]

        # Combined weight: similarity * corroboration
        weights = sims[active_and_written] * corr_weights
        weights = np.maximum(weights, 0)  # ensure non-negative
        weights = weights / (weights.sum() + 1e-12)

        # Weighted sum of centroids
        result = (centroids * weights[:, None]).sum(axis=0)

        return normalize(result)

    def read_with_confidence(self, query: np.ndarray) -> dict:
        """Read from SDM with full confidence and conflict information.

        Returns the result vector PLUS metadata about corroboration
        and any detected conflicts at the activated locations.

        Args:
            query: Shape (D,), should be normalized.

        Returns:
            dict with keys:
              'result': the retrieved vector (normalized)
              'mean_corroboration': average corroboration at activated locations
              'n_conflicts': number of activated locations with competing hypotheses
              'conflict_magnitude': mean conflict magnitude (0=none, 1=strong)
              'trust_score': overall confidence in the result (0-1)
              'competing_result': alternative vector if conflicts exist (or None)
        """
        mask = self._compute_activation_mask(query)
        active_and_written = mask & (self.counters > 0)

        if not active_and_written.any():
            return {
                'result': normalize(query.copy()),
                'mean_corroboration': 1.0,
                'n_conflicts': 0,
                'conflict_magnitude': 0.0,
                'trust_score': 1.0,
                'competing_result': None,
            }

        # Main result (as before, with corroboration weighting)
        centroids = (
            self.accumulators[active_and_written] /
            self.counters[active_and_written, None].astype(np.float32)
        )
        sims = self.addresses @ query.astype(np.float32)
        corr_weights = self.corroboration[active_and_written]
        weights = sims[active_and_written] * corr_weights
        weights = np.maximum(weights, 0)
        weights = weights / (weights.sum() + 1e-12)
        result = normalize((centroids * weights[:, None]).sum(axis=0))

        # ── Conflict detection ──
        active_conflicts = active_and_written & self.has_conflict
        n_conflicts = int(active_conflicts.sum())
        conflict_mag = float(
            self.conflict_magnitude[active_conflicts].mean()
        ) if n_conflicts > 0 else 0.0

        # Competing result (from alt accumulators at conflict locations)
        competing_result = None
        if n_conflicts > 0:
            alt_active = active_conflicts & (self.alt_counters > 0)
            if alt_active.any():
                alt_centroids = (
                    self.alt_accumulators[alt_active] /
                    self.alt_counters[alt_active, None].astype(np.float32)
                )
                alt_corr = self.alt_corroboration[alt_active]
                alt_weights = sims[alt_active] * alt_corr
                alt_weights = np.maximum(alt_weights, 0)
                alt_sum = alt_weights.sum()
                if alt_sum > 1e-12:
                    alt_weights = alt_weights / alt_sum
                    competing_result = normalize(
                        (alt_centroids * alt_weights[:, None]).sum(axis=0)
                    )

        # Trust score: combines corroboration and absence of conflicts
        mean_corr = float(corr_weights.mean())
        trust_penalty = min(conflict_mag * 0.5, 0.5)  # max 50% penalty
        trust_score = max(0.1, mean_corr - trust_penalty) / max(mean_corr, 0.1)
        trust_score = float(np.clip(trust_score * mean_corr / 5.0, 0.0, 1.0))

        return {
            'result': result,
            'mean_corroboration': float(mean_corr),
            'n_conflicts': n_conflicts,
            'conflict_magnitude': conflict_mag,
            'trust_score': trust_score,
            'competing_result': competing_result,
        }

    def write_batch(self, vectors: np.ndarray) -> int:
        """Write multiple vectors to the SDM.

        Args:
            vectors: Shape (n_vectors, D), each row a normalized vector.

        Returns:
            Total number of location activations across all writes.
        """
        total_activations = 0
        for i in range(vectors.shape[0]):
            total_activations += self.write(vectors[i])
        return total_activations

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_location_theme(
        self,
        idx: int,
        word_vectors: np.ndarray,
        i2w: dict[int, str],
        top_k: int = 10
    ) -> list[Tuple[str, float]]:
        """Get the nearest words to a location's stored centroid.

        This reveals what "theme" emerged at that location from the
        data that activated it — without any explicit labeling.

        Args:
            idx: Location index (0 to n_locations-1)
            word_vectors: All word vectors, shape (vocab_size, D)
            i2w: Index-to-word mapping
            top_k: Number of nearest words to return

        Returns:
            List of (word, similarity) tuples sorted by similarity.
        """
        if self.counters[idx] == 0:
            return []

        centroid = self.accumulators[idx] / self.counters[idx]
        centroid = normalize(centroid)

        sims = word_vectors @ centroid
        top_indices = np.argsort(sims)[::-1][:top_k]

        return [(i2w[i], float(sims[i])) for i in top_indices]

    def get_location_address_theme(
        self,
        idx: int,
        word_vectors: np.ndarray,
        i2w: dict[int, str],
        top_k: int = 10
    ) -> list[Tuple[str, float]]:
        """Get the nearest words to a location's ADDRESS vector.

        This reveals what the location is "looking for" — its address
        in semantic space — independent of what has been stored there.

        Args:
            idx: Location index
            word_vectors: All word vectors, shape (vocab_size, D)
            i2w: Index-to-word mapping
            top_k: Number of nearest words to return

        Returns:
            List of (word, similarity) tuples.
        """
        address = self.addresses[idx]
        sims = word_vectors @ address
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(i2w[i], float(sims[i])) for i in top_indices]

    @property
    def stats(self) -> dict:
        """Return memory statistics."""
        n_written = int((self.counters > 0).sum())
        n_untouched = self.n_locations - n_written
        avg_writes = float(self.counters[self.counters > 0].mean()) if n_written > 0 else 0.0
        max_writes = int(self.counters.max())

        # Memory usage estimate
        addr_mb = self.addresses.nbytes / (1024 * 1024)
        acc_mb = self.accumulators.nbytes / (1024 * 1024)
        cnt_mb = self.counters.nbytes / (1024 * 1024)

        return {
            'n_locations': self.n_locations,
            'n_written': n_written,
            'n_untouched': n_untouched,
            'avg_writes_per_location': round(avg_writes, 2),
            'max_writes_per_location': max_writes,
            'total_writes': self.total_writes,
            'last_threshold': round(self._last_threshold, 6),
            'last_n_activated': self._last_n_activated,
            'memory_addresses_mb': round(addr_mb, 1),
            'memory_accumulators_mb': round(acc_mb, 1),
            'memory_counters_mb': round(cnt_mb, 1),
            'memory_total_mb': round(addr_mb + acc_mb + cnt_mb, 1),
        }
