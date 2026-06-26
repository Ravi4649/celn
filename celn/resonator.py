"""
CELN v3 — Resonator Network Decoder
====================================
Iterative factorization of composite vectors into constituent codebook
elements, based on Resonator Networks (Frady, Kleyko & Sommer, 2020).

Adapted for CELN v3:
  - Binding: circular convolution via FFT (bind/unbind from core.py)
  - Vectors: real-valued, L2-normalized to unit hypersphere (not bipolar)
  - Clean-up: nearest-neighbor by cosine similarity (not sign projection)
  - Codebook: shared across all factors (same word vectors)

Complexity: O(F · N_iter · V · D) vs exhaustive O(V^F) where
  F = number of factors, V = vocabulary size, D = 10k

Reference:
  Frady, Kent, Olshausen, Sommer. "Resonator Networks, 1: An Efficient
  Solution for Factoring High-Dimensional, Distributed Representations
  of Data Structures." Neural Computation 32(12), 2020.
"""

import numpy as np
from numpy.fft import fft, ifft
from typing import Optional, Tuple
from numba import njit

from .core import D, normalize, similarity


# ---------------------------------------------------------------------------
# Binding/unbinding for circular convolution
# ---------------------------------------------------------------------------

def bind_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular convolution of two vectors via FFT."""
    return np.real(ifft(fft(a) * fft(b)))


def unbind_vec(c: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Circular correlation — approximate inverse of bind.

    If c ≈ bind(a, b), then unbind(c, a) ≈ b (spectrally weighted).
    """
    return np.real(ifft(fft(c) * np.conj(fft(a))))


# ---------------------------------------------------------------------------
# Directional unbinding for M (projective_resonance)
# ---------------------------------------------------------------------------

def _compute_M_weights(
    y: np.ndarray,
    gamma: float = 1.0,
    bilateral: bool = True,
    x: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the real-valued spectral weights used in M(x, y).

    These are the phi weights that amplify dominant frequencies in y
    (unilateral) or frequencies where y has more energy than x (bilateral).

    Args:
        y: The "new info" vector.
        gamma: Amplification exponent.
        bilateral: If True, use bilateral (differential) weights.
        x: The "context" vector (required for bilateral).

    Returns:
        Real-valued weight array, shape (D,), same as |FFT(y)|.
    """
    Y = fft(y)
    mag_y = np.abs(Y)

    if bilateral and x is not None:
        X = fft(x)
        mag_x = np.abs(X)
        ratio = mag_y / (mag_x + 1e-12)
        median_ratio = np.median(ratio)
        if median_ratio > 1e-12:
            rel_weight = ratio / median_ratio
            weight_mag = np.tanh(rel_weight ** gamma)
        else:
            weight_mag = np.ones_like(mag_y)
    else:
        median_mag = np.median(mag_y)
        if median_mag > 1e-12:
            rel_mag = mag_y / median_mag
            weight_mag = np.tanh(rel_mag ** gamma)
        else:
            weight_mag = np.ones_like(mag_y)

    return weight_mag.astype(np.float32)


def unbind_M_forward(
    composite: np.ndarray,
    b: np.ndarray,
    gamma: float = 1.0,
    bilateral: bool = True,
    x: np.ndarray | None = None,
) -> np.ndarray:
    """Recover A from composite = M(A, B) — EXACT when B is known.

    M(A, B) = IFFT(FFT(A) * FFT(B) * w(|B|))
    → FFT(A) = FFT(composite) / (FFT(B) * w(|B|))
    → A = IFFT(FFT(composite) / (FFT(B) * w(|B|)))

    This is the KEY innovation: forward unbinding inverts M exactly
    because we know all the spectral weights applied to B.

    Args:
        composite: M(A, B), the bound state.
        b: The "new info" vector B (or estimate thereof).
        gamma, bilateral: M parameters matching the encoding.
        x: Context for bilateral weight computation (if B was encoded
           with bilateral=True, this should be the A estimate).

    Returns:
        Recovered A vector (not normalized — caller should nearest() it).
    """
    C = fft(composite)
    B = fft(b.astype(np.float32))

    # Compute the EXACT weights used when B was the "new info" in M(A, B)
    w = _compute_M_weights(b, gamma=gamma, bilateral=bilateral, x=x)

    # Spectral division: C / (B * w) = FFT(A)
    denominator = B * w

    # Safe division with magnitude threshold
    denom_mag = np.abs(denominator)
    safe_denom = np.where(denom_mag > 1e-10, denominator, 1.0)
    # Where denominator is too small, zero out the result (unreliable frequencies)
    A_hat = C / safe_denom
    A_hat = np.where(denom_mag > 1e-10, A_hat, 0.0)

    return np.real(ifft(A_hat))


def unbind_M_reverse(
    composite: np.ndarray,
    a: np.ndarray,
    gamma: float = 1.0,
    bilateral: bool = True,
    n_refine: int = 5,
) -> np.ndarray:
    """Recover B from composite = M(A, B) — ITERATIVE refinement.

    M(A, B) = IFFT(FFT(A) * FFT(B) * w(|B|))
    → FFT(B) * w(|B|) = FFT(composite) / FFT(A)

    This is NON-LINEAR in B because w depends on |FFT(B)|.
    Solved by iterative refinement: start with initial B estimate
    ignoring w, then recompute w from B estimate and repeat.

    Args:
        composite: M(A, B), the bound state.
        a: The "context" vector A (or estimate thereof).
        gamma, bilateral: M parameters matching the encoding.
        n_refine: Number of refinement iterations (default 5).

    Returns:
        Recovered B vector (not normalized).
    """
    C = fft(composite)
    A = fft(a.astype(np.float32))

    # Initial estimate: ignore w, just divide
    B_hat = C / (A + 1e-12)

    # Iterative refinement: recompute w from current B estimate
    for _ in range(n_refine):
        b_current = np.real(ifft(B_hat))
        w = _compute_M_weights(b_current, gamma=gamma, bilateral=bilateral, x=a)
        B_hat = C / (A * w + 1e-12)

    return np.real(ifft(B_hat))


# ---------------------------------------------------------------------------
# Resonator Decoder
# ---------------------------------------------------------------------------

@njit(cache=True)
def _nearest_idx_numba(codebook: np.ndarray, vec_norm: np.ndarray) -> int:
    sims = codebook @ vec_norm.astype(codebook.dtype)
    return int(np.argmax(sims))


@njit(cache=True)
def _nearest_score_numba(codebook: np.ndarray, vec_norm: np.ndarray) -> tuple[int, float]:
    sims = codebook @ vec_norm.astype(codebook.dtype)
    idx = int(np.argmax(sims))
    return idx, float(sims[idx])


class ResonatorDecoder:
    """Factorize composite vectors into constituent codebook elements.

    Given a composite vector c = bind(x1, bind(x2, x3)) (or similar),
    iteratively recover each factor xi from a shared codebook.

    Parameters
    ----------
    codebook : np.ndarray, shape (V, D)
        Normalized word vectors (the codebook). All factors are drawn
        from this same codebook.
    max_iter : int
        Maximum iterations per restart (default 20).
    n_restarts : int
        Number of random initializations to try (default 3).
        The best result (highest average similarity) is returned.
    convergence_patience : int
        Stop early if estimates haven't changed for this many iterations.
    seed : int or None
        RNG seed for reproducible random initializations.
    """

    def __init__(
        self,
        codebook: np.ndarray,
        max_iter: int = 20,
        n_restarts: int = 3,
        convergence_patience: int = 4,
        seed: Optional[int] = None
    ):
        self.codebook = codebook.astype(np.float32)
        self.V, self.dim = codebook.shape
        self.max_iter = max_iter
        self.n_restarts = n_restarts
        self.convergence_patience = convergence_patience
        self.rng = np.random.RandomState(seed)

        # Precompute spectral magnitudes for resonance scoring (optional)
        self._codebook_spectra = np.abs(fft(codebook, axis=1))

    # ------------------------------------------------------------------
    # Core resonator iteration
    # ------------------------------------------------------------------

    def _nearest(self, vec: np.ndarray, top_k: int = 1) -> np.ndarray:
        vec_norm = normalize(vec)
        if top_k == 1:
            return np.array([_nearest_idx_numba(self.codebook, vec_norm)])
        sims = self.codebook @ vec_norm.astype(np.float32)
        return np.argsort(sims)[-top_k:][::-1]

    def _nearest_with_score(self, vec: np.ndarray) -> Tuple[int, float]:
        vec_norm = normalize(vec)
        return _nearest_score_numba(self.codebook, vec_norm)

    def _compute_unbound(
        self,
        composite: np.ndarray,
        other_factors: list[np.ndarray],
        binding_op: str = 'bind'
    ) -> np.ndarray:
        """Remove all other factors from the composite to isolate one factor.

        For binding_op='bind':
            Unbind each other factor: unbind(composite, prod(other_factors))

        For binding_op='M' (projective_resonance):
            Uses circular correlation as an approximate inverse.
            The phi weights from M are not exactly inverted, but the
            nearest-neighbor cleanup handles the resulting noise.

        Args:
            composite: The composite vector.
            other_factors: List of current estimates for all OTHER factors.
            binding_op: 'bind' or 'M'

        Returns:
            Noisy estimate of the target factor.
        """
        if not other_factors:
            return composite.copy()

        # Bind all other factors together
        bound_others = other_factors[0].copy()
        for f in other_factors[1:]:
            bound_others = bind_vec(bound_others, f)

        # Unbind them from the composite
        return unbind_vec(composite, bound_others)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode_2factor(
        self,
        composite: np.ndarray,
        binding_op: str = 'bind'
    ) -> dict:
        """Recover (a, b) from composite = bind(a, b) or M(a, b).

        For M encoding, uses DIRECTIONAL unbinding:
          - unbind_M_forward(composite, b) → recovers A EXACTLY
          - unbind_M_reverse(composite, a) → recovers B iteratively

        This respects M's non-commutativity and dramatically improves
        recovery of the first factor (A, the context).

        Args:
            composite: The composite vector, shape (D,).
            binding_op: 'bind' for pure circular convolution,
                       'M' if composite was created with projective_resonance.

        Returns:
            dict with keys:
                'indices': [idx_a, idx_b] — recovered codebook indices
                'similarities': [sim_a, sim_b]
                'iterations': number of iterations used
                'converged': whether convergence was reached
                'history': list of (idx_a, idx_b) per iteration
        """
        use_M = (binding_op == 'M')
        best_result = None
        best_avg_sim = -1.0

        for restart in range(self.n_restarts):
            # Random initialization
            a_idx = self.rng.randint(0, self.V)
            b_idx = self.rng.randint(0, self.V)

            history = [(a_idx, b_idx)]
            patience_counter = 0

            for iteration in range(self.max_iter):
                if use_M:
                    # ── DIRECTIONAL M-UNBINDING ──
                    b_vec = self.codebook[b_idx]

                    # Recover A (context): EXACT forward unbinding
                    a_tilde = unbind_M_forward(composite, b_vec)
                    a_new, a_sim = self._nearest_with_score(a_tilde)

                    # Recover B (new info): ITERATIVE reverse unbinding
                    a_vec = self.codebook[a_new]
                    b_tilde = unbind_M_reverse(composite, a_vec)
                    b_new, b_sim = self._nearest_with_score(b_tilde)
                else:
                    # ── SYMMETRIC BIND-UNBIND (original) ──
                    b_vec = self.codebook[b_idx]
                    a_tilde = unbind_vec(composite, b_vec)
                    a_new, a_sim = self._nearest_with_score(a_tilde)

                    a_vec = self.codebook[a_new]
                    b_tilde = unbind_vec(composite, a_vec)
                    b_new, b_sim = self._nearest_with_score(b_tilde)

                history.append((a_new, b_new))

                # Check convergence
                if a_new == a_idx and b_new == b_idx:
                    patience_counter += 1
                    if patience_counter >= self.convergence_patience:
                        break
                else:
                    patience_counter = 0
                    a_idx, b_idx = a_new, b_new

            avg_sim = (a_sim + b_sim) / 2.0 if iteration > 0 else 0.0

            if avg_sim > best_avg_sim:
                best_avg_sim = avg_sim
                best_result = {
                    'indices': [a_idx, b_idx],
                    'similarities': [float(a_sim), float(b_sim)],
                    'iterations': iteration + 1,
                    'converged': patience_counter >= self.convergence_patience,
                    'history': history,
                    'restart': restart,
                    'avg_similarity': avg_sim,
                }

        return best_result

    def decode_3factor(
        self,
        composite: np.ndarray,
        binding_op: str = 'bind'
    ) -> dict:
        """Recover (a, b, c) from composite = M(a, M(b, c)).

        For M encoding, uses DIRECTIONAL SEQUENTIAL unbinding:
        Composite = M(A, M(B, C)), where:
          - A is outermost (context for M(B,C))
          - B is middle (context for C)
          - C is innermost (new info, amplified twice)

        Sequential algorithm (outside-in):
          1. inner = unbind_M_reverse(composite, A)  → M(B, C)
          2. C = unbind_M_reverse(inner, B)          → C (innermost)
          3. B = unbind_M_forward(inner, C)          → B (EXACT)
          4. A = unbind_M_forward(composite, inner)  → A (EXACT)

        Each step peels off one layer, respecting M's directionality.

        Args:
            composite: The composite vector, shape (D,).
            binding_op: 'bind' for pure circular convolution,
                       'M' if composite was created with projective_resonance.

        Returns:
            dict with 'indices', 'similarities', 'iterations', 'converged', 'history'.
        """
        use_M = (binding_op == 'M')
        best_result = None
        best_avg_sim = -1.0

        for restart in range(self.n_restarts):
            # Random initialization
            a_idx = self.rng.randint(0, self.V)
            b_idx = self.rng.randint(0, self.V)
            c_idx = self.rng.randint(0, self.V)

            history = [(a_idx, b_idx, c_idx)]
            patience_counter = 0

            for iteration in range(self.max_iter):
                if use_M:
                    # ── DIRECTIONAL SEQUENTIAL M-UNBINDING ──
                    # Composite = M(A, M(B, C))
                    # Work OUTSIDE-IN: peel layers to reach innermost factor
                    #   inner = M(B, C)  →  C is last arg to inner M
                    #   composite = M(A, inner)  →  inner is last arg to outer M

                    # Step 1: Recover inner ≈ M(B, C) by peeling A from composite
                    # unbind_M_reverse recovers the "new info" (inner) from M(A, inner)
                    inner_est = unbind_M_reverse(
                        composite, self.codebook[a_idx]
                    )

                    # Step 2: Recover C (innermost) from inner = M(B, C)
                    # C was the "new info" in M(B, C); peel B off to get C
                    c_tilde = unbind_M_reverse(
                        inner_est, self.codebook[b_idx]
                    )
                    c_new, c_sim = self._nearest_with_score(c_tilde)

                    # Step 3: Recover B (middle) from inner using known C
                    # unbind_M_forward is EXACT: peel C off M(B, C) → B
                    # Use current B estimate as context for bilateral weights
                    b_tilde = unbind_M_forward(
                        inner_est, self.codebook[c_new],
                        x=self.codebook[b_idx]  # B_est for bilateral w
                    )
                    b_new, b_sim = self._nearest_with_score(b_tilde)

                    # Step 4: Reconstruct inner = M(B_new, C_new) — now exact
                    from .core import projective_resonance as M
                    inner_refined = M(
                        self.codebook[b_new], self.codebook[c_new],
                        gamma=1.0, bilateral=True
                    )

                    # Step 5: Recover A (outermost) using exact inner — EXACT
                    a_tilde = unbind_M_forward(composite, inner_refined)
                    a_new, a_sim = self._nearest_with_score(a_tilde)
                else:
                    # ── SYMMETRIC BIND-UNBIND (original) ──
                    bc_bound = bind_vec(
                        self.codebook[b_idx], self.codebook[c_idx]
                    )
                    a_tilde = unbind_vec(composite, bc_bound)
                    a_new, a_sim = self._nearest_with_score(a_tilde)

                    ac_bound = bind_vec(
                        self.codebook[a_new], self.codebook[c_idx]
                    )
                    b_tilde = unbind_vec(composite, ac_bound)
                    b_new, b_sim = self._nearest_with_score(b_tilde)

                    ab_bound = bind_vec(
                        self.codebook[a_new], self.codebook[b_new]
                    )
                    c_tilde = unbind_vec(composite, ab_bound)
                    c_new, c_sim = self._nearest_with_score(c_tilde)

                history.append((a_new, b_new, c_new))

                # Check convergence
                if (a_new == a_idx and b_new == b_idx and
                        c_new == c_idx):
                    patience_counter += 1
                    if patience_counter >= self.convergence_patience:
                        break
                else:
                    patience_counter = 0
                    a_idx, b_idx, c_idx = a_new, b_new, c_new

            avg_sim = (a_sim + b_sim + c_sim) / 3.0

            if avg_sim > best_avg_sim:
                best_avg_sim = avg_sim
                best_result = {
                    'indices': [a_idx, b_idx, c_idx],
                    'similarities': [
                        float(a_sim), float(b_sim), float(c_sim)
                    ],
                    'iterations': iteration + 1,
                    'converged': patience_counter >= self.convergence_patience,
                    'history': history,
                    'restart': restart,
                    'avg_similarity': avg_sim,
                }

        return best_result

    def decode(
        self,
        composite: np.ndarray,
        n_factors: int = 2,
        binding_op: str = 'bind'
    ) -> dict:
        """General F-factor decoder.

        Args:
            composite: Composite vector.
            n_factors: Number of factors to recover (2 or 3 supported).
            binding_op: 'bind' or 'M'.

        Returns:
            Decoding result dict (see decode_2factor/decode_3factor).
        """
        if n_factors == 2:
            return self.decode_2factor(composite, binding_op)
        elif n_factors == 3:
            return self.decode_3factor(composite, binding_op)
        else:
            raise ValueError(f"n_factors={n_factors} not supported (use 2 or 3)")

    # ------------------------------------------------------------------
    # Baseline: direct cosine similarity
    # ------------------------------------------------------------------

    def decode_direct(
        self,
        composite: np.ndarray,
        n_factors: int = 2,
        top_k: int = 20
    ) -> dict:
        """Baseline: top-N words by direct cosine similarity to composite.

        This is the current decoder approach — should perform poorly
        on M-encoded or bind-encoded composites.

        Returns:
            dict with 'indices' (top-N), 'similarities'.
        """
        vec_norm = normalize(composite)
        sims = self.codebook @ vec_norm.astype(np.float32)
        top_indices = np.argsort(sims)[-top_k:][::-1]

        return {
            'indices': [int(i) for i in top_indices[:n_factors]],
            'similarities': [float(sims[i]) for i in top_indices[:n_factors]],
            'all_top_k': [int(i) for i in top_indices],
            'all_similarities': [float(sims[i]) for i in top_indices],
        }

    # ------------------------------------------------------------------
    # Composite construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_composite_bind(factors: list[np.ndarray]) -> np.ndarray:
        """Create composite via pure circular convolution.

        c = bind(f0, bind(f1, bind(f2, ...)))
        """
        result = factors[0].copy()
        for f in factors[1:]:
            result = bind_vec(result, f)
        return normalize(result)

    @staticmethod
    def make_composite_M(
        factors: list[np.ndarray],
        gamma: float = 1.0,
        bilateral: bool = True
    ) -> np.ndarray:
        """Create composite via projective_resonance (M operation).

        c = M(f0, M(f1, M(f2, ...)))
        Non-commutative — order matters.
        """
        from .core import projective_resonance

        result = factors[0].copy()
        for f in factors[1:]:
            result = projective_resonance(
                result, f, gamma=gamma, bilateral=bilateral
            )
        return normalize(result)


# ---------------------------------------------------------------------------
# Utility: top-K accuracy
# ---------------------------------------------------------------------------

def top_k_accuracy(
    recovered_indices: list[int],
    ground_truth_indices: list[int],
    k_values: list[int] = [1, 3, 5, 10]
) -> dict[int, float]:
    """Compute top-K accuracy: fraction of factors within top-K.

    Args:
        recovered_indices: Indices returned by decoder (exact).
        ground_truth_indices: True indices.
        k_values: K values to evaluate.

    Returns:
        Dict mapping K to accuracy (fraction of factors correct at top-K).
        For top-1, this is exact match rate.
    """
    # For exact recovery, check each factor individually
    results = {}
    for k in k_values:
        correct = sum(
            1 for r, gt in zip(recovered_indices, ground_truth_indices)
            if r == gt
        )
        results[k] = correct / len(ground_truth_indices) if ground_truth_indices else 0.0
    return results
