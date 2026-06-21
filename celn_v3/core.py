"""
CELN v3 — Core Vector Operations
=================================
Projective Resonance: a single operation that unifies binding and attention
in the frequency domain, without backprop, running on CPU.

Mathematical foundation:
  M(x, y) = FFT⁻¹( FFT(x) ⊙ FFT(y) ⊙ φ_weight(FFT(y)) )

where φ_weight amplifies dominant frequencies and suppresses noise,
using the median magnitude as self-calibrating reference.

Key properties:
  - Non-commutative: M(x,y) ≠ M(y,x) — preserves word order
  - Associativity-preserving in scan mode (consistent left-to-right application)
  - Self-attentive: the binding itself encodes relevance via spectral emphasis
  - Self-calibrating: median-based, no fixed thresholds
  - O(d log d) via FFT — runs on CPU at ~microsecond scale
"""

import numpy as np
from numpy.fft import fft, ifft
from numba import njit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
D = 10_000  # Vector dimensionality (10k from VSA 2.0 heritage)

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    """L2 normalization to unit hypersphere."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def batch_normalize(M: np.ndarray) -> np.ndarray:
    """Normalize each row of M to unit length."""
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return M / norms


# ---------------------------------------------------------------------------
# FFT-based binding (circular convolution) and unbinding (correlation)
# ---------------------------------------------------------------------------

def bind(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Circular convolution via FFT: x * y.

    In the Fourier domain, convolution becomes element-wise multiplication.
    Complexity: O(d log d) for FFT, O(d) for multiplication.
    """
    X = fft(x)
    Y = fft(y)
    return ifft(X * Y).real


def unbind(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Circular correlation via FFT: approximate inverse of bind.

    Given c = bind(a, b), unbind(c, a) ≈ b  (noisy recovery).
    Uses complex conjugate in Fourier domain.
    """
    X = fft(x)
    Y = fft(y)
    return ifft(X * np.conj(Y)).real


# ---------------------------------------------------------------------------
# φ: The Frequency-Domain Amplifier (the "attention" in Projective Resonance)
# ---------------------------------------------------------------------------

def phi_weights(y: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Compute frequency amplification weights for vector y.

    This is the core innovation. It identifies which frequencies in y
    are "dominant" (above median magnitude) and returns weights that:
      - Amplify dominant frequencies (signal)
      - Suppress weak frequencies (noise)
      - Self-calibrate via median (no fixed threshold)

    Args:
        y: Real vector in time domain
        gamma: Amplification exponent.
               0.0 = identity (no amplification)
               1.0 = linear relative to median
               2.0 = aggressive winner-take-all

    Returns:
        Complex weight vector (same shape as FFT(y))
        to be multiplied element-wise in the Fourier domain.
    """
    Y = fft(y)
    magnitude = np.abs(Y)
    median_mag = np.median(magnitude)

    if median_mag < 1e-12:
        # All magnitudes near zero — return unity weights
        return np.ones_like(Y)

    # Relative magnitude: how dominant is each frequency?
    rel_mag = magnitude / median_mag

    # Power-law amplification: dominant freqs get >1, weak get <1
    weight_mag = rel_mag ** gamma

    # Soft clipping: tanh prevents unbounded amplification
    # while preserving the ordering (monotonic transformation)
    weight_mag = np.tanh(weight_mag)

    # Apply magnitude weight to the complex frequency components
    # Phase is preserved; only magnitude is modulated
    phase = Y / (magnitude + 1e-12)
    return weight_mag * phase


def phi(y: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply frequency amplification to vector y and return to time domain.

    φ(y) = FFT⁻¹( FFT(y) ⊙ φ_weights(y) )
    """
    Y = fft(y)
    weights = phi_weights(y, gamma)
    return ifft(Y * weights).real


# ---------------------------------------------------------------------------
# M(x, y): Projective Resonance — the unified operation
# ---------------------------------------------------------------------------

@njit(cache=True)
def _proj_spectrum(X: np.ndarray, Y: np.ndarray,
                   gamma: float, bilateral: bool,
                   mag_x: np.ndarray, mag_y: np.ndarray) -> np.ndarray:
    """Inner spectral compute for projective_resonance (nopython-safe)."""
    if bilateral:
        ratio = mag_y / (mag_x + 1e-12)
        median_ratio = np.median(ratio)
        if median_ratio > 1e-12:
            rel_weight = ratio / median_ratio
            weight_mag = np.tanh(rel_weight ** gamma).astype(np.complex128)
        else:
            weight_mag = np.ones_like(mag_y, dtype=np.complex128)
    else:
        median_mag = np.median(mag_y)
        if median_mag > 1e-12:
            rel_mag = mag_y / median_mag
            weight_mag = np.tanh(rel_mag ** gamma).astype(np.complex128)
        else:
            weight_mag = np.ones_like(mag_y, dtype=np.complex128)
    return X * Y * weight_mag


def projective_resonance(x: np.ndarray, y: np.ndarray,
                         gamma: float = 1.0,
                         bilateral: bool = False,
                         normalize_output: bool = True) -> np.ndarray:
    X = fft(x)
    Y = fft(y)
    result_spectrum = _proj_spectrum(X, Y, gamma, bilateral, np.abs(X), np.abs(Y))
    result = ifft(result_spectrum).real
    if normalize_output:
        result = normalize(result)
    return result


# ---------------------------------------------------------------------------
# U(s, y): Inverse of Projective Resonance — the "unbinding" for M
# ---------------------------------------------------------------------------

def inverse_projective_resonance(s: np.ndarray, y: np.ndarray,
                                 gamma: float = 1.0,
                                 bilateral: bool = False,
                                 n_iter: int = 20) -> np.ndarray:
    """U(s, y) ≈ x  where s = M(x, y, gamma, bilateral).

    The inverse recovers x from the bound state s and the second operand y.

    UNILATERAL mode (bilateral=False):
      M(x,y) = IFFT(x̂ ⊙ ŷ ⊙ w(|ŷ|))
      Since w depends only on y (known), inversion is EXACT:
        U(s,y) = normalize(IFFT(ŝ / (ŷ ⊙ w(|ŷ|))))

    BILATERAL mode (bilateral=True):
      M(x,y) = IFFT(x̂ ⊙ ŷ ⊙ w(|ŷ|/|x̂|))
      Since w depends on |x̂| (unknown), we use fixed-point iteration:
        |x̂|^(0) = |ŝ| / |ŷ|
        |x̂|^(t+1) = |ŝ| / (|ŷ| · w(|ŷ|/|x̂|^(t)))
      Phase is always recoverable: phase(x̂_k) = phase(ŝ_k) - phase(ŷ_k)

    Args:
        s: The bound state M(x, y) — normalized vector in time domain
        y: The second operand (known)
        gamma: Must match the gamma used in M
        bilateral: Must match the bilateral flag used in M
        n_iter: Number of fixed-point iterations (bilateral only)

    Returns:
        Recovered x — normalized vector approximating the original
    """
    S = fft(s)
    Y = fft(y)

    if not bilateral:
        # ── UNILATERAL: exact inversion (up to normalization loss) ──
        mag_y = np.abs(Y)
        median_mag = np.median(mag_y)

        if median_mag > 1e-12:
            rel_mag = mag_y / median_mag
            weight_mag = np.tanh(rel_mag ** gamma)
        else:
            weight_mag = np.ones_like(mag_y)

        # x̂_k = ŝ_k / (ŷ_k · w_k)
        X_recovered = S / (Y * weight_mag + 1e-12)

    else:
        # ── BILATERAL: fixed-point iteration ──
        mag_y = np.abs(Y)

        # Initial guess: assume w=1 (plain circular correlation)
        X_recovered = S / (Y + 1e-12)
        mag_x = np.abs(X_recovered)

        for _ in range(n_iter):
            # Recompute weights using current magnitude estimate
            ratio = mag_y / (mag_x + 1e-12)
            median_ratio = np.median(ratio)

            if median_ratio > 1e-12:
                rel_weight = ratio / median_ratio
                weight_mag = np.tanh(rel_weight ** gamma)
            else:
                weight_mag = np.ones_like(mag_y)

            # Update x̂ estimate
            X_recovered = S / (Y * weight_mag + 1e-12)
            mag_x = np.abs(X_recovered)

    return normalize(ifft(X_recovered).real)


def recover_from_chain(bound_state: np.ndarray,
                       word_sequence: list[np.ndarray],
                       gamma: float = 1.0,
                       bilateral: bool = False) -> np.ndarray:
    """Recover the FIRST word from a chain-encoded sequence.

    Given: state = M(w1, M(w2, M(w3, ...)))
    and: the full sequence [w1, w2, w3, ...]
    Recover: w1 by peeling off layers from the outside.

    This demonstrates the "infinite context" property:
    the first word of a conversation can be recovered
    from the final state if all subsequent words are known.
    """
    current = bound_state.copy()
    # Peel from last to first
    for w in reversed(word_sequence[1:]):
        current = inverse_projective_resonance(
            current, w, gamma=gamma, bilateral=bilateral
        )
    return current


def recover_first_word(bound_state: np.ndarray,
                       all_words_except_first: list[np.ndarray],
                       gamma: float = 1.0,
                       bilateral: bool = False) -> np.ndarray:
    """Shorthand: recover the first word given the state and remaining words."""
    return recover_from_chain(
        bound_state, [np.zeros_like(all_words_except_first[0])] + all_words_except_first,
        gamma=gamma, bilateral=bilateral
    )


# ---------------------------------------------------------------------------
# Resonance Score: measuring "fit" between state and candidate word
# ---------------------------------------------------------------------------

def resonance_score(state: np.ndarray, word: np.ndarray) -> float:
    """How much does a word resonate with the current state?

    Measures the overlap of dominant frequency components.
    Words whose spectral profile matches the state's profile score higher.

    This replaces attention's Q·K similarity with a spectral-domain
    comparison that runs in O(d) after precomputed FFTs.

    Args:
        state: Current state vector (time domain)
        word: Word vector (time domain)

    Returns:
        Scalar score in [0, 1] — higher = better fit
    """
    S = np.abs(fft(state))
    W = np.abs(fft(word))
    # Cosine similarity of magnitude spectra
    return float(np.dot(S, W) / (np.linalg.norm(S) * np.linalg.norm(W) + 1e-12))


def resonance_scores_batch(state: np.ndarray,
                           word_spectra: np.ndarray) -> np.ndarray:
    """Compute resonance scores for all candidate words simultaneously.

    Args:
        state: Current state vector (time domain)
        word_spectra: Precomputed |FFT| for all words, shape (vocab_size, D)

    Returns:
        Array of scores, shape (vocab_size,)
    """
    S = np.abs(fft(state))
    S_norm = np.linalg.norm(S) + 1e-12
    # Vectorized cosine similarity
    dots = word_spectra @ S  # (vocab_size,)
    word_norms = np.linalg.norm(word_spectra, axis=1)
    return dots / (S_norm * word_norms + 1e-12)


# ---------------------------------------------------------------------------
# Sequence encoding & decoding (scan operations)
# ---------------------------------------------------------------------------

def encode_sequence(words: list[np.ndarray],
                    gamma: float = 1.0,
                    bilateral: bool = False) -> np.ndarray:
    """Encode a sequence of word vectors into a single state vector.

    Uses left-to-right scan: state_i = M(state_{i-1}, word_i)
    This preserves order without requiring associativity.

    Args:
        words: List of word vectors in sequence order
        gamma: Amplification exponent for projective_resonance
        bilateral: If True, use bilateral φ for stronger non-commutativity

    Returns:
        State vector encoding the entire sequence
    """
    if not words:
        return np.zeros(D)
    state = words[0].copy()
    for w in words[1:]:
        state = projective_resonance(state, w, gamma=gamma, bilateral=bilateral)
    return state


def encode_sequence_plain(words: list[np.ndarray]) -> np.ndarray:
    """Encode using plain circular convolution (baseline, no φ)."""
    if not words:
        return np.zeros(D)
    state = words[0].copy()
    for w in words[1:]:
        X = fft(state)
        Y = fft(w)
        state = normalize(ifft(X * Y).real)
    return state


def decode_next_candidates(state: np.ndarray,
                           candidate_vecs: np.ndarray,
                           word_spectra: np.ndarray,
                           temperature: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Score all candidate words for next position given current state.

    Args:
        state: Current state vector
        candidate_vecs: All word vectors, shape (vocab_size, D)
        word_spectra: Precomputed |FFT| for all words, shape (vocab_size, D)

    Returns:
        (scores, probabilities) — arrays of shape (vocab_size,)
    """
    scores = resonance_scores_batch(state, word_spectra)

    # Temperature scaling with auto-calibration:
    # Scale by the spread of scores to avoid temperature being too hot/cold
    score_std = np.std(scores)
    if score_std > 1e-12:
        effective_temp = temperature * score_std
    else:
        effective_temp = temperature

    # Softmax-like via temperature-scaled normalization
    # Avoid overflow by subtracting max
    scores_centered = scores - np.max(scores)
    exp_scores = np.exp(scores_centered / (effective_temp + 1e-12))
    probs = exp_scores / exp_scores.sum()

    return scores, probs


# ---------------------------------------------------------------------------
# Self-calibrating operations (no fixed thresholds)
# ---------------------------------------------------------------------------

def auto_threshold(values: np.ndarray, percentile: float = 90.0) -> float:
    """Self-calibrating threshold at given percentile of distribution."""
    return float(np.percentile(values, percentile))


def competitive_filter(scores: np.ndarray,
                       percentile: float = 90.0) -> np.ndarray:
    """Zero out scores below the auto-calibrated competitive threshold.

    This replaces fixed "top-k" with a data-driven selection:
    only elements scoring above the 90th percentile survive.
    """
    threshold = auto_threshold(scores, percentile)
    filtered = scores.copy()
    filtered[filtered < threshold] = 0.0
    return filtered


# ---------------------------------------------------------------------------
# Phase Rotation Lens — Context-dependent similarity deformation
# ---------------------------------------------------------------------------

def phase_lens(word_vec: np.ndarray,
               context_vec: np.ndarray,
               alpha: float = 0.5) -> np.ndarray:
    """Deform a word vector by rotating its phases toward the context.

    THE CONTEXT LENS — the algebraic equivalent of attention for CELN.

    In the Fourier domain:
      - Magnitude = WHAT the word is (identity, meaning)
      - Phase = HOW the word relates (structure, relationships)

    Phase rotation preserves the word's identity (magnitude) while
    shifting its relational profile (phase) toward the context.
    This changes cosine similarity to other words in a context-
    dependent way — without backprop, without re-training.

    Mathematically:
      result[k] = |FFT(word)[k]| * e^{i((1-α)θ_w[k] + αθ_ctx[k])}

    Properties:
      - α=0: identity (returns original word)
      - α=1: word magnitude + context phase (full deformation)
      - Preserves L2 norm (|result| = |word|)
      - O(D log D) via FFT — runs on CPU at microsecond scale
      - ZERO backprop, ZERO templates, purely algebraic

    Args:
        word_vec: The word vector to deform (normalized, shape (D,))
        context_vec: The context vector to deform toward (normalized, shape (D,))
        alpha: Deformation strength [0, 1].
               0 = no deformation, 1 = full context phase replacement

    Returns:
        Deformed word vector (normalized, shape (D,))
    """
    W = fft(word_vec)
    C = fft(context_vec)

    # Extract magnitude and phase
    W_mag = np.abs(W)
    W_phase = W / (W_mag + 1e-12)   # e^{iθ_w}
    C_phase = C / (np.abs(C) + 1e-12)  # e^{iθ_c}

    # Phase interpolation on the complex unit circle
    # phase_diff = e^{i(θ_c - θ_w)}
    phase_diff = C_phase / (W_phase + 1e-12)

    # new_phase = e^{i(θ_w + α(θ_c - θ_w))} = e^{iθ_w} · (e^{i(θ_c-θ_w)})^α
    new_phase = W_phase * (phase_diff ** alpha)

    # Reconstruct: preserve magnitude, shift phase
    result_spectrum = W_mag * new_phase
    result = ifft(result_spectrum).real
    return normalize(result)


def phase_lens_scores_batch(
    word_vec: np.ndarray,
    context_vec: np.ndarray,
    candidate_spectra: np.ndarray,
    candidate_mags: np.ndarray,
    candidate_phases: np.ndarray,
    query_vec: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Phase-rotate ALL candidate words toward context, score vs query.

    Fully vectorized batch operation. Uses precomputed FFT components
    to avoid per-word FFT calls at generation time.

    Algorithm:
      1. Phase-rotate each candidate toward context: (1-α)·θ_w + α·θ_ctx
      2. Compute similarity to query via Parseval's theorem:
         dot(deformed_word, query) = (1/D) · dot(FFT(deformed), conj(FFT(query)))

    Complexity: O(VD) per call — two matrix multiplications.
    With precomputed spectra: NO FFT calls needed at generation time.

    Args:
        word_vec: Not used directly (kept for API consistency)
        context_vec: The context vector to deform toward (shape (D,))
        candidate_spectra: Precomputed FFT of all candidates (V, D) complex
        candidate_mags: Precomputed |FFT| of all candidates (V, D) real
        candidate_phases: Precomputed FFT/|FFT| of all candidates (V, D) complex
        query_vec: The query vector to compare against (shape (D,))
        alpha: Deformation strength [0, 1]

    Returns:
        Similarity scores for all candidates, shape (V,)
    """
    V, D = candidate_spectra.shape

    # ── Extract context phase ──
    C = fft(context_vec)
    C_phase = C / (np.abs(C) + 1e-12)  # (D,)

    # ── Phase-rotate all candidates at once ──
    # phase_diff = e^{i(θ_c - θ_w)} for each candidate
    phase_diff = C_phase[None, :] / (candidate_phases + 1e-12)  # (V, D)

    # new_phase = e^{iθ_w} · (phase_diff)^α
    rotated_phases = candidate_phases * (phase_diff ** alpha)  # (V, D)

    # Reconstruct deformed FFT: mag * rotated_phase
    deformed_fft = candidate_mags * rotated_phases  # (V, D)

    # ── Score vs query using Parseval ──
    # dot(IFFT(X), y) = (1/D) · dot(X, conj(FFT(y)))
    query_fft_conj = np.conj(fft(query_vec))  # (D,)
    scores = np.real(deformed_fft @ query_fft_conj) / D  # (V,)

    # Normalize by word norms (candidate mags preserve norm, query is unit)
    # |deformed_word| = |original_word| since magnitude preserved
    word_norms = np.linalg.norm(candidate_mags, axis=1) / np.sqrt(D)  # Parseval
    scores = scores / (word_norms + 1e-12)

    return scores.astype(np.float32)


def precompute_word_spectra(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute FFT components for all word vectors.

    Returns:
        (spectra, mags, phases) where:
          spectra: complex FFT of each word, shape (V, D)
          mags: |FFT| of each word, shape (V, D) real
          phases: FFT/|FFT| of each word, shape (V, D) complex (unit circle)
    """
    V, D = vectors.shape
    spectra = np.zeros((V, D), dtype=np.complex128)
    mags = np.zeros((V, D), dtype=np.float64)
    phases = np.zeros((V, D), dtype=np.complex128)

    for i in range(V):
        s = fft(vectors[i].astype(np.float64))
        spectra[i] = s
        m = np.abs(s)
        mags[i] = m
        phases[i] = s / (m + 1e-12)

    return spectra.astype(np.complex64), mags.astype(np.float32), phases.astype(np.complex64)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def make_random_vector(seed: int = None) -> np.ndarray:
    """Generate a random normalized vector (for word initialization)."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    v = rng.randn(D)
    return normalize(v)


def spectral_entropy(v: np.ndarray) -> float:
    """Entropy of the magnitude spectrum — measures frequency concentration.

    Low entropy → few dominant frequencies (peaked)
    High entropy → flat spectrum (noisy / unfocused)
    """
    mag = np.abs(fft(v))
    mag = mag / (mag.sum() + 1e-12)
    mag = mag[mag > 1e-12]
    return float(-np.sum(mag * np.log(mag)))
