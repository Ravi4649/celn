"""Minimal comparison test between Hadamard‑based binding (HLB) and
Projective‑Resonance (PR) on low‑dimensional vectors.

The test runs a small number of trials (50) with vectors of size 1024 – a
power‑of‑two dimension suitable for the Fast Walsh‑Hadamard Transform (FWHT).

For each trial we:
  * generate two random normalized vectors ``x`` and ``y``
  * bind them with HLB and with PR
  * unbind to recover ``y``
  * record L2 reconstruction error and cosine similarity
  * run a discrimination sub‑test: among 10 candidate vectors (the true ``y``
    plus nine random vectors) we bind‑and‑unbind each candidate and measure the
    L2 distance to the true ``y``.  The correct candidate should have the smallest
    distance.  We count how many trials each method gets right.

The results are printed to ``stdout`` – this file is meant to be executed
directly (``python experiments/test_hlb_minimal.py``).
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

# Import the Projective Resonance implementation from the library.
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from celn.core import projective_resonance, similarity
from celn.resonator import unbind_M_reverse

# ---------------------------------------------------------------------------
# Fast Walsh‑Hadamard Transform (FWHT)
# ---------------------------------------------------------------------------

def fwht(a: np.ndarray) -> np.ndarray:
    """In‑place Fast Walsh‑Hadamard Transform.

    The implementation follows the classic O(N log N) algorithm where ``a``
    must have a length that is a power of two.  The transform is its own inverse
    up to a scaling factor of ``len(a)`` – we therefore apply the scaling when
    required (see ``hadamard_bind``).
    """
    h = 1
    n = a.shape[0]
    a = a.copy()
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return a


def hadamard_bind(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bind two vectors using the Hadamard (XOR‑convolution) operation.

    ``x`` and ``y`` must be 1‑D arrays of equal length ``N`` where ``N`` is a power
    of two.  The binding is defined as the inverse FWHT of the element‑wise
    product of the FWHTs of the operands, normalized by ``N``.
    """
    n = x.shape[0]
    fx = fwht(x)
    fy = fwht(y)
    bound = fwht(fx * fy) / n  # inverse transform (same as forward) with scaling
    return bound


def hadamard_unbind(bundle: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Recover ``y`` from ``bundle = bind(x, y)`` using the Hadamard inverse.

    The operation performs element‑wise division in the transform domain.
    ``x`` must have non‑zero entries in its FWHT (true for random vectors).
    """
    n = x.shape[0]
    fx = fwht(x)
    fb = fwht(bundle)
    # Avoid division by zero – add a tiny epsilon.
    recovered_fwht = fb / (fx + 1e-12)
    recovered = fwht(recovered_fwht) / n
    return recovered

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def random_normalized(dim: int) -> np.ndarray:
    vec = np.random.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-12)

def l2_error(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_trials(
    dim: int = 1024,
    n_trials: int = 50,
    n_candidates: int = 10,
) -> Tuple[dict, dict]:
    """Run the comparison and return dictionaries with aggregated metrics.

    Returns ``(hlb_stats, pr_stats)`` where each dict contains the mean L2
    error, mean cosine similarity and the count of correct discrimination
    decisions.
    """
    hlb_errors = []
    pr_errors = []
    hlb_cos = []
    pr_cos = []
    hlb_correct = 0
    pr_correct = 0

    for _ in range(n_trials):
        x = random_normalized(dim)
        y = random_normalized(dim)

        # ---- HLB binding / unbinding ----
        bound_hlb = hadamard_bind(x, y)
        rec_hlb = hadamard_unbind(bound_hlb, x)
        hlb_errors.append(l2_error(rec_hlb, y))
        hlb_cos.append(similarity(rec_hlb, y))

        # ---- PR binding / unbinding ----
        bound_pr = projective_resonance(x, y, bilateral=False)
        rec_pr = unbind_M_reverse(bound_pr, x, bilateral=False)
        pr_errors.append(l2_error(rec_pr, y))
        pr_cos.append(similarity(rec_pr, y))

        # ---- Discrimination test ----
        candidates = [y]
        while len(candidates) < n_candidates:
            candidates.append(random_normalized(dim))
        np.random.shuffle(candidates)

        # HLB discrimination
        dists_hlb = []
        for cand in candidates:
            bound = hadamard_bind(x, cand)
            rec = hadamard_unbind(bound, x)
            dists_hlb.append(l2_error(rec, y))
        if np.argmin(dists_hlb) == 0:
            hlb_correct += 1

        # PR discrimination
        dists_pr = []
        for cand in candidates:
            bound = projective_resonance(x, cand, bilateral=False)
            rec = unbind_M_reverse(bound, x, bilateral=False)
            dists_pr.append(l2_error(rec, y))
        if np.argmin(dists_pr) == 0:
            pr_correct += 1

    hlb_stats = {
        "mean_l2": float(np.mean(hlb_errors)),
        "mean_cos": float(np.mean(hlb_cos)),
        "correct": hlb_correct,
    }
    pr_stats = {
        "mean_l2": float(np.mean(pr_errors)),
        "mean_cos": float(np.mean(pr_cos)),
        "correct": pr_correct,
    }
    return hlb_stats, pr_stats

if __name__ == "__main__":
    np.random.seed(42)
    hlb_stats, pr_stats = run_trials()
    print("=== Hadamard‑Based Binding (HLB) ===")
    print(f"Mean L2 error      : {hlb_stats['mean_l2']:.6f}")
    print(f"Mean cosine sim.  : {hlb_stats['mean_cos']:.6f}")
    print(f"Discrimination correct out of 50 trials: {hlb_stats['correct']}/50")
    print("\n=== Projective Resonance (PR) ===")
    print(f"Mean L2 error      : {pr_stats['mean_l2']:.6f}")
    print(f"Mean cosine sim.  : {pr_stats['mean_cos']:.6f}")
    print(f"Discrimination correct out of 50 trials: {pr_stats['correct']}/50")
