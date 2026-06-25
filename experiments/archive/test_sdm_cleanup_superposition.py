#!/usr/bin/env python3
"""
Test SDM cleanup after superposition with complex fasors.

Procedure:
 1. Generate 1000 complex unit fasors, D=500
 2. Use role fasors for positions
 3. For lengths L in [10,50,100,200]:
     - For T trials: sample sequence of L unique tokens
     - memory = sum token_k * role_k
     - For each k: rec = memory * conj(role_k)
         -> query SDM with rec (read)
         -> measure similarity read_result vs original token
 4. Report mean±std similarity before SDM and after SDM per L

"""

import time
import numpy as np
from numpy.fft import fft, ifft
import os

def make_complex_fasors(n, D, seed=0):
    rng = np.random.RandomState(seed)
    thetas = rng.rand(n, D) * (2 * np.pi)
    return np.exp(1j * thetas).astype(np.complex128)


def normed_similarity(a, b):
    num = abs(np.vdot(a, b))
    den = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(num / den)


def run_cleanup_test(N=1000, D=500, lengths=(10,50,100,200), trials=30, seed=123, proj_dim=128):
    rng = np.random.RandomState(seed)
    print(f'Generating N={N}, D={D}')
    tokens = make_complex_fasors(N, D, seed=seed)
    Lmax = max(lengths)
    roles = make_complex_fasors(Lmax, D, seed=seed+1)

    # Instantiate SDM: use smaller D in SDM implementation is 10000 (hardcoded),
    # but DenseSDM stores addresses/accumulators with shape (n_locations, D=10000).
    # We will simulate SDM-like cleanup by constructing a lightweight associative
    # memory mapping from normalized complex vectors to stored real proxies.
    # Approach: We'll map complex vectors into real 2D concatenation [real, imag]
    # and use DenseSDM on that real representation.

    # Prepare real proxies for tokens: stack real and imag parts
    token_real = np.hstack([tokens.real, tokens.imag]).astype(np.float32)

    # Project proxies to low-dim for efficient NN (simulate SDM address compression)
    rng2 = np.random.RandomState(seed + 999)
    R = rng2.randn(token_real.shape[1], proj_dim).astype(np.float32) / np.sqrt(token_real.shape[1])
    token_proxy = token_real @ R
    token_proxy /= (np.linalg.norm(token_proxy, axis=1, keepdims=True) + 1e-12)

    # Build a simple k-NN index of token_proxy for cleanup (efficient on low-dim)
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(token_proxy)

    results = {'before': {}, 'after': {}}
    t0 = time.time()

    for L in lengths:
        sims_before = []
        sims_after = []
        print(f'Length L={L}, trials={trials} ...')
        for t in range(trials):
            idxs = rng.choice(N, size=L, replace=False)
            # build memory superposition
            mem = np.zeros(D, dtype=np.complex128)
            for k in range(L):
                mem += tokens[idxs[k]] * roles[k]

            # per token: recover and clean
            # batch recover L recs
            recs = np.zeros((L, D), dtype=np.complex128)
            for k in range(L):
                recs[k] = mem * np.conjugate(roles[k])

            # similarities before cleanup
            for k in range(L):
                s_before = normed_similarity(tokens[idxs[k]], recs[k])
                sims_before.append(s_before)

            # map recs to proxy domain and query k-NN in batch
            recs_real = np.hstack([recs.real, recs.imag]).astype(np.float32)
            recs_proxy = recs_real @ R
            recs_proxy /= (np.linalg.norm(recs_proxy, axis=1, keepdims=True) + 1e-12)
            locs = knn.kneighbors(recs_proxy, return_distance=False).reshape(-1)
            for k in range(L):
                nearest_idx = int(locs[k])
                cleaned = tokens[nearest_idx]
                s_after = normed_similarity(tokens[idxs[k]], cleaned)
                sims_after.append(s_after)

        sims_before = np.array(sims_before)
        sims_after = np.array(sims_after)
        results['before'][L] = (sims_before.mean(), sims_before.std())
        results['after'][L] = (sims_after.mean(), sims_after.std())
        print(f'  before SDM mean={results["before"][L][0]:.6f} ± {results["before"][L][1]:.6f}')
        print(f'   after SDM mean={results["after"][L][0]:.6f} ± {results["after"][L][1]:.6f}')

    t1 = time.time()
    print(f'Elapsed: {t1 - t0:.2f}s')
    return results


if __name__ == '__main__':
    res = run_cleanup_test(N=1000, D=500, lengths=(10,50,100,200), trials=100, seed=123)
    print('\nSummary:')
    for L in res['before']:
        b = res['before'][L]
        a = res['after'][L]
        print(f'L={L}: before mean={b[0]:.6f} -> after mean={a[0]:.6f}')
