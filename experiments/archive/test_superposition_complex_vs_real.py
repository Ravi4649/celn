#!/usr/bin/env python3
"""
Compare superposition robustness: Complex fasors (multiplicative binding)
vs Real vectors with circular convolution (HRR-style binding).

Experiment:
 1. Generate N token vectors (complex fasors and real gaussian)
 2. Create positional role vectors for up to L_max positions
 3. For lengths in [10,50,100,200]:
     - Repeat T trials: sample a sequence of length L
     - Build memory = sum_k bind(token_k, role_k)
     - Recover each token by unbinding with role_k
     - Measure similarity(original, recovered)
 4. Report mean±std similarity per length for complex and real

This directly tests whether superposition collapses as context grows.
"""

import time
import numpy as np
from numpy.fft import fft, ifft
import os


def make_complex_fasors(n, D, seed=0):
    rng = np.random.RandomState(seed)
    thetas = rng.rand(n, D) * (2 * np.pi)
    return np.exp(1j * thetas).astype(np.complex128)


def make_real_vectors(n, D, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, D)).astype(np.float64)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X


def normed_similarity(a, b):
    # normalized absolute inner product
    num = abs(np.vdot(a, b))
    den = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(num / den)


def run_experiment(N=1000, D=500, lengths=(10, 50, 100, 200), trials=100, seed=42):
    rng = np.random.RandomState(seed)
    print(f'Generating N={N} tokens, D={D} dims')
    tokens_c = make_complex_fasors(N, D, seed=seed)
    tokens_r = make_real_vectors(N, D, seed=seed + 1)

    Lmax = max(lengths)
    # positional role vectors
    print(f'Generating role vectors for up to L={Lmax}')
    role_c = make_complex_fasors(Lmax, D, seed=seed + 10)  # complex roles
    role_r = make_real_vectors(Lmax, D, seed=seed + 20)    # real roles

    # Precompute FFTs for real baseline
    print('Precomputing FFTs for real baseline...')
    tokens_r_fft = fft(tokens_r, axis=1)
    role_r_fft = fft(role_r, axis=1)

    results = {'complex': {}, 'real': {}}

    t0 = time.time()
    for L in lengths:
        sims_c = []
        sims_r = []
        print(f'Running L={L}, trials={trials} ...')
        for t in range(trials):
            # sample unique tokens for the sequence
            idxs = rng.choice(N, size=L, replace=False)

            # ------- Complex pipeline -------
            # memory = sum_k v_k * p_k
            mem_c = np.zeros(D, dtype=np.complex128)
            for k in range(L):
                mem_c += tokens_c[idxs[k]] * role_c[k]

            # recover each token by mem * conj(role_k)
            for k in range(L):
                rec = mem_c * np.conjugate(role_c[k])
                sim = normed_similarity(tokens_c[idxs[k]], rec)
                sims_c.append(sim)

            # ------- Real (HRR) pipeline -------
            # Do binding in frequency domain: multiply FFTs and sum
            seq_token_fft = tokens_r_fft[idxs]        # (L, D)
            seq_role_fft = role_r_fft[:L]             # (L, D)
            mem_freq = np.sum(seq_token_fft * seq_role_fft, axis=0)  # (D,)
            # recover each token: rec_freq = mem_freq * conj(role_fft)
            for k in range(L):
                rec_freq = mem_freq * np.conjugate(role_r_fft[k])
                rec = ifft(rec_freq)
                rec = rec.real
                sim = normed_similarity(tokens_r[idxs[k]], rec)
                sims_r.append(sim)

        sims_c = np.array(sims_c, dtype=np.float64)
        sims_r = np.array(sims_r, dtype=np.float64)

        results['complex'][L] = (sims_c.mean(), sims_c.std())
        results['real'][L] = (sims_r.mean(), sims_r.std())
        print(f'  Complex mean sim: {results["complex"][L][0]:.6f} ± {results["complex"][L][1]:.6f}')
        print(f'  Real    mean sim: {results["real"][L][0]:.6f} ± {results["real"][L][1]:.6f}')

    t1 = time.time()
    print(f'Elapsed: {t1 - t0:.2f}s')
    return results


if __name__ == '__main__':
    res = run_experiment(N=1000, D=500, lengths=(10, 50, 100, 200), trials=100, seed=42)
    print('\nFinal results:')
    for mode in ('complex', 'real'):
        print(f'--- {mode.upper()} ---')
        for L, (m, s) in res[mode].items():
            print(f'L={L:3d}: mean={m:.6f}, std={s:.6f}')
