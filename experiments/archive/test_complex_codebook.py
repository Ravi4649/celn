#!/usr/bin/env python3
"""
Quick experiment: Complex fasors + Factorized Codebooks (Kimi proposal)

Steps:
 1. Generate N complex unit vectors (fasors) of dimension D
 2. Build factorized codebooks: B blocks, each of size K and block_dim = D//B
 3. Test binding/unbinding stability in complex domain (should be numerically exact)
 4. Quantize vectors with codebooks and report quantization similarity
 5. Compare semantic gap (related vs unrelated pairs) to a real 10k-D baseline

Principles: zero backprop, pure numpy, CPU.
"""

import time
import numpy as np
import os


def kmeans_complex_block(data, K=64, max_iter=200, seed=1):
    """Simple Lloyd's k-means for complex vectors (per-block).

    data: (n, dim) complex array
    returns: centroids (K, dim) complex unitary per coordinate
    """
    rng = np.random.RandomState(seed)
    n, dim = data.shape
    # init centroids by sampling existing points
    init_idx = rng.choice(n, size=K, replace=False)
    centroids = data[init_idx].copy()
    # normalize centroids to unit phase per coordinate
    centroids = np.exp(1j * np.angle(centroids))

    labels = np.full(n, -1, dtype=np.int32)
    for it in range(max_iter):
        # compute squared distances: sum |x - c|^2 across dims
        # shape (n, K)
        # use broadcasting carefully to avoid huge memory
        diffs = np.abs(data[:, None, :] - centroids[None, :, :]) ** 2
        dists = diffs.sum(axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # recompute centroids
        for k in range(K):
            pts = data[labels == k]
            if pts.shape[0] == 0:
                centroids[k] = data[rng.randint(n)]
            else:
                mean = pts.mean(axis=0)
                # convert mean to unit-phase per coord
                ang = np.angle(mean)
                centroids[k] = np.exp(1j * ang)
    return centroids, labels


def make_complex_fasors(n=500, D=100, seed=1234, dtype=np.complex128):
    rng = np.random.RandomState(seed)
    thetas = rng.rand(n, D) * (2 * np.pi)
    vecs = np.exp(1j * thetas).astype(dtype)
    return vecs


def quantize_with_codebooks(vecs, codebooks):
    # vecs: (n, D) complex
    n, D = vecs.shape
    B = len(codebooks)
    block_dim = D // B
    q = np.zeros_like(vecs)
    for b in range(B):
        cbs = codebooks[b]  # (K, block_dim)
        block = vecs[:, b * block_dim:(b + 1) * block_dim]
        # compute distances in batch: (n, K)
        # avoid huge mem: compute via (a-b)**2 sum
        # using broadcasting is fine here (n small)
        diffs = np.abs(block[:, None, :] - cbs[None, :, :]) ** 2
        dists = diffs.sum(axis=2)
        idx = np.argmin(dists, axis=1)
        q[:, b * block_dim:(b + 1) * block_dim] = cbs[idx]
    return q


def complex_similarity(a, b):
    # a, b: (..., D) complex, returns real similarity in [0, 1]
    # use magnitude-normalized inner product per-vector
    # cos_sim = |<a*, b>| / D
    num = np.abs(np.sum(np.conjugate(a) * b, axis=-1))
    denom = a.shape[-1]
    return (num / denom).astype(float)


def real_cosine_similarity(a, b):
    # a, b: (..., D) real or complex; we treat as real vectors
    # returns normalized dot product
    a_f = a.reshape(a.shape[0], -1)
    b_f = b.reshape(b.shape[0], -1)
    an = np.linalg.norm(a_f, axis=1)
    bn = np.linalg.norm(b_f, axis=1)
    dots = np.sum(a_f * b_f, axis=1)
    return dots / (an * bn + 1e-12)


def baseline_real_space_gap(n=500, seed=1337):
    """Load real 10k vectors from repo and compute related/unrelated gap.

    Related = add small gaussian noise to make near neighbor; unrelated = random other vector.
    """
    rng = np.random.RandomState(seed)
    path = os.path.join(os.path.dirname(__file__), os.pardir, 'celn_v3_full_vectors.npz')
    path = os.path.normpath(path)
    if not os.path.exists(path):
        # try absolute path used elsewhere
        path = '/home/ravizin/celn-v3/celn_v3_full_vectors.npz'
    npz = np.load(path)
    # Try many possible keys
    if 'vectors' in npz:
        V = npz['vectors']
    else:
        # take first array
        key = list(npz.files)[0]
        V = npz[key]
    V = V.astype(np.float64)
    # select n random rows
    tot = V.shape[0]
    idx = rng.choice(tot, size=n, replace=False)
    samp = V[idx, :]
    # normalize
    samp = samp / (np.linalg.norm(samp, axis=1, keepdims=True) + 1e-12)

    # create related: add tiny gaussian noise and renormalize
    noise = rng.normal(scale=0.02, size=samp.shape)
    related = samp + noise
    related = related / (np.linalg.norm(related, axis=1, keepdims=True) + 1e-12)

    # unrelated: shuffle
    perm = rng.permutation(n)
    unrelated = samp[perm]

    rel_sim = np.sum(samp * related, axis=1)  # dot since normalized
    unrel_sim = np.sum(samp * unrelated, axis=1)
    return float(rel_sim.mean()), float(unrel_sim.mean()), float(rel_sim.mean() - unrel_sim.mean())


def main():
    rng = np.random.RandomState(42)
    n = 500
    D = 100
    B = 10
    block_dim = D // B
    K = 64

    print('\n=== Complex Fasors + Factorized Codebooks Test ===')
    t0 = time.time()

    # Step 1: generate fasors
    vecs = make_complex_fasors(n=n, D=D, seed=42, dtype=np.complex128)
    print(f'Generated {n} complex unit fasors, D={D}')

    # Step 2: build codebooks per block
    codebooks = []
    for b in range(B):
        block = vecs[:, b * block_dim:(b + 1) * block_dim]
        print(f'Clustering block {b+1}/{B} (shape {block.shape}) ...', end=' ')
        cb, labels = kmeans_complex_block(block, K=K, max_iter=100, seed=100 + b)
        codebooks.append(cb)
        print('done')

    # Step 3: binding/unbinding stability
    m = 200
    pairs = rng.randint(0, n, size=(m, 2))
    errs = []
    for i, j in pairs:
        a = vecs[i]
        b = vecs[j]
        bound = a * b
        rec = bound * np.conjugate(b)
        err = np.max(np.abs(rec - a))
        errs.append(err)
    errs = np.array(errs)
    print('\nBinding/unbinding stability (complex fasors):')
    print('  pairs:', m)
    print('  max error:', errs.max())
    print('  mean error:', errs.mean())

    # check threshold
    ok_reconstruction = float(errs.max() < 1e-6)
    print('  reconstruction < 1e-6?', bool(ok_reconstruction))

    # Step 4: quantize and measure quantization fidelity
    q = quantize_with_codebooks(vecs, codebooks)
    sims = complex_similarity(vecs, q)
    print('\nQuantization results:')
    print('  mean similarity (orig vs quantized):', float(sims.mean()))
    print('  median similarity:', float(np.median(sims)))
    print('  fraction exact matches (1.0):', float(np.sum(np.isclose(sims, 1.0)) / len(sims)))

    # Step 5: similarity gap test
    # Generate related vectors by small per-dimension phase perturbation
    noise = rng.normal(scale=0.2, size=(n, D))  # radians
    related = vecs * np.exp(1j * noise)
    rel_sims = complex_similarity(vecs, related)
    # unrelated pairs: random pairing
    perm = rng.permutation(n)
    unrelated = vecs[perm]
    unrel_sims = complex_similarity(vecs, unrelated)
    print('\nComplex space similarity:')
    print('  related mean sim:', float(rel_sims.mean()))
    print('  unrelated mean sim:', float(unrel_sims.mean()))
    print('  gap (related - unrelated):', float(rel_sims.mean() - unrel_sims.mean()))

    # Baseline: real 10k-D SVD vectors
    try:
        rel_b, unrel_b, gap_b = baseline_real_space_gap(n=n, seed=123)
        print('\nBaseline real 10k-D (sampled) similarity:')
        print('  related mean sim:', rel_b)
        print('  unrelated mean sim:', unrel_b)
        print('  gap (related - unrelated):', gap_b)
    except Exception as e:
        print('\nBaseline real space test failed:', e)

    t1 = time.time()
    print(f'\nTotal time: {t1 - t0:.2f}s')


if __name__ == '__main__':
    main()
