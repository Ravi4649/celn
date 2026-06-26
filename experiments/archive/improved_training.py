#!/usr/bin/env python3
"""
Train word vectors using SVD on PPMI matrix + spectral initialization.
This produces much better word vectors than pure Hebbian updates.

The key idea: instead of learning vectors via iterative Hebbian updates,
compute the PPMI matrix and use Truncated SVD to get low-rank embeddings.
This is essentially a simplified GloVe without the weighting function.
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celn.core import D, normalize, batch_normalize, similarity
from celn.train import load_corpus, tokenize, build_cooccurrence, compute_ppmi


def train_svd_vectors(sentences, dim=D, verbose=True):
    """Train word vectors via Truncated SVD on PPMI matrix.

    This is mathematically equivalent to a first-order approximation
    of the PMI factorization that word2vec and GloVe perform,
    but computed directly via SVD without backprop or SGD.
    """
    # Build co-occurrence
    if verbose:
        print("Building co-occurrence statistics...")
    word_counts, cooc_counts, w2i, i2w = build_cooccurrence(
        sentences, window_size=5
    )
    vocab_size = len(w2i)
    if verbose:
        print(f"  Vocabulary: {vocab_size} words")
        print(f"  Co-occurrence pairs: {len(cooc_counts)}")

    # Compute PPMI
    if verbose:
        print("Computing PPMI...")
    ppmi = compute_ppmi(word_counts, cooc_counts, w2i)
    nonzero = np.count_nonzero(ppmi) / ppmi.size
    if verbose:
        print(f"  PPMI density: {nonzero:.2%}")

    # Truncated SVD
    if verbose:
        print(f"Running Truncated SVD ({vocab_size} x {vocab_size} → {dim})...")
    t0 = time.time()

    # For efficiency: if vocab is large, use randomized SVD
    try:
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=min(dim, vocab_size - 1),
                          n_iter=7, random_state=42)
        vectors = svd.fit_transform(ppmi)
    except ImportError:
        # Fallback: numpy SVD
        U, S, Vt = np.linalg.svd(ppmi.astype(np.float64), full_matrices=False)
        k = min(dim, vocab_size - 1)
        vectors = U[:, :k] * S[:k]

    if verbose:
        print(f"  SVD completed in {time.time() - t0:.1f}s")
        print(f"  Vectors shape: {vectors.shape}")

    # Spectral weighting: scale by singular values to emphasize dominant dimensions
    # (This is a form of "spectral attention" applied to the embedding space)
    variances = np.var(vectors, axis=0)
    # Sort dimensions by variance, boost top dimensions
    spectral_weights = np.sqrt(variances)
    spectral_weights = spectral_weights / spectral_weights.max()

    # Apply spectral weighting with auto-calibrated emphasis
    # Top dimensions (high variance) get boosted; low dims get suppressed
    median_weight = np.median(spectral_weights)
    emphasis = np.where(
        spectral_weights > median_weight,
        spectral_weights / median_weight,  # boost top
        spectral_weights / median_weight   # suppress bottom
    )
    emphasis = np.tanh(np.sqrt(emphasis))  # soft clip

    vectors = vectors * emphasis
    vectors = batch_normalize(vectors)

    return vectors.astype(np.float32), w2i, i2w, ppmi


def main():
    corpus_path = sys.argv[1] if len(sys.argv) > 1 else 'corpus_final.txt'
    max_sent = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print("=" * 60)
    print("CELN v3 — SVD-based Word Vector Training")
    print("=" * 60)

    sentences = load_corpus(corpus_path, max_sentences=max_sent)
    print(f"Loaded {len(sentences)} sentences")

    vectors, w2i, i2w, ppmi = train_svd_vectors(sentences, verbose=True)

    # Quality check
    print("\nSemantic quality (nearest neighbors):")
    test_words = ['cobre', 'onça', 'revolução', 'metal', 'animal', 'frança',
                  'água', 'elétrico', 'floresta', 'rei', 'python', 'gato']
    for word in test_words:
        if word in w2i:
            idx = w2i[word]
            sims = [(similarity(vectors[idx], vectors[j]), i2w[j])
                    for j in range(len(vectors)) if j != idx]
            sims.sort(reverse=True)
            print(f"  {word:>12} → {', '.join(w for _, w in sims[:6])}")

    # Save vectors
    outfile = f'celn_vectors_{len(w2i)}.npz'
    np.savez(outfile, vectors=vectors, vocab=np.array(list(w2i.keys())))
    print(f"\nSaved vectors to {outfile}")

    return vectors, w2i, i2w, ppmi


if __name__ == '__main__':
    main()
