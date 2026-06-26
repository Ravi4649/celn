"""
CELN v3 — HDC Type Vectors
===========================
Pure Hyperdimensional Computing (HDC) type vectors with Hebbian
distributional learning. No semantic leakage, no pre-defined categories.

Core insight (Kanerva, BEAGLE): words that appear in the same POSITIONS
should have similar type vectors. "o" and "a" are both articles not
because someone labeled them, but because they both precede nouns and
follow nothing in particular. "gato" and "cachorro" are both nouns
because they follow articles and precede verbs.

Architecture:
  - H = 4096 dimensional bipolar vectors {-1, +1}^H
  - Initialize: random bipolar (nearly orthogonal)
  - Learn: Hebbian — pull each word's vector toward the centroid
    of its context window's type vectors
  - Bipolarize after each update: sign(x)
  - After N epochs, words with similar distributional behavior
    have similar type vectors (cosine similarity → 1.0)
  - Words with different behavior remain orthogonal (cosine → 0.0)

This works for ANY language or domain — the clusters emerge from
positional distribution, not linguistic categories.
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# HDC Type Vector Training
# ---------------------------------------------------------------------------

def train_hdc_type_vectors(
    sentences: list[list[str]],
    w2i: dict[str, int],
    vocab_size: int,
    hdc_dim: int = 4096,
    context_window: int = 3,
    n_epochs: int = 5,
    learning_rate: float = 0.05,
    seed: int = 42,
    verbose: bool = True,
) -> np.ndarray:
    """Train HDC type vectors via Hebbian distributional learning.

    Algorithm:
      For each word w in the corpus:
        1. Collect type vectors of words within ±context_window
        2. Sum them to form the "context centroid"
        3. Pull w's type vector toward this centroid (Hebbian)
        4. Bipolarize: type_vec[w] = sign(type_vec[w])

    Words that share contexts converge to similar type vectors.
    Words in different contexts remain orthogonal.

    Args:
        sentences: Tokenized sentences.
        w2i: Word-to-index mapping.
        vocab_size: Vocabulary size.
        hdc_dim: HDC vector dimensionality (power of 2 recommended).
        context_window: Number of words to each side for context.
        n_epochs: Training epochs over the corpus.
        learning_rate: Hebbian update strength.
        seed: Random seed for initialization.
        verbose: Print progress.

    Returns:
        Type vectors, shape (V, H), in {-1, +1}^{H}, L2-normalized
        (cosine similarity = dot product / H).
    """
    rng = np.random.RandomState(seed)

    # Initialize: random bipolar vectors {-1, +1}
    # Each word starts orthogonal to all others
    type_vecs = rng.choice(
        [-1.0, 1.0], size=(vocab_size, hdc_dim)
    ).astype(np.float32)

    n_sentences = len(sentences)
    total_updates = 0

    for epoch in range(n_epochs):
        epoch_updates = 0
        # Decay learning rate over epochs
        lr = learning_rate * (1.0 - epoch / (n_epochs + 1))

        for sent_idx, tokens in enumerate(sentences):
            indices = [w2i[w] for w in tokens if w in w2i]
            if len(indices) < 2:
                continue

            for pos, word_idx in enumerate(indices):
                # Context: words within window, excluding the word itself
                start = max(0, pos - context_window)
                end = min(len(indices), pos + context_window + 1)
                context_indices = [
                    indices[j] for j in range(start, end) if j != pos
                ]

                if not context_indices:
                    continue

                # Context centroid: sum of context type vectors (real-valued)
                ctx_centroid = type_vecs[context_indices].sum(axis=0)

                # Hebbian update: pull toward context
                type_vecs[word_idx] += lr * ctx_centroid

                # Bipolarize: threshold at 0 to keep in {-1, +1}
                type_vecs[word_idx] = np.sign(type_vecs[word_idx])
                # sign(0) = 0 in numpy, fix to +1
                type_vecs[word_idx][type_vecs[word_idx] == 0] = 1.0

                epoch_updates += 1

        total_updates += epoch_updates
        if verbose:
            # Compute clustering quality
            avg_sim = _compute_avg_similarity(type_vecs, vocab_size, hdc_dim)
            print(f"    Epoch {epoch+1}/{n_epochs}: {epoch_updates} updates, "
                  f"lr={lr:.3f}, avg_pairwise_sim={avg_sim:.4f}")

    # Normalize so that cosine_sim(a,b) = dot(a,b) / H
    # (bipolar vectors have norm = sqrt(H))
    type_vecs = type_vecs / np.sqrt(hdc_dim)

    if verbose:
        print(f"  Total updates: {total_updates}")
        print(f"  Type vectors: ({vocab_size}, {hdc_dim}), "
              f"values in {{{type_vecs.min():.4f}, {type_vecs.max():.4f}}}")

    return type_vecs


def _compute_avg_similarity(
    type_vecs: np.ndarray, vocab_size: int, hdc_dim: int
) -> float:
    """Average pairwise cosine similarity (sample-based for speed)."""
    sample = min(200, vocab_size)
    indices = np.random.choice(vocab_size, size=sample, replace=False)
    sub = type_vecs[indices]
    sims = sub @ sub.T
    # Exclude diagonal
    mask = ~np.eye(sample, dtype=bool)
    return float(sims[mask].mean())


# ---------------------------------------------------------------------------
# Type Vector Analysis
# ---------------------------------------------------------------------------

def analyze_type_clusters(
    type_vecs: np.ndarray,
    w2i: dict[str, int],
    i2w: dict[int, str],
    words_of_interest: list[str] | None = None,
    top_k: int = 10,
) -> dict:
    """Analyze which words cluster together in type space.

    Args:
        type_vecs: Normalized type vectors (V, H).
        w2i, i2w: Word-index mappings.
        words_of_interest: Words to analyze (default: common words).
        top_k: Number of nearest neighbors to report.

    Returns:
        Dict mapping word → [(neighbor, similarity), ...].
    """
    if words_of_interest is None:
        words_of_interest = [
            'o', 'a', 'os', 'as', 'um', 'uma',
            'de', 'do', 'da', 'em', 'no', 'na', 'por', 'para', 'com',
            'e', 'que', 'se', 'é', 'foi', 'era', 'não',
            'gato', 'cachorro', 'peixe', 'onça', 'cobra', 'lobo',
            'corre', 'nadou', 'voou', 'atacou', 'devorou',
            'cobre', 'metal', 'água', 'coração', 'carro',
        ]

    results = {}
    for word in words_of_interest:
        if word not in w2i:
            continue
        idx = w2i[word]
        vec = type_vecs[idx]
        sims = type_vecs @ vec  # cosine similarity (vectors are normalized)
        top_indices = np.argsort(sims)[-(top_k + 1):][::-1]
        neighbors = [
            (i2w[int(i)], float(sims[i]))
            for i in top_indices
            if int(i) != idx
        ][:top_k]
        results[word] = neighbors

    return results


# ---------------------------------------------------------------------------
# Type Field (learned in type space)
# ---------------------------------------------------------------------------

def learn_type_field(
    sentences: list[list[str]],
    w2i: dict[str, int],
    type_vecs: np.ndarray,
) -> np.ndarray:
    """Learn the TYPE FIELD from corpus transitions.

    For each transition w1 → w2:
      type_field[w1] accumulates type_vec[w2]

    After all transitions: type_field[w1] = centroid of type vectors
    of words that follow w1 in the corpus.

    This field, in HDC type space, points FROM each word's type
    TOWARD the type of its typical followers.

    Args:
        sentences: Tokenized sentences.
        w2i: Word-to-index mapping.
        type_vecs: Normalized type vectors (V, H).

    Returns:
        Type field array, shape (V, H), normalized.
    """
    H = type_vecs.shape[1]
    V = type_vecs.shape[0]
    accum = np.zeros((V, H), dtype=np.float32)
    counts = np.zeros(V, dtype=np.int32)

    for tokens in sentences:
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            if w1 not in w2i or w2 not in w2i:
                continue
            i1, i2 = w2i[w1], w2i[w2]
            accum[i1] += type_vecs[i2]
            counts[i1] += 1

    field = np.zeros((V, H), dtype=np.float32)
    for i in range(V):
        if counts[i] >= 3:
            # Normalize the centroid
            centroid = accum[i] / counts[i]
            norm = np.linalg.norm(centroid)
            if norm > 1e-12:
                field[i] = centroid / norm

    return field
