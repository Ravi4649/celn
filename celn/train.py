"""
CELN v3 — Word Vector Training
===============================
Learn 10k-dimensional word vectors from the corpus using PMI-based
Hebbian updates. No backprop, no gradients — just co-occurrence
statistics projected into the vector space.

Algorithm:
  1. Tokenize corpus, build vocabulary
  2. Count co-occurrences in sliding window
  3. Compute PPMI (Positive Pointwise Mutual Information)
  4. Initialize random normalized vectors
  5. Hebbian update: v_i += η * Σ_j PPMI(i,j) * v_j
  6. Normalize and iterate until convergence
"""

import re
import numpy as np
from collections import Counter, defaultdict
from typing import Optional

from .core import D, normalize, batch_normalize, make_random_vector

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize(text: str, min_len: int = 2) -> list[str]:
    """Simple Portuguese-aware tokenizer.

    Lowercases, splits on whitespace/punctuation.

    Args:
        text: Input text in Portuguese.
        min_len: Minimum word length to keep (default 2).
                 Use min_len=1 to include articles/prepositions
                 ('a', 'o', 'e', 'é') for fluent generation.
    """
    text = text.lower()
    tokens = re.findall(rf'[a-zà-úçãõâêôéáíóú]{{{min_len},}}', text)
    return tokens


# ---------------------------------------------------------------------------
# Co-occurrence statistics
# ---------------------------------------------------------------------------

def build_cooccurrence(sentences: list[list[str]],
                       window_size: int = 5) -> tuple[dict[str, int],
                                                       dict[tuple[int, int], float],
                                                       dict[str, int],
                                                       dict[int, str]]:
    """Build co-occurrence statistics from tokenized sentences.

    Args:
        sentences: List of tokenized sentences
        window_size: Symmetric context window (words on each side)

    Returns:
        word_counts: {word: count}
        cooc_counts: {(word_i_idx, word_j_idx): count}
        word2idx: {word: index}
        idx2word: {index: word}
    """
    word_counts = Counter()
    for sent in sentences:
        word_counts.update(sent)

    # Build vocabulary: words appearing at least min_count times
    min_count = max(2, int(len(sentences) * 0.001))  # dynamic threshold
    vocab_words = [w for w, c in word_counts.items() if c >= min_count]
    word2idx = {w: i for i, w in enumerate(vocab_words)}
    idx2word = {i: w for i, w in enumerate(vocab_words)}

    # Count co-occurrences
    cooc_counts = Counter()
    half_window = window_size // 2

    for sent in sentences:
        indices = [word2idx[w] for w in sent if w in word2idx]
        for i, center in enumerate(indices):
            start = max(0, i - half_window)
            end = min(len(indices), i + half_window + 1)
            for j in range(start, end):
                if i != j:
                    ctx = indices[j]
                    cooc_counts[(center, ctx)] += 1

    return dict(word_counts), dict(cooc_counts), word2idx, idx2word


# ---------------------------------------------------------------------------
# PPMI computation
# ---------------------------------------------------------------------------

def compute_ppmi(word_counts: dict[str, int],
                 cooc_counts: dict[tuple[int, int], float],
                 word2idx: dict[str, int],
                 shift: float = 1.0) -> np.ndarray:
    """Compute Positive Pointwise Mutual Information matrix.

    PPMI(i,j) = max(0, log(P(i,j) / (P(i) * P(j))))

    Args:
        shift: Smoothing shift for log (higher = more conservative)

    Returns:
        PPMI matrix of shape (vocab_size, vocab_size)
    """
    vocab_size = len(word2idx)
    total_cooc = sum(cooc_counts.values())

    # Word probabilities (handle words in vocab that don't appear in counts)
    total_words = sum(word_counts.get(w, 0) for w in word2idx)
    word_probs = np.zeros(vocab_size)
    for w, idx in word2idx.items():
        word_probs[idx] = word_counts.get(w, 0) / max(total_words, 1)

    # Build PPMI matrix
    ppmi = np.zeros((vocab_size, vocab_size), dtype=np.float32)

    for (i, j), count in cooc_counts.items():
        p_ij = count / total_cooc
        p_i = word_probs[i]
        p_j = word_probs[j]
        # PMI = log(P(i,j) / (P(i) * P(j)))
        if p_ij > 0 and p_i > 0 and p_j > 0:
            pmi = np.log(p_ij / (p_i * p_j) + 1e-12)
            # Positive PMI with shift
            ppmi[i, j] = max(0.0, pmi - shift)

    return ppmi


# ---------------------------------------------------------------------------
# Hebbian vector learning
# ---------------------------------------------------------------------------

def initialize_vectors(vocab_size: int, dim: int = D) -> np.ndarray:
    """Initialize word vectors randomly on the unit hypersphere."""
    vecs = np.random.randn(vocab_size, dim).astype(np.float32)
    return batch_normalize(vecs)


def hebbian_update(vectors: np.ndarray,
                   ppmi: np.ndarray,
                   learning_rate: float = 0.05) -> np.ndarray:
    """One step of Hebbian learning.

    For each word i:
        v_i += η * Σ_j PPMI(i,j) * v_j

    This pulls word vectors toward the vectors of words they
    frequently co-occur with, weighted by PMI strength.

    Vectorized for efficiency.
    """
    # ppmi @ vectors gives Σ_j PPMI(i,j) * v_j for all i simultaneously
    updates = ppmi @ vectors  # (vocab_size, D)
    new_vectors = vectors + learning_rate * updates
    return batch_normalize(new_vectors)


def train_vectors(sentences: list[list[str]],
                  dim: int = D,
                  learning_rate: float = 0.05,
                  epochs: int = 10,
                  window_size: int = 5,
                  verbose: bool = True) -> tuple[np.ndarray, dict, dict, np.ndarray]:
    """Full training pipeline: co-occurrence → PPMI → Hebbian learning.

    Returns:
        word_vectors: (vocab_size, D) array of trained vectors
        word2idx: {word: index} mapping
        idx2word: {index: word} mapping
        ppmi: PPMI matrix for reference
    """
    # Step 1: Co-occurrence statistics
    if verbose:
        print(f"Building co-occurrence from {len(sentences)} sentences...")
    word_counts, cooc_counts, word2idx, idx2word = build_cooccurrence(
        sentences, window_size=window_size
    )
    vocab_size = len(word2idx)
    if verbose:
        print(f"  Vocabulary: {vocab_size} words")
        print(f"  Co-occurrence pairs: {len(cooc_counts)}")

    # Step 2: PPMI matrix
    if verbose:
        print("Computing PPMI matrix...")
    ppmi = compute_ppmi(word_counts, cooc_counts, word2idx)

    # PPMI matrix sparsity
    nonzero_ratio = np.count_nonzero(ppmi) / ppmi.size
    if verbose:
        print(f"  PPMI non-zero: {nonzero_ratio:.2%}")

    # Step 3: Initialize random vectors
    if verbose:
        print(f"Initializing {vocab_size} vectors in {dim}D...")
    vectors = initialize_vectors(vocab_size, dim)

    # Step 4: Hebbian iterations
    if verbose:
        print(f"Training for {epochs} epochs (η={learning_rate})...")

    for epoch in range(epochs):
        vectors = hebbian_update(vectors, ppmi, learning_rate)

        if verbose and (epoch + 1) % 5 == 0:
            # Check convergence: average cosine change
            if epoch >= 5:
                # Sample some nearest neighbors for quality check
                sample_idx = 0
                sims = vectors @ vectors[sample_idx]
                top_k = np.argsort(sims)[-6:-1][::-1]
                top_words = [idx2word[i] for i in top_k]
                print(f"  Epoch {epoch+1}: top neighbors of "
                      f"'{idx2word[0]}': {', '.join(top_words)}")

    # Step 5: Precompute magnitude spectra for fast scoring
    if verbose:
        print("Precomputing word spectra for generation...")

    return vectors, word2idx, idx2word, ppmi


def precompute_spectra(vectors: np.ndarray) -> np.ndarray:
    """Precompute |FFT(word)| for all words (used in resonance scoring)."""
    spectra = np.zeros_like(vectors)
    for i in range(len(vectors)):
        spectra[i] = np.abs(np.fft.fft(vectors[i]))
    return spectra


def train_vectors_rp(sentences: list[list[str]],
                     dim: int = D,
                     window_size: int = 5,
                     seed: int = 42,
                     min_count: int = 2,
                     verbose: bool = True) -> tuple[np.ndarray, dict, dict, np.ndarray]:
    """PPMI → Random Projection: non-collapsing 10k-dim word vectors.

    Steps:
      1. Co-occurrence statistics (same as Hebbian)
      2. PPMI matrix (same)
      3. Random projection matrix R (V × D) with entries ±1/√V
      4. vectors = PPMI @ R  (V × D)
      5. Normalize each row to unit norm

    Properties:
      - ZERO collapse: no iterative updates, transformação linear fixa
      - Preserva similaridades do espaço PPMI (Johnson-Lindenstrauss)
      - O(V²D) tempo, ~0.5s em CPU para V=2526, D=10000
      - Sem backprop, gradients, ou thresholds
    """
    if verbose:
        print(f"Building co-occurrence from {len(sentences)} sentences (min_count={min_count})...")
    
    # Override build_cooccurrence's internal min_count by building manually
    word_counts = Counter()
    for sent in sentences:
        word_counts.update(sent)
    min_count = max(2, int(len(sentences) * 0.001)) if min_count is None else min_count
    vocab_words = [w for w, c in word_counts.items() if c >= min_count]
    word2idx = {w: i for i, w in enumerate(vocab_words)}
    idx2word = {i: w for i, w in enumerate(vocab_words)}
    
    cooc_counts = Counter()
    half_window = window_size // 2
    for sent in sentences:
        indices = [word2idx[w] for w in sent if w in word2idx]
        for i, center in enumerate(indices):
            start = max(0, i - half_window)
            end = min(len(indices), i + half_window + 1)
            for j in range(start, end):
                if i != j:
                    ctx = indices[j]
                    cooc_counts[(center, ctx)] += 1

    word_counts_dict = dict(word_counts)
    cooc_counts_dict = dict(cooc_counts)
    vocab_size = len(word2idx)
    if verbose:
        print(f"  Vocabulary: {vocab_size} words")
        print(f"  Co-occurrence pairs: {len(cooc_counts)}")

    if verbose:
        print("Computing PPMI matrix...")
    ppmi = compute_ppmi(word_counts_dict, cooc_counts_dict, word2idx)
    nonzero_ratio = np.count_nonzero(ppmi) / ppmi.size
    if verbose:
        print(f"  PPMI non-zero: {nonzero_ratio:.2%}")

    if verbose:
        print(f"Random projection to {dim} dimensions...")
    rng = np.random.RandomState(seed)
    scale = 1.0 / np.sqrt(float(vocab_size))
    R = rng.choice([-scale, scale], size=(vocab_size, dim)).astype(np.float32)
    vectors = ppmi.astype(np.float32) @ R
    vectors = batch_normalize(vectors)

    if verbose:
        print(f"  Vectors: {vectors.shape[0]} words × {vectors.shape[1]} dims")

    return vectors, word2idx, idx2word, ppmi


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(filepath: str,
                max_sentences: Optional[int] = None,
                min_len: int = 2) -> list[list[str]]:
    """Load and tokenize the corpus.

    Args:
        filepath: Path to corpus_final.txt
        max_sentences: Cap sentences for quick experiments (None = all)
        min_len: Minimum word length (default 2). Use 1 to include
                 articles and prepositions for fluent generation.

    Returns:
        List of tokenized sentences
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split into sentences (., !, ?, newlines)
    raw_sentences = re.split(r'[.!?\n]+', text)
    sentences = []
    for s in raw_sentences:
        tokens = tokenize(s, min_len=min_len)
        if len(tokens) >= 3:  # meaningful sentences only
            sentences.append(tokens)

    if max_sentences is not None:
        sentences = sentences[:max_sentences]

    return sentences


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from .core import similarity
    import sys

    corpus_path = sys.argv[1] if len(sys.argv) > 1 else 'corpus_final.txt'
    max_sent = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print("=" * 60)
    print("CELN v3 — Word Vector Training")
    print("=" * 60)

    sentences = load_corpus(corpus_path, max_sentences=max_sent)
    print(f"Loaded {len(sentences)} sentences")

    vectors, w2i, i2w, ppmi = train_vectors(
        sentences, verbose=True, epochs=10
    )

    # Quality checks
    print("\nSemantic similarity examples:")
    test_words = ['cobre', 'onça', 'revolução', 'metal', 'animal', 'frança']
    for word in test_words:
        if word in w2i:
            idx = w2i[word]
            sims = [(similarity(vectors[idx], vectors[j]), i2w[j])
                    for j in range(len(vectors)) if j != idx]
            sims.sort(reverse=True)
            print(f"  {word}: {', '.join(w for _, w in sims[:5])}")
