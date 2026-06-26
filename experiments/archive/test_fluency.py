"""
Test Fluent Generator — CELN v3 (Hybrid Approach)
===================================================
Hybrid approach: use high-quality SVD-trained vectors (existing .npz)
for semantic channel + add 1-char function words for structural channel.

Key insight: we don't need perfect semantic vectors for function words
because the directional bigram channel handles their placement.
The semantic channel only needs good vectors for content words.
"""

import sys
import os
import re
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celn.train import tokenize, build_cooccurrence, compute_ppmi
from celn.core import normalize, similarity, batch_normalize, make_random_vector
from celn.fluency import (
    build_directional_bigrams,
    DirectionalGenerator,
)
from celn.generate import ContextWindow, generate


# Portuguese function words (for evaluation only, never for generation/scoring)
FUNCTION_WORDS = {
    'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas',
    'de', 'do', 'da', 'dos', 'das',
    'em', 'no', 'na', 'nos', 'nas', 'num', 'numa',
    'por', 'pelo', 'pela', 'pelos', 'pelas',
    'para', 'pra', 'pro', 'pros',
    'com', 'sem', 'sob', 'sobre', 'entre', 'até',
    'e', 'ou', 'mas', 'que', 'se', 'nem',
    'é', 'foi', 'era', 'são', 'está', 'ser',
    'não', 'sim',
    'me', 'te', 'lhe', 'nos', 'vos',
    'este', 'essa', 'isto', 'isso', 'aquele',
    'ele', 'ela', 'eles', 'elas',
    'muito', 'pouco', 'mais', 'menos',
    'como', 'quando', 'onde', 'porque',
}


# ---------------------------------------------------------------------------
# Phase 1: Load data - hybrid approach
# ---------------------------------------------------------------------------

def train_svd_vectors(sentences, dim=10000, verbose=True):
    """Train word vectors using Truncated SVD on PPMI matrix.

    Based on experiments/improved_training.py approach.
    """
    from sklearn.decomposition import TruncatedSVD

    word_counts, cooc_counts, w2i, i2w = build_cooccurrence(sentences, window_size=5)
    vocab_size = len(w2i)

    ppmi = compute_ppmi(word_counts, cooc_counts, w2i)

    # Truncated SVD
    n_components = min(dim, vocab_size - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    vecs_reduced = svd.fit_transform(ppmi)  # (vocab_size, n_components)

    # Weight by singular value variance
    singular_values = svd.singular_values_
    var_ratio = singular_values ** 2 / (singular_values ** 2).sum()
    weights = var_ratio / var_ratio.max()
    vecs_weighted = vecs_reduced * weights[None, :]

    if verbose:
        print(f"    SVD: {n_components} components, "
              f"explained variance: {var_ratio[:50].sum():.1%}")

    # Expand to target dimension via random projection
    if n_components < dim:
        rng = np.random.RandomState(42)
        R = rng.randn(n_components, dim) / np.sqrt(n_components)
        vectors = vecs_weighted @ R
    else:
        vectors = vecs_weighted

    vectors = batch_normalize(vectors)
    return vectors, ppmi, w2i, i2w


def phase1_load_svd():
    """Load corpus with min_len=1 and train SVD vectors on full vocab."""
    print("=" * 72)
    print("PHASE 1: Load Corpus (min_len=1) and Train SVD Vectors")
    print("=" * 72)

    corpus_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'corpus_final.txt'
    )
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    raw_sentences = re.split(r'[.!?\n]+', text)
    sentences_full = []
    for s in raw_sentences:
        tokens = tokenize(s, min_len=1)
        if len(tokens) >= 3:
            sentences_full.append(tokens)

    print(f"  Loaded {len(sentences_full)} sentences (min_len=1)")

    all_words = [w for s in sentences_full for w in s]
    unique = set(all_words)
    func_in_vocab = FUNCTION_WORDS & unique
    print(f"  Unique words: {len(unique)}")
    print(f"  Function words available: {len(func_in_vocab)} (e.g. {sorted(func_in_vocab)[:10]})")

    print(f"\n  Training SVD word vectors...")
    vectors, ppmi, w2i, i2w = train_svd_vectors(sentences_full, dim=10000)
    print(f"  Vectors: {vectors.shape}")

    return sentences_full, vectors, ppmi, w2i, i2w


# ---------------------------------------------------------------------------
# Phase 2: Build directional bigrams
# ---------------------------------------------------------------------------

def phase2_build_bigrams(sentences, w2i, i2w, vocab_size):
    """Build directional bigram probability matrix."""
    print()
    print("=" * 72)
    print("PHASE 2: Build Directional Bigram Model")
    print("=" * 72)

    bigram_prob = build_directional_bigrams(sentences, w2i, vocab_size, smoothing=0.01)
    print(f"  Bigram matrix: {bigram_prob.shape}")

    non_zero = (bigram_prob > 0.001).sum()
    print(f"  Non-zero entries (>0.1%): {non_zero} "
          f"({non_zero / (vocab_size * vocab_size):.2%} density)")

    common_words = ['o', 'a', 'de', 'do', 'é', 'um', 'cobre', 'onça']
    for w in common_words:
        if w in w2i:
            idx = w2i[w]
            top5 = np.argsort(bigram_prob[idx])[-5:][::-1]
            transitions = [(i2w[i], float(bigram_prob[idx][i]))
                          for i in top5 if bigram_prob[idx][i] > 0.01]
            print(f"  '{w}' → {transitions}")

    return bigram_prob


# ---------------------------------------------------------------------------
# Phase 3: Compare generators
# ---------------------------------------------------------------------------

def phase3_compare(vectors, bigram_prob, w2i, i2w):
    """Compare old generator vs new DirectionalGenerator.

    Note: the old generator only works with the old vocabulary,
    so we map generated tokens accordingly.
    """
    print()
    print("=" * 72)
    print("PHASE 3: Compare Generators — Old vs New")
    print("=" * 72)

    test_prefixes = [
        "o cobre é um",
        "a onça pintada",
        "a revolução francesa",
        "python é uma linguagem",
        "o coração humano",
        "o carro elétrico",
        "a água do rio",
        "o gato doméstico",
        "a fotossíntese",
        "o leite materno",
    ]

    gen_length = 10
    temperature = 0.8
    seed = 42

    results = []

    for prefix_str in test_prefixes:
        prefix_tokens = tokenize(prefix_str, min_len=1)
        prefix_known = [w for w in prefix_tokens if w in w2i]

        if len(prefix_known) < 2:
            print(f"\n  Prefix '{prefix_str}' — too few known words, skipping")
            continue

        # --- OLD Generator (baseline, old vocab only) ---
        # Map prefix to old vocab for fair comparison
        old_vocab_words = {w for w in w2i if len(w) >= 2}
        prefix_old = [w for w in prefix_known if w in old_vocab_words]
        if len(prefix_old) < 2:
            prefix_old = prefix_known[:2]  # fallback

        try:
            # Old generator uses old vocab vectors from the .npz
            # We need to create a compatible setup
            from celn.generate import ContextWindow as CW
            from celn.core import normalize

            # Build old PPMI from co-occurrence only within old vocab
            # For simplicity, we rebuild a PPMI limited to new vocab
            # that maps correctly for the old generate function
            old_window = CW(max_window=8, decay=0.7)
            old_pref_idx = [w2i[w] for w in prefix_old]
            for idx in old_pref_idx:
                old_window.add(vectors[idx])

            # Build a PPMI-like transition matrix from our bigram model
            # For fair comparison: use a symmetric PPMI from co-occurrence
            from celn.train import build_cooccurrence, compute_ppmi
            # Just use the bigram_prob as ppmi for testing (both old and new
            # generators get the same transition info)
            old_ppmi = np.maximum(bigram_prob, bigram_prob.T)  # symmetrize

            old_generated, _ = generate(
                old_window, vectors, i2w, w2i, old_ppmi,
                max_len=gen_length, temperature=temperature,
                boost_weight=0.3, seed=seed
            )
        except Exception as e:
            old_generated = []
            print(f"  OLD generator ERROR: {e}")

        # --- NEW Generator (directional bigram + auto-calibrated blend) ---
        try:
            new_gen = DirectionalGenerator(
                vectors, bigram_prob, w2i, i2w,
                window_size=8, window_decay=0.7,
                base_structure_weight=0.35,
            )
            new_generated = new_gen.generate(
                prefix_known, max_len=gen_length,
                temperature=temperature, seed=seed
            )
        except Exception as e:
            new_generated = []
            print(f"  NEW generator ERROR: {e}")

        # --- Metrics ---
        print(f"\n  {'─'*66}")
        print(f"  Prefix: '{' '.join(prefix_known)}'")
        print(f"  {'─'*66}")

        # Function word ratio
        old_func = sum(1 for w in old_generated if w in FUNCTION_WORDS) / max(len(old_generated), 1)
        new_func = sum(1 for w in new_generated if w in FUNCTION_WORDS) / max(len(new_generated), 1)

        # Bigram authenticity
        if len(old_generated) >= 2:
            old_bigrams = list(zip(old_generated[:-1], old_generated[1:]))
            old_auth = sum(1 for (a, b) in old_bigrams
                          if a in w2i and b in w2i and
                          bigram_prob[w2i[a], w2i[b]] > 0.01) / len(old_bigrams)
        else:
            old_auth = 0.0
        if len(new_generated) >= 2:
            new_bigrams = list(zip(new_generated[:-1], new_generated[1:]))
            new_auth = sum(1 for (a, b) in new_bigrams
                          if a in w2i and b in w2i and
                          bigram_prob[w2i[a], w2i[b]] > 0.01) / len(new_bigrams)
        else:
            new_auth = 0.0

        # Topic coherence
        p_indices = [w2i[w] for w in prefix_known if w in w2i]
        if p_indices:
            p_centroid = normalize(vectors[p_indices].mean(axis=0))
            old_indices = [w2i[w] for w in old_generated if w in w2i]
            new_indices = [w2i[w] for w in new_generated if w in w2i]
            old_coh = float(np.dot(p_centroid, normalize(vectors[old_indices].mean(axis=0)))) if old_indices else 0.0
            new_coh = float(np.dot(p_centroid, normalize(vectors[new_indices].mean(axis=0)))) if new_indices else 0.0
        else:
            old_coh = new_coh = 0.0

        print(f"  OLD: {' '.join(old_generated)}")
        print(f"  NEW: {' '.join(new_generated)}")
        print(f"  Func%: OLD={old_func:.0%}  NEW={new_func:.0%}")
        print(f"  BigramAuth: OLD={old_auth:.0%}  NEW={new_auth:.0%}")
        print(f"  TopicCoh: OLD={old_coh:.3f}  NEW={new_coh:.3f}")

        results.append({
            'prefix': prefix_known,
            'old_func': old_func, 'new_func': new_func,
            'old_auth': old_auth, 'new_auth': new_auth,
            'old_coh': old_coh, 'new_coh': new_coh,
        })

    return results


# ---------------------------------------------------------------------------
# Phase 4: Analyze
# ---------------------------------------------------------------------------

def phase4_analyze(results):
    """Aggregate metrics."""
    print()
    print("=" * 72)
    print("PHASE 4: Aggregate Analysis")
    print("=" * 72)

    old_func = [r['old_func'] for r in results]
    new_func = [r['new_func'] for r in results]
    old_auth = [r['old_auth'] for r in results]
    new_auth = [r['new_auth'] for r in results]
    old_coh = [r['old_coh'] for r in results]
    new_coh = [r['new_coh'] for r in results]

    print(f"\n  {'Metric':<28} {'OLD':>10} {'NEW':>10} {'Δ':>10}")
    print(f"  {'─'*28} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'Function Word Ratio':<28} {np.mean(old_func):>10.1%} {np.mean(new_func):>10.1%} {np.mean(new_func)-np.mean(old_func):>+10.1%}")
    print(f"  {'Bigram Authenticity':<28} {np.mean(old_auth):>10.1%} {np.mean(new_auth):>10.1%} {np.mean(new_auth)-np.mean(old_auth):>+10.1%}")
    print(f"  {'Topic Coherence':<28} {np.mean(old_coh):>10.4f} {np.mean(new_coh):>10.4f} {np.mean(new_coh)-np.mean(old_coh):>+10.4f}")

    return {
        'func_delta': np.mean(new_func) - np.mean(old_func),
        'auth_delta': np.mean(new_auth) - np.mean(old_auth),
        'coh_delta': np.mean(new_coh) - np.mean(old_coh),
    }


# ===================================================================
# Main
# ===================================================================

def main():
    print("╔" + "═" * 70 + "╗")
    print("║  Fluent Generator — CELN v3 (Hybrid Vectors)                  ║")
    print("║  SVD content vectors + directional bigram structure           ║")
    print("╚" + "═" * 70 + "╝")

    # Phase 1: Load and train SVD vectors
    sentences, vectors, ppmi, w2i, i2w = phase1_load_svd()

    # Phase 2: Build bigrams
    bigram_prob = phase2_build_bigrams(sentences, w2i, i2w, vectors.shape[0])

    # Phase 3: Compare
    results = phase3_compare(vectors, bigram_prob, w2i, i2w)

    # Phase 4: Analyze
    deltas = phase4_analyze(results)

    # ================================================================
    print()
    print("╔" + "═" * 70 + "╗")
    print("║  FINAL REPORT                                                ║")
    print("╚" + "═" * 70 + "╝")

    ok = [deltas['func_delta'] > 0.05,
          deltas['auth_delta'] > 0.05,
          deltas['coh_delta'] > -0.05]
    passed = sum(ok)

    print(f"\n  1. Func% increase > 5%: {deltas['func_delta']:+.1%} {'✓' if ok[0] else '✗'}")
    print(f"  2. BigramAuth increase > 5%: {deltas['auth_delta']:+.1%} {'✓' if ok[1] else '✗'}")
    print(f"  3. TopicCoh preserved: {deltas['coh_delta']:+.4f} {'✓' if ok[2] else '✗'}")
    print(f"\n  Result: {passed}/3 criteria passed")

    if passed >= 2:
        print(f"\n  CONCLUSION: Directional bigrams + auto-calibrated structure")
        print(f"  blending produces more fluent text (+{deltas['func_delta']:+.0%} func words,")
        print(f"  +{deltas['auth_delta']:+.0%} authentic bigrams) while preserving")
        print(f"  topic coherence ({deltas['coh_delta']:+.3f}).")
    else:
        print(f"\n  CONCLUSION: Needs tuning. Try higher base_structure_weight,")
        print(f"  better vector training, or trigram extension.")

    print()
    return passed


if __name__ == '__main__':
    main()
