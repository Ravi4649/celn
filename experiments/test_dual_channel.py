#!/usr/bin/env python3
"""
CELN v3 — Dual-Channel Generator Test (NO Bigram)
===================================================
Compares the NEW pure-algebraic DualChannelGenerator
(type field + semantic, auto-calibrated, ZERO bigram)
against the OLD DirectionalGenerator (semantic + bigram).

Hypothesis: removing the bigram channel and using auto-calibrated
type field + semantic produces MORE DIVERSE text without losing
grammatical structure or topic coherence.

Metrics:
  1. Function Word Ratio (>35% target)
  2. Topic Coherence (cosine similarity to prefix centroid)
  3. Bigram Authenticity (corpus-attested transitions)
  4. Diversity — repetition rate, unique bigrams generated

Usage:
  python experiments/test_dual_channel.py [--quick]
"""

import sys
import os
import re
import time
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celn_v3.train import tokenize, build_cooccurrence, compute_ppmi, load_corpus
from celn_v3.core import normalize, similarity, batch_normalize
from celn_v3.dual_channel import DualChannelGenerator, extract_type_vectors
from celn_v3.hdc_types import train_hdc_type_vectors, learn_type_field
from celn_v3.fluency import DirectionalGenerator, build_directional_bigrams


# ---------------------------------------------------------------------------
# Portuguese function words (for evaluation ONLY — never for generation)
# ---------------------------------------------------------------------------
FUNCTION_WORDS = {
    'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas',
    'de', 'do', 'da', 'dos', 'das',
    'em', 'no', 'na', 'nos', 'nas', 'num', 'numa',
    'por', 'pelo', 'pela', 'pelos', 'pelas',
    'para', 'pra', 'pro', 'pros',
    'com', 'sem', 'sob', 'sobre', 'entre', 'até',
    'e', 'ou', 'mas', 'que', 'se', 'nem',
    'é', 'foi', 'era', 'são', 'está', 'ser', 'sendo',
    'não', 'sim',
    'me', 'te', 'lhe', 'nos', 'vos',
    'este', 'essa', 'isto', 'isso', 'aquele',
    'ele', 'ela', 'eles', 'elas',
    'muito', 'pouco', 'mais', 'menos',
    'como', 'quando', 'onde', 'porque',
}


# ---------------------------------------------------------------------------
# Phase 1: Train vectors
# ---------------------------------------------------------------------------

def train_svd_vectors(sentences, dim=10000, verbose=True):
    """Train SVD semantic word vectors from corpus."""
    from sklearn.decomposition import TruncatedSVD

    word_counts, cooc_counts, w2i, i2w = build_cooccurrence(sentences, window_size=5)
    vocab_size = len(w2i)
    ppmi = compute_ppmi(word_counts, cooc_counts, w2i)

    n_components = min(dim, vocab_size - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    vecs_reduced = svd.fit_transform(ppmi)

    singular_values = svd.singular_values_
    var_ratio = singular_values ** 2 / (singular_values ** 2).sum()
    weights = var_ratio / var_ratio.max()
    vecs_weighted = vecs_reduced * weights[None, :]

    if verbose:
        print(f"    SVD: {n_components} components, "
              f"explained variance: {var_ratio[:50].sum():.1%}")

    if n_components < dim:
        rng = np.random.RandomState(42)
        R = rng.randn(n_components, dim) / np.sqrt(n_components)
        vectors = vecs_weighted @ R
    else:
        vectors = vecs_weighted

    vectors = batch_normalize(vectors)
    return vectors, ppmi, w2i, i2w


def phase1_train(corpus_path='corpus_final.txt', quick=False):
    """Train all vector representations."""
    print("=" * 72)
    print("PHASE 1: Training Vector Representations")
    print("=" * 72)

    # Load corpus
    corpus_full = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        corpus_path
    )
    with open(corpus_full, 'r', encoding='utf-8') as f:
        text = f.read()
    raw_sentences = re.split(r'[.!?\n]+', text)
    sentences_full = []
    for s in raw_sentences:
        tokens = tokenize(s, min_len=1)
        if len(tokens) >= 3:
            sentences_full.append(tokens)

    if quick:
        sentences_full = sentences_full[:500]

    print(f"  Loaded {len(sentences_full)} sentences")

    all_words = [w for s in sentences_full for w in s]
    unique = set(all_words)
    func_in_vocab = FUNCTION_WORDS & unique
    print(f"  Unique words: {len(unique)}")
    print(f"  Function words available: {len(func_in_vocab)}")

    # ── Train SVD semantic vectors ──
    print(f"\n  Training SVD semantic vectors...")
    t0 = time.time()
    vectors, ppmi, w2i, i2w = train_svd_vectors(sentences_full, dim=10000)
    vocab_size = len(w2i)
    print(f"  Semantic vectors: {vectors.shape} in {time.time()-t0:.1f}s")

    # ── Train HDC type vectors ──
    print(f"\n  Training HDC type vectors (Hebbian, positional)...")
    t0 = time.time()
    hdc_dim = 4096
    type_vecs = train_hdc_type_vectors(
        sentences_full, w2i, vocab_size,
        hdc_dim=hdc_dim,
        context_window=3,
        n_epochs=3 if quick else 5,
        learning_rate=0.05,
        seed=42,
        verbose=True,
    )
    print(f"  HDC type vectors: {type_vecs.shape} in {time.time()-t0:.1f}s")

    # ── Build directional bigrams (for baseline ONLY) ──
    print(f"\n  Building directional bigrams (for baseline comparison)...")
    t0 = time.time()
    bigram_prob = build_directional_bigrams(sentences_full, w2i, vocab_size, smoothing=0.01)
    non_zero = (bigram_prob > 0.001).sum()
    print(f"  Bigram matrix: {bigram_prob.shape}, "
          f"{non_zero} non-zero ({non_zero/(vocab_size*vocab_size):.2%} density) "
          f"in {time.time()-t0:.1f}s")

    return sentences_full, vectors, type_vecs, ppmi, bigram_prob, w2i, i2w


# ---------------------------------------------------------------------------
# Phase 2: Create and train generators
# ---------------------------------------------------------------------------

def phase2_setup(sentences, vectors, type_vecs, bigram_prob, w2i, i2w):
    """Create and train both generators."""
    print()
    print("=" * 72)
    print("PHASE 2: Setup Generators")
    print("=" * 72)

    # ── NEW: Dual-Channel Generator (NO bigram, auto-calibrated) ──
    print("\n  Creating DualChannelGenerator (type field + semantic, NO bigram)...")
    t0 = time.time()
    dual_gen = DualChannelGenerator(
        semantic_vectors=vectors,
        type_vectors=type_vecs,
        w2i=w2i,
        i2w=i2w,
        window_size=5,
        window_decay=0.7,
    )
    dual_gen.learn_type_field(sentences)
    words_with_field = int((np.linalg.norm(dual_gen.type_field, axis=1) > 1e-12).sum())
    print(f"  Type field learned: {words_with_field}/{len(w2i)} words have field "
          f"({words_with_field/len(w2i):.1%})")
    print(f"  Channels: type_field + semantic, AUTO-CALIBRATED per step")
    print(f"  Setup in {time.time()-t0:.1f}s")

    # ── OLD: DirectionalGenerator (WITH bigram, for comparison) ──
    print("\n  Creating DirectionalGenerator (semantic + bigram, for baseline)...")
    t0 = time.time()
    base_gen = DirectionalGenerator(
        word_vectors=vectors,
        bigram_prob=bigram_prob,
        w2i=w2i,
        i2w=i2w,
        window_size=5,
        window_decay=0.7,
        base_structure_weight=0.35,
    )
    print(f"  Channels: semantic + directional bigram (fixed base weight)")
    print(f"  Setup in {time.time()-t0:.1f}s")

    return dual_gen, base_gen


# ---------------------------------------------------------------------------
# Phase 3: Generate and compare
# ---------------------------------------------------------------------------

def phase3_compare(dual_gen, base_gen, vectors, w2i, i2w, bigram_prob,
                   temperature=0.8, gen_length=12, seed=42):
    """Generate text with both generators and compare."""
    print()
    print("=" * 72)
    print("PHASE 3: Head-to-Head Comparison")
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
        "a fotossíntese é",
        "o leite materno",
        "produção de petróleo",
        "o rei luís",
        "a floresta tropical",
        "o sistema solar",
        "a poluição do ar",
    ]

    results = []

    for i, prefix_str in enumerate(test_prefixes):
        prefix_tokens = tokenize(prefix_str, min_len=1)
        prefix_known = [w for w in prefix_tokens if w in w2i]

        if len(prefix_known) < 2:
            print(f"\n  [{i+1}/{len(test_prefixes)}] '{prefix_str}' — too few known words, skipping")
            continue

        # ── Generate with NEW (dual channel, no bigram) ──
        try:
            new_generated = dual_gen.generate(
                prefix_known,
                max_len=gen_length,
                temperature=temperature,
                seed=seed + i,
            )
        except Exception as e:
            print(f"  NEW generator ERROR: {e}")
            new_generated = []

        # ── Generate with OLD (semantic + bigram baseline) ──
        try:
            old_generated = base_gen.generate(
                prefix_known,
                max_len=gen_length,
                temperature=temperature,
                seed=seed + i,
            )
        except Exception as e:
            print(f"  OLD generator ERROR: {e}")
            old_generated = []

        # ── Compute metrics ──
        # 1. Function Word Ratio
        new_func = sum(1 for w in new_generated if w in FUNCTION_WORDS) / max(len(new_generated), 1)
        old_func = sum(1 for w in old_generated if w in FUNCTION_WORDS) / max(len(old_generated), 1)

        # 2. Topic Coherence (cosine sim to prefix centroid)
        p_indices = [w2i[w] for w in prefix_known if w in w2i]
        if p_indices:
            p_centroid = normalize(vectors[p_indices].mean(axis=0))
            new_indices = [w2i[w] for w in new_generated if w in w2i]
            old_indices = [w2i[w] for w in old_generated if w in w2i]
            new_coh = float(np.dot(
                p_centroid,
                normalize(vectors[new_indices].mean(axis=0))
            )) if new_indices else 0.0
            old_coh = float(np.dot(
                p_centroid,
                normalize(vectors[old_indices].mean(axis=0))
            )) if old_indices else 0.0
        else:
            new_coh = old_coh = 0.0

        # 3. Bigram Authenticity (corpus-attested transitions)
        if len(new_generated) >= 2:
            new_bigrams_list = list(zip(new_generated[:-1], new_generated[1:]))
            new_auth = sum(1 for (a, b) in new_bigrams_list
                         if a in w2i and b in w2i and
                         bigram_prob[w2i[a], w2i[b]] > 0.01) / len(new_bigrams_list)
        else:
            new_auth = 0.0
        if len(old_generated) >= 2:
            old_bigrams_list = list(zip(old_generated[:-1], old_generated[1:]))
            old_auth = sum(1 for (a, b) in old_bigrams_list
                         if a in w2i and b in w2i and
                         bigram_prob[w2i[a], w2i[b]] > 0.01) / len(old_bigrams_list)
        else:
            old_auth = 0.0

        # 4. Diversity — unique bigrams ratio (higher = more diverse, less repetitive)
        if new_generated:
            new_unique_ratio = len(set(new_generated)) / len(new_generated)
        else:
            new_unique_ratio = 0.0
        if old_generated:
            old_unique_ratio = len(set(old_generated)) / len(old_generated)
        else:
            old_unique_ratio = 0.0

        # 5. Diversity — unique bigram transitions
        if len(new_generated) >= 2:
            new_bigram_set = set(zip(new_generated[:-1], new_generated[1:]))
            new_bigram_uniqueness = len(new_bigram_set) / len(new_bigrams_list)
        else:
            new_bigram_uniqueness = 0.0
        if len(old_generated) >= 2:
            old_bigram_set = set(zip(old_generated[:-1], old_generated[1:]))
            old_bigram_uniqueness = len(old_bigram_set) / len(old_bigrams_list)
        else:
            old_bigram_uniqueness = 0.0

        # ── Print ──
        prefix_display = ' '.join(prefix_known)
        print(f"\n  {'─'*66}")
        print(f"  [{i+1}/{len(test_prefixes)}] Prefix: '{prefix_display}'")
        print(f"  {'─'*66}")
        print(f"  NEW (dual, no bigram):  {' '.join(new_generated)}")
        print(f"  OLD (semantic+bigram):  {' '.join(old_generated)}")
        print(f"  Func%:      NEW={new_func:.0%}  OLD={old_func:.0%}")
        print(f"  TopicCoh:   NEW={new_coh:.3f}  OLD={old_coh:.3f}")
        print(f"  BigramAuth: NEW={new_auth:.0%}  OLD={old_auth:.0%}")
        print(f"  Diversity (unique words):   NEW={new_unique_ratio:.0%}  OLD={old_unique_ratio:.0%}")
        print(f"  Diversity (unique bigrams): NEW={new_bigram_uniqueness:.0%}  OLD={old_bigram_uniqueness:.0%}")

        results.append({
            'prefix': prefix_known,
            'new_generated': new_generated,
            'old_generated': old_generated,
            'new_func': new_func,
            'old_func': old_func,
            'new_coh': new_coh,
            'old_coh': old_coh,
            'new_auth': new_auth,
            'old_auth': old_auth,
            'new_unique_words': new_unique_ratio,
            'old_unique_words': old_unique_ratio,
            'new_unique_bigrams': new_bigram_uniqueness,
            'old_unique_bigrams': old_bigram_uniqueness,
        })

    return results


# ---------------------------------------------------------------------------
# Phase 4: Analysis
# ---------------------------------------------------------------------------

def phase4_analyze(results):
    """Aggregate and analyze all metrics."""
    print()
    print("=" * 72)
    print("PHASE 4: Aggregate Analysis")
    print("=" * 72)

    metrics = {
        'Function Word Ratio': ('new_func', 'old_func', '%'),
        'Topic Coherence': ('new_coh', 'old_coh', '.4f'),
        'Bigram Authenticity': ('new_auth', 'old_auth', '%'),
        'Unique Words': ('new_unique_words', 'old_unique_words', '%'),
        'Unique Bigrams': ('new_unique_bigrams', 'old_unique_bigrams', '%'),
    }

    deltas = {}

    for name, (new_key, old_key, fmt) in metrics.items():
        new_vals = [r[new_key] for r in results]
        old_vals = [r[old_key] for r in results]
        new_mean = np.mean(new_vals)
        old_mean = np.mean(old_vals)
        delta = new_mean - old_mean

        if fmt == '%':
            print(f"  {name:<28} NEW={new_mean:>8.1%}  OLD={old_mean:>8.1%}  Δ={delta:>+8.1%}")
            deltas[new_key] = delta
        else:
            print(f"  {name:<28} NEW={new_mean:>8.4f}  OLD={old_mean:>8.4f}  Δ={delta:>+8.4f}")
            deltas[new_key] = delta

    # ── Statistical significance check ──
    print()
    print("  Statistical check (NEW vs OLD):")
    for name, (new_key, old_key, _) in metrics.items():
        new_vals = np.array([r[new_key] for r in results])
        old_vals = np.array([r[old_key] for r in results])
        diff = new_vals - old_vals
        wins = (diff > 0).sum()
        losses = (diff < 0).sum()
        ties = (diff == 0).sum()
        print(f"    {name:<26} wins={wins} losses={losses} ties={ties}")

    return deltas


# ---------------------------------------------------------------------------
# Phase 5: Detailed diversity analysis
# ---------------------------------------------------------------------------

def phase5_diversity_analysis(results, w2i, i2w):
    """Deep dive into pattern diversity."""
    print()
    print("=" * 72)
    print("PHASE 5: Pattern Diversity Deep-Dive")
    print("=" * 72)

    # Collect all generated text
    all_new = []
    all_old = []
    for r in results:
        all_new.extend(r['new_generated'])
        all_old.extend(r['old_generated'])

    # Word frequency distributions
    new_word_counts = Counter(all_new)
    old_word_counts = Counter(all_old)

    # Top repeated words
    print("\n  Most frequent words:")
    print(f"  {'Word':<15} {'NEW count':>10} {'OLD count':>10}")
    print(f"  {'─'*15} {'─'*10} {'─'*10}")
    all_words = set(list(new_word_counts.keys()) + list(old_word_counts.keys()))
    sorted_words = sorted(all_words, key=lambda w: new_word_counts.get(w, 0) + old_word_counts.get(w, 0), reverse=True)
    for w in sorted_words[:15]:
        nc = new_word_counts.get(w, 0)
        oc = old_word_counts.get(w, 0)
        print(f"  {w:<15} {nc:>10} {oc:>10}")

    # Repetition rate (words appearing > once)
    new_repeats = sum(1 for w, c in new_word_counts.items() if c > 1)
    old_repeats = sum(1 for w, c in old_word_counts.items() if c > 1)
    new_total = len(all_new)
    old_total = len(all_old)
    print(f"\n  Words repeated >1x: NEW={new_repeats}/{len(new_word_counts)} "
          f"OLD={old_repeats}/{len(old_word_counts)}")
    print(f"  Total words generated: NEW={new_total} OLD={old_total}")

    # Bigram pattern check — does it repeat "é uma das maiores"?
    print("\n  Checking for 'dead patterns' (repeated 3+ word sequences):")
    for r in results:
        new_gen = r['new_generated']
        old_gen = r['old_generated']
        # Check 3-grams
        new_trigrams = set()
        old_trigrams = set()
        for i in range(len(new_gen) - 2):
            new_trigrams.add(' '.join(new_gen[i:i+3]))
        for i in range(len(old_gen) - 2):
            old_trigrams.add(' '.join(old_gen[i:i+3]))

    # Cross-sample trigram repetition
    all_new_trigrams = []
    all_old_trigrams = []
    for r in results:
        ng = r['new_generated']
        og = r['old_generated']
        for i in range(len(ng) - 2):
            all_new_trigrams.append(' '.join(ng[i:i+3]))
        for i in range(len(og) - 2):
            all_old_trigrams.append(' '.join(og[i:i+3]))

    new_tri_counts = Counter(all_new_trigrams)
    old_tri_counts = Counter(all_old_trigrams)

    new_repeated_tris = [(t, c) for t, c in new_tri_counts.items() if c > 1]
    old_repeated_tris = [(t, c) for t, c in old_tri_counts.items() if c > 1]

    print(f"  NEW: {len(new_repeated_tris)} repeated trigrams")
    for t, c in sorted(new_repeated_tris, key=lambda x: -x[1])[:5]:
        print(f"    '{t}' ×{c}")
    print(f"  OLD: {len(old_repeated_tris)} repeated trigrams")
    for t, c in sorted(old_repeated_tris, key=lambda x: -x[1])[:5]:
        print(f"    '{t}' ×{c}")

    return {
        'new_repeated_trigrams': len(new_repeated_tris),
        'old_repeated_trigrams': len(old_repeated_tris),
        'new_total_trigrams': len(all_new_trigrams),
        'old_total_trigrams': len(all_old_trigrams),
    }


# ===================================================================
# Main
# ===================================================================

def main():
    quick = '--quick' in sys.argv

    print("╔" + "═" * 70 + "╗")
    print("║  CELN v3 — Dual-Channel Generator: NO Bigram Test           ║")
    print("║  Type Field (HDC) + Semantic (SVD), AUTO-CALIBRATED         ║")
    print("╚" + "═" * 70 + "╝")

    start_time = time.time()

    # Phase 1: Train
    sentences, vectors, type_vecs, ppmi, bigram_prob, w2i, i2w = phase1_train(quick=quick)

    # Phase 2: Setup generators
    dual_gen, base_gen = phase2_setup(sentences, vectors, type_vecs, bigram_prob, w2i, i2w)

    # Phase 3: Compare
    results = phase3_compare(
        dual_gen, base_gen, vectors, w2i, i2w, bigram_prob,
        temperature=0.8, gen_length=8 if quick else 12, seed=42
    )

    # Phase 4: Analyze
    deltas = phase4_analyze(results)

    # Phase 5: Diversity deep-dive
    div_data = phase5_diversity_analysis(results, w2i, i2w)

    # ════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ════════════════════════════════════════════════════════════
    print()
    print("╔" + "═" * 70 + "╗")
    print("║  FINAL REPORT                                                ║")
    print("╚" + "═" * 70 + "╝")

    # Test criteria
    func_ok = deltas.get('new_func', 0) >= -0.05  # func% preserved
    coh_ok = deltas.get('new_coh', 0) >= -0.03    # coherence preserved
    div_ok = deltas.get('new_unique_bigrams', 0) >= 0.0  # diversity same or better
    rep_ok = div_data['new_repeated_trigrams'] <= div_data['old_repeated_trigrams']  # fewer repeats

    passed = sum([func_ok, coh_ok, div_ok, rep_ok])

    print(f"\n  Criteria:")
    print(f"  1. Function Word Ratio preserved (Δ≥-5%): "
          f"{deltas.get('new_func', 0):+.1%} {'✓' if func_ok else '✗'}")
    print(f"  2. Topic Coherence preserved (Δ≥-0.03): "
          f"{deltas.get('new_coh', 0):+.4f} {'✓' if coh_ok else '✗'}")
    print(f"  3. Bigram Diversity same/better (Δ≥0): "
          f"{deltas.get('new_unique_bigrams', 0):+.1%} {'✓' if div_ok else '✗'}")
    print(f"  4. Fewer repeated trigrams: "
          f"NEW={div_data['new_repeated_trigrams']} OLD={div_data['old_repeated_trigrams']} "
          f"{'✓' if rep_ok else '✗'}")
    print(f"\n  Result: {passed}/4 criteria passed")

    total_time = time.time() - start_time

    if passed >= 3:
        print(f"\n  ✅ CONCLUSION: The pure-algebraic DualChannelGenerator")
        print(f"     (type field + semantic, auto-calibrated, ZERO bigram)")
        print(f"     produces text as good or better than the bigram version")
        print(f"     in {total_time:.0f}s on CPU (Ryzen 2600, 16GB RAM).")
        print(f"\n     The bigram channel is confirmed redundant —")
        print(f"     HDC type field handles syntax with geometry, not frequency.")
    else:
        print(f"\n  ⚠️  CONCLUSION: Mixed results. The auto-calibration may need tuning.")
        print(f"     Check individual samples for qualitative evaluation.")

    print()
    return passed >= 3


if __name__ == '__main__':
    main()
