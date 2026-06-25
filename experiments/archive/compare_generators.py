#!/usr/bin/env python3
"""
CELN v3 — Head-to-Head Generator Comparison
============================================
Projective Resonance (M + anchor) vs VSA 2.0 Boca Universal (plain bind).

Answers: Does using M(x,y) with anchor produce more fluent,
thematically coherent text than plain convolution?

Usage:
  python experiments/compare_generators.py [--quick]
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celn_v3.core import D, normalize, batch_normalize, similarity, spectral_entropy
from celn_v3.encoder import Encoder, BaselineEncoder
from celn_v3.decoder import Decoder, BaselineDecoder
from celn_v3.generator import ProjectiveGenerator, BaselineGenerator
from celn_v3.eval import run_comparison, print_report
from celn_v3.train import load_corpus


def main():
    quick = '--quick' in sys.argv
    corpus_path = 'corpus_final.txt'
    vectors_path = 'celn_v3_native_vectors.npz'

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  CELN v3 — Generator Comparison                         ║")
    print("║  Projective Resonance vs VSA 2.0 Boca Universal         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # ── Load or Build Vectors ────────────────────────────
    if os.path.exists(vectors_path):
        print(f"Loading vectors from {vectors_path}...")
        data = np.load(vectors_path, allow_pickle=True)
        vectors = data['vectors']
        vocab = data['vocab']
        w2i = {w: i for i, w in enumerate(vocab)}
        i2w = {i: w for i, w in enumerate(vocab)}
        print(f"  {len(w2i)} words, {vectors.shape[1]}D (native SVD)")
    else:
        print("Building SVD vectors from corpus...")
        from experiments.improved_training import train_svd_vectors
        sentences_all = load_corpus(corpus_path)
        vectors, w2i, i2w, ppmi = train_svd_vectors(sentences_all, verbose=True)
        print(f"  {len(w2i)} words, {vectors.shape[1]}D (native SVD)")

    # Also build PPMI from corpus for transition statistics
    from celn_v3.train import build_cooccurrence, compute_ppmi
    all_sentences = load_corpus(corpus_path)
    filtered = [[w for w in s if w in w2i] for s in all_sentences]
    filtered = [s for s in filtered if len(s) >= 3]
    word_counts, cooc_counts, _, _ = build_cooccurrence(filtered)
    ppmi = compute_ppmi(word_counts, cooc_counts, w2i)
    print(f"  PPMI: {ppmi.shape}, density: {np.count_nonzero(ppmi)/ppmi.size:.2%}")

    # ── Semantic Quality Check ───────────────────────────
    print()
    print("Semantic quality (nearest neighbors):")
    for word in ['cobre', 'onça', 'revolução', 'python', 'gato', 'rei']:
        if word in w2i:
            idx = w2i[word]
            sims = [(similarity(vectors[idx], vectors[j]), i2w[j])
                    for j in range(len(vectors)) if j != idx]
            sims.sort(reverse=True)
            print(f"  {word:>12} → {', '.join(w for _, w in sims[:6])}")

    # ── Non-Commutativity Check ──────────────────────────
    print()
    print("Non-commutativity of M(x,y):")
    from celn_v3.core import projective_resonance
    for wa, wb in [('cobre','metal'), ('onça','pintada'), ('rei','água')]:
        if wa in w2i and wb in w2i:
            va, vb = vectors[w2i[wa]], vectors[w2i[wb]]
            s = similarity(
                projective_resonance(va, vb, gamma=1.0, bilateral=True),
                projective_resonance(vb, va, gamma=1.0, bilateral=True))
            print(f"  M({wa},{wb}) vs M({wb},{wa}): sim = {s:.4f} "
                  f"({'non-commutative' if s < 0.95 else 'similar'})")

    # ── Single Example ───────────────────────────────────
    print()
    print("Quick generation example:")
    # Very stable anchor (0.99 decay): anchor barely moves from prefix topic
    # High anchor weight (0.6): prioritize global coherence over local fit
    proj_gen = ProjectiveGenerator(vectors, gamma=1.0, bilateral=True,
                                   anchor_decay=0.99, anchor_weight=0.6)
    base_gen = BaselineGenerator(vectors)

    for prefix in ['a onça pintada', 'o cobre é um', 'python é uma']:
        p_words, _ = proj_gen.generate_from_words(
            ['a','onça','pintada'] if 'onça' in prefix else
            ['o','cobre','é','um'] if 'cobre' in prefix else
            ['python','é','uma'],
            w2i, i2w, max_len=6, temperature=0.8, seed=42)

        b_words, _ = base_gen.generate_from_words(
            ['a','onça','pintada'] if 'onça' in prefix else
            ['o','cobre','é','um'] if 'cobre' in prefix else
            ['python','é','uma'],
            w2i, i2w, max_len=6, temperature=0.8, seed=42)

        print(f"  {prefix}:")
        print(f"    Proj (M):  {' '.join(p_words)}")
        print(f"    Baseline:  {' '.join(b_words)}")

    # ── Full Comparison ──────────────────────────────────
    print()
    print("─" * 60)
    print("Running full comparison...")
    print("─" * 60)

    n_samples = 15 if quick else 40
    results = run_comparison(
        vectors, w2i, i2w, all_sentences,
        n_samples=n_samples, gen_length=8,
        temperature=0.8, seed=42
    )
    print_report(results)

    return results


if __name__ == '__main__':
    main()
