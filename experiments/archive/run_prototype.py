#!/usr/bin/env python3
"""
CELN v3 — Projective Resonance Prototype
========================================
Tests the central hypothesis:
  Can Projective Resonance M(x,y) = FFT⁻¹(FFT(x) ⊙ φ(FFT(y)))
  serve as a unified binding+attention operation for language?

Experiment design:
  - Train word vectors via PMI + Hebbian learning (no backprop)
  - Generate text continuations from prefixes
  - Compare: context window scoring WITH vs WITHOUT PMI boost
  - Measure: topic coherence, diversity, repetition, novelty

The key insight: Projective Resonance M(x,y) is used for ENCODING
sequences (preserving order, amplifying dominant features), while
a Context Window with PMI boost handles next-word SCORING.
This avoids the "binding feedback loop" where bound states are
spectrally dominated by the most recent word.

Usage:
  python experiments/run_prototype.py [--quick] [--full]
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celn_v3.core import D, similarity, projective_resonance, spectral_entropy
from celn_v3.train import load_corpus, train_vectors, precompute_spectra
from celn_v3.generate import (
    ContextWindow, generate, generate_baseline, generate_from_prefix,
)
from celn_v3.evaluate import run_experiment, print_report


def main():
    parser = argparse.ArgumentParser(description='CELN v3 Prototype')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with 200 sentences (~30s)')
    parser.add_argument('--full', action='store_true',
                       help='Full experiment with all sentences (~2min)')
    parser.add_argument('--corpus', default='corpus_final.txt',
                       help='Path to corpus file')
    args = parser.parse_args()

    if args.quick:
        max_sentences = 200
        epochs = 5
        n_samples = 15
        label = "QUICK"
    elif args.full:
        max_sentences = None
        epochs = 15
        n_samples = 50
        label = "FULL"
    else:
        max_sentences = 500
        epochs = 10
        n_samples = 30
        label = "STANDARD"

    print("╔══════════════════════════════════════════════════════════╗")
    print("║       CELN v3 — Projective Resonance Prototype          ║")
    print("║   M(x,y) = FFT⁻¹(FFT(x) ⊙ φ(FFT(y)))                  ║")
    print(f"║   Mode: {label:<47s}║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print(f"Corpus: {args.corpus}")
    print(f"Sentences: {max_sentences or 'ALL'}")
    print(f"Dimensionality: {D}")
    print(f"Epochs: {epochs}")
    print(f"Test samples: {n_samples}")
    print()

    # ══════════════════════════════════════════════════════════
    # Phase 1: Load Corpus
    # ══════════════════════════════════════════════════════════
    print("─" * 60)
    print("Phase 1: Loading Corpus")
    print("─" * 60)
    sentences = load_corpus(args.corpus, max_sentences=max_sentences)
    print(f"Loaded {len(sentences)} sentences")
    print(f"Avg sentence length: {np.mean([len(s) for s in sentences]):.1f} words")
    print(f"Total tokens: {sum(len(s) for s in sentences)}")

    # ══════════════════════════════════════════════════════════
    # Phase 2: Train Word Vectors (PMI + Hebbian)
    # ══════════════════════════════════════════════════════════
    print()
    print("─" * 60)
    print("Phase 2: Training Word Vectors")
    print("─" * 60)
    vectors, w2i, i2w, ppmi = train_vectors(
        sentences, epochs=epochs, verbose=True
    )
    spectra = precompute_spectra(vectors)
    Vocab = len(w2i)
    print(f"Vocabulary: {Vocab} words")
    print(f"Memory: {vectors.nbytes / 1024 / 1024:.1f} MB")

    # ══════════════════════════════════════════════════════════
    # Phase 3: Semantic Quality Check
    # ══════════════════════════════════════════════════════════
    print()
    print("─" * 60)
    print("Phase 3: Semantic Quality (nearest neighbors)")
    print("─" * 60)
    test_words = ['cobre', 'onça', 'revolução', 'metal', 'animal', 'frança',
                  'água', 'elétrico', 'floresta', 'rei']
    for word in test_words:
        if word in w2i:
            idx = w2i[word]
            sims = [(similarity(vectors[idx], vectors[j]), i2w[j])
                    for j in range(Vocab) if j != idx]
            sims.sort(reverse=True)
            print(f"  {word:>12} → {', '.join(w for _, w in sims[:5])}")

    # ══════════════════════════════════════════════════════════
    # Phase 4: Quick Sanity Check (manual inspection)
    # ══════════════════════════════════════════════════════════
    print()
    print("─" * 60)
    print("Phase 4: Quick Generation Sanity Check")
    print("─" * 60)
    test_prefixes = [
        "o cobre é um",
        "a onça pintada",
        "a revolução francesa",
        "o metal é",
        "a água do",
        "produção de petróleo",
        "o rei luís",
    ]

    for prefix in test_prefixes:
        ptoks_proj, gen_proj = generate_from_prefix(
            prefix, vectors, w2i, i2w, ppmi,
            max_len=6, temperature=0.8, boost_weight=0.3, seed=42,
            use_projective=True
        )
        _, gen_base = generate_from_prefix(
            prefix, vectors, w2i, i2w, ppmi,
            max_len=6, temperature=0.8, boost_weight=0.3, seed=42,
            use_projective=False
        )
        print(f"\n  Prefix:    {prefix}")
        print(f"  Projective: {' '.join(gen_proj)}")
        print(f"  Baseline:   {' '.join(gen_base)}")

    # ══════════════════════════════════════════════════════════
    # Phase 5: Full Evaluation
    # ══════════════════════════════════════════════════════════
    print()
    print("─" * 60)
    print("Phase 5: Full Comparative Evaluation")
    print("─" * 60)

    # Test multiple boost_weight values
    boost_weights = [0.0, 0.2, 0.3, 0.5]

    results = run_experiment(
        vectors, w2i, i2w, ppmi, sentences,
        n_samples=n_samples,
        gen_length=8,
        temperature=0.8,
        boost_weights=boost_weights,
        seed=42
    )

    print_report(results)

    # ══════════════════════════════════════════════════════════
    # Phase 6: Spectral Analysis of Projective Resonance
    # ══════════════════════════════════════════════════════════
    print()
    print("─" * 60)
    print("Phase 6: Spectral Analysis of M(x,y)")
    print("─" * 60)

    # Pick some word pairs and analyze their bound states
    pairs = [
        ('cobre', 'metal'),
        ('onça', 'pintada'),
        ('revolução', 'francesa'),
        ('luís', 'rei'),
    ]

    available_pairs = [(a, b) for a, b in pairs if a in w2i and b in w2i]

    print("\nSpectral entropy of M(x,y) vs plain bind(x,y):")
    print(f"{'Words':<25} {'γ=0.5':>8} {'γ=1.0':>8} {'γ=2.0':>8} {'plain':>8}")
    print("-" * 57)

    for wa, wb in available_pairs:
        va = vectors[w2i[wa]]
        vb = vectors[w2i[wb]]
        row = f"{wa}+{wb:<22}"
        for gamma in [0.5, 1.0, 2.0]:
            m = projective_resonance(va, vb, gamma=gamma)
            row += f" {spectral_entropy(m):>8.4f}"
        # Plain bind
        from celn_v3.core import bind
        pb = bind(va, vb)
        row += f" {spectral_entropy(pb):>8.4f}"
        print(row)

    print("\nLower entropy = more focused attention (φ amplifies dominant freqs)")

    # ══════════════════════════════════════════════════════════
    # Phase 7: Non-commutativity Verification
    # ══════════════════════════════════════════════════════════
    print()
    print("─" * 60)
    print("Phase 7: Non-Commutativity Check")
    print("─" * 60)
    print("M(x,y) should ≠ M(y,x) — order matters for language")
    print()

    for wa, wb in available_pairs[:3]:
        va = vectors[w2i[wa]]
        vb = vectors[w2i[wb]]
        forward = projective_resonance(va, vb, gamma=1.0)
        backward = projective_resonance(vb, va, gamma=1.0)
        sim = similarity(forward, backward)
        print(f"  M({wa}, {wb}) vs M({wb}, {wa}): similarity = {sim:.4f}")
        print(f"    Non-commutative: {'YES' if sim < 0.95 else 'WEAK'}")

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              Experiment Complete                         ║")
    print("╚══════════════════════════════════════════════════════════╝")

    return results


if __name__ == '__main__':
    main()
