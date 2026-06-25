#!/usr/bin/env python3
"""
CELN v3 — Final Experiment
===========================
Uses SVD-trained word vectors (high-quality semantics) to test
Projective Resonance generation.

This is the definitive test: with good word vectors, can
Projective Resonance produce fluent, coherent text?
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celn_v3.train import load_corpus, tokenize, precompute_spectra
from celn_v3.generate import (
    ContextWindow, generate, generate_baseline, generate_from_prefix,
    context_window_scores,
)
from celn_v3.evaluate import run_experiment, print_report
from experiments.improved_training import train_svd_vectors

# Import core with variable dimensionality
from celn_v3.core import (
    normalize, batch_normalize,
    bind, unbind,
    projective_resonance,
    encode_sequence, encode_sequence_plain,
    resonance_score, resonance_scores_batch,
    similarity, spectral_entropy,
)


def main():
    corpus_path = sys.argv[1] if len(sys.argv) > 1 else 'corpus_final.txt'
    quick = '--quick' in sys.argv

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     CELN v3 — Projective Resonance: Final Test          ║")
    print("║     SVD-trained vectors + Context Window generation     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # ══════════════════════════════════════════════════════════
    # Phase 1: Train SVD word vectors
    # ══════════════════════════════════════════════════════════
    print("─" * 60)
    print("Phase 1: Training SVD Word Vectors")
    print("─" * 60)
    sentences = load_corpus(corpus_path, max_sentences=500 if quick else None)
    vectors_svd, w2i, i2w, ppmi = train_svd_vectors(sentences, verbose=True)
    native_dim = vectors_svd.shape[1]
    print(f"Native dimension: {native_dim}")

    # Optionally expand to 10k via random projection
    target_dim = 10000
    if native_dim < target_dim:
        print(f"Expanding to {target_dim}D via random projection...")
        rng = np.random.RandomState(42)
        # Random projection matrix with orthonormal-ish rows
        R = rng.randn(native_dim, target_dim) / np.sqrt(native_dim)
        vectors = vectors_svd @ R  # (vocab, native) @ (native, target) = (vocab, target)
        vectors = batch_normalize(vectors)
        print(f"  Expanded: {vectors.shape}")
    else:
        vectors = vectors_svd
        target_dim = native_dim

    spectra = precompute_spectra(vectors)
    vocab_size = len(w2i)
    print(f"Final vocabulary: {vocab_size} words in {target_dim}D")

    # ══════════════════════════════════════════════════════════
    # Phase 2: Semantic quality check
    # ══════════════════════════════════════════════════════════
    print()
    print("─" * 60)
    print("Phase 2: Semantic Quality Check")
    print("─" * 60)
    test_words = ['cobre', 'onça', 'revolução', 'metal', 'animal', 'frança',
                  'água', 'elétrico', 'floresta', 'rei', 'python', 'gato',
                  'fotossíntese', 'petróleo', 'leite', 'café']
    for word in test_words:
        if word in w2i:
            idx = w2i[word]
            sims = [(similarity(vectors[idx], vectors[j]), i2w[j])
                    for j in range(vocab_size) if j != idx]
            sims.sort(reverse=True)
            print(f"  {word:>14} → {', '.join(w for _, w in sims[:5])}")

    # ══════════════════════════════════════════════════════════
    # Phase 3: Generation sanity check
    # ══════════════════════════════════════════════════════════
    print()
    print("─" * 60)
    print("Phase 3: Generation Examples")
    print("─" * 60)

    prefixes = [
        "o cobre é um",
        "a onça pintada",
        "a revolução francesa",
        "o gato doméstico",
        "a fotossíntese é",
        "produção de petróleo",
        "o rei luís",
        "python é uma",
        "a água do rio",
        "o leite materno",
    ]

    for prefix in prefixes:
        ptoks, gen = generate_from_prefix(
            prefix, vectors, w2i, i2w, ppmi,
            max_len=6, temperature=0.8, boost_weight=0.3, seed=42,
            use_projective=True
        )
        print(f"  {prefix:>22} → {' '.join(gen)}")

    # ══════════════════════════════════════════════════════════
    # Phase 4: Non-commutativity test
    # ══════════════════════════════════════════════════════════
    print()
    print("─" * 60)
    print("Phase 4: Non-Commutativity (Bilateral φ)")
    print("─" * 60)
    pairs = [
        ('cobre', 'metal'),
        ('onça', 'pintada'),
        ('revolução', 'francesa'),
        ('rei', 'água'),
        ('cobre', 'onça'),
        ('gato', 'python'),
    ]
    print(f"{'Pair':<22} {'Uni γ=1':>10} {'Bi γ=1':>10} {'Bi γ=2':>10}")
    print("-" * 52)
    for wa, wb in pairs:
        if wa not in w2i or wb not in w2i:
            continue
        va, vb = vectors[w2i[wa]], vectors[w2i[wb]]

        u_sim = similarity(
            projective_resonance(va, vb, gamma=1.0, bilateral=False),
            projective_resonance(vb, va, gamma=1.0, bilateral=False))
        b1_sim = similarity(
            projective_resonance(va, vb, gamma=1.0, bilateral=True),
            projective_resonance(vb, va, gamma=1.0, bilateral=True))
        b2_sim = similarity(
            projective_resonance(va, vb, gamma=2.0, bilateral=True),
            projective_resonance(vb, va, gamma=2.0, bilateral=True))

        print(f"  {wa}+{wb:<19} {u_sim:>10.4f} {b1_sim:>10.4f} {b2_sim:>10.4f}")

    # ══════════════════════════════════════════════════════════
    # Phase 5: Full evaluation
    # ══════════════════════════════════════════════════════════
    print()
    print("─" * 60)
    print("Phase 5: Full Evaluation")
    print("─" * 60)

    n_samples = 15 if quick else 40
    results = run_experiment(
        vectors, w2i, i2w, ppmi, sentences,
        n_samples=n_samples,
        gen_length=8,
        temperature=0.8,
        boost_weights=[0.0, 0.2, 0.3, 0.5],
        seed=42
    )
    print_report(results)

    # ══════════════════════════════════════════════════════════
    # Phase 6: Spectral analysis
    # ══════════════════════════════════════════════════════════
    print()
    print("─" * 60)
    print("Phase 6: Spectral Analysis")
    print("─" * 60)
    print(f"{'Words':<22} {'γ=0.5':>8} {'γ=1.0':>8} {'γ=2.0':>8} {'plain':>8}")
    print("-" * 46)
    for wa, wb in pairs[:4]:
        if wa not in w2i or wb not in w2i:
            continue
        va, vb = vectors[w2i[wa]], vectors[w2i[wb]]
        row = f"{wa}+{wb:<19}"
        for gamma in [0.5, 1.0, 2.0]:
            m = projective_resonance(va, vb, gamma=gamma, bilateral=False)
            row += f" {spectral_entropy(m):>8.4f}"
        pb = bind(va, vb)
        row += f" {spectral_entropy(pb):>8.4f}"
        print(row)

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              Final Experiment Complete                   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    return vectors, w2i, i2w, ppmi


if __name__ == '__main__':
    main()
