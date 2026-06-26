"""
Test SDM ↔ Generator Integration
=================================
Compare text generation with and without SDM knowledge grounding.

Experiment:
  1. Load corpus + word vectors + init SDM + write sentences
  2. Generate with and without SDM on 6 factual prefixes
  3. Measure topic_coherence, knowledge_alignment, bigram_diversity
  4. Report side-by-side comparisons

Key question: Does SDM-grounded generation produce more relevant,
contextualized content than generation without memory?
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celn.train import load_corpus, tokenize
from celn.core import normalize, similarity
from celn.memory import DenseSDM, sentence_to_centroid
from celn.eval import (
    topic_coherence, source_alignment, knowledge_alignment,
    bigram_diversity, repetition_rate, semantic_novelty,
)
from celn.generator import ProjectiveGenerator


# ---------------------------------------------------------------------------
# Factual test prefixes — topics well-covered in corpus_final.txt
# ---------------------------------------------------------------------------

FACTUAL_PREFIXES = [
    ("o cobre é um", "Chemistry / Metals"),
    ("a revolução francesa foi", "History / French Revolution"),
    ("a onça pintada é", "Biology / Jaguars"),
    ("python é uma linguagem", "Programming / Technology"),
    ("o coração humano é", "Human Body / Anatomy"),
    ("o carro elétrico funciona", "Transport / Technology"),
]


# ---------------------------------------------------------------------------
# Experiment phases
# ---------------------------------------------------------------------------

def phase1_setup():
    """Load corpus, vectors, init SDM, write all sentences."""
    print("=" * 72)
    print("PHASE 1: Setup — Load Data, Init SDM, Write Corpus")
    print("=" * 72)

    # Load corpus
    corpus_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'corpus_final.txt'
    )
    sentences = load_corpus(corpus_path)
    print(f"  Loaded {len(sentences)} sentences")

    # Load word vectors
    vecs_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'celn_full_vectors.npz'
    )
    data = np.load(vecs_path)
    vectors = data['vectors']
    words = list(data['vocab'])
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)

    w2i = {w: i for i, w in enumerate(words)}
    i2w = {i: w for i, w in enumerate(words)}
    print(f"  Loaded {len(words)} word vectors × {vectors.shape[1]}D")

    # Init SDM
    print(f"  Computing sentence centroids...")
    centroids = np.zeros((len(sentences), vectors.shape[1]), dtype=np.float32)
    valid = 0
    for tokens in sentences:
        c = sentence_to_centroid(tokens, vectors, w2i)
        if np.linalg.norm(c) > 1e-12:
            centroids[valid] = c
            valid += 1
    centroids = centroids[:valid]
    print(f"  Valid centroids: {valid}")

    sdm = DenseSDM(n_locations=4096, activation_pct=0.01, seed=42)
    sdm.initialize_addresses(centroids)
    print(f"  SDM initialized: {sdm.n_locations} locations, {sdm.stats['memory_total_mb']} MB")

    # Write all sentences
    print(f"  Writing {len(sentences)} sentences to SDM...")
    for tokens in sentences:
        c = sentence_to_centroid(tokens, vectors, w2i)
        if np.linalg.norm(c) > 1e-12:
            sdm.write(c)
    print(f"  SDM filled: {sdm.stats['n_written']} locations written, "
          f"{sdm.stats['avg_writes_per_location']} avg writes/location")

    return sentences, vectors, words, w2i, i2w, sdm


def phase2_generate(vectors, w2i, i2w, sdm):
    """Generate with and without SDM on factual prefixes."""
    print()
    print("=" * 72)
    print("PHASE 2: Generate With vs Without SDM")
    print("=" * 72)

    # Create generators
    gen_standard = ProjectiveGenerator(
        vectors, gamma=1.0, bilateral=True,
        anchor_decay=0.9, anchor_weight=0.3,
        sdm=None  # No SDM — original behavior
    )
    gen_sdm = ProjectiveGenerator(
        vectors, gamma=1.0, bilateral=True,
        anchor_decay=0.9, anchor_weight=0.3,
        sdm=sdm, sdm_knowledge_weight=0.25
    )

    gen_length = 10
    temperature = 0.8

    results = []

    for prefix_str, domain in FACTUAL_PREFIXES:
        prefix_tokens = tokenize(prefix_str)

        print(f"\n  {'─'*66}")
        print(f"  Prefix: '{prefix_str}' ({domain})")
        print(f"  {'─'*66}")

        # Generate WITHOUT SDM
        try:
            std_words, std_state = gen_standard.generate_from_words(
                prefix_tokens, w2i, i2w,
                max_len=gen_length, temperature=temperature, seed=42
            )
        except Exception as e:
            std_words, std_state = [], None
            print(f"  ERROR (standard): {e}")

        # Generate WITH SDM
        try:
            sdm_words, sdm_state = gen_sdm.generate_from_words(
                prefix_tokens, w2i, i2w,
                max_len=gen_length, temperature=temperature, seed=42
            )
        except Exception as e:
            sdm_words, sdm_state = [], None
            print(f"  ERROR (SDM): {e}")

        # Compute metrics
        tc_std = topic_coherence(prefix_tokens, std_words, vectors, w2i)
        tc_sdm = topic_coherence(prefix_tokens, sdm_words, vectors, w2i)

        ka_std = knowledge_alignment(std_words, sdm, prefix_tokens, vectors, w2i)
        ka_sdm = knowledge_alignment(sdm_words, sdm, prefix_tokens, vectors, w2i)

        bd_std = bigram_diversity(std_words)
        bd_sdm = bigram_diversity(sdm_words)

        rr_std = repetition_rate(std_words)
        rr_sdm = repetition_rate(sdm_words)

        sn_std = semantic_novelty(std_words, vectors, w2i)
        sn_sdm = semantic_novelty(sdm_words, vectors, w2i)

        print(f"  Standard:  {' '.join(std_words)}")
        print(f"  SDM-aug:   {' '.join(sdm_words)}")

        print(f"\n  {'Metric':<28} {'Standard':>12} {'SDM-aug':>12} {'Δ':>10}")
        print(f"  {'─'*28} {'─'*12} {'─'*12} {'─'*10}")
        print(f"  {'Topic Coherence':<28} {tc_std:>12.4f} {tc_sdm:>12.4f} {tc_sdm-tc_std:>+10.4f}")
        print(f"  {'Knowledge Alignment':<28} {ka_std:>12.4f} {ka_sdm:>12.4f} {ka_sdm-ka_std:>+10.4f}")
        print(f"  {'Bigram Diversity':<28} {bd_std:>12.4f} {bd_sdm:>12.4f} {bd_sdm-bd_std:>+10.4f}")
        print(f"  {'Repetition Rate':<28} {rr_std:>12.4f} {rr_sdm:>12.4f} {rr_sdm-rr_std:>+10.4f}")
        print(f"  {'Semantic Novelty':<28} {sn_std:>12.4f} {sn_sdm:>12.4f} {sn_sdm-sn_std:>+10.4f}")

        # Show SDM's stored knowledge near the prefix topic
        prefix_indices = [w2i[w] for w in prefix_tokens if w in w2i]
        if prefix_indices:
            prefix_centroid = normalize(vectors[prefix_indices].mean(axis=0))
            sdm_knowledge = sdm.read(prefix_centroid)
            # Top words in SDM knowledge vector
            sims = vectors @ sdm_knowledge
            top_k = np.argsort(sims)[-8:][::-1]
            top_words = [i2w[i] for i in top_k if sims[i] > 0.1]
            if top_words:
                print(f"  SDM knowledge near '{prefix_str}': {', '.join(top_words[:6])}")

        results.append({
            'prefix': prefix_str,
            'domain': domain,
            'standard': {
                'words': std_words,
                'topic_coherence': tc_std,
                'knowledge_alignment': ka_std,
                'bigram_diversity': bd_std,
                'repetition_rate': rr_std,
                'semantic_novelty': sn_std,
            },
            'sdm_aug': {
                'words': sdm_words,
                'topic_coherence': tc_sdm,
                'knowledge_alignment': ka_sdm,
                'bigram_diversity': bd_sdm,
                'repetition_rate': rr_sdm,
                'semantic_novelty': sn_sdm,
            },
        })

    return results


def phase3_analyze(results):
    """Aggregate metrics and report."""
    print()
    print("=" * 72)
    print("PHASE 3: Aggregate Analysis")
    print("=" * 72)

    metrics = ['topic_coherence', 'knowledge_alignment', 'bigram_diversity',
               'repetition_rate', 'semantic_novelty']

    print(f"\n  {'Metric':<28} {'Standard':>12} {'SDM-aug':>12} {'Δ Mean':>10}  {'Wins':>6}")
    print(f"  {'─'*28} {'─'*12} {'─'*12} {'─'*10}  {'─'*6}")

    for metric in metrics:
        std_vals = [r['standard'][metric] for r in results]
        sdm_vals = [r['sdm_aug'][metric] for r in results]
        std_mean = np.mean(std_vals)
        sdm_mean = np.mean(sdm_vals)
        delta = sdm_mean - std_mean

        # Count wins (higher is better for all these metrics except repetition_rate)
        if metric == 'repetition_rate':
            wins = sum(1 for s, a in zip(std_vals, sdm_vals) if a < s)
        else:
            wins = sum(1 for s, a in zip(std_vals, sdm_vals) if a > s)

        print(f"  {metric:<28} {std_mean:>12.4f} {sdm_mean:>12.4f} {delta:>+10.4f}  {wins:>3}/{len(results)}")

    # Overall
    ka_deltas = [r['sdm_aug']['knowledge_alignment'] - r['standard']['knowledge_alignment']
                 for r in results]
    avg_ka_delta = np.mean(ka_deltas)

    tc_deltas = [r['sdm_aug']['topic_coherence'] - r['standard']['topic_coherence']
                 for r in results]
    avg_tc_delta = np.mean(tc_deltas)

    print(f"\n  Summary:")
    print(f"    Knowledge Alignment Δ: {avg_ka_delta:+.4f} "
          f"({'SDM adds factual grounding' if avg_ka_delta > 0 else 'SDM reduces grounding'})")
    print(f"    Topic Coherence Δ:     {avg_tc_delta:+.4f} "
          f"({'SDM preserves coherence' if avg_tc_delta >= -0.02 else 'SDM degrades coherence'})")

    return avg_ka_delta, avg_tc_delta


def phase4_qualitative(results):
    """Show detailed side-by-side word choices."""
    print()
    print("=" * 72)
    print("PHASE 4: Qualitative Word-Choice Analysis")
    print("=" * 72)

    for r in results:
        prefix = r['prefix']
        std_words = r['standard']['words']
        sdm_words = r['sdm_aug']['words']

        # Find words that differ between the two
        print(f"\n  Prefix: '{prefix}'")
        print(f"  Standard: {' '.join(std_words)}")
        print(f"  SDM-aug:  {' '.join(sdm_words)}")

        # Highlight different words
        diff_positions = []
        for i, (sw, sdm_w) in enumerate(zip(std_words, sdm_words)):
            if sw != sdm_w:
                diff_positions.append((i, sw, sdm_w))

        if diff_positions:
            print(f"  Different words ({len(diff_positions)}/{len(std_words)}):")
            for pos, sw, sdm_w in diff_positions:
                print(f"    pos {pos}: standard='{sw}' → sdm='{sdm_w}'")
        else:
            print(f"  (identical — SDM had no effect on this prefix)")


# ===================================================================
# Main
# ===================================================================

def main():
    print("╔" + "═" * 70 + "╗")
    print("║  SDM ↔ Generator Integration Test                              ║")
    print("║  Does SDM-grounded generation improve content relevance?       ║")
    print("╚" + "═" * 70 + "╝")

    # Phase 1: Setup
    sentences, vectors, words, w2i, i2w, sdm = phase1_setup()

    # Phase 2: Generate
    results = phase2_generate(vectors, w2i, i2w, sdm)

    # Phase 3: Analyze
    avg_ka_delta, avg_tc_delta = phase3_analyze(results)

    # Phase 4: Qualitative
    phase4_qualitative(results)

    # ================================================================
    # Final Report
    # ================================================================
    print()
    print("╔" + "═" * 70 + "╗")
    print("║  FINAL REPORT                                                ║")
    print("╚" + "═" * 70 + "╝")

    ka_pass = avg_ka_delta > 0
    tc_pass = avg_tc_delta >= -0.02

    print(f"\n  1. Knowledge Alignment increase: {avg_ka_delta:+.4f} {'✓' if ka_pass else '✗'}")
    print(f"  2. Topic Coherence preserved (Δ ≥ -0.02): {avg_tc_delta:+.4f} {'✓' if tc_pass else '✗'}")

    passed = sum([ka_pass, tc_pass])

    if passed == 2:
        print(f"\n  CONCLUSION: Integrating the Dense SDM with the generator")
        print(f"  INCREASES content relevance ({avg_ka_delta:+.3f} knowledge alignment gain)")
        print(f"  while PRESERVING topic coherence ({avg_tc_delta:+.3f} change).")
        print(f"  The SDM provides factual grounding without degrading fluency.")
    elif ka_pass:
        print(f"\n  CONCLUSION: SDM integration improves knowledge alignment")
        print(f"  but may slightly impact coherence. Adjust sdm_knowledge_weight.")
    else:
        print(f"\n  CONCLUSION: SDM integration did not improve content relevance.")
        print(f"  Consider: higher sdm_knowledge_weight, different query strategy,")
        print(f"  or SDM read during anchor update instead of scoring.")

    print()

    # Print SDM stats for reference
    stats = sdm.stats
    print(f"  SDM stats: {stats['n_written']}/{stats['n_locations']} locations, "
          f"{stats['avg_writes_per_location']} avg writes, "
          f"{stats['memory_total_mb']} MB")

    return passed


if __name__ == '__main__':
    main()
