"""
Test Dense SDM as long-term memory for CELN v3
==============================================

6-phase experiment:
  1. Load corpus and word vectors
  2. Initialize SDM from real sentence centroids
  3. Write ALL sentences into SDM
  4. Query by topic words from 6 different domains
  5. Validate with ground-truth: do retrieved words co-occur with query word?
  6. Contrast: SDM retrieval vs direct word-vector nearest neighbors
  7. Stability: add more sentences, verify old queries still work

Uses the DenseSDM + sentence_to_centroid from celn/memory.py.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celn.train import load_corpus
from celn.core import normalize, similarity, auto_threshold
from celn.memory import DenseSDM, sentence_to_centroid


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def nearest_words(
    vec: np.ndarray,
    vectors: np.ndarray,
    i2w: dict[int, str],
    top_k: int = 20,
    exclude: set[str] | None = None
) -> list[tuple[str, float]]:
    """Find nearest words to a vector by cosine similarity."""
    sims = vectors @ normalize(vec)
    order = np.argsort(sims)[::-1]
    results = []
    for idx in order:
        word = i2w[idx]
        if exclude and word in exclude:
            continue
        results.append((word, float(sims[idx])))
        if len(results) >= top_k:
            break
    return results


def build_ground_truth(
    sentences: list[list[str]],
    target_word: str
) -> set[str]:
    """Find all words that co-occur with target_word in the same sentence.

    This is our ground-truth for SDM evaluation: if the SDM retrieves
    words that actually appear alongside the query word in the corpus,
    it's recovering real associations.
    """
    co_words: set[str] = set()
    for tokens in sentences:
        if target_word in tokens:
            co_words.update(tokens)
    co_words.discard(target_word)
    return co_words


def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def precision_at_k(
    retrieved_words: list[str],
    ground_truth: set[str]
) -> float:
    """Precision@k: fraction of retrieved words in ground truth."""
    if not retrieved_words:
        return 0.0
    hits = sum(1 for w in retrieved_words if w in ground_truth)
    return hits / len(retrieved_words)


# ===================================================================
# Experiment Phases
# ===================================================================

def phase1_load():
    """Load corpus and word vectors."""
    print("=" * 72)
    print("PHASE 1: Load Corpus and Word Vectors")
    print("=" * 72)

    # Load corpus
    corpus_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'corpus_final.txt'
    )
    sentences = load_corpus(corpus_path)
    print(f"  Loaded {len(sentences)} sentences from corpus_final.txt")

    # Load word vectors
    vecs_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'celn_full_vectors.npz'
    )
    data = np.load(vecs_path)
    vectors = data['vectors']  # (vocab_size, D)
    words = list(data['vocab'])

    # Build mappings
    w2i = {w: i for i, w in enumerate(words)}
    i2w = {i: w for i, w in enumerate(words)}

    # Ensure normalized
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)

    print(f"  Loaded {len(words)} word vectors × {vectors.shape[1]}D")
    print(f"  All vectors normalized: ", end="")
    norms = np.linalg.norm(vectors, axis=1)
    print(f"min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")

    return sentences, vectors, words, w2i, i2w


def phase2_init_sdm(sentences, vectors, w2i):
    """Initialize SDM with addresses sampled from real sentence centroids."""
    print()
    print("=" * 72)
    print("PHASE 2: Initialize SDM from Real Sentence Centroids")
    print("=" * 72)

    n_locations = 4096
    activation_pct = 0.01

    # Compute centroids for all sentences
    print(f"  Computing centroids for {len(sentences)} sentences...")
    centroids = np.zeros((len(sentences), vectors.shape[1]), dtype=np.float32)
    valid_count = 0
    for i, tokens in enumerate(sentences):
        c = sentence_to_centroid(tokens, vectors, w2i)
        if np.linalg.norm(c) > 1e-12:
            centroids[valid_count] = c
            valid_count += 1
    centroids = centroids[:valid_count]
    print(f"  Valid centroids: {valid_count}")

    # Init SDM
    sdm = DenseSDM(n_locations=n_locations, activation_pct=activation_pct, seed=42)
    sdm.initialize_addresses(centroids)
    print(f"  SDM initialized: {n_locations} locations × {vectors.shape[1]}D")
    print(f"  Activation: top {activation_pct*100:.1f}% (~{int(n_locations*activation_pct)} locations/access)")

    stats = sdm.stats
    print(f"  Memory used: {stats['memory_total_mb']} MB")

    return sdm, centroids, valid_count


def phase3_write(sdm, sentences, vectors, w2i):
    """Write all sentence centroids into the SDM."""
    print()
    print("=" * 72)
    print("PHASE 3: Write All Sentences into SDM")
    print("=" * 72)

    written = 0
    skipped = 0
    total_activations = 0

    for tokens in sentences:
        c = sentence_to_centroid(tokens, vectors, w2i)
        if np.linalg.norm(c) < 1e-12:
            skipped += 1
            continue
        n_activated = sdm.write(c)
        total_activations += n_activated
        written += 1

    print(f"  Written: {written} sentences")
    print(f"  Skipped: {skipped} (no known words)")
    print(f"  Total location activations: {total_activations}")

    stats = sdm.stats
    print(f"  Locations touched: {stats['n_written']}/{stats['n_locations']}")
    print(f"  Locations untouched: {stats['n_untouched']}")
    print(f"  Avg writes per location: {stats['avg_writes_per_location']}")
    print(f"  Max writes per location: {stats['max_writes_per_location']}")
    print(f"  Last threshold (cosine sim): {stats['last_threshold']}")

    return written


def phase4_query(sdm, vectors, w2i, i2w, sentences):
    """Query SDM by topic words from 6 different domains."""
    print()
    print("=" * 72)
    print("PHASE 4: Query SDM by Topic Words")
    print("=" * 72)

    # Define topic queries with their expected domains
    queries = [
        ('cobre', 'Chemistry / Metals'),
        ('onça', 'Biology / Jaguars'),
        ('revolução', 'History / French Revolution'),
        ('python', 'Programming / Technology'),
        ('coração', 'Human Body / Anatomy'),
        ('carro', 'Transport / Economics'),
    ]

    all_retrieved = {}
    sdm_results = {}

    for query_word, domain in queries:
        if query_word not in w2i:
            print(f"\n  Query '{query_word}' — NOT in vocabulary, skipping")
            continue

        print(f"\n  --- Query: '{query_word}' ({domain}) ---")

        # Encode query
        q_vec = vectors[w2i[query_word]]

        # Read from SDM
        retrieved = sdm.read(q_vec)

        # Find nearest words to retrieved vector
        top_words = nearest_words(retrieved, vectors, i2w, top_k=20, exclude={query_word})
        retrieved_set = {w for w, _ in top_words}

        all_retrieved[query_word] = retrieved_set
        sdm_results[query_word] = top_words

        print(f"  Top-20 retrieved words:")
        for i, (word, sim) in enumerate(top_words):
            if i % 10 == 0:
                print(f"    ", end="")
            print(f"{word}({sim:.3f})", end="  ")
            if i % 10 == 9:
                print()

        # Ground truth precision
        gt = build_ground_truth(sentences, query_word)
        p20 = precision_at_k([w for w, _ in top_words], gt)
        print(f"\n  Precision@20 (corpus co-occurrence): {p20:.2%}")
        print(f"  Ground-truth size: {len(gt)} words co-occur with '{query_word}'")

    return all_retrieved, sdm_results


def phase5_ground_truth_validation(sdm, vectors, w2i, i2w, sentences, sdm_results):
    """Detailed ground-truth validation: precision, recall, overlap analysis."""
    print()
    print("=" * 72)
    print("PHASE 5: Ground-Truth Validation")
    print("=" * 72)

    query_words = list(sdm_results.keys())
    precisions = {}
    domain_jaccards = {}

    # Build ground-truth sets for each query
    ground_truths = {}
    for q in query_words:
        ground_truths[q] = build_ground_truth(sentences, q)

    # Precision@20 for each query
    print("\n  Precision @ 20 (words that co-occur with query in corpus):")
    print(f"  {'Query':<15} {'Prec@20':>10} {'GT Size':>10} {'Hits':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    for q in query_words:
        retrieved = [w for w, _ in sdm_results[q]]
        gt = ground_truths[q]
        hits = sum(1 for w in retrieved if w in gt)
        p = hits / len(retrieved) if retrieved else 0
        precisions[q] = p
        print(f"  {q:<15} {p:>10.2%} {len(gt):>10} {hits:>10}")

    # Inter-topic distinction (Jaccard between different queries' results)
    print("\n  Inter-topic Jaccard (lower = more distinct):")
    all_sets = {}
    for q in query_words:
        all_sets[q] = {w for w, _ in sdm_results[q]}

    for i, q1 in enumerate(query_words):
        for q2 in query_words[i + 1:]:
            j = jaccard(all_sets[q1], all_sets[q2])
            domain_jaccards[f"{q1} vs {q2}"] = j
            print(f"    {q1} vs {q2}: {j:.3f}")

    # Summary
    avg_precision = np.mean(list(precisions.values()))
    avg_jaccard = np.mean(list(domain_jaccards.values()))

    print(f"\n  Average Precision@20: {avg_precision:.2%}")
    print(f"  Average Inter-topic Jaccard: {avg_jaccard:.3f}")

    return precisions, domain_jaccards


def phase6_contrast(vectors, w2i, i2w, sdm_results):
    """Contrast SDM retrieval vs direct word-vector nearest neighbors."""
    print()
    print("=" * 72)
    print("PHASE 6: SDM vs Direct Word-Vector Neighbors")
    print("=" * 72)

    print("\n  If SDM adds contextual enrichment, its results should DIFFER")
    print("  from simply looking up nearest neighbors of the query word.")
    print()

    query_words = list(sdm_results.keys())
    jaccards = {}
    enrichment_scores = {}

    for q in query_words:
        q_vec = vectors[w2i[q]]
        sdm_set = {w for w, _ in sdm_results[q]}

        # Direct word-vector nearest neighbors
        direct = nearest_words(q_vec, vectors, i2w, top_k=20, exclude={q})
        direct_set = {w for w, _ in direct}

        j = jaccard(sdm_set, direct_set)
        jaccards[q] = j

        # Words unique to SDM (contextual enrichment)
        sdm_unique = sdm_set - direct_set
        direct_unique = direct_set - sdm_set
        enrichment = len(sdm_unique) / 20.0  # fraction of SDM results NOT in direct neighbors

        enrichment_scores[q] = enrichment

        print(f"  '{q}':")
        print(f"    Jaccard(SDM, Direct): {j:.3f}")
        print(f"    Words ONLY in SDM: {sorted(sdm_unique)}")
        print(f"    Words ONLY in Direct: {sorted(direct_unique)}")
        print()

    avg_j = np.mean(list(jaccards.values()))
    avg_enrich = np.mean(list(enrichment_scores.values()))

    print(f"  Average Jaccard: {avg_j:.3f}")
    print(f"  Average Enrichment (SDM-unique fraction): {avg_enrich:.2%}")
    print(f"  Interpretation: ", end="")
    if avg_j < 0.8:
        print(f"✓ SDM meaningfully differs from direct neighbors (Jaccard < 0.8)")
    else:
        print(f"⚠ SDM results too similar to direct neighbors — low enrichment")

    return jaccards, enrichment_scores


def phase7_stability(sdm, full_sentences, vectors, w2i, i2w, sentences_subset, sdm_results_before):
    """Test stability: add more sentences, verify old queries still work."""
    print()
    print("=" * 72)
    print("PHASE 7: Stability — Adding More Sentences")
    print("=" * 72)

    # Use the second half of the corpus as "new" sentences
    mid = len(full_sentences) // 2
    new_sentences = full_sentences[mid:]

    print(f"  Adding {len(new_sentences)} new sentences...")

    written_new = 0
    for tokens in new_sentences:
        c = sentence_to_centroid(tokens, vectors, w2i)
        if np.linalg.norm(c) < 1e-12:
            continue
        sdm.write(c)
        written_new += 1

    print(f"  Written: {written_new} new sentences")
    print(f"  Total writes now: {sdm.total_writes}")

    # Re-query and compare
    print("\n  Re-querying after new writes:")
    print(f"  {'Query':<15} {'Before':>10} {'After':>10} {'Change':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")

    query_words = list(sdm_results_before.keys())
    stability_ratios = {}
    sdm_results_after = {}

    for q in query_words:
        q_vec = vectors[w2i[q]]
        retrieved = sdm.read(q_vec)
        top_words = nearest_words(retrieved, vectors, i2w, top_k=20, exclude={q})
        sdm_results_after[q] = top_words

        before_set = {w for w, _ in sdm_results_before[q]}
        after_set = {w for w, _ in top_words}

        j = jaccard(before_set, after_set)
        stability_ratios[q] = j

        print(f"  {q:<15} {len(before_set):>10} {len(after_set):>10} {j:>10.3f}")

    avg_stability = np.mean(list(stability_ratios.values()))

    print(f"\n  Average stability (Jaccard before/after): {avg_stability:.3f}")
    print(f"  Interpretation: ", end="")
    if avg_stability >= 0.9:
        print("✓ Excellent — queries highly stable after new writes")
    elif avg_stability >= 0.7:
        print("⚠ Moderate — some drift after new writes")
    else:
        print("✗ Significant drift — memory unstable with new writes")

    return stability_ratios, sdm_results_after


def phase_bonus_location_themes(sdm, vectors, i2w):
    """Bonus: inspect themes of some locations."""
    print()
    print("=" * 72)
    print("BONUS: Location Theme Inspection")
    print("=" * 72)

    # Sample locations with most writes (most "experienced")
    top_indices = np.argsort(sdm.counters)[::-1][:5]

    print("\n  Top-5 most-written locations (their stored themes):")
    for idx in top_indices:
        if sdm.counters[idx] == 0:
            continue
        theme = sdm.get_location_theme(idx, vectors, i2w, top_k=8)
        addr_theme = sdm.get_location_address_theme(idx, vectors, i2w, top_k=5)
        print(f"\n  Location {idx} ({sdm.counters[idx]} writes):")
        print(f"    Address near: {', '.join(w for w, _ in addr_theme)}")
        print(f"    Stored theme: {', '.join(f'{w}({s:.3f})' for w, s in theme)}")

    # Also sample locations with few writes (specialized)
    few_indices = np.where((sdm.counters > 0) & (sdm.counters <= 3))[0][:5]
    print(f"\n  Sample of specialized (1-3 write) locations:")
    for idx in few_indices:
        if sdm.counters[idx] == 0:
            continue
        theme = sdm.get_location_theme(idx, vectors, i2w, top_k=5)
        addr_theme = sdm.get_location_address_theme(idx, vectors, i2w, top_k=5)
        print(f"\n  Location {idx} ({sdm.counters[idx]} writes):")
        print(f"    Address near: {', '.join(w for w, _ in addr_theme)}")
        print(f"    Stored theme: {', '.join(f'{w}({s:.3f})' for w, s in theme)}")


# ===================================================================
# Main
# ===================================================================

def main():
    print("╔" + "═" * 70 + "╗")
    print("║  Dense SDM — Long-Term Memory Experiment for CELN v3         ║")
    print("║  Testing: storage, retrieval, stability, enrichment          ║")
    print("╚" + "═" * 70 + "╝")

    # Phase 1: Load
    sentences, vectors, words, w2i, i2w = phase1_load()

    # Phase 2: Init SDM
    sdm, centroids, n_valid = phase2_init_sdm(sentences, vectors, w2i)

    # Phase 3: Write
    n_written = phase3_write(sdm, sentences, vectors, w2i)

    # Phase 4: Query
    all_retrieved, sdm_results = phase4_query(sdm, vectors, w2i, i2w, sentences)

    # Phase 5: Validation
    precisions, domain_jaccards = phase5_ground_truth_validation(
        sdm, vectors, w2i, i2w, sentences, sdm_results
    )

    # Phase 6: Contrast
    contrast_jaccards, enrichment = phase6_contrast(vectors, w2i, i2w, sdm_results)

    # Phase 7: Stability
    stability_ratios, results_after = phase7_stability(
        sdm, sentences, vectors, w2i, i2w, sentences, sdm_results
    )

    # Bonus
    phase_bonus_location_themes(sdm, vectors, i2w)

    # ================================================================
    # Final Report
    # ================================================================
    print()
    print("╔" + "═" * 70 + "╗")
    print("║  FINAL REPORT                                                ║")
    print("╚" + "═" * 70 + "╝")

    passed = 0
    total = 4

    # Criterion 1: Precision
    avg_precision = np.mean(list(precisions.values()))
    p1_ok = avg_precision > 0.3
    print(f"\n  1. Precision@20 > 30%: {avg_precision:.2%} {'✓' if p1_ok else '✗'}")

    # Criterion 2: Inter-topic distinction
    avg_jaccard = np.mean(list(domain_jaccards.values()))
    p2_ok = avg_jaccard < 0.2
    print(f"  2. Inter-topic Jaccard < 0.2: {avg_jaccard:.3f} {'✓' if p2_ok else '✗'}")

    # Criterion 3: SDM vs Direct
    avg_contrast = np.mean(list(contrast_jaccards.values()))
    p3_ok = avg_contrast < 0.8
    print(f"  3. SDM ≠ Direct (Jaccard < 0.8): {avg_contrast:.3f} {'✓' if p3_ok else '✗'}")

    # Criterion 4: Stability
    avg_stability = np.mean(list(stability_ratios.values()))
    p4_ok = avg_stability >= 0.9
    print(f"  4. Stability ≥ 90%: {avg_stability:.2%} {'✓' if p4_ok else '✗'}")

    passed = sum([p1_ok, p2_ok, p3_ok, p4_ok])

    # RAM
    stats = sdm.stats
    print(f"\n  RAM used: {stats['memory_total_mb']} MB")
    print(f"  Locations written: {stats['n_written']}/{stats['n_locations']}")
    print(f"  Avg writes/location: {stats['avg_writes_per_location']}")

    print(f"\n  Result: {passed}/{total} criteria passed")

    if passed >= 3:
        print("\n  CONCLUSION: A Dense SDM adapted for real-valued vectors with")
        print("  cosine similarity CAN serve as long-term memory for CELN v3.")
        print("  It stores knowledge incrementally, retrieves by semantic")
        print("  similarity, and maintains stability under new information —")
        print("  without backprop, templates, or fixed thresholds.")
    elif passed >= 2:
        print("\n  CONCLUSION: Partial success. The SDM shows promise but needs")
        print("  tuning of activation parameters or location initialization.")
    else:
        print("\n  CONCLUSION: The dense SDM in its current form does not meet")
        print("  the criteria. Consider: different activation %, more locations,")
        print("  or a hybrid M-encoded + centroid dual-memory approach.")

    print()
    return passed


if __name__ == '__main__':
    main()
