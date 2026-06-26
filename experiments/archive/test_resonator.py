"""
Test Resonator Network Decoder for CELN v3
==========================================
5-phase experiment evaluating iterative factorization of composite vectors.

Key question: Can Resonator Networks recover constituent words from
bind-encoded and M-encoded composite vectors with high accuracy?
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celn.core import normalize, similarity
from celn.resonator import (
    ResonatorDecoder,
    bind_vec, unbind_vec,
    top_k_accuracy,
)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_codebook():
    """Load word vectors as the shared codebook."""
    vecs_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data/celn_full_vectors.npz'
    )
    data = np.load(vecs_path)
    vectors = data['vectors']
    words = list(data['vocab'])
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    i2w = {i: w for i, w in enumerate(words)}
    return vectors, words, i2w


def generate_test_examples(vectors, n_examples=100, n_factors=2, seed=42):
    """Generate random factor combinations for testing.

    Returns:
        factors: list of (indices_list) for ground truth
    """
    rng = np.random.RandomState(seed)
    V = vectors.shape[0]
    examples = []
    for _ in range(n_examples):
        indices = [rng.randint(0, V) for _ in range(n_factors)]
        examples.append(indices)
    return examples


# ---------------------------------------------------------------------------
# Phase 1: 2-Factor Recovery (pure bind)
# ---------------------------------------------------------------------------

def phase1_2factor_bind(vectors, words, i2w, n_examples=100):
    """Test 2-factor recovery with pure circular convolution binding."""
    print("=" * 72)
    print("PHASE 1: 2-Factor Recovery — bind(a, b)")
    print("=" * 72)

    decoder = ResonatorDecoder(
        vectors, max_iter=20, n_restarts=3,
        convergence_patience=3, seed=42
    )
    examples = generate_test_examples(vectors, n_examples, n_factors=2)

    exact_recoveries = 0
    set_recoveries = 0  # Both words correct, any order
    factor_recoveries = {'a': 0, 'b': 0}
    swap_recoveries = 0  # Words recovered but swapped
    wrong_recoveries = 0  # At least one word completely wrong
    iterations = []
    converged_count = 0
    sims_recovered = []
    sims_random = []

    for i, (a_idx, b_idx) in enumerate(examples):
        a_vec = vectors[a_idx]
        b_vec = vectors[b_idx]

        # Create composite via bind (circular convolution)
        composite = ResonatorDecoder.make_composite_bind([a_vec, b_vec])

        # Decode
        result = decoder.decode_2factor(composite, binding_op='bind')
        recovered_a, recovered_b = result['indices']

        # Track recovery types
        gt_set = {a_idx, b_idx}
        recovered_set = {recovered_a, recovered_b}
        set_match = gt_set == recovered_set
        ordered_match = (recovered_a == a_idx and recovered_b == b_idx)
        swap_match = set_match and not ordered_match

        if ordered_match:
            exact_recoveries += 1
        if set_match:
            set_recoveries += 1
        if swap_match:
            swap_recoveries += 1
        if not set_match:
            wrong_recoveries += 1
        if recovered_a == a_idx:
            factor_recoveries['a'] += 1
        if recovered_b == b_idx:
            factor_recoveries['b'] += 1

        iterations.append(result['iterations'])
        if result['converged']:
            converged_count += 1

        # Similarity of recovered vectors to ground truth
        sim_a = similarity(vectors[recovered_a], vectors[a_idx])
        sim_b = similarity(vectors[recovered_b], vectors[b_idx])
        sims_recovered.append((sim_a + sim_b) / 2.0)

        # Baseline: random guess similarity
        random_idx = np.random.randint(0, vectors.shape[0])
        sim_rand = (similarity(vectors[random_idx], vectors[a_idx]) +
                    similarity(vectors[random_idx], vectors[b_idx])) / 2.0
        sims_random.append(sim_rand)

    # Report
    print(f"\n  Tested {n_examples} random word pairs")
    print(f"  Exact ordered match:     {exact_recoveries}/{n_examples} "
          f"({exact_recoveries/n_examples:.1%})")
    print(f"  Set match (any order):   {set_recoveries}/{n_examples} "
          f"({set_recoveries/n_examples:.1%})")
    print(f"  Swap recovery:           {swap_recoveries}/{n_examples} "
          f"({swap_recoveries/n_examples:.1%})")
    print(f"  Wrong (≥1 word wrong):   {wrong_recoveries}/{n_examples} "
          f"({wrong_recoveries/n_examples:.1%})")
    print(f"  Factor A recovery: {factor_recoveries['a']}/{n_examples} "
          f"({factor_recoveries['a']/n_examples:.1%})")
    print(f"  Factor B recovery: {factor_recoveries['b']}/{n_examples} "
          f"({factor_recoveries['b']/n_examples:.1%})")
    print(f"  Converged: {converged_count}/{n_examples}")
    print(f"  Iterations: median={np.median(iterations):.0f}, "
          f"mean={np.mean(iterations):.1f}±{np.std(iterations):.1f}, "
          f"max={np.max(iterations)}")
    print(f"  Similarity recovered vs GT: {np.mean(sims_recovered):.4f}")
    print(f"  Similarity random vs GT:     {np.mean(sims_random):.4f}")

    # Show some examples
    n_show = 5
    print(f"\n  Example decodings (first {n_show}):")
    for i in range(min(n_show, n_examples)):
        a_idx, b_idx = examples[i]
        a_vec = vectors[a_idx]
        b_vec = vectors[b_idx]
        composite = ResonatorDecoder.make_composite_bind([a_vec, b_vec])
        result = decoder.decode_2factor(composite, binding_op='bind')
        ra, rb = result['indices']
        correct = "✓" if (ra == a_idx and rb == b_idx) else "✗"
        print(f"    [{i}] {correct} GT=({i2w[a_idx]}, {i2w[b_idx]}) → "
              f"Recovered=({i2w[ra]}, {i2w[rb]}) "
              f"in {result['iterations']} iters")

    return {
        'exact_rate': exact_recoveries / n_examples,
        'set_rate': set_recoveries / n_examples,
        'swap_rate': swap_recoveries / n_examples,
        'wrong_rate': wrong_recoveries / n_examples,
        'factor_a_rate': factor_recoveries['a'] / n_examples,
        'factor_b_rate': factor_recoveries['b'] / n_examples,
        'median_iterations': np.median(iterations),
        'mean_similarity': np.mean(sims_recovered),
    }


# ---------------------------------------------------------------------------
# Phase 2: 3-Factor Recovery (pure bind)
# ---------------------------------------------------------------------------

def phase2_3factor_bind(vectors, words, i2w, n_examples=100):
    """Test 3-factor recovery with pure circular convolution binding."""
    print()
    print("=" * 72)
    print("PHASE 2: 3-Factor Recovery — bind(bind(a, b), c)")
    print("=" * 72)

    decoder = ResonatorDecoder(
        vectors, max_iter=30, n_restarts=3,
        convergence_patience=4, seed=42
    )
    examples = generate_test_examples(vectors, n_examples, n_factors=3)

    exact_recoveries = 0
    factor_recoveries = [0, 0, 0]
    iterations = []
    converged_count = 0

    for i, (a_idx, b_idx, c_idx) in enumerate(examples):
        a_vec = vectors[a_idx]
        b_vec = vectors[b_idx]
        c_vec = vectors[c_idx]

        # Composite: bind(bind(a, b), c)
        composite = ResonatorDecoder.make_composite_bind([a_vec, b_vec, c_vec])

        # Decode
        result = decoder.decode_3factor(composite, binding_op='bind')
        ra, rb, rc = result['indices']

        all_correct = (ra == a_idx and rb == b_idx and rc == c_idx)
        if all_correct:
            exact_recoveries += 1
        for j, (r, gt) in enumerate(zip([ra, rb, rc], [a_idx, b_idx, c_idx])):
            if r == gt:
                factor_recoveries[j] += 1

        iterations.append(result['iterations'])
        if result['converged']:
            converged_count += 1

    # Report
    print(f"\n  Tested {n_examples} random word triples")
    print(f"  Exact recovery (all 3 factors): {exact_recoveries}/{n_examples} "
          f"({exact_recoveries/n_examples:.1%})")
    for j, label in enumerate(['A', 'B', 'C']):
        print(f"  Factor {label} recovery: {factor_recoveries[j]}/{n_examples} "
              f"({factor_recoveries[j]/n_examples:.1%})")
    print(f"  Converged: {converged_count}/{n_examples}")
    print(f"  Iterations: median={np.median(iterations):.0f}, "
          f"mean={np.mean(iterations):.1f}±{np.std(iterations):.1f}")

    n_show = 5
    print(f"\n  Example decodings (first {n_show}):")
    for i in range(min(n_show, n_examples)):
        a_idx, b_idx, c_idx = examples[i]
        a_vec, b_vec, c_vec = vectors[a_idx], vectors[b_idx], vectors[c_idx]
        composite = ResonatorDecoder.make_composite_bind([a_vec, b_vec, c_vec])
        result = decoder.decode_3factor(composite, binding_op='bind')
        ra, rb, rc = result['indices']
        correct = "✓" if (ra == a_idx and rb == b_idx and rc == c_idx) else "✗"
        print(f"    [{i}] {correct} GT=({i2w[a_idx]}, {i2w[b_idx]}, {i2w[c_idx]})")
        print(f"        Recovered=({i2w[ra]}, {i2w[rb]}, {i2w[rc]}) "
              f"in {result['iterations']} iters")

    return {
        'exact_rate': exact_recoveries / n_examples,
        'factor_rates': [f/n_examples for f in factor_recoveries],
        'median_iterations': np.median(iterations),
    }


# ---------------------------------------------------------------------------
# Phase 3: M (projective_resonance) Recovery
# ---------------------------------------------------------------------------

def phase3_M_recovery(vectors, words, i2w, n_examples=100):
    """Test 2-factor recovery when composite uses projective_resonance."""
    print()
    print("=" * 72)
    print("PHASE 3: 2-Factor Recovery — M(a, b) via projective_resonance")
    print("=" * 72)

    from celn.core import projective_resonance

    decoder = ResonatorDecoder(
        vectors, max_iter=20, n_restarts=3,
        convergence_patience=3, seed=42
    )
    examples = generate_test_examples(vectors, n_examples, n_factors=2)

    exact_recoveries = 0
    set_recoveries = 0
    factor_recoveries = {'a': 0, 'b': 0}
    top5_recoveries = 0
    iterations = []

    for i, (a_idx, b_idx) in enumerate(examples):
        a_vec = vectors[a_idx]
        b_vec = vectors[b_idx]

        # Composite using M (projective_resonance, bilateral for strong non-commutativity)
        composite = ResonatorDecoder.make_composite_M(
            [a_vec, b_vec], gamma=1.0, bilateral=True
        )

        # Decode — uses unbind (circular correlation) which is approximate for M
        result = decoder.decode_2factor(composite, binding_op='M')
        ra, rb = result['indices']

        gt_set = {a_idx, b_idx}
        recovered_set = {ra, rb}

        if ra == a_idx and rb == b_idx:
            exact_recoveries += 1
        if gt_set == recovered_set:
            set_recoveries += 1
        if ra == a_idx:
            factor_recoveries['a'] += 1
        if rb == b_idx:
            factor_recoveries['b'] += 1

        # Top-5: is the ground truth in top-5 nearest neighbors of recovered?
        vec_norm_a = normalize(unbind_vec(composite, vectors[rb]))
        top5_a = set(decoder._nearest(vec_norm_a, top_k=5))
        vec_norm_b = normalize(unbind_vec(composite, vectors[ra]))
        top5_b = set(decoder._nearest(vec_norm_b, top_k=5))
        if a_idx in top5_a and b_idx in top5_b:
            top5_recoveries += 1

        iterations.append(result['iterations'])

    print(f"\n  Tested {n_examples} random word pairs with M(a,b) encoding")
    print(f"  Exact ordered match:     {exact_recoveries}/{n_examples} "
          f"({exact_recoveries/n_examples:.1%})")
    print(f"  Set match (any order):   {set_recoveries}/{n_examples} "
          f"({set_recoveries/n_examples:.1%})")
    print(f"  Factor A (context) recovery: {factor_recoveries['a']}/{n_examples} "
          f"({factor_recoveries['a']/n_examples:.1%})")
    print(f"  Factor B (new word) recovery: {factor_recoveries['b']}/{n_examples} "
          f"({factor_recoveries['b']/n_examples:.1%})")
    print(f"  Top-5 recovery (both factors in top-5): {top5_recoveries}/{n_examples} "
          f"({top5_recoveries/n_examples:.1%})")
    print(f"  Iterations: median={np.median(iterations):.0f}, "
          f"mean={np.mean(iterations):.1f}±{np.std(iterations):.1f}")

    # Show examples
    n_show = 5
    print(f"\n  Example decodings (first {n_show}):")
    for i in range(min(n_show, n_examples)):
        a_idx, b_idx = examples[i]
        a_vec, b_vec = vectors[a_idx], vectors[b_idx]
        composite = ResonatorDecoder.make_composite_M(
            [a_vec, b_vec], gamma=1.0, bilateral=True
        )
        result = decoder.decode_2factor(composite, binding_op='M')
        ra, rb = result['indices']
        correct = "✓" if (ra == a_idx and rb == b_idx) else "✗"
        a_correct = "A✓" if ra == a_idx else "A✗"
        b_correct = "B✓" if rb == b_idx else "B✗"
        print(f"    [{i}] {correct} ({a_correct},{b_correct}) "
              f"GT=({i2w[a_idx]}, {i2w[b_idx]}) → "
              f"R=({i2w[ra]}, {i2w[rb]}) "
              f"in {result['iterations']} iters")

    return {
        'exact_rate': exact_recoveries / n_examples,
        'set_rate': set_recoveries / n_examples,
        'factor_a_rate': factor_recoveries['a'] / n_examples,
        'factor_b_rate': factor_recoveries['b'] / n_examples,
        'top5_rate': top5_recoveries / n_examples,
        'median_iterations': np.median(iterations),
    }


# ---------------------------------------------------------------------------
# Phase 4: Convergence Analysis
# ---------------------------------------------------------------------------

def phase4_convergence(vectors, words, i2w, n_examples=50):
    """Analyze convergence dynamics of Resonator Network."""
    print()
    print("=" * 72)
    print("PHASE 4: Convergence Dynamics")
    print("=" * 72)

    decoder = ResonatorDecoder(
        vectors, max_iter=30, n_restarts=1,
        convergence_patience=30, seed=42  # high patience = no early stop
    )
    examples = generate_test_examples(vectors, n_examples, n_factors=2)

    # Track per-iteration accuracy
    max_iter = 30
    per_iter_accuracy = np.zeros(max_iter)
    per_iter_similarity = np.zeros(max_iter)
    converged_by_iter = np.zeros(max_iter)

    for a_idx, b_idx in examples:
        a_vec, b_vec = vectors[a_idx], vectors[b_idx]
        composite = ResonatorDecoder.make_composite_bind([a_vec, b_vec])

        result = decoder.decode_2factor(composite, binding_op='bind')
        gt_set = {a_idx, b_idx}

        for t, (ra, rb) in enumerate(result['history']):
            if t < max_iter:
                # Set match: both words correct, any order
                if {ra, rb} == gt_set:
                    per_iter_accuracy[t] += 1
                per_iter_similarity[t] += (
                    similarity(vectors[ra], a_vec) +
                    similarity(vectors[rb], b_vec)
                ) / 2.0

        # When did it converge to correct set?
        for t, (ra, rb) in enumerate(result['history'][1:], 1):
            if {ra, rb} == gt_set:
                converged_by_iter[t] += 1
                break

    per_iter_accuracy /= n_examples
    per_iter_similarity /= n_examples
    converged_by_iter = np.cumsum(converged_by_iter) / n_examples

    # Find convergence speed
    median_convergence = np.argmax(converged_by_iter >= 0.5) + 1
    p90_convergence = np.argmax(converged_by_iter >= 0.9) + 1

    print(f"\n  Accuracy vs iteration:")
    for t in [0, 2, 5, 10, 15, 20, 29]:
        print(f"    iter {t:>2}: accuracy={per_iter_accuracy[t]:.1%}  "
              f"avg_sim={per_iter_similarity[t]:.4f}  "
              f"cumulative_converged={converged_by_iter[t]:.1%}")

    print(f"\n  Median convergence: iteration {median_convergence}")
    print(f"  90% converged by:   iteration {p90_convergence}")

    return {
        'median_convergence': median_convergence,
        'p90_convergence': p90_convergence,
        'final_accuracy': per_iter_accuracy[-1],
        'final_similarity': per_iter_similarity[-1],
    }


# ---------------------------------------------------------------------------
# Phase 5: Baseline Comparison — Resonator vs Direct Cosine
# ---------------------------------------------------------------------------

def phase5_baseline(vectors, words, i2w, n_examples=100):
    """Compare Resonator Network against direct cosine similarity baseline."""
    print()
    print("=" * 72)
    print("PHASE 5: Resonator vs Cosine Similarity Baseline")
    print("=" * 72)

    decoder = ResonatorDecoder(
        vectors, max_iter=20, n_restarts=3,
        convergence_patience=3, seed=42
    )
    examples = generate_test_examples(vectors, n_examples, n_factors=2)

    resonator_exact = 0
    cosine_exact = 0
    resonator_ranks = []
    cosine_ranks = []

    resonator_set = 0
    cosine_set = 0

    for a_idx, b_idx in examples:
        a_vec, b_vec = vectors[a_idx], vectors[b_idx]
        composite = ResonatorDecoder.make_composite_bind([a_vec, b_vec])

        # Resonator
        res_result = decoder.decode_2factor(composite, binding_op='bind')
        gt_set = {a_idx, b_idx}
        if res_result['indices'][0] == a_idx and res_result['indices'][1] == b_idx:
            resonator_exact += 1
        if {res_result['indices'][0], res_result['indices'][1]} == gt_set:
            resonator_set += 1

        # Direct cosine baseline
        cos_result = decoder.decode_direct(composite, n_factors=2, top_k=50)

        # Where do a and b rank in cosine similarity?
        all_sims = decoder.codebook @ normalize(composite).astype(np.float32)
        rank_a = int((all_sims > all_sims[a_idx]).sum()) + 1
        rank_b = int((all_sims > all_sims[b_idx]).sum()) + 1
        resonator_ranks.append(1 if res_result['indices'][0] == a_idx else 999)
        cosine_ranks.append(rank_a)

        # Check if cosine gets both in top-2
        if (a_idx in cos_result['indices'][:2] and
                b_idx in cos_result['indices'][:2]):
            cosine_exact += 1

    print(f"\n  Tested {n_examples} word pairs with bind(a,b) composite")
    print(f"\n  Resonator ordered match:  {resonator_exact}/{n_examples} "
          f"({resonator_exact/n_examples:.1%})")
    print(f"  Resonator set match:      {resonator_set}/{n_examples} "
          f"({resonator_set/n_examples:.1%})")
    print(f"  Cosine baseline exact:    {cosine_exact}/{n_examples} "
          f"({cosine_exact/n_examples:.1%})")

    # Rank statistics for cosine
    print(f"\n  Cosine similarity rank of ground-truth words:")
    print(f"    Median rank of factor A: {np.median(cosine_ranks):.0f}")
    print(f"    Mean rank of factor A:   {np.mean(cosine_ranks):.1f}")
    print(f"    Rank ≤ 2:                {sum(1 for r in cosine_ranks if r <= 2)}/{n_examples}")
    print(f"    Rank ≤ 10:               {sum(1 for r in cosine_ranks if r <= 10)}/{n_examples}")

    improvement = resonator_exact / n_examples - cosine_exact / n_examples
    print(f"\n  Resonator improvement over cosine: {improvement:+.1%}")

    return {
        'resonator_exact_rate': resonator_exact / n_examples,
        'resonator_set_rate': resonator_set / n_examples,
        'cosine_exact_rate': cosine_exact / n_examples,
        'improvement': improvement,
        'cosine_median_rank': np.median(cosine_ranks),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("╔" + "═" * 70 + "╗")
    print("║  Resonator Network Decoder — CELN v3                           ║")
    print("║  Can we recover word factors from composite vectors?           ║")
    print("╚" + "═" * 70 + "╝")

    vectors, words, i2w = load_codebook()
    print(f"\n  Codebook: {vectors.shape[0]} words × {vectors.shape[1]}D")

    # Phase 1: 2-factor bind
    r1 = phase1_2factor_bind(vectors, words, i2w, n_examples=100)

    # Phase 2: 3-factor bind
    r2 = phase2_3factor_bind(vectors, words, i2w, n_examples=100)

    # Phase 3: M (projective_resonance)
    r3 = phase3_M_recovery(vectors, words, i2w, n_examples=100)

    # Phase 4: Convergence
    r4 = phase4_convergence(vectors, words, i2w, n_examples=50)

    # Phase 5: Baseline comparison
    r5 = phase5_baseline(vectors, words, i2w, n_examples=100)

    # ================================================================
    # Final Report
    # ================================================================
    print()
    print("╔" + "═" * 70 + "╗")
    print("║  FINAL REPORT                                                ║")
    print("╚" + "═" * 70 + "╝")

    # Criteria
    c1 = r1['set_rate'] > 0.7   # 2-factor set match > 70%
    c2 = r2['exact_rate'] > 0.1  # 3-factor exact > 10% (much harder)
    c3 = r3['set_rate'] > 0.5    # M-encoded set match > 50%
    c4 = r4['median_convergence'] <= 15  # Fast convergence
    c5 = r5['improvement'] > 0.3  # Resonator significantly beats cosine

    passed = sum([c1, c2, c3, c4, c5])

    print(f"\n  1. 2-factor set match > 70%: {r1['set_rate']:.1%} {'✓' if c1 else '✗'}")
    print(f"  2. 3-factor exact > 10%: {r2['exact_rate']:.1%} {'✓' if c2 else '✗'}")
    print(f"  3. M-encoded set match > 50%: {r3['set_rate']:.1%} {'✓' if c3 else '✗'}")
    print(f"  4. Median convergence ≤ 15: {r4['median_convergence']:.0f} {'✓' if c4 else '✗'}")
    print(f"  5. Resonator beats cosine > 30%: {r5['improvement']:+.1%} {'✓' if c5 else '✗'}")

    print(f"\n  Result: {passed}/5 criteria passed")

    if passed >= 4:
        print("\n  CONCLUSION: Resonator Networks WORK for CELN v3.")
        print("  They can recover constituent factors from both bind and")
        print("  M-encoded composites, dramatically outperforming direct")
        print("  cosine similarity. This enables a new decoding pathway:")
        print("  store M-encoded vectors, recover constituents via resonance.")
    elif passed >= 3:
        print("\n  CONCLUSION: Resonator Networks show promise but need")
        print("  tuning. The core mechanism works — iterative unbinding +")
        print("  clean-up converges — but accuracy on M-encoded composites")
        print("  is limited by phi-weight distortion.")
    else:
        print("\n  CONCLUSION: Resonator Networks in current form do not")
        print("  reliably decode CELN composites. Consider: different")
        print("  binding operations, larger codebooks, or alternative")
        print("  cleanup mechanisms.")

    print()
    return passed


if __name__ == '__main__':
    main()
