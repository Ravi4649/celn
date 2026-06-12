"""
Test Operator Algebra — CELN v3
================================
Function words as linear operators (low-rank transformation matrices),
content words as vectors. Alternating structure/content generation.

Hypothesis: Operators direct the state toward regions where the next
content word lives, enabling natural alternation between function and
content words without frequency counting.
"""

import sys, os, re, numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celn_v3.train import tokenize, build_cooccurrence, compute_ppmi, batch_normalize
from celn_v3.core import normalize, similarity, projective_resonance
from celn_v3.operators import (
    identify_operators, OperatorMemory, OperatorGenerator
)
from celn_v3.fluency import build_directional_bigrams, DirectionalGenerator

FUNCTION_WORDS = {
    'o','a','os','as','um','uma','uns','umas','de','do','da','dos','das',
    'em','no','na','nos','nas','por','pelo','pela','para','com','sem',
    'e','ou','mas','que','se','nem','é','foi','era','são','não','sim',
    'como','quando','onde','porque','até','sob','sobre','entre',
}

# ---------------------------------------------------------------------------
# Phase 1: Setup
# ---------------------------------------------------------------------------

def phase1_setup():
    """Load corpus, train SVD vectors, identify operators."""
    print("=" * 72)
    print("PHASE 1: Setup — Load Corpus & Identify Operators")
    print("=" * 72)

    corpus_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'corpus_final.txt'
    )
    with open(corpus_path) as f:
        text = f.read()
    raw = re.split(r'[.!?\n]+', text)
    sentences_full = []
    for s in raw:
        tokens = tokenize(s, min_len=1)
        if len(tokens) >= 3:
            sentences_full.append(tokens)
    print(f"  Loaded {len(sentences_full)} sentences (min_len=1)")

    # Train SVD vectors on full vocabulary
    from sklearn.decomposition import TruncatedSVD
    word_counts, cooc_counts, w2i, i2w = build_cooccurrence(sentences_full, window_size=5)
    vocab_size = len(w2i)
    ppmi = compute_ppmi(word_counts, cooc_counts, w2i)
    n_components = min(10000, vocab_size - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    vecs_reduced = svd.fit_transform(ppmi)
    sv = svd.singular_values_
    var_ratio = sv**2 / (sv**2).sum()
    weights = var_ratio / var_ratio.max()
    vecs_weighted = vecs_reduced * weights[None, :]
    rng = np.random.RandomState(42)
    R = rng.randn(n_components, 10000) / np.sqrt(n_components)
    vectors = vecs_weighted @ R
    vectors = batch_normalize(vectors)
    print(f"  SVD vectors: {vectors.shape}")

    # Identify operators: take top 20 by composite score (freq * entropy)
    all_ops = identify_operators(sentences_full, w2i, vectors, freq_percentile=95)
    # Cap at 20 operators for memory efficiency
    operators = all_ops[:20]
    print(f"\n  Auto-identified {len(all_ops)} operators, using top {len(operators)}:")
    for i, op in enumerate(operators):
        freq = word_counts.get(op, 0)
        is_func = "FUNC" if op in FUNCTION_WORDS else "?"
        print(f"    [{i:>2}] {op:<12} freq={freq:>5}  {is_func}")

    func_hits = sum(1 for op in operators if op in FUNCTION_WORDS)
    print(f"  Functional word hits: {func_hits}/{len(operators)}")

    return sentences_full, vectors, ppmi, w2i, i2w, operators


# ---------------------------------------------------------------------------
# Phase 2: Learn operator matrices
# ---------------------------------------------------------------------------

def phase2_learn(sentences, vectors, w2i, i2w, operators):
    """Learn operator bias vectors from corpus followers."""
    print()
    print("=" * 72)
    print("PHASE 2: Learn Operator Biases (Directional Vectors)")
    print("=" * 72)

    op_memory = OperatorMemory(operators, w2i, dim=vectors.shape[1], alpha=0.3)

    print(f"  Collecting followers for {len(operators)} operators...")
    op_memory.learn_from_corpus(sentences, w2i, vectors)

    stats = op_memory.stats
    print(f"  Samples per operator: {stats['samples_per_op'][:5]}... (first 5)")
    print(f"  Memory: {stats['memory_mb']} MB")

    # Finalize: bias = centroid(content_followers)
    # Content followers are words that follow the operator
    # but are NOT themselves operators
    op_memory.finalize()
    print(f"  Biases computed (content-word followers only). Done.")

    return op_memory


# ---------------------------------------------------------------------------
# Phase 3: Validate operator directions
# ---------------------------------------------------------------------------

def phase3_validate(sentences, vectors, w2i, i2w, operators, op_memory):
    """Verify that operators map states toward their actual followers."""
    print()
    print("=" * 72)
    print("PHASE 3: Validate Operator Directions")
    print("=" * 72)

    # For each operator, compute what words M_op @ state points to
    # and compare with actual followers in the corpus
    print(f"\n  Operator → Top-5 words pointed to | Actual top-5 followers")
    print(f"  {'─'*55}")

    # Build follower sets from corpus
    follower_counts = {}
    for s in sentences:
        for i in range(len(s) - 1):
            w1, w2 = s[i], s[i + 1]
            if w1 not in follower_counts:
                follower_counts[w1] = Counter()
            follower_counts[w1][w2] += 1

    precisions = {}

    for op_word in operators[:15]:
        if op_word not in w2i or op_word not in follower_counts:
            continue

        # Use a neutral state (average of all content vectors)
        # to test what the operator does generically
        neutral_state = normalize(vectors.mean(axis=0))

        # Apply operator
        transformed = op_memory.apply(op_word, neutral_state)

        # Find nearest words
        sims = vectors @ transformed
        top_indices = np.argsort(sims)[-10:][::-1]
        top_words = [(i2w[i], float(sims[i])) for i in top_indices if sims[i] > 0]

        # Actual followers
        actual = [w for w, _ in follower_counts[op_word].most_common(10)]
        if len(actual) > 5:
            actual = actual[:5]

        # Precision: how many top-10 pointed-to words are actual followers?
        pointed_set = {w for w, _ in top_words[:10]}
        actual_set = set(actual)
        hits = len(pointed_set & actual_set)
        prec = hits / min(10, len(actual_set)) if actual_set else 0
        precisions[op_word] = prec

        top_str = ', '.join(f'{w}({s:.2f})' for w, s in top_words[:5])
        actual_str = ', '.join(actual[:5])
        status = '✓' if prec > 0.3 else ('△' if prec > 0.1 else '✗')
        print(f"  {status} {op_word:<12} → {top_str}")
        print(f"    {'':12}   actual: {actual_str}")

    avg_prec = np.mean(list(precisions.values())) if precisions else 0
    print(f"\n  Average Precision@10: {avg_prec:.1%}")
    print(f"  Operators above 30%: {sum(1 for p in precisions.values() if p > 0.3)}")

    return avg_prec


# ---------------------------------------------------------------------------
# Phase 4: Generate and compare
# ---------------------------------------------------------------------------

def phase4_generate(vectors, w2i, i2w, operators, op_memory):
    """Compare OperatorGenerator vs DirectionalGenerator baseline."""
    print()
    print("=" * 72)
    print("PHASE 4: Generate — Operator Algebra vs Directional Bigrams")
    print("=" * 72)

    # Rebuild sentences for bigram model
    corpus_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'corpus_final.txt'
    )
    with open(corpus_path) as f:
        text = f.read()
    raw = re.split(r'[.!?\n]+', text)
    sents = []
    for s in raw:
        tokens = tokenize(s, min_len=1)
        if len(tokens) >= 3:
            sents.append(tokens)
    bigram_prob = build_directional_bigrams(sents, w2i, vectors.shape[0])

    baseline_gen = DirectionalGenerator(
        vectors, bigram_prob, w2i, i2w,
        window_size=8, window_decay=0.7, base_structure_weight=0.35
    )

    op_gen = OperatorGenerator(
        vectors, op_memory, w2i, i2w, operators,
        window_size=8, window_decay=0.7, benefit_percentile=70
    )

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

    gen_length = 12
    temperature = 0.8
    seed = 42
    results = []

    for prefix_str in test_prefixes:
        prefix_tokens = tokenize(prefix_str, min_len=1)
        prefix_known = [w for w in prefix_tokens if w in w2i]

        if len(prefix_known) < 2:
            continue

        # Baseline
        try:
            bl_words = baseline_gen.generate(
                prefix_known, gen_length, temperature, seed=seed
            )
        except Exception as e:
            bl_words = [f"ERR:{e}"]

        # Operator
        try:
            op_words = op_gen.generate(
                prefix_known, gen_length, temperature, seed=seed
            )
        except Exception as e:
            op_words = [f"ERR:{e}"]

        # Metrics
        bl_func = sum(1 for w in bl_words if w in FUNCTION_WORDS) / max(len(bl_words), 1)
        op_func = sum(1 for w in op_words if w in FUNCTION_WORDS) / max(len(op_words), 1)

        # Bigram authenticity
        if len(bl_words) >= 2:
            bl_bigrams = list(zip(bl_words[:-1], bl_words[1:]))
            bl_auth = sum(1 for (a,b) in bl_bigrams
                         if a in w2i and b in w2i and
                         bigram_prob[w2i[a], w2i[b]] > 0.01) / len(bl_bigrams)
        else:
            bl_auth = 0
        if len(op_words) >= 2:
            op_bigrams = list(zip(op_words[:-1], op_words[1:]))
            op_auth = sum(1 for (a,b) in op_bigrams
                         if a in w2i and b in w2i and
                         bigram_prob[w2i[a], w2i[b]] > 0.01) / len(op_bigrams)
        else:
            op_auth = 0

        # Topic coherence
        p_indices = [w2i[w] for w in prefix_known if w in w2i]
        p_centroid = normalize(vectors[p_indices].mean(axis=0))
        bl_idx = [w2i[w] for w in bl_words if w in w2i]
        op_idx = [w2i[w] for w in op_words if w in w2i]
        bl_coh = float(np.dot(p_centroid,
            normalize(vectors[bl_idx].mean(axis=0)))) if bl_idx else 0
        op_coh = float(np.dot(p_centroid,
            normalize(vectors[op_idx].mean(axis=0)))) if op_idx else 0

        # Operator alternation
        op_count = sum(1 for w in op_words if w in operators)
        op_ratio = op_count / max(len(op_words), 1)

        print(f"\n  Prefix: '{' '.join(prefix_known)}'")
        print(f"  Baseline:  {' '.join(bl_words)}")
        print(f"  Operators: {' '.join(op_words)}")
        print(f"  Func%: BL={bl_func:.0%}  OP={op_func:.0%}")
        print(f"  BigrAuth: BL={bl_auth:.0%}  OP={op_auth:.0%}")
        print(f"  TopicCoh: BL={bl_coh:.3f}  OP={op_coh:.3f}")
        print(f"  OpsUsed: {op_ratio:.0%}")

        results.append(dict(
            bl_func=bl_func, op_func=op_func,
            bl_auth=bl_auth, op_auth=op_auth,
            bl_coh=bl_coh, op_coh=op_coh,
            op_ratio=op_ratio,
        ))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("╔" + "═" * 70 + "╗")
    print("║  Operator Algebra — CELN v3                                  ║")
    print("║  Function words = linear operators, content = vectors        ║")
    print("╚" + "═" * 70 + "╝")

    # Phase 1
    sentences, vectors, ppmi, w2i, i2w, operators = phase1_setup()

    # Phase 2
    op_memory = phase2_learn(sentences, vectors, w2i, i2w, operators)

    # Phase 3
    avg_prec = phase3_validate(sentences, vectors, w2i, i2w, operators, op_memory)

    # Phase 4
    results = phase4_generate(vectors, w2i, i2w, operators, op_memory)

    # ================================================================
    print()
    print("╔" + "═" * 70 + "╗")
    print("║  FINAL REPORT                                                ║")
    print("╚" + "═" * 70 + "╝")

    bl_func = np.mean([r['bl_func'] for r in results])
    op_func = np.mean([r['op_func'] for r in results])
    bl_auth = np.mean([r['bl_auth'] for r in results])
    op_auth = np.mean([r['op_auth'] for r in results])
    bl_coh = np.mean([r['bl_coh'] for r in results])
    op_coh = np.mean([r['op_coh'] for r in results])
    op_ratio = np.mean([r['op_ratio'] for r in results])

    print(f"\n  {'Metric':<28} {'Baseline':>10} {'Operators':>10} {'Δ':>10}")
    print(f"  {'─'*28} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'Function Word Ratio':<28} {bl_func:>10.1%} {op_func:>10.1%} {op_func-bl_func:>+10.1%}")
    print(f"  {'Bigram Authenticity':<28} {bl_auth:>10.1%} {op_auth:>10.1%} {op_auth-bl_auth:>+10.1%}")
    print(f"  {'Topic Coherence':<28} {bl_coh:>10.4f} {op_coh:>10.4f} {op_coh-bl_coh:>+10.4f}")
    print(f"  {'Operator Usage':<28} {'—':>10} {op_ratio:>10.1%}")

    # Criteria
    c1 = avg_prec > 0.3  # Operators learn correct directions
    c2 = op_func > 0.20  # Operators appear naturally
    c3 = op_coh > bl_coh - 0.05  # Coherence not degraded

    passed = sum([c1, c2, c3])
    print(f"\n  1. Operator precision@10 > 30%: {avg_prec:.1%} {'✓' if c1 else '✗'}")
    print(f"  2. Function word ratio > 20%: {op_func:.1%} {'✓' if c2 else '✗'}")
    print(f"  3. Topic coherence preserved: {op_coh-bl_coh:+.4f} {'✓' if c3 else '✗'}")
    print(f"\n  Result: {passed}/3 criteria passed")

    if passed >= 2:
        print(f"\n  CONCLUSION: The operator algebra approach shows that")
        print(f"  function words CAN be modeled as linear transformations.")
        print(f"  Precision@10={avg_prec:.1%} indicates the matrices learn")
        print(f"  directional mappings from corpus statistics.")
        if op_func > bl_func:
            print(f"  Operator usage ({op_func:.1%}) demonstrates natural")
            print(f"  alternation between structure and content.")
    else:
        print(f"\n  CONCLUSION: The operator algebra needs refinement.")
        print(f"  Consider: higher rank, more training data, or different")
        print(f"  identification criteria for operators.")

    print()
    return passed


if __name__ == '__main__':
    main()
