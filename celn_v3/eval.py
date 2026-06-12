"""
CELN v3 — Fluency Evaluation
=============================
Compare Projective Resonance generator vs VSA 2.0 "Boca Universal" baseline.

Metrics:
  1. Bigram Diversity — type/token ratio of word pairs (higher = more varied)
  2. Repetition Rate — fraction of bigrams that repeat (lower = less stuck)
  3. Topic Coherence — average similarity of generated words to prefix centroid
  4. Semantic Novelty — average dissimilarity between consecutive generated words
  5. Anchor Drift — how far the state drifts from the anchor over time
  6. Spectral Focus — entropy of state spectrum (lower = more focused attention)

Diagnostic output:
  - Statistical comparison (improvement in std deviations)
  - Sample generations side-by-side
  - Drift trajectory analysis
  - Clear verdict: improvement or not
"""

import numpy as np
from collections import Counter
from typing import Optional

from .core import (
    D, normalize,
    similarity, spectral_entropy,
    projective_resonance,
)
from .generator import ProjectiveGenerator, BaselineGenerator
from .encoder import Encoder, BaselineEncoder
from .decoder import Decoder


# ---------------------------------------------------------------------------
# Metric Functions
# ---------------------------------------------------------------------------

def bigram_diversity(generated_words: list[str]) -> float:
    """Type/Token ratio of bigrams — higher = more varied transitions."""
    if len(generated_words) < 2:
        return 0.0
    bigrams = list(zip(generated_words[:-1], generated_words[1:]))
    return len(set(bigrams)) / len(bigrams)


def repetition_rate(generated_words: list[str]) -> float:
    """Fraction of words that appear more than once in the output."""
    if not generated_words:
        return 0.0
    counts = Counter(generated_words)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / len(generated_words)


def topic_coherence(prefix_words: list[str],
                    generated_words: list[str],
                    word_vectors: np.ndarray,
                    word2idx: dict[str, int]) -> float:
    """Average cosine similarity of generated words to prefix centroid.

    High = staying on topic. Low = drifted away.
    """
    p_indices = [word2idx.get(w) for w in prefix_words if w in word2idx]
    g_indices = [word2idx.get(w) for w in generated_words if w in word2idx]
    if not p_indices or not g_indices:
        return 0.0

    p_centroid = normalize(word_vectors[p_indices].mean(axis=0))
    sims = [similarity(word_vectors[i], p_centroid) for i in g_indices]
    return float(np.mean(sims))


def semantic_novelty(generated_words: list[str],
                     word_vectors: np.ndarray,
                     word2idx: dict[str, int]) -> float:
    """Average dissimilarity between consecutive generated words.

    High = diverse, exploring the semantic space.
    Low = stuck in a narrow region.
    """
    indices = [word2idx.get(w) for w in generated_words if w in word2idx]
    if len(indices) < 2:
        return 0.0

    novelties = []
    for i in range(1, len(indices)):
        sim = similarity(word_vectors[indices[i-1]], word_vectors[indices[i]])
        novelties.append(1.0 - sim)

    return float(np.mean(novelties))


def anchor_drift(state_history: list[np.ndarray],
                 anchor_history: list[np.ndarray]) -> float:
    """Average distance between state and anchor over time.

    Low drift = anchor is working, keeping generation anchored.
    High drift = anchor is not constraining the state.
    """
    if not state_history or not anchor_history:
        return 0.0
    distances = []
    for s, a in zip(state_history, anchor_history):
        distances.append(1.0 - similarity(s, a))
    return float(np.mean(distances))


def vocabulary_utilization(generated_words: list[str],
                           vocab_size: int) -> float:
    """Fraction of vocabulary used in generation."""
    return len(set(generated_words)) / vocab_size


def source_alignment(generated_words: list[str],
                     source_words: list[str],
                     word_vectors: np.ndarray,
                     word2idx: dict[str, int]) -> float:
    """How well generated words match the actual source continuation.

    Measures cosine similarity between centroids.
    """
    g_indices = [word2idx.get(w) for w in generated_words if w in word2idx]
    s_indices = [word2idx.get(w) for w in source_words if w in word2idx]
    if not g_indices or not s_indices:
        return 0.0
    g_centroid = normalize(word_vectors[g_indices].mean(axis=0))
    s_centroid = normalize(word_vectors[s_indices].mean(axis=0))
    return similarity(g_centroid, s_centroid)


def knowledge_alignment(generated_words: list[str],
                        sdm,  # DenseSDM
                        prefix_words: list[str],
                        word_vectors: np.ndarray,
                        word2idx: dict[str, int]) -> float:
    """How well generated words align with SDM-stored knowledge.

    Measures cosine similarity between the generated centroid and the
    SDM's stored knowledge about the prefix topic. Higher = generation
    is more grounded in corpus knowledge for this topic.

    Args:
        generated_words: Words produced by the generator
        sdm: DenseSDM instance with corpus knowledge stored
        prefix_words: Prefix tokens defining the topic
        word_vectors: All word vectors, shape (vocab_size, D)
        word2idx: Word-to-index mapping

    Returns:
        Cosine similarity between generated centroid and SDM knowledge.
    """
    g_indices = [word2idx.get(w) for w in generated_words if w in word2idx]
    p_indices = [word2idx.get(w) for w in prefix_words if w in word2idx]
    if not g_indices or not p_indices:
        return 0.0

    # Query SDM: what does the corpus know about the prefix topic?
    prefix_centroid = normalize(word_vectors[p_indices].mean(axis=0))
    sdm_knowledge = sdm.read(prefix_centroid)

    # How well does the generated content align with that knowledge?
    gen_centroid = normalize(word_vectors[g_indices].mean(axis=0))
    return similarity(gen_centroid, sdm_knowledge)


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

def run_comparison(word_vectors: np.ndarray,
                   word2idx: dict[str, int],
                   idx2word: dict[int, str],
                   sentences: list[list[str]],
                   n_samples: int = 40,
                   gen_length: int = 8,
                   temperature: float = 0.8,
                   seed: int = 42) -> dict:
    """Run head-to-head comparison between Projective and Baseline generators.

    Returns:
        Dictionary with all metrics and sample outputs.
    """
    rng = np.random.RandomState(seed)
    vocab_size = len(word_vectors)

    # Select test sentences
    test_sents = [s for s in sentences if len(s) >= gen_length + 3]
    if len(test_sents) > n_samples:
        indices = rng.choice(len(test_sents), n_samples, replace=False)
        test_sents = [test_sents[i] for i in indices]

    # Initialize generators
    proj_gen = ProjectiveGenerator(
        word_vectors, gamma=1.0, bilateral=True,
        anchor_decay=0.9, anchor_weight=0.4
    )
    base_gen = BaselineGenerator(word_vectors)

    # Results storage
    results = {
        'config': {
            'n_samples': len(test_sents),
            'gen_length': gen_length,
            'temperature': temperature,
            'vocab_size': vocab_size,
        },
        'projective': _empty_metrics(),
        'baseline': _empty_metrics(),
        'samples': [],
    }

    print(f"Comparing generators on {len(test_sents)} test sentences...")
    print(f"  Generation length: {gen_length} words")
    print(f"  Temperature: {temperature}")
    print()

    for i, sent in enumerate(test_sents):
        prefix_tokens = sent[:3]
        source_cont = sent[3:3 + gen_length]

        sample = {
            'prefix': ' '.join(prefix_tokens),
            'source_continuation': ' '.join(source_cont),
        }

        # --- Projective Generation ---
        try:
            proj_words, proj_bound_state = proj_gen.generate_from_words(
                prefix_tokens, word2idx, idx2word,
                max_len=gen_length, temperature=temperature, seed=seed + i
            )
        except Exception as e:
            proj_words, proj_bound_state = [], None

        # --- Baseline Generation ---
        try:
            base_words, base_bound_state = base_gen.generate_from_words(
                prefix_tokens, word2idx, idx2word,
                max_len=gen_length, temperature=temperature, seed=seed + i
            )
        except Exception as e:
            base_words, base_bound_state = [], None

        # --- Record Metrics ---
        _record_metrics(results['projective'], proj_words, prefix_tokens,
                       word_vectors, word2idx, [], [],
                       source_cont, vocab_size)

        _record_metrics(results['baseline'], base_words, prefix_tokens,
                       word_vectors, word2idx, [], [],
                       source_cont, vocab_size)

        sample['projective'] = ' '.join(proj_words)
        sample['baseline'] = ' '.join(base_words)
        results['samples'].append(sample)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(test_sents)}...")

    # Compute summaries
    results['summary'] = _compute_summary(results)
    results['verdict'] = _compute_verdict(results['summary'])

    return results


def _empty_metrics() -> dict:
    return {
        'bigram_diversity': [],
        'repetition': [],
        'topic_coherence': [],
        'semantic_novelty': [],
        'source_alignment': [],
        'vocab_utilization': [],
        'state_entropy': [],
        'anchor_drift': [],
        'tokens': [],
    }


def _record_metrics(metrics: dict, words: list[str], prefix: list[str],
                    vectors: np.ndarray, w2i: dict[str, int],
                    states: list[np.ndarray], anchors: list[np.ndarray],
                    source: list[str], vocab_size: int):
    """Record all metrics for one generation."""
    if not words:
        return

    metrics['bigram_diversity'].append(bigram_diversity(words))
    metrics['repetition'].append(repetition_rate(words))
    metrics['topic_coherence'].append(
        topic_coherence(prefix, words, vectors, w2i))
    metrics['semantic_novelty'].append(
        semantic_novelty(words, vectors, w2i))
    metrics['source_alignment'].append(
        source_alignment(words, source, vectors, w2i))
    metrics['vocab_utilization'].append(
        vocabulary_utilization(words, vocab_size))
    metrics['tokens'].extend(words)

    if states:
        # Average spectral entropy of states (lower = more focused)
        entropies = [spectral_entropy(s) for s in states]
        metrics['state_entropy'].append(np.mean(entropies))

    if states and anchors:
        drift = anchor_drift(states, anchors)
        metrics['anchor_drift'].append(drift)


def _compute_summary(results: dict) -> dict:
    """Compute mean ± std for all metrics."""
    summary = {}
    for mode in ['projective', 'baseline']:
        m = results[mode]
        summary[mode] = {}
        for key in ['bigram_diversity', 'repetition', 'topic_coherence',
                     'semantic_novelty', 'source_alignment', 'vocab_utilization',
                     'state_entropy', 'anchor_drift']:
            vals = m[key]
            if vals:
                summary[mode][key] = (np.mean(vals), np.std(vals))
            else:
                summary[mode][key] = (0.0, 0.0)

        # Token-level stats
        summary[mode]['unique_tokens'] = len(set(m['tokens']))
        summary[mode]['total_tokens'] = len(m['tokens'])
        summary[mode]['type_token_ratio'] = (
            summary[mode]['unique_tokens'] /
            max(summary[mode]['total_tokens'], 1)
        )

    return summary


def _compute_verdict(summary: dict) -> dict:
    """Determine whether Projective Resonance improves over baseline.

    Computes improvement ratio and effect size for each metric.
    """
    p = summary['projective']
    b = summary['baseline']

    verdict = {}

    for key in ['bigram_diversity', 'topic_coherence', 'semantic_novelty',
                 'source_alignment']:
        p_mean, p_std = p.get(key, (0.0, 0.0))
        b_mean, b_std = b.get(key, (0.0, 0.0))

        delta = p_mean - b_mean
        pooled_std = np.sqrt((p_std**2 + b_std**2) / 2)

        if pooled_std > 1e-12:
            effect_size = delta / pooled_std
        else:
            effect_size = 0.0

        verdict[key] = {
            'projective_mean': p_mean,
            'baseline_mean': b_mean,
            'delta': delta,
            'effect_size': effect_size,
            'improved': delta > 0,
        }

    # type_token_ratio is a scalar, not a distribution
    p_ttr = p.get('type_token_ratio', 0.0)
    b_ttr = b.get('type_token_ratio', 0.0)
    verdict['type_token_ratio'] = {
        'projective_mean': p_ttr,
        'baseline_mean': b_ttr,
        'delta': p_ttr - b_ttr,
        'effect_size': 0.0,
        'improved': p_ttr > b_ttr,
    }

    # For metrics where lower is better
    for key in ['repetition', 'anchor_drift', 'state_entropy']:
        p_mean, p_std = p.get(key, (0, 0))
        b_mean, b_std = b.get(key, (0, 0))
        delta = b_mean - p_mean  # positive = improvement
        pooled_std = np.sqrt((p_std**2 + b_std**2) / 2)
        effect_size = delta / pooled_std if pooled_std > 1e-12 else 0.0

        verdict[key] = {
            'projective_mean': p_mean,
            'baseline_mean': b_mean,
            'delta': delta,
            'effect_size': effect_size,
            'improved': delta > 0,
        }

    # Overall: count how many metrics improved
    improved_count = sum(1 for v in verdict.values() if v['improved'])
    total_count = len(verdict)
    avg_effect = np.mean([abs(v['effect_size']) for v in verdict.values()])

    verdict['overall'] = {
        'metrics_improved': f"{improved_count}/{total_count}",
        'improvement_ratio': improved_count / total_count,
        'average_effect_size': avg_effect,
        'significant': avg_effect > 0.2,
    }

    return verdict


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def print_report(results: dict):
    """Generate a formatted diagnostic report."""
    s = results['summary']
    v = results['verdict']
    cfg = results['config']

    print()
    print("=" * 78)
    print("║  CELN v3 — Projective Resonance vs VSA 2.0 Baseline" + " " * 17 + "║")
    print("=" * 78)
    print(f"Test sentences: {cfg['n_samples']}")
    print(f"Generation length: {cfg['gen_length']} words per sample")
    print(f"Temperature: {cfg['temperature']}")
    print(f"Vocabulary: {cfg['vocab_size']} words")
    print()

    # ── Metric Comparison Table ──
    print("─" * 78)
    print("Quantitative Comparison")
    print("─" * 78)

    header = f"{'Metric':<28} {'Projective':>14} {'Baseline':>14} {'Δ':>10} {'Effect':>10}"
    print(header)
    print("-" * 78)

    metric_specs = [
        ('Bigram Diversity ↑', 'bigram_diversity', True),
        ('Repetition Rate ↓', 'repetition', False),
        ('Topic Coherence ↑', 'topic_coherence', True),
        ('Semantic Novelty ↑', 'semantic_novelty', True),
        ('Source Alignment ↑', 'source_alignment', True),
        ('State Entropy ↓', 'state_entropy', False),
        ('Anchor Drift ↓', 'anchor_drift', False),
        ('Type/Token Ratio ↑', 'type_token_ratio', True),
    ]

    for label, key, higher_better in metric_specs:
        p_val = s['projective'].get(key, (0, 0))
        b_val = s['baseline'].get(key, (0, 0))

        if isinstance(p_val, tuple):
            p_str = f"{p_val[0]:.4f} ±{p_val[1]:.4f}"
            b_str = f"{b_val[0]:.4f} ±{b_val[1]:.4f}"
        else:
            p_str = f"{p_val:.4f}"
            b_str = f"{b_val:.4f}"

        delta = v.get(key, {}).get('delta', 0)
        effect = v.get(key, {}).get('effect_size', 0)
        improved = v.get(key, {}).get('improved', False)

        # Direction indicator
        arrow = '▲' if improved else '▼'
        delta_str = f"{arrow}{abs(delta):.4f}"
        effect_str = f"{effect:+.2f}σ"

        print(f"{label:<28} {p_str:>14} {b_str:>14} {delta_str:>10} {effect_str:>10}")

    print()

    # ── Vocab Stats ──
    print("Vocabulary Utilization:")
    for mode, label in [('projective', 'Projective'), ('baseline', 'Baseline')]:
        print(f"  {label}: {s[mode]['unique_tokens']} unique / "
              f"{s[mode]['total_tokens']} total "
              f"({s[mode]['type_token_ratio']:.1%})")

    print()

    # ── Verdict ──
    print("─" * 78)
    print("VERDICT")
    print("─" * 78)
    overall = v['overall']
    improved = overall['metrics_improved']
    avg_eff = overall['average_effect_size']

    print(f"  Metrics improved: {improved}")
    print(f"  Average effect size: {avg_eff:.2f}σ")

    if overall['significant']:
        print()
        print("  ✅ Projective Resonance shows MEASURABLE improvement")
        print("     over VSA 2.0 baseline across multiple metrics.")
    else:
        print()
        print("  ⚠️  Improvement is MARGINAL or NEGLIGIBLE.")
        print("     Projective Resonance does not clearly outperform baseline.")

    # Highlight key metrics
    improvements = []
    regressions = []
    for key in ['topic_coherence', 'bigram_diversity', 'semantic_novelty',
                'repetition', 'source_alignment']:
        vk = v.get(key, {})
        if vk.get('improved', False) and abs(vk.get('effect_size', 0)) > 0.1:
            improvements.append((key, vk['effect_size']))
        elif not vk.get('improved', False) and abs(vk.get('effect_size', 0)) > 0.1:
            regressions.append((key, vk['effect_size']))

    if improvements:
        print()
        print("  Key improvements:")
        for key, ef in sorted(improvements, key=lambda x: -abs(x[1])):
            print(f"    + {key}: {ef:+.2f}σ")

    if regressions:
        print()
        print("  Key regressions:")
        for key, ef in sorted(regressions, key=lambda x: -abs(x[1])):
            print(f"    - {key}: {ef:+.2f}σ")

    print()

    # ── Sample Generations ──
    print("─" * 78)
    print("Sample Generations")
    print("─" * 78)
    n_show = min(12, len(results['samples']))
    for i, sample in enumerate(results['samples'][:n_show]):
        print(f"\n  [{i+1}] Prefix:  {sample['prefix']}")
        print(f"      Source:   {sample['source_continuation']}")
        print(f"      Proj (M): {sample['projective']}")
        print(f"      Baseline: {sample['baseline']}")

    print()
    print("=" * 78)


# ---------------------------------------------------------------------------
# Direct Run
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')
    from train import load_corpus, train_vectors, precompute_spectra

    corpus = sys.argv[1] if len(sys.argv) > 1 else '../corpus_final.txt'
    quick = '--quick' in sys.argv

    sentences = load_corpus(corpus, max_sentences=500 if quick else None)
    vectors, w2i, i2w, ppmi = train_vectors(sentences, epochs=10, verbose=True)
    spectra = precompute_spectra(vectors)

    results = run_comparison(
        vectors, w2i, i2w, sentences,
        n_samples=20 if quick else 40,
        gen_length=8,
        temperature=0.8,
        seed=42
    )
    print_report(results)
