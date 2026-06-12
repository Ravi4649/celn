"""
CELN v3 — Evaluation Framework
===============================
Compare Projective Resonance (with PMI-boosted context window) vs
plain context window baseline on:
  1. Topic Coherence — does generated text stay on-topic?
  2. Source Alignment — does generation match the source continuation?
  3. Diversity — how much of the vocabulary is used?
  4. Repetition — does the system get stuck in loops?
  5. Fluency (subjective) — sample outputs for human judgment
"""

import numpy as np
from collections import Counter
from typing import Optional

from .core import (
    D, normalize, similarity,
    projective_resonance,
    encode_sequence, encode_sequence_plain,
    spectral_entropy,
)
from .generate import (
    ContextWindow, generate, generate_baseline, generate_from_prefix,
    context_window_scores,
)
from .train import tokenize, precompute_spectra


# ---------------------------------------------------------------------------
# Topic Coherence
# ---------------------------------------------------------------------------

def topic_coherence(prefix_tokens: list[str],
                    generated_tokens: list[str],
                    word_vectors: np.ndarray,
                    word2idx: dict[str, int]) -> float:
    """Average cosine similarity of generated words to prefix centroid."""
    prefix_indices = [word2idx[w] for w in prefix_tokens if w in word2idx]
    gen_indices = [word2idx[w] for w in generated_tokens if w in word2idx]

    if not prefix_indices or not gen_indices:
        return 0.0

    prefix_centroid = normalize(word_vectors[prefix_indices].mean(axis=0))
    sims = [similarity(word_vectors[i], prefix_centroid) for i in gen_indices]
    return float(np.mean(sims))


def source_topic_alignment(generated_tokens: list[str],
                           source_tokens: list[str],
                           word_vectors: np.ndarray,
                           word2idx: dict[str, int]) -> float:
    """Similarity between generated centroid and source continuation centroid."""
    gen_indices = [word2idx[w] for w in generated_tokens if w in word2idx]
    src_indices = [word2idx[w] for w in source_tokens if w in word2idx]

    if not gen_indices or not src_indices:
        return 0.0

    gen_centroid = normalize(word_vectors[gen_indices].mean(axis=0))
    src_centroid = normalize(word_vectors[src_indices].mean(axis=0))
    return similarity(gen_centroid, src_centroid)


# ---------------------------------------------------------------------------
# Diversity & Repetition
# ---------------------------------------------------------------------------

def vocabulary_diversity(all_generated: list[list[str]],
                         vocab_size: int) -> dict:
    """Measure vocabulary utilization."""
    all_tokens = []
    for seq in all_generated:
        all_tokens.extend(seq)

    total = len(all_tokens)
    unique = len(set(all_tokens))
    freq_dist = Counter(all_tokens)

    return {
        'unique_words': unique,
        'vocab_coverage': unique / vocab_size,
        'type_token_ratio': unique / max(total, 1),
        'total_tokens': total,
        'top_10_words': freq_dist.most_common(10),
        'hapax_legomena': sum(1 for _, c in freq_dist.items() if c == 1),
    }


def repetition_rate(generated_tokens: list[str]) -> float:
    """Bigram repetition rate."""
    if len(generated_tokens) < 3:
        return 0.0
    bigrams = list(zip(generated_tokens[:-1], generated_tokens[1:]))
    bigram_counts = Counter(bigrams)
    repeated = sum(1 for _, c in bigram_counts.items() if c > 1)
    return repeated / max(len(bigrams), 1)


def prefix_word_overlap(generated_tokens: list[str],
                        prefix_tokens: list[str]) -> float:
    """Fraction of generated words that appear in the prefix."""
    prefix_set = set(prefix_tokens)
    if not generated_tokens:
        return 0.0
    return sum(1 for w in generated_tokens if w in prefix_set) / len(generated_tokens)


# ---------------------------------------------------------------------------
# Novelty (information gain per step)
# ---------------------------------------------------------------------------

def novelty_scores(generated_tokens: list[str],
                   word_vectors: np.ndarray,
                   word2idx: dict[str, int]) -> list[float]:
    """Measure how much each new word differs from the running context.

    High novelty = diverse generation. Low novelty = repetitive.
    """
    novelties = []
    context_vecs = []

    for w in generated_tokens:
        if w in word2idx:
            w_vec = word_vectors[word2idx[w]]
            if context_vecs:
                centroid = normalize(np.mean(context_vecs, axis=0))
                nov = 1.0 - similarity(w_vec, centroid)
                novelties.append(float(nov))
            else:
                novelties.append(0.0)
            context_vecs.append(w_vec)

    return novelties


# ---------------------------------------------------------------------------
# Full Experiment Runner
# ---------------------------------------------------------------------------

def run_experiment(word_vectors: np.ndarray,
                   word2idx: dict[str, int],
                   idx2word: dict[int, str],
                   ppmi_matrix: np.ndarray,
                   sentences: list[list[str]],
                   n_samples: int = 30,
                   gen_length: int = 8,
                   temperature: float = 0.8,
                   boost_weights: list[float] = None,
                   seed: int = 42) -> dict:
    """Run full comparison experiment.

    Compares:
      - Baseline: context window scoring ONLY (no PMI boost)
      - Projective: context window + PMI boost (the full CELN v3 approach)

    across multiple boost_weight values.
    """
    if boost_weights is None:
        boost_weights = [0.0, 0.2, 0.3, 0.5]

    rng = np.random.RandomState(seed)

    # Filter test sentences
    test_sentences = [s for s in sentences if len(s) >= 8]
    if len(test_sentences) > n_samples:
        test_indices = rng.choice(len(test_sentences), n_samples, replace=False)
        test_sentences = [test_sentences[i] for i in test_indices]

    results = {
        'config': {
            'n_samples': len(test_sentences),
            'gen_length': gen_length,
            'temperature': temperature,
            'vocab_size': len(word2idx),
        },
        'conditions': {},  # condition_name → metrics
        'samples': [],
    }

    # Initialize metrics for each condition
    conditions = {f'boost_{bw}': {'boost': bw} for bw in boost_weights}
    conditions['baseline'] = {'boost': None}

    for cond_name in conditions:
        results['conditions'][cond_name] = {
            'coherence': [],
            'source_alignment': [],
            'repetition': [],
            'prefix_overlap': [],
            'novelty': [],
            'diversity_tokens': [],
        }

    print(f"Running experiment on {len(test_sentences)} test sentences...")
    print(f"Conditions: {list(conditions.keys())}")
    print()

    for i, sent in enumerate(test_sentences):
        prefix_tokens = sent[:3]
        prefix_str = ' '.join(prefix_tokens)
        source_continuation = sent[3:3 + gen_length]

        sample = {
            'prefix': prefix_str,
            'source_continuation': ' '.join(source_continuation),
        }

        for cond_name, cond_cfg in conditions.items():
            bw = cond_cfg['boost']
            is_baseline = (bw is None)

            # Build context window from prefix
            window = ContextWindow(max_window=8, decay=0.7)
            prefix_indices = []
            for w in prefix_tokens:
                if w in word2idx:
                    window.add(word_vectors[word2idx[w]])
                    prefix_indices.append(word2idx[w])

            if not prefix_indices:
                continue

            if is_baseline:
                gen, _ = generate_baseline(
                    window, word_vectors, idx2word, word2idx,
                    gen_length, temperature, seed=seed + i
                )
            else:
                gen, _ = generate(
                    window, word_vectors, idx2word, word2idx, ppmi_matrix,
                    gen_length, temperature, gamma=1.0,
                    boost_weight=bw, seed=seed + i
                )

            # Record metrics
            r = results['conditions'][cond_name]
            r['coherence'].append(
                topic_coherence(prefix_tokens, gen, word_vectors, word2idx)
            )
            r['source_alignment'].append(
                source_topic_alignment(gen, source_continuation,
                                       word_vectors, word2idx)
            )
            r['repetition'].append(repetition_rate(gen))
            r['prefix_overlap'].append(
                prefix_word_overlap(gen, prefix_tokens)
            )
            r['novelty'].append(
                np.mean(novelty_scores(gen, word_vectors, word2idx))
            )
            r['diversity_tokens'].extend(gen)

            sample[cond_name] = ' '.join(gen)

        results['samples'].append(sample)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(test_sentences)}...")

    # Aggregate
    results['summary'] = _aggregate(results, list(conditions.keys()))
    return results


def _aggregate(results: dict, condition_names: list[str]) -> dict:
    """Compute mean and std for all metrics across conditions."""
    summary = {}

    for cond_name in condition_names:
        r = results['conditions'][cond_name]
        div_tokens = r['diversity_tokens']
        unique = len(set(div_tokens))
        total = len(div_tokens)

        summary[cond_name] = {
            'coherence': (np.mean(r['coherence']), np.std(r['coherence'])),
            'source_alignment': (np.mean(r['source_alignment']),
                                np.std(r['source_alignment'])),
            'repetition': (np.mean(r['repetition']), np.std(r['repetition'])),
            'prefix_overlap': (np.mean(r['prefix_overlap']),
                              np.std(r['prefix_overlap'])),
            'novelty': (np.mean(r['novelty']), np.std(r['novelty'])),
            'unique_words': unique,
            'total_tokens': total,
            'type_token_ratio': unique / max(total, 1),
        }

    return summary


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: dict):
    """Formatted comparison report."""
    s = results['summary']
    condition_names = list(s.keys())
    cfg = results['config']

    print()
    print("=" * 80)
    print("CELN v3 — Projective Resonance Evaluation Report")
    print("=" * 80)
    print(f"Test sentences: {cfg['n_samples']}")
    print(f"Generation length: {cfg['gen_length']} words each")
    print(f"Temperature: {cfg['temperature']}")
    print(f"Vocabulary: {cfg['vocab_size']} words")
    print()

    # Table header
    name_width = max(len(n) for n in condition_names) + 2
    print(f"{'Metric':<28}", end="")
    for name in condition_names:
        print(f" {name:>{name_width}}", end="")
    print()
    print("-" * (28 + (name_width + 1) * len(condition_names)))

    metrics = [
        ('Topic Coherence ↑', 'coherence'),
        ('Source Alignment ↑', 'source_alignment'),
        ('Repetition Rate ↓', 'repetition'),
        ('Prefix Overlap ↓', 'prefix_overlap'),
        ('Novelty Rate ↑', 'novelty'),
        ('Type/Token Ratio ↑', 'type_token_ratio'),
    ]

    for label, key in metrics:
        print(f"{label:<28}", end="")
        for name in condition_names:
            val = s[name][key]
            if isinstance(val, tuple):
                print(f" {val[0]:>{name_width}.4f}", end="")
            else:
                print(f" {val:>{name_width}.4f}", end="")
        print()

    # Vocabulary stats
    print(f"{'Unique Words ↑':<28}", end="")
    for name in condition_names:
        print(f" {s[name]['unique_words']:>{name_width}d}", end="")
    print()

    print()
    print("Arrows: ↑ higher is better, ↓ lower is better")
    print()

    # Sample generations
    print("=" * 80)
    print("Sample Generations (for manual fluency assessment)")
    print("=" * 80)
    n_show = min(12, len(results['samples']))
    for i, sample in enumerate(results['samples'][:n_show]):
        print(f"\n--- Sample {i+1} ---")
        print(f"  Prefix:         {sample['prefix']}")
        print(f"  Source (real):  {sample['source_continuation']}")
        for name in condition_names:
            if name in sample:
                print(f"  {name:<15}: {sample[name]}")
