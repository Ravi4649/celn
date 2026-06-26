#!/usr/bin/env python3
"""
CELN v3 — Full Digestion Cycle for Expanded Corpus
===================================================
Pipeline:
  1. Load 16k-sentence expanded Portuguese corpus
  2. Tokenize, build co-occurrence, compute PPMI
  3. Train 10k-D word vectors via Hebbian updates
  4. Initialize SDM with sentence centroids
  5. Digest all sentences via write_corroborated
  6. Evaluate semantic concentration
  7. Save vectors and SDM state
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celn.train import load_corpus, build_cooccurrence, compute_ppmi, train_vectors
from celn.core import normalize, batch_normalize, similarity, projective_resonance as M
from celn.memory import DenseSDM

D = 10_000  # dimensionality

t0 = time.time()

print("=" * 70)
print("CELN v3 — Full Digestion Cycle (Expanded Corpus)")
print("=" * 70)

# ═══════════════════════════════════════════════════
# PHASE 1: Load and tokenize expanded corpus
# ═══════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("PHASE 1: Loading expanded corpus")
print(f"{'─'*70}")

corpus_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'corpus_pt_expandido.txt'
)

# Load with min_len=1 to keep articles/prepositions for fluency
sentences = load_corpus(corpus_path, min_len=1)
print(f"  Sentences after tokenization: {len(sentences)}")

# Show stats
all_tokens = [w for s in sentences for w in s]
print(f"  Total tokens: {len(all_tokens)}")
print(f"  Unique types: {len(set(all_tokens))}")

# ═══════════════════════════════════════════════════
# PHASE 2: Train word vectors
# ═══════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("PHASE 2: Training word vectors (Hebbian on PPMI)")
print(f"{'─'*70}")

vectors, w2i, i2w, ppmi = train_vectors(
    sentences, dim=D, learning_rate=0.05, epochs=10, window_size=5, verbose=True
)
V = vectors.shape[0]
print(f"  Vocabulary: {V} words")

# ═══════════════════════════════════════════════════
# PHASE 3: Semantic concentration test
# ═══════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("PHASE 3: Semantic concentration (cosine similarity check)")
print(f"{'─'*70}")

TEST_WORDS = ['cobre', 'eletricidade', 'brasil', 'água', 'música', 'gato', 'amor', 'futebol', 'internet', 'energia']
found = [w for w in TEST_WORDS if w in w2i]

all_cosines = []
word_top_sims = {}
for word in found:
    idx = w2i[word]
    sims = vectors @ vectors[idx]
    top5 = sorted([(sims[j], i2w[j]) for j in range(V) if j != idx], reverse=True)[:5]
    word_top_sims[word] = top5
    all_cosines.extend([s[0] for s in top5])
    print(f"  {word}: {', '.join(f'{w}({s:.3f})' for s, w in top5)}")

# Key metrics: mean top-5 similarity, and cosine concentration
mean_top5 = np.mean(all_cosines)
print(f"\n  Mean top-5 cosine similarity: {mean_top5:.4f}")

# Measure angular concentration: P75-P10 of all pairwise cosines
# (lower spread = more concentrated = better for generation)
if V <= 10000:
    sample = min(5000, V)
    ridx = np.random.RandomState(42).choice(V, sample, replace=False)
    sample_vecs = vectors[ridx]
    norms = np.linalg.norm(sample_vecs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    sample_vecs = sample_vecs / norms
    pairwise = sample_vecs @ sample_vecs.T
    triu = pairwise[np.triu_indices_from(pairwise, k=1)]
    p10 = float(np.percentile(triu, 10))
    p75 = float(np.percentile(triu, 75))
    p90 = float(np.percentile(triu, 90))
    print(f"  Cosine P10: {p10:.4f}")
    print(f"  Cosine P75: {p75:.4f}")
    print(f"  Cosine P90: {p90:.4f}")
    print(f"  Dispersion (P90-P10): {p90-p10:.4f}")
    print(f"  (Lower dispersion = more semantic concentration)")
else:
    print(f"  Skipping pairwise (V={V} is large)")

# ═══════════════════════════════════════════════════
# PHASE 4: Initialize SDM
# ═══════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("PHASE 4: Initializing SDM from sentence centroids")
print(f"{'─'*70}")

sdm = DenseSDM(n_locations=8192, activation_pct=0.005, seed=42)

seed_centroids = []
for tokens in sentences[:min(len(sentences), 5000)]:
    idxs = [w2i[w] for w in tokens if w in w2i]
    if len(idxs) >= 3:
        seed_centroids.append(normalize(vectors[idxs].mean(axis=0)))

sdm.initialize_addresses(np.array(seed_centroids))
print(f"  SDM: {sdm.n_locations} locations from {len(seed_centroids)} centroids")
print(f"  Activation: top {sdm.activation_pct:.1%}")

# ═══════════════════════════════════════════════════
# PHASE 5: Digest all sentences into SDM
# ═══════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("PHASE 5: Digesting corpus into SDM (write_corroborated)")
print(f"{'─'*70}")

total_corr = 0
total_contra = 0
total_neutral = 0

digest_start = time.time()
for i, tokens in enumerate(sentences):
    idxs = [w2i[w] for w in tokens if w in w2i]
    if len(idxs) >= 3:
        centroid = normalize(vectors[idxs].mean(axis=0))
        r = sdm.write_corroborated(centroid)
        total_corr += r['corroborating']
        total_contra += r['contradictory']
        total_neutral += r['neutral']

    if (i + 1) % 2000 == 0:
        elapsed = time.time() - digest_start
        print(f"  [{i+1}/{len(sentences)}] corr={total_corr} contra={total_contra} "
              f"neut={total_neutral} ({elapsed:.0f}s)")

digest_time = time.time() - digest_start

print(f"\n  ── Digestion Summary ──")
print(f"  Sentences digested: {len(sentences)}")
print(f"  Total corroborations: {total_corr}")
print(f"  Total contradictions: {total_contra}")
print(f"  Total neutral writes: {total_neutral}")
print(f"  Conflicts isolated:   {sdm.total_conflicts_detected}")
print(f"  Conflict locations:   {sdm.has_conflict.sum()}/{sdm.n_locations}")
print(f"  Digestion time:       {digest_time:.0f}s")

# ═══════════════════════════════════════════════════
# PHASE 6: Evaluate SDM knowledge
# ═══════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("PHASE 6: SDM Knowledge Evaluation")
print(f"{'─'*70}")

# Query by reading known concepts
QUERIES = ['cobre', 'eletricidade', 'brasil', 'água', 'internet', 'amor', 'futebol']
for word in QUERIES:
    if word not in w2i:
        continue
    q_vec = vectors[w2i[word]]
    r = sdm.read_with_confidence(q_vec)
    sims = vectors @ normalize(r['result'])
    top = np.argsort(sims)[-10:][::-1]
    top_words = [f"{i2w[j]}({sims[j]:.3f})" for j in top]
    print(f"  {word}: {', '.join(top_words[:5])}")
    print(f"         trust={r['trust_score']:.3f} conflicts={r['n_conflicts']}")

# ═══════════════════════════════════════════════════
# PHASE 7: Save
# ═══════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("PHASE 7: Saving trained vectors and SDM")
print(f"{'─'*70}")

save_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Save word vectors
npz_path = os.path.join(save_dir, 'celn_expanded_vectors.npz')
np.savez_compressed(npz_path, vectors=vectors, vocab=np.array(list(w2i.keys()), dtype=object))
print(f"  Vectors saved: {npz_path} ({os.path.getsize(npz_path)/1024/1024:.0f} MB)")

# Learn Type Field for generation
print("\n  Learning Type Field from corpus pairs...")
type_dim = 2000
pair_src = []
pair_fol = []
for tokens in sentences:
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i+1]
        if w1 in w2i and w2 in w2i:
            pair_src.append(w2i[w1])
            pair_fol.append(w2i[w2])

# Type vectors from random indexing
rng_type = np.random.RandomState(42)
type_vecs = rng_type.randn(V, type_dim).astype(np.float32)
type_vecs = batch_normalize(type_vecs)

print(f"  Type Field: {V} words × {type_dim} dims")
print(f"  Pairs: {len(pair_src)} edge pairs")

total_time = time.time() - t0
print(f"\n{'='*70}")
print(f"Digestion Cycle Complete — {total_time:.0f}s total")
print(f"  Corpus:  {len(sentences)} sentences → {V} words")
print(f"  Vectors: {npz_path}")
print(f"  SDM:     {sdm.n_locations} locations, {sdm.total_writes} writes")
print(f"  Conflicts: {sdm.total_conflicts_detected}")
print(f"{'='*70}")
