"""
Test: PMI-Weighted Random Indexing — Concentrating without Collapsing
======================================================================
The pure Random Indexing collapsed because function words act as hubs.
This variant weights each context neighbor by PPMI(word, neighbor):
  - Strong co-occurrence → strong pull
  - Weak/accidental co-occurrence → weak pull
  - No backprop, no fixed lists, no classification
"""
import sys, numpy as np, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/ravizin/celn-v3')
from celn.train import load_corpus, build_cooccurrence, compute_ppmi, tokenize

def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(a @ b)

# ------------------------------------------------------------------
# 1. LOAD & BUILD PPMI
# ------------------------------------------------------------------
print("=" * 70)
print("PMI-WEIGHTED RANDOM INDEXING — ANTI-COLLAPSE TEST")
print("=" * 70)

data = np.load('/home/ravizin/celn-v3/celn_full_vectors.npz', allow_pickle=True)
vectors = data['vectors'].astype(np.float32)
vocab = data['vocab']
w2i = {w: i for i, w in enumerate(vocab)}
V, D = vectors.shape
print(f"\nLoaded {V} words × {D} dimensions")

sentences = load_corpus('/home/ravizin/celn-v3/corpus_final.txt', min_len=2)
print(f"Corpus: {len(sentences)} sentences")

word_counts, cooc_counts, _, _ = build_cooccurrence(sentences, window_size=5)
ppmi = compute_ppmi(word_counts, cooc_counts, w2i)
print(f"PPMI matrix: {ppmi.shape}")

# ------------------------------------------------------------------
# 2. PAIRS TO TRACK
# ------------------------------------------------------------------
related = [
    ('cobre', 'conduz'),
    ('gato', 'cachorro'),
    ('brasil', 'américa'),
    ('água', 'líquido'),
]
unrelated = [
    ('cobre', 'gato'),
    ('brasil', 'líquido'),
    ('animal', 'planeta'),
]

def filter_pairs(pairs):
    return [(a, b) for a, b in pairs if a in w2i and b in w2i]

related = filter_pairs(related)
unrelated = filter_pairs(unrelated)
print(f"Tracking {len(related)} related, {len(unrelated)} unrelated pairs")

# ------------------------------------------------------------------
# 3. BASELINE
# ------------------------------------------------------------------
def measure(pairs, vecs):
    return [cosine(vecs[w2i[a]], vecs[w2i[b]]) for a, b in pairs]

base_rel = measure(related, vectors)
base_unr = measure(unrelated, vectors)

print("\n" + "-" * 70)
print("BASELINE (SVD vectors)")
print("-" * 70)
for (a, b), s in zip(related, base_rel):
    print(f"  {a:<12s} ↔ {b:<12s}  {s:.4f}")
print(f"  Mean related:    {np.mean(base_rel):.4f}")
for (a, b), s in zip(unrelated, base_unr):
    print(f"  {a:<12s} ↔ {b:<12s}  {s:.4f}")
print(f"  Mean unrelated:  {np.mean(base_unr):.4f}")
print(f"  Gap (rel-unr):   {np.mean(base_rel) - np.mean(base_unr):.4f}")

# ------------------------------------------------------------------
# 4. PMI-WEIGHTED RANDOM INDEXING
# ------------------------------------------------------------------
print("\n" + "-" * 70)
print("PMI-WEIGHTED RI — 10 passes, lr=0.01, top-20 PMI neighbors")
print("-" * 70)

ri_vectors = vectors.copy()
lr = 0.01
n_passes = 10

# Precompute top PMI neighbors for each word (efficiency)
print("Precomputing PMI neighbor lists...")
top_neighbors = {}
for i in range(V):
    row = ppmi[i].copy()
    row[i] = -1.0  # exclude self
    # Take top 20 by PMI (positive only)
    top_k = min(20, V)
    top_idx = np.argpartition(row, -top_k)[-top_k:]
    top_idx = top_idx[row[top_idx] > 0]
    if len(top_idx) > 0:
        weights = row[top_idx]
        weights = weights / (weights.sum() + 1e-12)
        top_neighbors[i] = (top_idx, weights)

for p in range(n_passes):
    np.random.shuffle(sentences)
    for sent in sentences:
        tokens = [t for t in sent if t in w2i]
        if len(tokens) < 2:
            continue
        for i, w in enumerate(tokens):
            wi = w2i[w]
            if wi not in top_neighbors:
                continue
            # Use precomputed PMI-weighted neighbors
            nbr_idx, nbr_weights = top_neighbors[wi]
            # Context words in this sentence that are also top PMI neighbors
            ctx_idx = []
            ctx_weights = []
            for j in range(max(0, i - 5), min(len(tokens), i + 6)):
                if j == i:
                    continue
                nj = w2i[tokens[j]]
                # Find if this neighbor is in the top PMI list
                mask = nbr_idx == nj
                if mask.any():
                    pos = np.where(mask)[0][0]
                    ctx_idx.append(nj)
                    ctx_weights.append(nbr_weights[pos])
            if not ctx_idx:
                continue
            # Weighted context mean
            ctx_weights = np.array(ctx_weights, dtype=np.float32)
            ctx_weights = ctx_weights / (ctx_weights.sum() + 1e-12)
            ctx_mean = np.sum(ri_vectors[ctx_idx] * ctx_weights[:, None], axis=0)
            # Pull toward weighted context mean
            ri_vectors[wi] += lr * (ctx_mean - ri_vectors[wi])
    # Normalize per pass
    norms = np.linalg.norm(ri_vectors, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    ri_vectors = ri_vectors / norms
    print(f"  Pass {p+1:2d} done")

# ------------------------------------------------------------------
# 5. POST-RI SIMILARITIES
# ------------------------------------------------------------------
ri_rel = measure(related, ri_vectors)
ri_unr = measure(unrelated, ri_vectors)

print("\n" + "-" * 70)
print("AFTER PMI-WEIGHTED RANDOM INDEXING")
print("-" * 70)
for (a, b), s in zip(related, ri_rel):
    print(f"  {a:<12s} ↔ {b:<12s}  {s:.4f}")
print(f"  Mean related:    {np.mean(ri_rel):.4f}")
for (a, b), s in zip(unrelated, ri_unr):
    print(f"  {a:<12s} ↔ {b:<12s}  {s:.4f}")
print(f"  Mean unrelated:  {np.mean(ri_unr):.4f}")
print(f"  Gap (rel-unr):   {np.mean(ri_rel) - np.mean(ri_unr):.4f}")

# ------------------------------------------------------------------
# 6. CHANGE ANALYSIS
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("CHANGE ANALYSIS")
print("=" * 70)
rel_delta = np.mean(ri_rel) - np.mean(base_rel)
unr_delta = np.mean(ri_unr) - np.mean(base_unr)
gap_before = np.mean(base_rel) - np.mean(base_unr)
gap_after = np.mean(ri_rel) - np.mean(ri_unr)

print(f"Related   change: {rel_delta:+.4f}")
print(f"Unrelated change: {unr_delta:+.4f}")
print(f"Gap BEFORE: {gap_before:.4f}")
print(f"Gap AFTER:  {gap_after:.4f}")
print(f"Gap WIDENED: {gap_after > gap_before} (Δ {gap_after - gap_before:+.4f})")

print("\nPer-pair delta:")
for (a, b), b4, af in zip(related, base_rel, ri_rel):
    print(f"  {a:<12s} ↔ {b:<12s}  {b4:.4f} → {af:.4f}  ({af-b4:+.4f})")
for (a, b), b4, af in zip(unrelated, base_unr, ri_unr):
    print(f"  {a:<12s} ↔ {b:<12s}  {b4:.4f} → {af:.4f}  ({af-b4:+.4f})")

out_path = '/home/ravizin/celn-v3/celn_pmi_ri_vectors.npz'
np.savez_compressed(
    out_path,
    vectors=ri_vectors.astype(np.float32),
    vocab=vocab,
    source='data/celn_full_vectors.npz',
    method='pmi_weighted_random_indexing',
    n_passes=n_passes,
    learning_rate=lr,
    top_pmi_neighbors=20,
    gap_before=gap_before,
    gap_after=gap_after,
)
print(f"\nSaved refined vectors: {out_path}")
