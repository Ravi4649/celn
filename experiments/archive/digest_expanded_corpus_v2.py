#!/usr/bin/env python3
"""
CELN v3 — Digestion Cycle v2: SVD-based vectors (no Hebbian collapse)
====================================================================
Uses TruncatedSVD on PPMI (like test_digestion_cycle.py) to avoid
vector collapse from Hebbian over-training.
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celn_v3.train import load_corpus, build_cooccurrence, compute_ppmi
from celn_v3.core import normalize, batch_normalize, projective_resonance as M
from celn_v3.memory import DenseSDM
from sklearn.decomposition import TruncatedSVD

t0 = time.time()
D = 10_000

print("=" * 70)
print("CELN v3 — Digestion v2 (SVD vectors)")
print("=" * 70)

# ═══════════════════════════════════════════════════
# PHASE 1: Load corpus
# ═══════════════════════════════════════════════════
corpus_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'corpus_pt_expandido.txt'
)
sentences = load_corpus(corpus_path, min_len=1)
print(f"\nCorpus: {len(sentences)} sentences after tokenization")

# ═══════════════════════════════════════════════════
# PHASE 2: PPMI + SVD (no Hebbian)
# ═══════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("PHASE 2: PPMI + SVD → 10k-D vectors")
print(f"{'─'*70}")

word_counts, cooc_counts, w2i, i2w = build_cooccurrence(sentences, window_size=5)
V = len(w2i)
print(f"  Vocabulary: {V} words")
print(f"  Co-occurrence pairs: {len(cooc_counts)}")

ppmi = compute_ppmi(word_counts, cooc_counts, w2i)
nonzero = np.count_nonzero(ppmi) / ppmi.size
print(f"  PPMI non-zero: {nonzero:.2%}")

nc = min(5000, V - 1)
svd = TruncatedSVD(n_components=nc, random_state=42)
vr = svd.fit_transform(ppmi)
sv = svd.singular_values_
var = sv**2 / (sv**2).sum()
vr = vr * (var / var.max())[None, :]
print(f"  SVD components: {nc} (explained var ratio: {var.sum():.2%})")

# Project to 10k-D
if nc < 10000:
    R_mat = np.random.RandomState(42).randn(nc, 10000).astype(np.float32) / np.sqrt(nc)
    vectors = vr.astype(np.float32) @ R_mat
else:
    vectors = vr

vectors = batch_normalize(vectors)
print(f"  Vectors: {V} words × {vectors.shape[1]}D")

# ═══════════════════════════════════════════════════
# PHASE 3: Semantic concentration
# ═══════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("PHASE 3: Semantic concentration")
print(f"{'─'*70}")

TEST_WORDS = ['cobre', 'eletricidade', 'brasil', 'água', 'música', 'gato', 'futebol', 'internet', 'energia', 'amor']
for word in TEST_WORDS:
    if word not in w2i:
        continue
    idx = w2i[word]
    sims = vectors @ vectors[idx]
    top5 = sorted([(sims[j], i2w[j]) for j in range(V) if j != idx], reverse=True)[:5]
    print(f"  {word}: {', '.join(f'{w}({s:.3f})' for s, w in top5)}")

# Cosine concentration (pairwise)
sample = min(5000, V)
ridx = np.random.RandomState(42).choice(V, sample, replace=False)
sv = vectors[ridx]
norms = np.linalg.norm(sv, axis=1, keepdims=True)
norms[norms < 1e-12] = 1.0
sv = sv / norms
pairwise = sv @ sv.T
triu = pairwise[np.triu_indices_from(pairwise, k=1)]
p10 = float(np.percentile(triu, 10))
p75 = float(np.percentile(triu, 75))
p90 = float(np.percentile(triu, 90))
print(f"\n  Cosine distribution (n={sample}):")
print(f"    P10={p10:.4f}  P75={p75:.4f}  P90={p90:.4f}")
print(f"    Spread (P90-P10)={p90-p10:.4f}")
print(f"    Mean={triu.mean():.4f}  Std={triu.std():.4f}")
print(f"    (Original corpus had P90-P10 ≈ 0.80 — lower spread = better)")

# ═══════════════════════════════════════════════════
# PHASE 4: SDM
# ═══════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("PHASE 4: SDM digestion")
print(f"{'─'*70}")

sdm = DenseSDM(n_locations=8192, activation_pct=0.005, seed=42)

seed_centroids = []
for tokens in sentences[:min(len(sentences), 5000)]:
    idxs = [w2i[w] for w in tokens if w in w2i]
    if len(idxs) >= 3:
        seed_centroids.append(normalize(vectors[idxs].mean(axis=0)))
sdm.initialize_addresses(np.array(seed_centroids))
print(f"  SDM: {sdm.n_locations} locations")

total_corr = 0
total_contra = 0
t_neut = 0
for i, tokens in enumerate(sentences):
    idxs = [w2i[w] for w in tokens if w in w2i]
    if len(idxs) >= 3:
        centroid = normalize(vectors[idxs].mean(axis=0))
        r = sdm.write_corroborated(centroid)
        total_corr += r['corroborating']
        total_contra += r['contradictory']
        t_neut += r['neutral']
    if (i + 1) % 3000 == 0:
        print(f"  [{i+1}/{len(sentences)}] corr={total_corr} contra={total_contra}")

print(f"\n  Digestion: {len(sentences)} sentences → {sdm.total_writes} writes")
print(f"  Corroborations: {total_corr}  Contradictions: {total_contra}")
print(f"  Conflicts: {sdm.total_conflicts_detected} at {sdm.has_conflict.sum()} locations")

# SDM queries
print(f"\n{'─'*70}")
print("PHASE 5: SDM Knowledge Query")
print(f"{'─'*70}")
for word in ['cobre', 'eletricidade', 'brasil', 'água', 'internet', 'futebol']:
    if word not in w2i:
        continue
    q_vec = vectors[w2i[word]]
    r = sdm.read_with_confidence(q_vec)
    sims = vectors @ normalize(r['result'])
    top = np.argsort(sims)[-10:][::-1]
    top_w = [f"{i2w[j]}({sims[j]:.3f})" for j in top]
    print(f"  {word}: {', '.join(top_w[:5])} trust={r['trust_score']:.3f}")

# ═══════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("PHASE 6: Saving")
print(f"{'─'*70}")
save_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
npz_path = os.path.join(save_dir, 'celn_v3_expanded_svd_vectors.npz')
np.savez_compressed(npz_path, vectors=vectors, vocab=np.array(list(w2i.keys()), dtype=object))
print(f"  Vectors saved: {npz_path} ({os.path.getsize(npz_path)/1024/1024:.0f} MB)")

# Learn Type Field
type_dim = 2000
rng_t = np.random.RandomState(42)
type_vecs = rng_t.randn(V, type_dim).astype(np.float32)
type_vecs = batch_normalize(type_vecs)

pair_src, pair_fol = [], []
for tokens in sentences:
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i+1]
        if w1 in w2i and w2 in w2i:
            pair_src.append(w2i[w1])
            pair_fol.append(w2i[w2])

type_field = np.zeros((V, type_dim), dtype=np.float32)
accum = np.zeros((V, type_dim), dtype=np.float32)
counts = np.zeros(V, dtype=np.int32)
for i in range(len(pair_src)):
    src, fol = pair_src[i], pair_fol[i]
    accum[src] += type_vecs[fol]
    counts[src] += 1
for i in range(V):
    if counts[i] > 0:
        type_field[i] = normalize(accum[i] / counts[i])

npz_type_path = os.path.join(save_dir, 'celn_v3_expanded_type_field.npz')
np.savez_compressed(npz_type_path, type_field=type_field, type_vecs=type_vecs,
                    pair_src=np.array(pair_src, dtype=np.int32),
                    pair_fol=np.array(pair_fol, dtype=np.int32))
print(f"  Type Field saved: {npz_type_path}")

print(f"\n{'='*70}")
print(f"DIGESTION v2 COMPLETE — {time.time()-t0:.0f}s")
print(f"  Corpus: {len(sentences)} sentences → {V} words")
print(f"  Vectors: SVD-based (no Hebbian collapse)")
print(f"  Cosine spread (P90-P10): {p90-p10:.4f}")
print(f"{'='*70}")
