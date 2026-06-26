"""
Test: Parallel Transport as Generation Engine
===============================================
Encodes word pairs M(w_i, w_{i+1}) from corpus into SDM.
At generation, queries PairSDM with context-aware query pair,
extracts next word via Resonator unbinding.

Hypothesis: Transport replaces diffuse similarity with directional
extraction. M(A,B) stores the transition; unbind_M_reverse retrieves B.
"""

import sys
sys.path.insert(0, '/home/ravizin/celn-v3')

import numpy as np
from numpy.fft import fft, ifft
from celn.core import normalize, projective_resonance as M, similarity, phase_lens
from celn.dual_channel import DualChannelGenerator, extract_type_vectors
from celn.train import tokenize, load_corpus, build_cooccurrence, compute_ppmi
from celn.memory import DenseSDM
from celn.resonator import unbind_M_reverse, unbind_M_forward, ResonatorDecoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PARALLEL TRANSPORT — Pair SDM Engine Test")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[1] Loading vectors...")
data = np.load('/home/ravizin/celn-v3/celn_full_vectors.npz', allow_pickle=True)
sem_vecs = data['vectors']
vocab = data['vocab']
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for i, w in enumerate(vocab)}
V, D = sem_vecs.shape
print(f"    {V} words, {D}D")

print("\n[2] Loading corpus...")
sentences = load_corpus('/home/ravizin/celn-v3/corpus_final.txt', min_len=2)
print(f"    {len(sentences)} sentences")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ENCODE PAIRS → PAIR SDM
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3] Encoding word pairs and storing in PairSDM...")

# Collect all adjacent pairs
pair_vectors = []
pair_word_pairs = []  # track which words each pair encodes

for sent in sentences[:2000]:  # Use most sentences
    for i in range(len(sent) - 1):
        w1, w2 = sent[i], sent[i+1]
        if w1 in w2i and w2 in w2i:
            v1 = sem_vecs[w2i[w1]]
            v2 = sem_vecs[w2i[w2]]
            # Encode directional pair: M(word_i, word_{i+1})
            pair_vec = M(v1, v2, gamma=1.0, bilateral=True)
            pair_vectors.append(pair_vec)
            pair_word_pairs.append((w2i[w1], w2i[w2]))

print(f"    Encoded {len(pair_vectors)} pairs")

# Initialize PairSDM with pair vectors as addresses
pair_sdm = DenseSDM(n_locations=8192, activation_pct=0.005, seed=42)

# Sample pair vectors for address initialization
n_seed = min(len(pair_vectors), 8000)
indices = np.random.RandomState(42).choice(len(pair_vectors), n_seed, replace=False)
seed_vecs = np.array([pair_vectors[i] for i in indices])
pair_sdm.initialize_addresses(seed_vecs)

# Write all pair vectors to PairSDM
print("    Writing pairs to SDM...")
for i, pv in enumerate(pair_vectors):
    pair_sdm.write(pv)
    if (i+1) % 10000 == 0:
        print(f"      {i+1}/{len(pair_vectors)}")

print(f"    PairSDM: {pair_sdm.stats['n_locations']} locations, "
      f"{pair_sdm.stats['n_written']} written, "
      f"{pair_sdm.stats['total_writes']} total writes")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. INIT RESONATOR
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4] Initializing Resonator...")
resonator = ResonatorDecoder(sem_vecs, max_iter=20, n_restarts=2, convergence_patience=3, seed=42)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TEST: Transport Extraction from PairSDM
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TRANSPORT EXTRACTION TEST")
print("=" * 70)

def transport_extract(current_word, context_words, top_k=5):
    """Extract next word via Parallel Transport from PairSDM.

    1. Build context-aware query pair
    2. Query PairSDM → retrieve blended pair
    3. Unbind to extract next word
    4. Score candidates by similarity to extracted vector
    """
    if current_word not in w2i:
        return []

    v_current = sem_vecs[w2i[current_word]]

    # Build context centroid from recent words
    ctx_vecs = [sem_vecs[w2i[w]] for w in context_words if w in w2i]
    if ctx_vecs:
        ctx_centroid = normalize(np.mean(ctx_vecs, axis=0))
    else:
        ctx_centroid = v_current

    # Query pair: M(context, current_word) — "where are we, what word"
    query_pair = M(ctx_centroid, v_current, gamma=1.0, bilateral=True)

    # Query PairSDM
    sdm_result = pair_sdm.read(query_pair)

    # Extract B from SDM result via Resonator reverse unbinding
    b_recovered = unbind_M_reverse(sdm_result, v_current, gamma=1.0, bilateral=True, n_refine=5)

    # Score all words by similarity to recovered B
    b_norm = normalize(b_recovered)
    scores = sem_vecs @ b_norm.astype(np.float32)

    # Top-k
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(i2w[i], float(scores[i])) for i in top_idx]


# Compare with baseline SDM (word-level read)
def baseline_sdm(current_word, context_words, top_k=5):
    """Baseline: SDM read with word vectors directly."""
    if current_word not in w2i:
        return []

    ctx_vecs = [sem_vecs[w2i[w]] for w in context_words if w in w2i]
    ctx_centroid = normalize(np.mean(ctx_vecs, axis=0)) if ctx_vecs else sem_vecs[w2i[current_word]]

    sdm_result = pair_sdm.read(ctx_centroid)
    scores = sem_vecs @ sdm_result.astype(np.float32)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(i2w[i], float(scores[i])) for i in top_idx]


# Test cases
test_cases = [
    ("cobre", ["metal"]),
    ("cobra", ["animal"]),
    ("frança", ["europa"]),
    ("água", ["lago"]),
    ("célula", ["tecido"]),
]

print("\n  Comparison: Word → Extracted next words")
print(f"  {'Query':<12s} {'Transport (M unbind)':<50s} {'Baseline SDM':<50s}")
print("  " + "-" * 112)

for word, ctx in test_cases:
    transport = transport_extract(word, ctx)
    baseline = baseline_sdm(word, ctx)

    t_str = ' | '.join(f"{w}({s:.3f})" for w, s in transport)
    b_str = ' | '.join(f"{w}({s:.3f})" for w, s in baseline)

    print(f"  {word:<12s} {t_str:<50s} {b_str:<50s}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. TEST: Direct pair extraction accuracy
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("DIRECT PAIR EXTRACTION ACCURACY")
print("=" * 70)

# Test: for known pairs, can we extract the second word?
correct = 0
total = 0
test_pairs = pair_word_pairs[-200:]  # last 200 pairs

for w1_idx, w2_idx in test_pairs:
    v1 = sem_vecs[w1_idx]
    v2 = sem_vecs[w2_idx]
    pair = M(v1, v2, gamma=1.0, bilateral=True)

    # Extract B from pair
    b_rec = unbind_M_reverse(pair, v1, gamma=1.0, bilateral=True, n_refine=5)
    b_idx, b_sim = resonator._nearest_with_score(b_rec)

    total += 1
    if b_idx == w2_idx:
        correct += 1

print(f"  Direct extraction: {correct}/{total} = {correct/total:.1%}")

# Top-5 accuracy
top5_correct = 0
for w1_idx, w2_idx in test_pairs[-100:]:
    v1 = sem_vecs[w1_idx]
    v2 = sem_vecs[w2_idx]
    pair = M(v1, v2, gamma=1.0, bilateral=True)

    b_rec = unbind_M_reverse(pair, v1, gamma=1.0, bilateral=True, n_refine=5)
    top5 = resonator._nearest(b_rec, top_k=5)

    if w2_idx in top5:
        top5_correct += 1

print(f"  Top-5 extraction: {top5_correct}/100 = {top5_correct}%")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. QUANTITATIVE: Score concentration comparison
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SCORE CONCENTRATION: Transport vs SDM Baseline")
print("=" * 70)

query_word = "cobre"
v_q = sem_vecs[w2i[query_word]]
ctx_vec = sem_vecs[w2i["metal"]]

# Transport extraction
query_pair = M(ctx_vec, v_q, gamma=1.0, bilateral=True)
sdm_result = pair_sdm.read(query_pair)
b_rec = unbind_M_reverse(sdm_result, v_q, gamma=1.0, bilateral=True, n_refine=5)
transport_scores = sem_vecs @ normalize(b_rec).astype(np.float32)

# Baseline SDM
baseline_scores = sem_vecs @ pair_sdm.read(v_q).astype(np.float32)

# Static SVD
static_scores = sem_vecs @ v_q

def concentration(scores, name):
    valid = scores.copy()
    top10 = np.argpartition(valid, -10)[-10:]
    top10_mean = valid[top10].mean()
    top50 = np.argpartition(valid, -50)[-50:]
    top50_mean = valid[top50].mean()
    all_mean = valid.mean()
    all_std = valid.std()
    peak = valid.max()

    print(f"\n  {name}:")
    print(f"    mean={all_mean:.4f}, std={all_std:.4f}, max={peak:.4f}")
    print(f"    top10_mean={top10_mean:.4f}, top50_mean={top50_mean:.4f}")
    print(f"    gini10={top10_mean/(all_mean+1e-12):.1f}, peak_ratio={peak/(all_mean+1e-12):.1f}")

    top_words = [(i2w[i], float(valid[i])) for i in np.argsort(valid)[::-1][:10]]
    print(f"    Top-10: {', '.join(f'{w}({s:.3f})' for w,s in top_words)}")

concentration(transport_scores, "TRANSPORT (M unbind from PairSDM)")
concentration(baseline_scores, "Baseline SDM")
concentration(static_scores, "Static SVD")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
