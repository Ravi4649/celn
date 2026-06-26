"""
Test: Can M (projective_resonance) deform SVD space dynamically?
=================================================================
HYPOTHESIS: The state M can act as a "context lens" that makes word
similarity context-dependent.

  contextualized = M(word, context_domain, bilateral=True)
  score = cos(contextualized, candidate)

If context=electricity: "cobre" should approach "conduz", "corrente", "elétrica"
If context=mining:     "cobre" should approach "minério", "metal", "extração"

Purely algebraic, no backprop, no re-training.
"""

import sys
sys.path.insert(0, '/home/ravizin/celn-v3')

import numpy as np
from celn.core import (
    normalize, projective_resonance, similarity, D, phi_weights, phi
)
from celn.train import tokenize
import re
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD PRE-TRAINED VECTORS
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("Loading pre-trained vectors...")
data = np.load('/home/ravizin/celn-v3/celn_full_vectors.npz', allow_pickle=True)
vectors = data['vectors']  # (3007, 10000)
vocab = data['vocab']      # (3007,) strings
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for i, w in enumerate(vocab)}

print(f"  Vocabulary: {len(vocab)} words, {vectors.shape[1]}D")

# Verify normalization
norms = np.linalg.norm(vectors, axis=1)
print(f"  Vector norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. BASELINE: static similarity of "cobre"
# ═══════════════════════════════════════════════════════════════════════════════

def top_similar(word_vec, vectors, vocab, top_k=15, exclude_words=None):
    """Return top-k most similar words (cosine) to word_vec."""
    sims = vectors @ word_vec
    sims = sims / (np.linalg.norm(vectors, axis=1) + 1e-12)

    # Exclude self and specified words
    if exclude_words:
        for w in exclude_words:
            if w in w2i:
                sims[w2i[w]] = -1.0

    top_idx = np.argsort(sims)[::-1][:top_k]
    return [(vocab[i], float(sims[i])) for i in top_idx]


print("\n" + "=" * 70)
print("BASELINE: Static SVD similarity for 'cobre'")
print("=" * 70)

v_cobre = vectors[w2i['cobre']]
baseline_top = top_similar(v_cobre, vectors, vocab, top_k=15, exclude_words=['cobre'])
for i, (w, s) in enumerate(baseline_top):
    print(f"  {i+1:2d}. {w:20s}  cos={s:.4f}")

cobre_idx = w2i['cobre']

# ═══════════════════════════════════════════════════════════════════════════════
# 3. CREATE DOMAIN CONTEXT VECTORS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Creating domain context vectors")
print("=" * 70)

# Load corpus and split into domain groups
with open('/home/ravizin/celn-v3/corpus_final.txt', 'r', encoding='utf-8') as f:
    text = f.read()

raw_sentences = re.split(r'[.!?\n]+', text)
raw_sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 20]

# Domain keyword sets (carefully chosen to avoid overlap)
eletric_kw = {'cobre', 'conduz', 'corrente', 'eletricidade', 'elétrica', 'elétrico',
              'condutor', 'condutividade', 'térmica', 'transmissão', 'energia',
              'painéis', 'turbinas', 'baterias', 'eletrônica', 'eletrônico',
              'tensão', 'voltagem', 'circuito', 'fio', 'cabo', 'resistência'}

mining_kw = {'minério', 'mina', 'extração', 'metal', 'metálico', 'metálica',
             'ferro', 'liga', 'bronze', 'latão', 'zinco', 'estanho',
             'fundição', 'refino', 'britado', 'processado', 'indústria',
             'industrial', 'produção', 'reservas', 'extrativismo'}

bio_kw = {'célula', 'animal', 'tecido', 'fotossíntese', 'planta',
          'organismo', 'bactéria', 'coração', 'sangue', 'câncer',
          'hormônio', 'nervo', 'molecular', 'proteína', 'enzima',
          'membrana', 'respiração', 'digestão', 'nutriente', 'vitamina'}

def classify_sentence(sentence):
    """Classify sentence into domain based on keyword overlap."""
    words = set(tokenize(sentence, min_len=2))
    e_score = len(words & eletric_kw)
    m_score = len(words & mining_kw)
    b_score = len(words & bio_kw)

    # Only classify if ONE domain clearly dominates
    scores = {'eletric': e_score, 'mining': m_score, 'bio': b_score}
    best = max(scores, key=scores.get)
    best_score = scores[best]

    if best_score == 0:
        return None
    # Return domain only if the best is at least 2x the second best
    second_best = sorted(scores.values(), reverse=True)[1]
    if best_score >= 2 and best_score > second_best:
        return best
    elif best_score >= 3:
        return best
    return None

eletric_sents = []
mining_sents = []
bio_sents = []

for s in raw_sentences:
    domain = classify_sentence(s)
    if domain == 'eletric':
        eletric_sents.append(s)
    elif domain == 'mining':
        mining_sents.append(s)
    elif domain == 'bio':
        bio_sents.append(s)

print(f"  Electricity sentences: {len(eletric_sents)}")
print(f"  Mining sentences:      {len(mining_sents)}")
print(f"  Biology sentences:     {len(bio_sents)}")

def sentences_to_centroid(sentences, vectors, w2i):
    """Create domain centroid from sentences: mean of word vectors in those sentences."""
    all_vecs = []
    for s in sentences:
        tokens = tokenize(s, min_len=2)
        for t in tokens:
            if t in w2i:
                all_vecs.append(vectors[w2i[t]])
    if not all_vecs:
        return np.zeros(vectors.shape[1])
    centroid = np.mean(all_vecs, axis=0)
    return normalize(centroid)

ctx_eletric = sentences_to_centroid(eletric_sents, vectors, w2i)
ctx_mining = sentences_to_centroid(mining_sents, vectors, w2i)
ctx_bio = sentences_to_centroid(bio_sents, vectors, w2i)

# Also create global centroid
all_vecs_list = [vectors[w2i[t]] for s in raw_sentences
                 for t in tokenize(s, min_len=2) if t in w2i]
global_centroid = normalize(np.mean(all_vecs_list, axis=0))

print(f"\n  Domain centroid norms: eletric={np.linalg.norm(ctx_eletric):.4f}, "
      f"mining={np.linalg.norm(ctx_mining):.4f}, bio={np.linalg.norm(ctx_bio):.4f}")
print(f"  Domain similarity: eletric↔mining={similarity(ctx_eletric, ctx_mining):.4f}, "
      f"eletric↔bio={similarity(ctx_eletric, ctx_bio):.4f}, "
      f"mining↔bio={similarity(ctx_mining, ctx_bio):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TEST: M as contextual lens
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TEST 1: M(word, context) — Bilateral amplification")
print("=" * 70)
print("Hypothesis: M(cobre, ctx_domain) amplifies frequencies where")
print("the domain differs from cobre → adds domain-specific signature.")
print()

for name, ctx in [('ELECTRICITY', ctx_eletric), ('MINING', ctx_mining), ('BIOLOGY', ctx_bio)]:
    # Contextualize cobre via M
    v_ctx = projective_resonance(v_cobre, ctx, gamma=1.0, bilateral=True)

    top = top_similar(v_ctx, vectors, vocab, top_k=10, exclude_words=['cobre'])
    label = f"M(cobre, {name.lower()}, bilateral)"
    print(f"  [{label}]")
    for i, (w, s) in enumerate(top):
        marker = ""
        if w == 'conduz': marker = " ← ELECTRIC"
        if w == 'corrente': marker = " ← ELECTRIC"
        if w == 'elétrica': marker = " ← ELECTRIC"
        if w == 'minério': marker = " ← MINING"
        if w == 'metal': marker = " ← MINING"
        if w == 'ferro': marker = " ← MINING"
        print(f"    {i+1:2d}. {w:20s}  cos={s:.4f}{marker}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# 5. TEST: M(context, word) — Reverse order
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TEST 2: M(context, word) — Reverse order")
print("=" * 70)
print("Encodes word into context state (like encoder scan).")
print()

for name, ctx in [('ELECTRICITY', ctx_eletric), ('MINING', ctx_mining), ('BIOLOGY', ctx_bio)]:
    v_ctx = projective_resonance(ctx, v_cobre, gamma=1.0, bilateral=True)

    top = top_similar(v_ctx, vectors, vocab, top_k=10, exclude_words=['cobre'])
    label = f"M({name.lower()}, cobre, bilateral)"
    print(f"  [{label}]")
    for i, (w, s) in enumerate(top):
        marker = ""
        if w in ['conduz', 'corrente', 'elétrica']: marker = " ← ELECTRIC"
        if w in ['minério', 'metal', 'ferro']: marker = " ← MINING"
        print(f"    {i+1:2d}. {w:20s}  cos={s:.4f}{marker}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# 6. TEST: Alternative — Simple vector shift
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TEST 3: Simple shift — word + alpha * (ctx - global)")
print("=" * 70)
print("Classic algebraic approach: shift word toward domain-specific direction.")
print()

for name, ctx in [('ELECTRICITY', ctx_eletric), ('MINING', ctx_mining), ('BIOLOGY', ctx_bio)]:
    for alpha in [0.1, 0.3, 0.5]:
        direction = ctx - global_centroid
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        v_shifted = normalize(v_cobre + alpha * direction)

        top = top_similar(v_shifted, vectors, vocab, top_k=10, exclude_words=['cobre'])
        label = f"shift(cobre, {name}, α={alpha})"
        print(f"  [{label}]")
        top_words = [f"{w}({s:.3f})" for w, s in top[:5]]
        print(f"    Top 5: {', '.join(top_words)}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# 7. TEST: Frequency amplification without binding
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TEST 4: Spectral lens — amplify cobre's freqs that ctx also has")
print("=" * 70)
print("Multiply word spectrum by context spectrum (no circular convolution phase)")
print()

from numpy.fft import fft, ifft

for name, ctx in [('ELECTRICITY', ctx_eletric), ('MINING', ctx_mining), ('BIOLOGY', ctx_bio)]:
    # Amplify cobre's frequencies where context has strong energy
    C_mag = np.abs(fft(v_cobre))
    X_mag = np.abs(fft(ctx))

    # Weight: amplify frequencies strong in BOTH cobre and context
    median_x = np.median(X_mag)
    if median_x > 1e-12:
        ctx_weight = np.tanh(X_mag / median_x)
    else:
        ctx_weight = np.ones_like(X_mag)

    # Apply contextual weight to cobre's own frequencies
    C_phase = fft(v_cobre) / (C_mag + 1e-12)
    contextualized_spectrum = C_mag * ctx_weight * C_phase
    v_spectral = normalize(ifft(contextualized_spectrum).real)

    top = top_similar(v_spectral, vectors, vocab, top_k=10, exclude_words=['cobre'])
    label = f"spectral_lens(cobre, {name})"
    print(f"  [{label}]")
    for i, (w, s) in enumerate(top):
        marker = ""
        if w in ['conduz', 'corrente', 'elétrica']: marker = " ← ELECTRIC"
        if w in ['minério', 'metal', 'ferro']: marker = " ← MINING"
        print(f"    {i+1:2d}. {w:20s}  cos={s:.4f}{marker}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# 8. QUANTITATIVE ANALYSIS: ranking shifts
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("QUANTITATIVE ANALYSIS: Ranking shifts for key words")
print("=" * 70)

# Compute full similarity rankings for all 3007 words under each condition
baseline_sims = vectors @ v_cobre
baseline_sims = baseline_sims / (np.linalg.norm(vectors, axis=1) + 1e-12)
baseline_ranks = np.argsort(baseline_sims)[::-1]

# M(word, ctx) for each domain
m_eletric = projective_resonance(v_cobre, ctx_eletric, gamma=1.0, bilateral=True)
m_mining = projective_resonance(v_cobre, ctx_mining, gamma=1.0, bilateral=True)
m_bio = projective_resonance(v_cobre, ctx_bio, gamma=1.0, bilateral=True)

eletric_sims = vectors @ m_eletric
eletric_sims = eletric_sims / (np.linalg.norm(vectors, axis=1) + 1e-12)

mining_sims = vectors @ m_mining
mining_sims = mining_sims / (np.linalg.norm(vectors, axis=1) + 1e-12)

bio_sims = vectors @ m_bio
bio_sims = bio_sims / (np.linalg.norm(vectors, axis=1) + 1e-12)

# Track key words
track_words = ['conduz', 'corrente', 'elétrica', 'condutor', 'térmica', 'transmissão',
               'minério', 'metal', 'ferro', 'liga', 'bronze', 'latão',
               'célula', 'animal', 'tecido', 'coração', 'sangue']

print(f"\n{'Word':<16s} {'Baseline':>8s} {'Eletric':>8s} {'Mining':>8s} {'Bio':>8s} | {'Δ(E-M)':>8s}")
print("-" * 70)

for w in track_words:
    if w not in w2i:
        continue
    idx = w2i[w]

    # Rankings (lower = better, 0 = best)
    r_baseline = np.where(baseline_ranks == idx)[0][0]
    r_eletric = np.where(np.argsort(eletric_sims)[::-1] == idx)[0][0]
    r_mining = np.where(np.argsort(mining_sims)[::-1] == idx)[0][0]
    r_bio = np.where(np.argsort(bio_sims)[::-1] == idx)[0][0]

    delta_em = r_eletric - r_mining  # negative = ranked higher in eletric vs mining

    print(f"{w:<16s} {r_baseline:>8d} {r_eletric:>8d} {r_mining:>8d} {r_bio:>8d} | {delta_em:>+8d}")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. TEST: M chain state as context (full domain encoding)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TEST 5: M-encoded domain state as context")
print("=" * 70)
print("Encode domain sentences via M scan → domain state → use as lens")
print()

def encode_sentences_M(sentences, vectors, w2i, max_sents=20):
    """Encode domain sentences via M scan, return final state."""
    state = None
    count = 0
    for s in sentences[:max_sents]:
        tokens = tokenize(s, min_len=2)
        indices = [w2i[t] for t in tokens if t in w2i]
        if len(indices) < 3:
            continue
        # Encode this sentence
        s_state = vectors[indices[0]].copy()
        for idx in indices[1:]:
            s_state = projective_resonance(s_state, vectors[idx], gamma=1.0, bilateral=True)
        if state is None:
            state = s_state
        else:
            state = normalize(state + s_state)
        count += 1
    return state if state is not None else np.zeros(vectors.shape[1])

m_state_eletric = encode_sentences_M(eletric_sents, vectors, w2i)
m_state_mining = encode_sentences_M(mining_sents, vectors, w2i)
m_state_bio = encode_sentences_M(bio_sents, vectors, w2i)

for name, m_state in [('ELECTRICITY', m_state_eletric), ('MINING', m_state_mining), ('BIOLOGY', m_state_bio)]:
    # Use M state as context lens: M(cobre, M_domain_state)
    v_ctx = projective_resonance(v_cobre, m_state, gamma=1.0, bilateral=True)

    top = top_similar(v_ctx, vectors, vocab, top_k=10, exclude_words=['cobre'])
    label = f"M(cobre, M_encoded_{name})"
    print(f"  [{label}]")
    for i, (w, s) in enumerate(top):
        marker = ""
        if w in ['conduz', 'corrente', 'elétrica']: marker = " ← ELECTRIC"
        if w in ['minério', 'metal', 'ferro']: marker = " ← MINING"
        print(f"    {i+1:2d}. {w:20s}  cos={s:.4f}{marker}")
    print()

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
