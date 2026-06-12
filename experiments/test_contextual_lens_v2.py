"""
Test v2: Better approaches for contextual similarity deformation
=================================================================
V1 showed M (projective_resonance) destroys similarity because
circular convolution moves to a different region of space.

V2 tests approaches that PRESERVE similarity structure:
  A. UNBIND from M-encoded domain state
  B. Context-gated similarity (multiplicative)
  C. Contextual projection (remove orthogonal components)
  D. HRR unbinding with phase-only adjustment (preserves magnitude structure)
"""

import sys
sys.path.insert(0, '/home/ravizin/celn-v3')

import numpy as np
from numpy.fft import fft, ifft
from celn_v3.core import (
    normalize, projective_resonance, similarity, D,
    inverse_projective_resonance, phi_weights
)
from celn_v3.train import tokenize
import re
import warnings
warnings.filterwarnings('ignore')

# Load
print("Loading vectors...")
data = np.load('/home/ravizin/celn-v3/celn_v3_full_vectors.npz', allow_pickle=True)
vectors = data['vectors']
vocab = data['vocab']
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for i, w in enumerate(vocab)}
v_cobre = vectors[w2i['cobre']]
cobre_idx = w2i['cobre']
V, D = vectors.shape

print(f"  {V} words, {D} dims")

def top_similar(word_vec, top_k=15, exclude=None):
    sims = vectors @ word_vec
    sims = sims / (np.linalg.norm(vectors, axis=1) + 1e-12)
    if exclude:
        for w in exclude:
            if w in w2i:
                sims[w2i[w]] = -1.0
    idx = np.argsort(sims)[::-1][:top_k]
    return [(vocab[i], float(sims[i])) for i in idx]

# ═══════════════════════════════════════════════════════════════════════════════
# Create domain contexts
# ═══════════════════════════════════════════════════════════════════════════════

# Load and classify sentences
with open('/home/ravizin/celn-v3/corpus_final.txt', 'r', encoding='utf-8') as f:
    text = f.read()
raw_sentences = re.split(r'[.!?\n]+', text)
raw_sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 20]

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
    words = set(tokenize(sentence, min_len=2))
    e_score = len(words & eletric_kw)
    m_score = len(words & mining_kw)
    b_score = len(words & bio_kw)
    scores = {'e': e_score, 'm': m_score, 'b': b_score}
    best = max(scores, key=scores.get)
    best_score = scores[best]
    if best_score == 0:
        return None
    second = sorted(scores.values(), reverse=True)[1]
    if best_score >= 2 and best_score > second:
        return best
    return None

eletric_sents = [s for s in raw_sentences if classify_sentence(s) == 'e']
mining_sents = [s for s in raw_sentences if classify_sentence(s) == 'm']
bio_sents = [s for s in raw_sentences if classify_sentence(s) == 'b']

# Domain centroids
def sents_to_centroid(sentences):
    all_vecs = []
    for s in sentences:
        for t in tokenize(s, min_len=2):
            if t in w2i:
                all_vecs.append(vectors[w2i[t]])
    return normalize(np.mean(all_vecs, axis=0)) if all_vecs else np.zeros(D)

ctx_eletric = sents_to_centroid(eletric_sents)
ctx_mining = sents_to_centroid(mining_sents)
ctx_bio = sents_to_centroid(bio_sents)

# Global centroid
all_vecs = []
for s in raw_sentences:
    for t in tokenize(s, min_len=2):
        if t in w2i:
            all_vecs.append(vectors[w2i[t]])
global_centroid = normalize(np.mean(all_vecs, axis=0))

# ═══════════════════════════════════════════════════════════════════════════════
# Encode domain sentences into M states
# ═══════════════════════════════════════════════════════════════════════════════

def encode_domain_M(sentences, max_sents=30):
    """Encode all domain sentences into a single M state via accumulation."""
    state = None
    count = 0
    for s in sentences[:max_sents]:
        indices = [w2i[t] for t in tokenize(s, min_len=2) if t in w2i]
        if len(indices) < 3:
            continue
        s_state = vectors[indices[0]].copy()
        for idx in indices[1:]:
            s_state = projective_resonance(s_state, vectors[idx], gamma=1.0, bilateral=True)
        if state is None:
            state = s_state.copy()
        else:
            # Accumulate: add and normalize (like SDM write)
            state = normalize(state + s_state)
        count += 1
    return state if state is not None else np.zeros(D)

print("\nEncoding domain M-states...")
M_eletric = encode_domain_M(eletric_sents)
M_mining = encode_domain_M(mining_sents)
M_bio = encode_domain_M(bio_sents)
print(f"  M-state norms: {np.linalg.norm(M_eletric):.4f}, {np.linalg.norm(M_mining):.4f}, {np.linalg.norm(M_bio):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("BASELINE: Static similarity for 'cobre'")
print("=" * 70)
for i, (w, s) in enumerate(top_similar(v_cobre, 15, ['cobre'])):
    print(f"  {i+1:2d}. {w:20s}  cos={s:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST A: UNBIND from M-encoded domain state
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TEST A: UNBINDING from M-encoded domain state")
print("=" * 70)
print("proposal = unbind(M_domain, cobre)")
print("This extracts what's associated with 'cobre' in the domain state.")
print()

# Standard HRR unbind: unbind(state, anchor) ≈ what_else_was_bound
# unbind(c, a) = IFFT(FFT(c) * conj(FFT(a)))
for name, M_state in [('ELECTRICITY', M_eletric), ('MINING', M_mining), ('BIOLOGY', M_bio)]:
    proposal = np.real(ifft(
        fft(M_state) * np.conj(fft(v_cobre))
    ))
    proposal = normalize(proposal)

    top = top_similar(proposal, 15)
    print(f"  [unbind(M_{name}, cobre)]")
    for i, (w, s) in enumerate(top):
        marker = ""
        if w in ['conduz', 'corrente', 'elétrica', 'elétricas', 'condutor']:
            marker = " ← ELECTRIC"
        if w in ['minério', 'metal', 'ferro', 'zinco', 'liga', 'bronze', 'latão']:
            marker = " ← MINING"
        if w in ['célula', 'animal', 'tecido', 'sangue', 'coração']:
            marker = " ← BIO"
        print(f"    {i+1:2d}. {w:20s}  cos={s:.4f}{marker}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# TEST A2: UNBIND with inverse_projective_resonance (bilateral-aware)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TEST A2: UNBIND via inverse_projective_resonance (bilateral-aware)")
print("=" * 70)
print("Uses the proper M inverse with fixed-point iteration for bilateral.")
print()

for name, M_state in [('ELECTRICITY', M_eletric), ('MINING', M_mining), ('BIOLOGY', M_bio)]:
    try:
        # inverse_projective_resonance(state, word) recovers what was composed
        # But we need to know the OPERAND ORDER. The M state was built as:
        #   state = M(... M(M(w0, w1), w2) ...)
        # inverse_projective_resonance(state, wn) recovers state before wn
        # We want: what's associated with "cobre" in the state
        # This is different — we want U(state, cobre) where "cobre" was part
        # of the encoding at some point.
        # If cobre was never encoded, this won't work well.
        proposal = inverse_projective_resonance(
            M_state, v_cobre, gamma=1.0, bilateral=True, n_iter=20
        )
        top = top_similar(proposal, 15)
        print(f"  [inverse_M(M_{name}, cobre)]")
        for i, (w, s) in enumerate(top):
            print(f"    {i+1:2d}. {w:20s}  cos={s:.4f}")
    except Exception as e:
        print(f"  [inverse_M(M_{name}, cobre)] ERROR: {e}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# TEST B: Context-Gated Similarity
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TEST B: Context-Gated Similarity")
print("=" * 70)
print("score(w) = cos(cobre, w) * cos(context, w)")
print("Boosts words that are similar to BOTH cobre and the context.")
print()

for name, ctx in [('ELECTRICITY', ctx_eletric), ('MINING', ctx_mining), ('BIOLOGY', ctx_bio)]:
    # Compute gated scores for all words
    sim_cobre = vectors @ v_cobre
    sim_ctx = vectors @ ctx
    # Normalize each
    sim_cobre = sim_cobre / (np.linalg.norm(vectors, axis=1) + 1e-12)
    sim_ctx = sim_ctx / (np.linalg.norm(vectors, axis=1) + 1e-12)

    # Gate: geometric mean preserves ranking within each channel
    gated = sim_cobre * sim_ctx
    gated[cobre_idx] = -1.0

    top_idx = np.argsort(gated)[::-1][:15]
    top = [(vocab[i], float(gated[i])) for i in top_idx]

    print(f"  [gated(cobre, {name})]")
    for i, (w, s) in enumerate(top):
        marker = ""
        if w in ['conduz', 'corrente', 'elétrica', 'elétricas', 'condutor']:
            marker = " ← ELECTRIC"
        if w in ['minério', 'metal', 'ferro', 'zinco', 'liga', 'bronze', 'latão']:
            marker = " ← MINING"
        if w in ['célula', 'animal', 'tecido', 'sangue', 'coração']:
            marker = " ← BIO"
        print(f"    {i+1:2d}. {w:20s}  gate={s:.4f}{marker}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# TEST C: Contextual Projection — remove components orthogonal to context
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TEST C: Contextual Projection")
print("=" * 70)
print("Keep only the subspace spanned by top-N words similar to context.")
print()

for name, ctx in [('ELECTRICITY', ctx_eletric), ('MINING', ctx_mining), ('BIOLOGY', ctx_bio)]:
    # Find top-N words most similar to context
    sim_ctx = vectors @ ctx
    sim_ctx = sim_ctx / (np.linalg.norm(vectors, axis=1) + 1e-12)

    N = 50  # subspace dimension
    top_N_idx = np.argsort(sim_ctx)[::-1][:N]
    # Build orthonormal basis for this subspace via QR
    subspace_vecs = vectors[top_N_idx].T  # (D, N)
    Q, _ = np.linalg.qr(subspace_vecs)  # (D, N)

    # Project cobre onto this subspace
    cobre_proj = Q @ (Q.T @ v_cobre)
    cobre_proj = normalize(cobre_proj)

    top = top_similar(cobre_proj, 15, ['cobre'])
    print(f"  [projection(cobre, {name}_subspace)]")
    for i, (w, s) in enumerate(top):
        marker = ""
        if w in ['conduz', 'corrente', 'elétrica', 'elétricas', 'condutor']:
            marker = " ← ELECTRIC"
        if w in ['minério', 'metal', 'ferro', 'zinco', 'liga', 'bronze', 'latão']:
            marker = " ← MINING"
        if w in ['célula', 'animal', 'tecido', 'sangue', 'coração']:
            marker = " ← BIO"
        print(f"    {i+1:2d}. {w:20s}  cos={s:.4f}{marker}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# TEST D: Novel — Phase Shift by Context (preserves magnitude, shifts phase)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TEST D: Phase Rotation by Context")
print("=" * 70)
print("Rotate word's frequency phases toward context's phases.")
print("Preserves magnitude → stays in similar region of space.")
print()

for name, ctx in [('ELECTRICITY', ctx_eletric), ('MINING', ctx_mining), ('BIOLOGY', ctx_bio)]:
    for alpha in [0.1, 0.3, 0.5]:
        C = fft(v_cobre)
        X = fft(ctx)

        # Interpolate phases: move cobre's phase toward context's phase
        C_mag = np.abs(C)
        C_phase = C / (C_mag + 1e-12)
        X_phase = X / (np.abs(X) + 1e-12)

        # Phase interpolation (on the unit circle)
        # phase_new = phase_cobre + alpha * (phase_ctx - phase_cobre)
        # But we need to handle angle wrapping properly
        phase_diff = X_phase / (C_phase + 1e-12)  # e^{i(θ_x - θ_c)}
        # Weighted: e^{i(θ_c + α*(θ_x - θ_c))} = e^{iθ_c} * (e^{i(θ_x - θ_c)})^{α}
        phase_shifted = C_phase * (phase_diff ** alpha)

        # Reconstruct: original magnitude, shifted phase
        result_spectrum = C_mag * phase_shifted
        v_rotated = normalize(ifft(result_spectrum).real)

        top = top_similar(v_rotated, 15, ['cobre'])
        print(f"  [phase_rotate(cobre, {name}, α={alpha})]")
        top_words = [f"{w}({s:.3f})" for w, s in top[:8]]
        print(f"    Top 8: {', '.join(top_words)}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY: Which approach works?
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("SUMMARY: Context-Dependent Ranking for Key Domain Words")
print("=" * 70)

def ranking(word, sims_vector):
    """Return rank of word (0 = best)."""
    idx = w2i[word]
    return int(np.sum(sims_vector > sims_vector[idx]))

# Baseline
base_sims = vectors @ v_cobre
base_sims = base_sims / (np.linalg.norm(vectors, axis=1) + 1e-12)

# Gate approach
gate_eletric = base_sims * (vectors @ ctx_eletric / (np.linalg.norm(vectors, axis=1) + 1e-12))
gate_mining = base_sims * (vectors @ ctx_mining / (np.linalg.norm(vectors, axis=1) + 1e-12))
gate_bio = base_sims * (vectors @ ctx_bio / (np.linalg.norm(vectors, axis=1) + 1e-12))

# Unbind approach
prop_eletric = normalize(np.real(ifft(fft(M_eletric) * np.conj(fft(v_cobre)))))
prop_mining = normalize(np.real(ifft(fft(M_mining) * np.conj(fft(v_cobre)))))
prop_bio = normalize(np.real(ifft(fft(M_bio) * np.conj(fft(v_cobre)))))
unbind_eletric = vectors @ prop_eletric / (np.linalg.norm(vectors, axis=1) + 1e-12)
unbind_mining = vectors @ prop_mining / (np.linalg.norm(vectors, axis=1) + 1e-12)
unbind_bio = vectors @ prop_bio / (np.linalg.norm(vectors, axis=1) + 1e-12)

# Shift approach (α=0.3)
direction_e = ctx_eletric - global_centroid
direction_e = direction_e / (np.linalg.norm(direction_e) + 1e-12)
direction_m = ctx_mining - global_centroid
direction_m = direction_m / (np.linalg.norm(direction_m) + 1e-12)
direction_b = ctx_bio - global_centroid
direction_b = direction_b / (np.linalg.norm(direction_b) + 1e-12)

shift_e = normalize(v_cobre + 0.3 * direction_e)
shift_m = normalize(v_cobre + 0.3 * direction_m)
shift_b = normalize(v_cobre + 0.3 * direction_b)
shift_e_sims = vectors @ shift_e / (np.linalg.norm(vectors, axis=1) + 1e-12)
shift_m_sims = vectors @ shift_m / (np.linalg.norm(vectors, axis=1) + 1e-12)
shift_b_sims = vectors @ shift_b / (np.linalg.norm(vectors, axis=1) + 1e-12)

track = {
    'ELECTRIC': ['conduz', 'corrente', 'elétrica', 'condutor', 'térmica', 'transmissão'],
    'MINING': ['minério', 'metal', 'ferro', 'liga', 'bronze', 'latão'],
    'BIO': ['célula', 'animal', 'tecido', 'coração', 'sangue'],
}

print(f"\n{'Word':<16s} {'Base':>5s} | {'G-E':>5s} {'G-M':>5s} {'G-B':>5s} | {'U-E':>5s} {'U-M':>5s} {'U-B':>5s} | {'S-E':>5s} {'S-M':>5s} {'S-B':>5s}")
print("-" * 90)

for domain, words in track.items():
    for w in words:
        if w not in w2i:
            continue
        r_base = ranking(w, base_sims)
        r_ge = ranking(w, gate_eletric)
        r_gm = ranking(w, gate_mining)
        r_gb = ranking(w, gate_bio)
        r_ue = ranking(w, unbind_eletric)
        r_um = ranking(w, unbind_mining)
        r_ub = ranking(w, unbind_bio)
        r_se = ranking(w, shift_e_sims)
        r_sm = ranking(w, shift_m_sims)
        r_sb = ranking(w, shift_b_sims)

        # Highlight best approach for each domain
        best_e = min(r_ge, r_ue, r_se)
        best_m = min(r_gm, r_um, r_sm)
        best_b = min(r_gb, r_ub, r_sb)

        print(f"{w:<16s} {r_base:>5d} | {r_ge:>5d} {r_gm:>5d} {r_gb:>5d} | {r_ue:>5d} {r_um:>5d} {r_ub:>5d} | {r_se:>5d} {r_sm:>5d} {r_sb:>5d}")

print("\nLegend: G=Gated, U=Unbind, S=Shift, E=Electricity, M=Mining, B=Biology")
print("Lower rank = better (0 = most similar)")
