"""
Test: Phase-Aware Dual-Channel Generation
==========================================
Diagnostic test of phase lens integration into DualChannelGenerator.

Tests:
  1. Word ranking shift: verify phase lens changes semantic scores
  2. Isolated semantic generation: bypass type field, test semantics only
  3. Full generation: baseline vs phase-aware with session context
"""

import sys
sys.path.insert(0, '/home/ravizin/celn-v3')

import numpy as np
from celn.core import normalize, similarity, projective_resonance, phase_lens
from celn.dual_channel import DualChannelGenerator, extract_type_vectors
from celn.train import tokenize, load_corpus, build_cooccurrence, compute_ppmi
from celn.memory import DenseSDM, sentence_to_centroid
import re
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PHASE LENS INTEGRATION — DIAGNOSTIC TEST")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[1] Loading vectors and building type space...")
data = np.load('/home/ravizin/celn-v3/celn_full_vectors.npz', allow_pickle=True)
sem_vecs = data['vectors']
vocab = data['vocab']
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for i, w in enumerate(vocab)}

sentences = load_corpus('/home/ravizin/celn-v3/corpus_final.txt', min_len=2)
word_counts, cooc_counts, _, _ = build_cooccurrence(sentences, window_size=5)
ppmi = compute_ppmi(word_counts, cooc_counts, w2i)
type_vecs = extract_type_vectors(ppmi, type_dim=2000)

# Both generators
gen_base = DualChannelGenerator(sem_vecs, type_vecs, w2i, i2w, window_size=5, window_decay=0.7,
                                 use_phase_lens=False)
gen_phase = DualChannelGenerator(sem_vecs, type_vecs, w2i, i2w, window_size=5, window_decay=0.7,
                                  use_phase_lens=True, phase_lens_max_alpha=0.6)
gen_base.learn_type_field(sentences)
gen_phase.learn_type_field(sentences)

print(f"    {gen_base.vocab_size} words, type fields trained")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. BUILD DOMAIN CONTEXTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2] Building domain contexts...")
eletric_kw = {'cobre','conduz','corrente','eletricidade','elétrica','elétrico',
              'condutor','condutividade','térmica','transmissão','energia','elétricas','eletrônica'}
mining_kw = {'minério','metal','ferro','liga','bronze','latão','zinco',
             'estanho','metálico','fundição','produção','industrial'}
bio_kw = {'célula','animal','tecido','fotossíntese','planta',
          'coração','sangue','câncer','hormônio','nervo','proteína'}

def domain_centroid_M(sents, max_s=20):
    """Build domain context vector from WORD CENTROIDS (not M-encoded).

    KEY FINDING: M-encoded states are quasi-orthogonal to word vectors.
    Phase rotation needs centroids that share the same vector space.
    Word centroids preserve semantic similarity and phase structure.
    """
    all_vecs = []
    for s in sents[:max_s]:
        for t in s:
            if t in w2i:
                all_vecs.append(sem_vecs[w2i[t]])
    if not all_vecs:
        return np.zeros(10000)
    return normalize(np.mean(all_vecs, axis=0))

def classify_s(s):
    words = set(tokenize(' '.join(s), min_len=2))
    sc = {'e': len(words & eletric_kw), 'm': len(words & mining_kw), 'b': len(words & bio_kw)}
    best = max(sc, key=sc.get)
    return best if sc[best] >= 2 and sc[best] > sorted(sc.values(), reverse=True)[1] else None

domain_sents = {'e': [], 'm': [], 'b': []}
for s in sentences:
    d = classify_s(s)
    if d:
        domain_sents[d].append(s)

ctx_e = domain_centroid_M(domain_sents['e'])
ctx_m = domain_centroid_M(domain_sents['m'])
ctx_b = domain_centroid_M(domain_sents['b'])

print(f"    E={len(domain_sents['e'])} M={len(domain_sents['m'])} B={len(domain_sents['b'])}")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: Word ranking shift under different contexts
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TEST 1: Word ranking shifts (phase lens vs static, same query)")
print("=" * 70)

v_cobre = sem_vecs[w2i['cobre']]
ctx_window = normalize(np.mean(sem_vecs[[w2i[w] for w in ['o','cobre','é'] if w in w2i]], axis=0))

# Static scores
static_scores = sem_vecs @ ctx_window

# Phase-aware scores for each domain
phase_e_scores = sem_vecs @ phase_lens(ctx_window, ctx_e, alpha=0.4)
phase_m_scores = sem_vecs @ phase_lens(ctx_window, ctx_m, alpha=0.4)
phase_b_scores = sem_vecs @ phase_lens(ctx_window, ctx_b, alpha=0.4)

def ranking(word, scores):
    idx = w2i[word]
    return int(np.sum(scores > scores[idx]))

track = {
    '⚡ ELECTRIC': ['conduz','corrente','elétrica','condutor','transmissão','energia','térmica'],
    '⛏ MINING': ['minério','metal','ferro','liga','bronze','latão','zinco'],
    '🧬 BIO': ['célula','animal','tecido','coração','sangue'],
}

print(f"\n{'Word':<16s} {'Static':>6s} {'PH-E':>6s} {'PH-M':>6s} {'PH-B':>6s} | Best")
print("-" * 65)

for label, words in track.items():
    for w in words:
        if w not in w2i:
            continue
        rs = ranking(w, static_scores)
        re_ = ranking(w, phase_e_scores)
        rm = ranking(w, phase_m_scores)
        rb = ranking(w, phase_b_scores)
        best_rank = min(re_, rm, rb)
        best_domain = ''
        if best_rank == re_: best_domain = 'E'
        elif best_rank == rm: best_domain = 'M'
        else: best_domain = 'B'
        print(f"{w:<16s} {rs:>6d} {re_:>6d} {rm:>6d} {rb:>6d} | {best_domain}")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: Semantic-only generation (bypass type field)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TEST 2: Semantic-only generation (type field bypassed)")
print("=" * 70)
print("Tests if phase lens improves semantic coherence when type")
print("doesn't dominate. Same prefix, different contexts.")
print()

prefix = ["o", "cobre", "é"]
prefix_idx = [w2i[w] for w in prefix if w in w2i]

def semantic_only_generate(gen, prefix_idx, session_ctx, max_len=8, temp=0.7):
    """Generate using ONLY semantic scores, no type blend."""
    rng = np.random.RandomState(42)
    recent = prefix_idx.copy()
    sem_recent = [gen.sem_vecs[i] for i in prefix_idx]
    generated = []

    for step in range(max_len):
        excluded = set(recent[-5:])
        centroid = gen._context_centroid(sem_recent)

        if session_ctx and gen.use_phase_lens:
            ctx_vec = normalize(np.mean([normalize(sv) for sv in session_ctx], axis=0))
            scores = gen._phase_aware_semantic_scores(centroid, ctx_vec, excluded)
        else:
            scores = gen.sem_vecs @ centroid.astype(np.float32)
            for idx in excluded:
                scores[idx] = -1.0

        # Temperature sampling
        valid_mask = np.ones(len(scores), dtype=bool)
        for idx in excluded:
            valid_mask[idx] = False
        valid = scores[valid_mask]
        if valid.max() - valid.min() < 1e-12:
            break
        scores_c = scores - valid.max()
        exp_s = np.exp(scores_c / (temp * max(np.std(valid), 1e-6)))
        probs = exp_s / exp_s.sum()

        idx = rng.choice(len(scores), p=probs)
        generated.append(gen.i2w[idx])
        recent.append(idx)
        sem_recent.append(gen.sem_vecs[idx])
        if len(sem_recent) > gen.window_size:
            sem_recent.pop(0)

    return generated

configs = [
    ("NO CONTEXT   ", None, False),
    ("ELECTRICITY ⚡", [ctx_e], False),
    ("MINING ⛏     ", [ctx_m], False),
    ("BIOLOGY 🧬   ", [ctx_b], False),
    ("ELECTRICITY ⚡", [ctx_e], True),
    ("MINING ⛏     ", [ctx_m], True),
    ("BIOLOGY 🧬   ", [ctx_b], True),
]

for label, ctx, use_phase in configs:
    gen = gen_phase if use_phase else gen_base
    output = semantic_only_generate(gen, prefix_idx, ctx)
    e_c = sum(1 for w in output if w in eletric_kw)
    m_c = sum(1 for w in output if w in mining_kw)
    b_c = sum(1 for w in output if w in bio_kw)
    output_str = ' '.join(output)
    phase_tag = "PHASE" if use_phase else "static"
    print(f"  [{label} | {phase_tag}] ⚡{e_c} ⛏{m_c} 🧬{b_c}")
    print(f"    → {output_str}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Full generator with session context
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TEST 3: Full generator (type maestro + phase lens)")
print("=" * 70)

prefix_words = ["o", "cobre"]

for label, ctx, use_phase in [
    ("NO CONTEXT", None, False),
    ("NO CONTEXT", None, True),
    ("ELECTRICITY", [ctx_e], False),
    ("ELECTRICITY", [ctx_e], True),
    ("MINING", [ctx_m], False),
    ("MINING", [ctx_m], True),
    ("BIOLOGY", [ctx_b], False),
    ("BIOLOGY", [ctx_b], True),
]:
    gen = gen_phase if use_phase else gen_base
    try:
        output = gen.generate(
            prefix_words=prefix_words, max_len=12, temperature=0.7, seed=42,
            session_context=ctx, thematic_state=None,
        )
        phase_tag = "PHASE" if use_phase else "BASE"
        e_c = sum(1 for w in output if w in eletric_kw)
        m_c = sum(1 for w in output if w in mining_kw)
        b_c = sum(1 for w in output if w in bio_kw)
        print(f"  [{label:<14s} | {phase_tag}] ⚡{e_c} ⛏{m_c} 🧬{b_c}")
        print(f"    → {' '.join(output)}")
    except Exception as e:
        print(f"  [{label:<14s} | {phase_tag}] ERROR: {e}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
