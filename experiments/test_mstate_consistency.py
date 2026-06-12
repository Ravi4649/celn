"""
Test: M-State Consistency Selection for Transport Engine
==========================================================
Tests whether re-ranking candidates by consistency with the
M-state of the conversation produces more coherent responses
than similarity-based selection alone.

Comparison:
  Baseline: Transport with similarity scoring only
  M-State:  Transport with M-state consistency re-ranking
"""

import sys
sys.path.insert(0, '/home/ravizin/celn-v3')

import numpy as np
from celn_v3.core import normalize, projective_resonance as M
from celn_v3.dual_channel import DualChannelGenerator, extract_type_vectors
from celn_v3.train import tokenize, load_corpus, build_cooccurrence, compute_ppmi
from celn_v3.memory import DenseSDM
import warnings
warnings.filterwarnings('ignore')

FUNCTION_WORDS = {
    'o','a','os','as','um','uma','uns','umas','de','do','da','dos','das',
    'em','no','na','nos','nas','e','ou','mas','que','se','nem','pois',
    'é','foi','era','são','está','ser','sendo','estava','foram',
    'não','sim','como','quando','onde','porque','muito','pouco',
    'mais','menos','tão','ele','ela','eles','elas','seu','sua',
    'para','com','por','pelo','pela','pelos','pelas','sem','sob',
    'sobre','entre','até',
}

print("=" * 70)
print("M-STATE CONSISTENCY SELECTION TEST")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & SETUP
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[1] Loading vectors...")
data = np.load('/home/ravizin/celn-v3/celn_v3_full_vectors.npz', allow_pickle=True)
sem_vecs = data['vectors']
vocab = data['vocab']
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for i, w in enumerate(vocab)}
V, D = sem_vecs.shape

sentences = load_corpus('/home/ravizin/celn-v3/corpus_final.txt', min_len=2)
word_counts, cooc_counts, _, _ = build_cooccurrence(sentences, window_size=5)
ppmi = compute_ppmi(word_counts, cooc_counts, w2i)
type_vecs = extract_type_vectors(ppmi, type_dim=2000)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ENCODE PAIRS → PAIR SDM
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2] Encoding word pairs for PairSDM...")
pair_vectors = []
for sent in sentences:
    for i in range(len(sent) - 1):
        w1, w2 = sent[i], sent[i+1]
        if w1 in w2i and w2 in w2i:
            pv = M(sem_vecs[w2i[w1]], sem_vecs[w2i[w2]], gamma=1.0, bilateral=True)
            pair_vectors.append(pv)

pair_sdm = DenseSDM(n_locations=8192, activation_pct=0.005, seed=42)
n_seed = min(len(pair_vectors), 8000)
seed_idx = np.random.RandomState(42).choice(len(pair_vectors), n_seed, replace=False)
seed_vecs = np.array([pair_vectors[i] for i in seed_idx])
pair_sdm.initialize_addresses(seed_vecs)
for pv in pair_vectors:
    pair_sdm.write(pv)
print(f"    {len(pair_vectors)} pairs in PairSDM")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. CREATE GENERATOR WITH M-STATE CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3] Creating generator...")
gen = DualChannelGenerator(
    sem_vecs, type_vecs, w2i, i2w, window_size=5, window_decay=0.7,
    sdm=None, use_phase_lens=True, phase_lens_max_alpha=0.6,
    pair_sdm=pair_sdm,
)
gen.learn_type_field(sentences)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TEST: Factual questions with domain context priming
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("GENERATION: M-State Consistency + Transport")
print("=" * 70)

# Domain-specific prefixes with strong M-state priming
test_cases = [
    # (question, prefix_words, domain_context_words)
    ("metais e condutividade", ["metal", "conduz"],
     ["eletricidade","corrente","transmissão","condutor","elétrica"]),
    ("cobre na indústria", ["cobre", "utilizado"],
     ["estanho","zinco","bronze","latão","produção"]),
    ("água e natureza", ["água", "rio"],
     ["lago","chuva","oceano","peixe","correnteza"]),
    ("cobra como predador", ["cobra", "devorou"],
     ["rato","presa","caça","animal","bote"]),
    ("energia e movimento", ["energia", "cinética"],
     ["força","trabalho","massa","corpo","movimento"]),
]

for question, prefix, domain_ctx in test_cases:
    # Build domain context vector for session
    ctx_vecs = [sem_vecs[w2i[w]] for w in domain_ctx if w in w2i]
    session_ctx = [normalize(np.mean(ctx_vecs, axis=0))] if ctx_vecs else None

    # Generate
    output = gen.generate(
        prefix_words=prefix, max_len=10, temperature=0.5, seed=42,
        session_context=session_ctx, thematic_state=None,
        dynamic_temperature=False,
    )

    func_ratio = sum(1 for w in output if w in FUNCTION_WORDS) / max(len(output), 1)
    output_str = ' '.join(output)

    print(f"\n  [{question}]")
    print(f"    → {output_str}")
    print(f"    func={func_ratio:.0%}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. DIRECT M-STATE CONSISTENCY TEST
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("DIRECT CONSISTENCY TEST: Which candidate fits the M-state?")
print("=" * 70)

# Build an M-state for an ELECTRICITY conversation
electricity_words = ["cobre","conduz","corrente","elétrica","energia","transmissão"]
e_vecs = [sem_vecs[w2i[w]] for w in electricity_words if w in w2i]
m_state_electric = e_vecs[0].copy()
for v in e_vecs[1:]:
    m_state_electric = M(m_state_electric, v, gamma=1.0, bilateral=True)

# Build an M-state for a MINING conversation
mining_words = ["minério","metal","ferro","extração","produção","fundição"]
m_vecs = [sem_vecs[w2i[w]] for w in mining_words if w in w2i]
m_state_mining = m_vecs[0].copy()
for v in m_vecs[1:]:
    m_state_mining = M(m_state_mining, v, gamma=1.0, bilateral=True)

# Test: for the same recent context ["o", "cobre", "é"], which follow-up
# word has HIGHER consistency with electricity vs mining M-state?
context_words = ["cobre"]
context_vecs = [sem_vecs[w2i[w]] for w in context_words if w in w2i]
recent_state = context_vecs[0].copy()

candidates = ["conduz", "metal", "elétrica", "minério", "corrente", "zinco"]

print(f"\n  Context: 'cobre'")
print(f"  {'Candidate':<12s} {'cos w/ELECTRIC M':>16s} {'cos w/MINING M':>16s} {'Δ(E-M)':>8s} {'Winner':>10s}")
print("  " + "-" * 65)

for w in candidates:
    if w not in w2i:
        continue
    v_w = sem_vecs[w2i[w]]
    simulated = M(recent_state, v_w, gamma=1.0, bilateral=True)

    sim_e = float(np.dot(simulated, m_state_electric) /
                   (np.linalg.norm(simulated) * np.linalg.norm(m_state_electric) + 1e-12))
    sim_m = float(np.dot(simulated, m_state_mining) /
                   (np.linalg.norm(simulated) * np.linalg.norm(m_state_mining) + 1e-12))

    delta = sim_e - sim_m
    winner = "ELECTRIC" if delta > 0 else "MINING"

    print(f"  {w:<12s} {sim_e:>16.4f} {sim_m:>16.4f} {delta:>+8.4f} {winner:>10s}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
