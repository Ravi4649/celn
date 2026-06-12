"""
Test: Parallel Transport Generation
=====================================
Full generation test using PairSDM + Resonator as primary engine.
Compares Transport vs SDM baseline on 10 factual questions.
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
    'o','a','os','as','um','uma','uns','umas',
    'de','do','da','dos','das','em','no','na','nos','nas','e','ou','mas','que','se','nem','pois',
    'é','foi','era','são','está','ser','sendo','estava','foram','não','sim','como','quando','onde',
    'porque','muito','pouco','mais','menos','tão','ele','ela','eles','elas','seu','sua',
    'para','com','por','pelo','pela','pelos','pelas','sem','sob','sobre','entre','até',
}

print("=" * 70)
print("PARALLEL TRANSPORT GENERATION TEST")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[1] Loading...")
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
# 2. TRAIN TYPE FIELD
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2] Training type field...")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ENCODE PAIRS & CREATE PAIR SDM
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3] Encoding word pairs for PairSDM...")
pair_vectors = []
for sent in sentences:
    for i in range(len(sent) - 1):
        w1, w2 = sent[i], sent[i+1]
        if w1 in w2i and w2 in w2i:
            pv = M(sem_vecs[w2i[w1]], sem_vecs[w2i[w2]], gamma=1.0, bilateral=True)
            pair_vectors.append(pv)

print(f"    {len(pair_vectors)} pairs encoded")

pair_sdm = DenseSDM(n_locations=8192, activation_pct=0.005, seed=42)
n_seed = min(len(pair_vectors), 8000)
seed_idx = np.random.RandomState(42).choice(len(pair_vectors), n_seed, replace=False)
seed_vecs = np.array([pair_vectors[i] for i in seed_idx])
pair_sdm.initialize_addresses(seed_vecs)

print("    Writing to PairSDM...")
for i, pv in enumerate(pair_vectors):
    pair_sdm.write(pv)
print(f"    PairSDM ready: {pair_sdm.stats['n_written']} locations written")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. CREATE GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4] Creating generators...")
sdm_word = DenseSDM(n_locations=4096, activation_pct=0.01, seed=42)
sc_list = []
for tokens in sentences[:2000]:
    idxs = [w2i[w] for w in tokens if w in w2i]
    if idxs:
        sc_list.append(normalize(sem_vecs[idxs].mean(axis=0)))
sdm_word.initialize_addresses(np.array(sc_list))
for i in range(V):
    sdm_word.write(sem_vecs[i])

gen_transport = DualChannelGenerator(
    sem_vecs, type_vecs, w2i, i2w, window_size=5, window_decay=0.7,
    sdm=None, use_phase_lens=True, phase_lens_max_alpha=0.6,
    pair_sdm=pair_sdm,
)

gen_sdm = DualChannelGenerator(
    sem_vecs, type_vecs, w2i, i2w, window_size=5, window_decay=0.7,
    sdm=sdm_word, use_phase_lens=True, phase_lens_max_alpha=0.6,
    pair_sdm=None,
)

gen_transport.learn_type_field(sentences)
gen_sdm.learn_type_field(sentences)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. TEST: 10 factual questions
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("GENERATION TEST: 10 Factual Questions")
print("=" * 70)

questions = [
    ("metais", ["metal"]),
    ("cobre", ["cobre"]),
    ("água", ["água"]),
    ("cobra", ["cobra"]),
    ("célula", ["célula"]),
    ("energia", ["energia"]),
    ("animal", ["animal"]),
    ("plantas", ["planta"]),
    ("frança", ["frança"]),
    ("sol", ["sol"]),
]

transport_results = []
sdm_results = []

for question, prefix in questions:
    t_out = gen_transport.generate(
        prefix_words=prefix, max_len=8, temperature=0.5, seed=42,
        session_context=None, thematic_state=None, dynamic_temperature=False,
    )
    s_out = gen_sdm.generate(
        prefix_words=prefix, max_len=8, temperature=0.5, seed=42,
        session_context=None, thematic_state=None, dynamic_temperature=False,
    )

    t_func = sum(1 for w in t_out if w in FUNCTION_WORDS) / max(len(t_out), 1)
    s_func = sum(1 for w in s_out if w in FUNCTION_WORDS) / max(len(s_out), 1)

    transport_results.append((question, ' '.join(t_out), t_func))
    sdm_results.append((question, ' '.join(s_out), s_func))

# ═══════════════════════════════════════════════════════════════════════════════
# 6. REPORT
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'Question':<12s} | {'TRANSPORT (PairSDM + Resonator)':<55s} | {'f%':>4s} | {'SDM BASELINE':<55s} | {'f%':>4s}")
print("-" * 135)

t_funcs = []
s_funcs = []

for i, ((q, t_out, tf), (_, s_out, sf)) in enumerate(zip(transport_results, sdm_results)):
    print(f"{q:<12s} | {t_out:<55s} | {tf:>3.0%} | {s_out:<55s} | {sf:>3.0%}")
    t_funcs.append(tf)
    s_funcs.append(sf)

print(f"\n  TRANSPORT mean func words: {np.mean(t_funcs):.1%}")
print(f"  SDM      mean func words: {np.mean(s_funcs):.1%}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
