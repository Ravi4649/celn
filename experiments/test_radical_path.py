"""
Test: Radical Path-Based Generation (ZERO cosine similarity)
Compares: baseline generate() vs generate_path_based()
"""
import sys, os, numpy as np
sys.path.insert(0, '/home/ravizin/celn-v3')
from celn_v3.train import load_corpus, build_cooccurrence, compute_ppmi
from celn_v3.dual_channel import extract_type_vectors
import warnings
warnings.filterwarnings('ignore')

FUNCTION_WORDS = {
    'o','a','os','as','um','uma','uns','umas','de','do','da','dos','das',
    'em','no','na','nos','nas','e','ou','mas','que','se','é','são','está',
    'não','como','por','para','com','ele','ela','este','essa','seu','sua',
}

print("="*70)
print("RADICAL PATH-BASED vs BASELINE")
print("="*70)

data = np.load('/home/ravizin/celn-v3/celn_v3_full_vectors.npz', allow_pickle=True)
sem_vecs = data['vectors']
vocab = data['vocab']
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for i, w in enumerate(vocab)}
V, D = sem_vecs.shape
print(f"Loaded {V} words x {D}D\n")

sentences = load_corpus('/home/ravizin/celn-v3/corpus_final.txt', min_len=2)
word_counts, cooc_counts, _, _ = build_cooccurrence(sentences, window_size=5)
ppmi = compute_ppmi(word_counts, cooc_counts, w2i)
type_vecs = extract_type_vectors(ppmi, type_dim=2000)

# Build pair indices
pair_src = []
pair_fol = []
for sent in sentences:
    for i in range(len(sent) - 1):
        w1, w2 = sent[i], sent[i+1]
        if w1 in w2i and w2 in w2i:
            pair_src.append(w2i[w1])
            pair_fol.append(w2i[w2])

# Radical generator
from celn_v3.radical_path import RadicalPathGenerator
rgen = RadicalPathGenerator(
    sem_vecs, type_vecs, w2i, i2w,
    pair_source_indices=np.asarray(pair_src, dtype=np.int32),
    pair_follower_indices=np.asarray(pair_fol, dtype=np.int32),
)
rgen.learn_type_field(sentences)

prefixes = [
    ["o", "cobre"],
    ["a", "eletricidade"],
    ["o", "metal"],
    ["o", "gato"],
]

print("--- Radical Path (n_depth=2) ---")
for prefix in prefixes:
    output = rgen.generate(prefix, max_len=10, temperature=0.5, seed=42, n_depth=2)
    func = sum(1 for w in output if w in FUNCTION_WORDS) / max(len(output), 1)
    print(f"  {prefix} → {' '.join(output)} (func={func:.0%})")

print("\n--- Baseline generate() ---")
from celn_v3.memory import DenseSDM
from celn_v3.core import projective_resonance as M
from celn_v3.dual_channel import DualChannelGenerator

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
for i, pv in enumerate(pair_vectors):
    pair_sdm.write(pv)

gen = DualChannelGenerator(
    sem_vecs, type_vecs, w2i, i2w,
    window_size=5, window_decay=0.7,
    pair_sdm=pair_sdm,
)
gen.learn_type_field(sentences)

for prefix in prefixes:
    output = gen.generate(prefix, max_len=10, temperature=0.5, seed=42)
    func = sum(1 for w in output if w in FUNCTION_WORDS) / max(len(output), 1)
    print(f"  {prefix} → {' '.join(output)} (func={func:.0%})")
