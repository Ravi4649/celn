#!/usr/bin/env python3
"""
Test generation with expanded corpus vectors.
Compares original vs expanded corpus generation quality.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celn.train import load_corpus, build_cooccurrence, compute_ppmi
from celn.core import normalize, batch_normalize, projective_resonance as M
from celn.dual_channel import DualChannelGenerator
from celn.memory import DenseSDM

FUNCTION_WORDS = {
    'o','a','os','as','um','uma','uns','umas','de','do','da','dos','das',
    'em','no','na','nos','nas','e','ou','mas','que','se','é','são','está',
    'não','como','por','para','com','ele','ela','este','essa','seu','sua',
    'seus','suas','meu','minha','tem','tinha','mais','muito','bem','já',
    'só','até','sem','entre','depois','antes','durante','contra','num',
    'numa','dos','pelo','pela','aos','às','pra','pro','naquele','nesse',
    'nessa','isto','isso','aquele','aquela','ali','aqui','lá','quem',
    'cujo','cuja','cujos','cujas','quanto','quanta','quão','cada',
    'todo','toda','todos','todas','algum','alguma','alguns','algumas',
    'nenhum','nenhuma','outro','outra','outros','outras','mesmo','mesma',
    'próprio','própria','tão','quase','demais','pouco','pouca',
    'bastante','vários','várias','diversos','diversas','pode','podem',
    'era','foram','ser','sendo','sido','estava','estavam','estiver',
    'houve','havia','existe','existem','existia','existiam','fica',
    'ficam','ficava','ficaram','tornou','torna','tornam','passa',
    'passam','passou','passaram','vai','vão','foi','fui','fomos',
    'esteve','estiveram','teve','tiveram','eram','será','serão',
}

print("=" * 70)
print("COMPARISON: Original vs Expanded Corpus Generation")
print("=" * 70)

# ── Load expanded vectors ──
data = np.load('/home/ravizin/celn-v3/celn_expanded_svd_vectors.npz', allow_pickle=True)
exp_vecs = data['vectors']
exp_vocab = list(data['vocab'])
exp_w2i = {w: i for i, w in enumerate(exp_vocab)}
exp_i2w = {i: w for i, w in enumerate(exp_vocab)}
print(f"\nExpanded vectors: {len(exp_vocab)} words x {exp_vecs.shape[1]}D")

# ── Load original vectors ──
orig_data = np.load('/home/ravizin/celn-v3/celn_full_vectors.npz', allow_pickle=True)
orig_vecs = orig_data['vectors']
orig_vocab = list(orig_data['vocab'])
orig_w2i = {w: i for i, w in enumerate(orig_vocab)}
orig_i2w = {i: w for i, w in enumerate(orig_vocab)}
print(f"Original vectors:  {len(orig_vocab)} words x {orig_vecs.shape[1]}D")

# ── Load expanded corpus for pairs ──
sentences_exp = load_corpus('/home/ravizin/celn-v3/corpus_pt_expandido.txt', min_len=1)
print(f"Expanded corpus: {len(sentences_exp)} sentences")

# ── Build type fields ──
rng_t = np.random.RandomState(42)
type_dim = 2000

def make_type_field(sentences, w2i, vectors):
    V = len(w2i)
    tvecs = rng_t.randn(V, type_dim).astype(np.float32)
    tvecs = batch_normalize(tvecs)
    type_f = np.zeros((V, type_dim), dtype=np.float32)
    accum = np.zeros((V, type_dim), dtype=np.float32)
    counts = np.zeros(V, dtype=np.int32)
    for tokens in sentences:
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i+1]
            if w1 in w2i and w2 in w2i:
                accum[w2i[w1]] += tvecs[w2i[w2]]
                counts[w2i[w1]] += 1
    for i in range(V):
        if counts[i] > 0:
            type_f[i] = normalize(accum[i] / counts[i])
    return type_f, tvecs

exp_type_field, exp_type_vecs = make_type_field(sentences_exp, exp_w2i, exp_vecs)
print("  Type Field learned (expanded)")

# Original corpus for original vectors
sentences_orig = load_corpus('/home/ravizin/celn-v3/corpus_final.txt', min_len=1)
orig_type_field, orig_type_vecs = make_type_field(sentences_orig, orig_w2i, orig_vecs)
print("  Type Field learned (original)")

# ── Build PairSDM for generation ──
def build_pair_sdm(sentences, w2i, vectors):
    pair_vecs = []
    for tokens in sentences:
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i+1]
            if w1 in w2i and w2 in w2i:
                pv = M(vectors[w2i[w1]], vectors[w2i[w2]], gamma=1.0, bilateral=True)
                pair_vecs.append(pv)
    sdm = DenseSDM(n_locations=4096, activation_pct=0.01, seed=42)
    n_seed = min(len(pair_vecs), 4000)
    seed_idx = np.random.RandomState(42).choice(len(pair_vecs), n_seed, replace=False)
    seed_vecs = np.array([pair_vecs[i] for i in seed_idx])
    sdm.initialize_addresses(seed_vecs)
    for pv in pair_vecs:
        sdm.write(pv)
    return sdm

print("\nBuilding PairSDMs (this takes a moment)...")
exp_pair_sdm = build_pair_sdm(sentences_exp, exp_w2i, exp_vecs)
print("  PairSDM (expanded): ready")
orig_pair_sdm = build_pair_sdm(sentences_orig, orig_w2i, orig_vecs)
print("  PairSDM (original): ready")

# ── Create generators ──
exp_gen = DualChannelGenerator(
    exp_vecs, exp_type_vecs, exp_w2i, exp_i2w,
    window_size=5, window_decay=0.7,
    pair_sdm=exp_pair_sdm,
    use_phase_lens=True,
)
exp_gen.type_field = exp_type_field

orig_gen = DualChannelGenerator(
    orig_vecs, orig_type_vecs, orig_w2i, orig_i2w,
    window_size=5, window_decay=0.7,
    pair_sdm=orig_pair_sdm,
    use_phase_lens=True,
)
orig_gen.type_field = orig_type_field

# ── Test generation ──
prefixes = [
    ["o", "cobre"],
    ["a", "eletricidade"],
    ["o", "brasil"],
    ["o", "gato"],
]

for label, gen, vecs, i2w, w2i in [
    ("ORIGINAL (2920 sentences)", orig_gen, orig_vecs, orig_i2w, orig_w2i),
    ("EXPANDED (15394 sentences)", exp_gen, exp_vecs, exp_i2w, exp_w2i),
]:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    for prefix in prefixes:
        try:
            output = gen.generate(prefix, max_len=10, temperature=0.5, seed=42)
            func = sum(1 for w in output if w in FUNCTION_WORDS) / max(len(output), 1)
            domain = sum(1 for w in output if w in {w.lower() for w in prefix})
            print(f"  {' '.join(prefix):<20} → {' '.join(output):<60} func={func:.0%} dom={domain}")
        except Exception as e:
            print(f"  {' '.join(prefix):<20} → ERROR: {e}")
