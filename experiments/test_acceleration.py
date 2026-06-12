#!/usr/bin/env python3
"""Teste de Geração por Aceleração — ZERO similaridade cosseno."""
import sys, numpy as np
sys.path.insert(0, '/home/ravizin/celn-v3')

from celn_v3.train import load_corpus, build_cooccurrence, compute_ppmi
from celn_v3.core import D, normalize, batch_normalize
from celn_v3.dual_channel import extract_type_vectors
from celn_v3.radical_path import RadicalPathGenerator
import warnings
warnings.filterwarnings('ignore')

FUNCTION_WORDS = {
    'o','a','os','as','um','uma','uns','umas','de','do','da','dos','das',
    'em','no','na','nos','nas','e','ou','mas','que','se','é','são','está',
    'não','como','por','para','com','ele','ela','este','essa','seu','sua',
    'seus','suas','tem','tinha','mais','muito','bem','já','só','até','sem',
    'entre','depois','antes','durante','contra','num','numa','pelo','pela',
    'aos','às','pra','pro','isto','isso','aquilo','quem','cuja','cujo',
    'quando','onde','porque','pois','então','assim','mesmo','também',
}

DOMAINS = {
    'cobre': ['cobre', 'metal', 'minério', 'elétrica', 'condutividade'],
    'electricidade':   ['electricidade', 'energia', 'elétrica', 'conduz', 'corrente'],
    'brasil': ['brasil', 'brasileiro', 'brasileira', 'nacional', 'brasília'],
    'gato':   ['gato', 'gatos', 'felino', 'felinos', 'animal'],
}

print("="*70)
print("TESTE DE GERAÇÃO POR ACELERAÇÃO")
print("  (ZERO similaridade cosseno, ZERO arestas de grafo)")
print("="*70)

# Carrega vetores SVD expandidos
data = np.load('/home/ravizin/celn-v3/celn_v3_expanded_svd_vectors.npz', allow_pickle=True)
sem_vecs = data['vectors']
vocab = list(data['vocab'])
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for i, w in enumerate(vocab)}
V, D = sem_vecs.shape
print(f"\nVetores: {V} palavras × {D}D (corpus expandido: 16.338 sentenças)\n")

# Carrega corpus expandido
sentences = load_corpus('/home/ravizin/celn-v3/corpus_pt_expandido.txt', min_len=1)

# Build Type vectors (random indexing)
type_dim = 2000
type_vecs = np.random.RandomState(42).randn(V, type_dim).astype(np.float32)
type_vecs = batch_normalize(type_vecs)

# Build pair indices
pair_src = []
pair_fol = []
for sent in sentences:
    for i in range(len(sent) - 1):
        w1, w2 = sent[i], sent[i+1]
        if w1 in w2i and w2 in w2i:
            pair_src.append(w2i[w1])
            pair_fol.append(w2i[w2])

print(f"Pares do corpus: {len(pair_src):,}")

# Cria generator com aceleração
rgen = RadicalPathGenerator(
    sem_vecs, type_vecs, w2i, i2w,
    pair_source_indices=np.asarray(pair_src, dtype=np.int32),
    pair_follower_indices=np.asarray(pair_fol, dtype=np.int32),
    window_size=5,
)
rgen.learn_type_field(sentences)

prefixes = [
    ["o", "cobre"],
    ["a", "eletricidade"],
    ["o", "brasil"],
    ["o", "gato"],
    ["amor", "e"],
    ["o", "futebol"],
]

print(f"\n{'─'*70}")
print("GERAÇÃO POR ACELERAÇÃO (temperature=0.3, seed=42)")
print(f"{'─'*70}")
for prefix in prefixes:
    output = rgen.generate(prefix, max_len=12, temperature=0.3, seed=42)
    func = sum(1 for w in output if w in FUNCTION_WORDS) / max(len(output), 1)
    
    # Verifica keywords semânticas
    dom = 0
    for dwords in DOMAINS.values():
        dom += sum(1 for w in output if w in dwords)
    
    print(f"  {' '.join(prefix):<20} → {' '.join(output)}")
    print(f"    (func={func:.0%} | dom_kwords={dom})")

# Testa com temperature mais baixa
print(f"\n{'─'*70}")
print("GERAÇÃO POR ACELERAÇÃO (temperature=0.15, seed=42) — mais focado")
print(f"{'─'*70}")
for prefix in prefixes:
    output = rgen.generate(prefix, max_len=12, temperature=0.15, seed=42)
    func = sum(1 for w in output if w in FUNCTION_WORDS) / max(len(output), 1)
    dom = 0
    for dwords in DOMAINS.values():
        dom += sum(1 for w in output if w in dwords)
    print(f"  {' '.join(prefix):<20} → {' '.join(output)} (func={func:.0%} | dom={dom})")

print(f"\n{'='*70}")
print("FIM DO TESTE")
print("="*70)
