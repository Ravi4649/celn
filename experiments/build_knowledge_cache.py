#!/usr/bin/env python3
"""
Build Knowledge Cache for GPVE's 5th channel (SDM Knowledge).
===============================================================
Roda UMA vez, offline. Constrói sentence_centroids.npz contendo:
  - centroids: (N_frases, D) — centróides IDF-ponderados de cada frase
  - idf: dict {palavra: peso_idf} — auto-calibrável do corpus

Uso:
  python experiments/build_knowledge_cache.py
  → produz sentence_centroids.npz (~150 MB)
  → GPVE carrega em < 1s
"""

import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import Counter

from celn_v3.core import normalize
from celn_v3.train import load_corpus, tokenize
from celn_v3.port_adapter import load_word_vectors

T = time.time

print("=" * 60)
print("Build Knowledge Cache — sentence_centroids.npz")
print("=" * 60)

# 1. Load vectors
t0 = T()
vectors, w2i = load_word_vectors("celn_v3_full_vectors.npz")
V, D = vectors.shape
print(f"[1] Vectors: {V} × {D}  ({T()-t0:.1f}s)")

# 2. Load corpus
t0 = T()
sentences = load_corpus("corpus_final.txt", min_len=2)
print(f"[2] Corpus: {len(sentences)} sentences  ({T()-t0:.1f}s)")

# 3. IDF weights
t0 = T()
doc_freq = Counter()
for toks in sentences:
    for w in set(toks):
        doc_freq[w] += 1
N = len(sentences)
idf = {w: float(np.log(max(N / max(df, 1), 1.0))) for w, df in doc_freq.items()}
print(f"[3] IDF: {len(idf)} words, range [{min(idf.values()):.1f}, {max(idf.values()):.1f}]  ({T()-t0:.1f}s)")

# 4. IDF-weighted centroids
t0 = T()
centroids = []
for toks in sentences:
    vecs, weights = [], []
    for w in toks:
        idx = w2i.get(w)
        if idx is not None:
            vecs.append(vectors[idx])
            weights.append(idf.get(w, 1.0))
    if vecs:
        arr = np.stack(vecs).astype(np.float32)
        w = np.array(weights, dtype=np.float32)
        c = normalize((arr.T @ w) / (w.sum() + 1e-12))
        centroids.append(c)

print(f"[4] Centroids: {len(centroids)} sentences  ({T()-t0:.1f}s)")

# 5. Save
t0 = T()
centroids_arr = np.stack(centroids).astype(np.float32)  # (N, D)
out_path = "sentence_centroids.npz"

# Save idf as dict for backward compat
idf_for_save = {}
for w, weight in idf.items():
    idf_for_save[w] = float(weight)

np.savez_compressed(
    out_path,
    centroids=centroids_arr,
    idf=np.array([idf_for_save], dtype=object),
)
print(f"[5] Saved {out_path} ({centroids_arr.nbytes / 1e6:.0f} MB raw)  ({T()-t0:.1f}s)")

# 6. Verify
t0 = T()
verify = np.load(out_path, allow_pickle=True)
v_centroids = verify["centroids"]
v_idf = dict(verify["idf"].item())
verif_words = list(v_idf.keys())[:5]
print(f"[6] Verify: {v_centroids.shape[0]} centroids × {v_centroids.shape[1]} dim, "
      f"{len(v_idf)} IDF words  ({T()-t0:.1f}s)")
print(f"    Sample IDF: {[(w, round(v_idf[w], 2)) for w in verif_words]}")

# 7. Load test
t0 = T()
from celn_v3.knowledge_channel import KnowledgeChannel
kc = KnowledgeChannel(out_path)
kv = kc.query_and_read(["fotossintese", "planta"], vectors, w2i)
scores = kc.score_candidates(["fotossintese", "planta", "carro", "o"], kv, vectors, w2i)
print(f"[7] Load+query: {T()-t0:.3f}s")
print(f"    Scores for [fotossintese, planta, carro, o]: "
      f"[{', '.join(f'{s:.3f}' for s in scores)}]")

print("\nDone.")
