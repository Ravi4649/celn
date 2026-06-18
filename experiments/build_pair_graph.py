#!/usr/bin/env python3
"""
Build PairGraph cache for GPVE's 6th channel (Trajectory Coherence).
=====================================================================
Roda UMA vez, offline. Extrai top-5 seguidores por palavra do corpus.
Produz pair_graph.npz com:
  - sources: (N,) int32 — word indices with known followers
  - followers: (N, 5) int32 — top-5 follower indices per source
  - n_followers: (N,) int32 — actual count per source (some may have <5)

Uso:
  python experiments/build_pair_graph.py [corpus_path] [output_path]
    corpus_path: path to corpus file (default: corpus_final.txt)
    output_path: path for output .npz (default: pair_graph.npz)

  Exemplo com corpus expandido:
  python experiments/build_pair_graph.py corpus_expanded.txt pair_graph_expanded.npz
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import defaultdict, Counter

from celn_v3.train import load_corpus, tokenize
from celn_v3.port_adapter import load_word_vectors

T = time.time

# Parse args
corpus_path = sys.argv[1] if len(sys.argv) > 1 else "corpus_final.txt"
output_path = sys.argv[2] if len(sys.argv) > 2 else "pair_graph.npz"

tag = os.path.splitext(os.path.basename(output_path))[0]
print("=" * 60)
print(f"Build PairGraph — {output_path}")
print("=" * 60)

# 1. Load vectors
t0 = T()
vectors, w2i = load_word_vectors("celn_v3_full_vectors.npz")
V, D = vectors.shape
print(f"[1] Vectors: {V} × {D}  ({T()-t0:.1f}s)")

# 2. Load corpus
t0 = T()
sentences = load_corpus(corpus_path, min_len=2)
print(f"[2] Corpus: {corpus_path} — {len(sentences)} sentences  ({T()-t0:.1f}s)")

# 3. Collect transition pairs
t0 = T()
pair_counts: dict[int, Counter] = defaultdict(Counter)
for toks in sentences:
    for i in range(len(toks) - 1):
        w1, w2 = toks[i], toks[i + 1]
        i1, i2 = w2i.get(w1), w2i.get(w2)
        if i1 is not None and i2 is not None:
            pair_counts[i1][i2] += 1
print(f"[3] Pairs: {len(pair_counts)} source words  ({T()-t0:.1f}s)")

# 4. Top-K per source
t0 = T()
K = 5
sources_list: list[int] = []
followers_list: list[np.ndarray] = []
n_followers_list: list[int] = []

for src_idx in sorted(pair_counts.keys()):
    cnt = pair_counts[src_idx]
    top = cnt.most_common(K)
    arr = np.full(K, -1, dtype=np.int32)
    for j, (f_idx, _) in enumerate(top):
        arr[j] = f_idx
    sources_list.append(src_idx)
    followers_list.append(arr)
    n_followers_list.append(len(top))

sources_arr = np.array(sources_list, dtype=np.int32)
followers_arr = np.stack(followers_list).astype(np.int32)
n_followers_arr = np.array(n_followers_list, dtype=np.int32)

print(f"[4] Top-5: {len(sources_list)} sources  ({T()-t0:.1f}s)")
print(f"    Avg followers: {np.mean(n_followers_arr):.1f}")

# 5. Save
t0 = T()
np.savez_compressed(
    output_path,
    sources=sources_arr,
    followers=followers_arr,
    n_followers=n_followers_arr,
)
print(f"[5] Saved {output_path}  ({T()-t0:.1f}s)")

# 6. Verify
t0 = T()
verify = np.load(output_path)
print(f"[6] Verify: {len(verify['sources'])} sources, "
      f"followers shape {verify['followers'].shape}  ({T()-t0:.1f}s)")

# 7. Lookahead test
t0 = T()
from celn_v3.pair_graph import PairGraph
from celn_v3.core import normalize
pg = PairGraph("pair_graph.npz")
# Test: fotossintese → should have followers
photo_idx = w2i.get("fotossintese")
print(f"[7] Load+lookahead test: {T()-t0:.3f}s")
if photo_idx is not None:
    followers = pg.get_followers(photo_idx)
    i2w = {i: w for w, i in w2i.items()}
    print(f"    fotossintese followers: {[i2w.get(f, '?') for f in followers]}")
    m_intent = normalize(vectors[photo_idx])
    coh = pg.lookahead_coherence(photo_idx, vectors, m_intent, depth=2, width=2)
    print(f"    lookahead_coherence: {coh:.4f}")

print("\nDone.")
