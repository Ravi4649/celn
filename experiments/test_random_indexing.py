"""
Test: Random Indexing — Concentrating Semantic Similarity
=========================================================
Hypothesis: Random Indexing (Jones & Mewhort, 2007) pulls vectors of
contextually co-occurring words closer, amplifying the signal difference
between related and unrelated pairs — without backprop, without collapsing.

Test:
  1. Load existing SVD vectors (3007 words, 10k dims)
  2. Apply Random Indexing: for each sentence, each word pulls its
     context-window neighbors with small learning rate (0.01)
  3. Run 10 passes over the corpus
  4. Measure cosine similarity BEFORE vs AFTER for:
     - Related pairs: cobre↔conduz, gato↔cachorro, Brasil↔América, água↔líquido
     - Unrelated pairs: cobre↔gato, Brasil↔líquido, animal↔planeta
  5. Check if the gap (related - unrelated) WIDENED
"""
import sys, numpy as np, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/ravizin/celn-v3')
from celn_v3.train import load_corpus, tokenize

def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(a @ b)

# ------------------------------------------------------------------
# 1. LOAD
# ------------------------------------------------------------------
print("=" * 70)
print("RANDOM INDEXING — SIMILARITY CONCENTRATION TEST")
print("=" * 70)

data = np.load('/home/ravizin/celn-v3/celn_v3_full_vectors.npz', allow_pickle=True)
vectors = data['vectors'].astype(np.float32)
vocab = data['vocab']
w2i = {w: i for i, w in enumerate(vocab)}
V, D = vectors.shape
print(f"\nLoaded {V} words × {D} dimensions")

sentences = load_corpus('/home/ravizin/celn-v3/corpus_final.txt', min_len=2)
print(f"Corpus: {len(sentences)} sentences")

# ------------------------------------------------------------------
# 2. PAIRS TO TRACK
# ------------------------------------------------------------------
related = [
    ('cobre', 'conduz'),
    ('gato', 'cachorro'),
    ('brasil', 'américa'),
    ('água', 'líquido'),
]
unrelated = [
    ('cobre', 'gato'),
    ('brasil', 'líquido'),
    ('animal', 'planeta'),
]

# Filter to existing words
def filter_pairs(pairs):
    return [(a, b) for a, b in pairs if a in w2i and b in w2i]

related = filter_pairs(related)
unrelated = filter_pairs(unrelated)
print(f"Tracking {len(related)} related, {len(unrelated)} unrelated pairs")

# ------------------------------------------------------------------
# 3. BASELINE SIMILARITIES
# ------------------------------------------------------------------
def measure(pairs, vecs):
    return [cosine(vecs[w2i[a]], vecs[w2i[b]]) for a, b in pairs]

base_rel = measure(related, vectors)
base_unr = measure(unrelated, vectors)

print("\n" + "-" * 70)
print("BASELINE (SVD vectors)")
print("-" * 70)
for (a, b), s in zip(related, base_rel):
    print(f"  {a:<12s} ↔ {b:<12s}  {s:.4f}")
print(f"  Mean related:    {np.mean(base_rel):.4f}")
for (a, b), s in zip(unrelated, base_unr):
    print(f"  {a:<12s} ↔ {b:<12s}  {s:.4f}")
print(f"  Mean unrelated:  {np.mean(base_unr):.4f}")
print(f"  Gap (rel-unr):   {np.mean(base_rel) - np.mean(base_unr):.4f}")

# ------------------------------------------------------------------
# 4. RANDOM INDEXING
# ------------------------------------------------------------------
print("\n" + "-" * 70)
print("RANDOM INDEXING — 10 passes, window=3, lr=0.01")
print("-" * 70)

ri_vectors = vectors.copy()
lr = 0.01
window = 3
n_passes = 10

for p in range(n_passes):
    np.random.shuffle(sentences)  # randomize order each pass
    for sent in sentences:
        tokens = [t for t in sent if t in w2i]
        if len(tokens) < 2:
            continue
        for i, w in enumerate(tokens):
            wi = w2i[w]
            # context window: words within ±window, excluding self
            ctx = []
            for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
                if j != i:
                    ctx.append(w2i[tokens[j]])
            if not ctx:
                continue
            # RI update: word vector pulls toward average of context vectors
            ctx_mean = np.mean(ri_vectors[ctx], axis=0)
            ri_vectors[wi] += lr * (ctx_mean - ri_vectors[wi])
    # Normalize every pass to prevent drift
    norms = np.linalg.norm(ri_vectors, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    ri_vectors = ri_vectors / norms
    print(f"  Pass {p+1:2d} done")

# ------------------------------------------------------------------
# 5. POST-RI SIMILARITIES
# ------------------------------------------------------------------
ri_rel = measure(related, ri_vectors)
ri_unr = measure(unrelated, ri_vectors)

print("\n" + "-" * 70)
print("AFTER RANDOM INDEXING")
print("-" * 70)
for (a, b), s in zip(related, ri_rel):
    print(f"  {a:<12s} ↔ {b:<12s}  {s:.4f}")
print(f"  Mean related:    {np.mean(ri_rel):.4f}")
for (a, b), s in zip(unrelated, ri_unr):
    print(f"  {a:<12s} ↔ {b:<12s}  {s:.4f}")
print(f"  Mean unrelated:  {np.mean(ri_unr):.4f}")
print(f"  Gap (rel-unr):   {np.mean(ri_rel) - np.mean(ri_unr):.4f}")

# ------------------------------------------------------------------
# 6. CHANGE ANALYSIS
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("CHANGE ANALYSIS")
print("=" * 70)
rel_delta = np.mean(ri_rel) - np.mean(base_rel)
unr_delta = np.mean(ri_unr) - np.mean(base_unr)
gap_before = np.mean(base_rel) - np.mean(base_unr)
gap_after = np.mean(ri_rel) - np.mean(ri_unr)

print(f"Related   change: {rel_delta:+.4f}")
print(f"Unrelated change: {unr_delta:+.4f}")
print(f"Gap BEFORE: {gap_before:.4f}")
print(f"Gap AFTER:  {gap_after:.4f}")
print(f"Gap WIDENED: {gap_after > gap_before} (Δ {gap_after - gap_before:+.4f})")

# Per-pair detail
print("\nPer-pair delta:")
for (a, b), b4, af in zip(related, base_rel, ri_rel):
    print(f"  {a:<12s} ↔ {b:<12s}  {b4:.4f} → {af:.4f}  ({af-b4:+.4f})")
for (a, b), b4, af in zip(unrelated, base_unr, ri_unr):
    print(f"  {a:<12s} ↔ {b:<12s}  {b4:.4f} → {af:.4f}  ({af-b4:+.4f})")
