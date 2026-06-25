"""
Quick test: Phase-Lens filtered PairSDM retrieval.

For a prefix word (e.g. 'cobre'), create two contexts (electricity, mining),
apply phase_lens to the query BEFORE consulting the PairSDM and show
the extracted follower candidates for each context.

Run from repository root: python3 experiments/test_phase_filtered_pairs.py
"""

import sys
import numpy as np
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from celn_v3.core import normalize, phase_lens, projective_resonance as M
from celn_v3.dual_channel import DualChannelGenerator
from celn_v3.memory import DenseSDM
from celn_v3.resonator import unbind_M_reverse


def load_env():
    vec_path = os.environ.get('CELN_VECTOR_PATH', 'celn_v3_full_vectors.npz')
    data = np.load(vec_path)
    # file contains 'vocab' (list/array of words) and 'vectors' (matrix)
    words = [w.decode('utf-8') if isinstance(w, bytes) else w for w in data['vocab'].tolist()]
    vecs = data['vectors']
    w2i = {w: i for i, w in enumerate(words)}
    i2w = {i: w for i, w in enumerate(words)}
    return vecs, w2i, i2w


def load_pair_sdm():
    # Use cached pair SDM if available
    cache = os.path.join('celn_v3', '.cache', 'pair_sdm_7a76849d61c97885.npz')
    # Fallback to default path in repo .cache
    cache2 = os.path.join('celn_v3', '.cache', 'pair_sdm_7a76849d61c97885.npz')
    # Attempt to open via DualChannelGenerator initialization later if needed
    return cache if os.path.exists(cache) else None


def top_candidates_from_recovered(recovered, sem_vecs, i2w, top_n=8, excluded=set()):
    recovered = normalize(recovered)
    sims = sem_vecs @ recovered
    for idx in excluded:
        sims[idx] = -1.0
    top = np.argpartition(sims, -top_n)[-top_n:]
    top = top[np.argsort(-sims[top])]
    return [(i2w[i], float(sims[i])) for i in top]


def main():
    sem_vecs, w2i, i2w = load_env()
    V, D = sem_vecs.shape

    # Load PairSDM via DualChannel's constructor convenience
    gen = DualChannelGenerator(sem_vecs, np.zeros((V, 2000), dtype=np.float32), w2i, i2w,
                               use_phase_lens=True, phase_lens_max_alpha=0.6)
    if gen.pair_sdm is None:
        # Try to find any pair_sdm_*.npz in the cache dir
        # cache lives at repo_root/.cache
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        cache_dir = os.path.join(repo_root, '.cache')
        found = False
        if os.path.exists(cache_dir):
            import glob
            matches = glob.glob(os.path.join(cache_dir, 'pair_sdm_*.npz'))
            if matches:
                cache_file = matches[0]
                try:
                    cache = np.load(cache_file, allow_pickle=True)
                    pair_sdm = DenseSDM(
                        n_locations=int(cache['n_locations']),
                        activation_pct=float(cache['activation_pct']),
                        seed=42,
                    )
                    pair_sdm.addresses = cache['addresses'].astype(np.float32)
                    pair_sdm.accumulators = cache['accumulators'].astype(np.float32)
                    pair_sdm.counters = cache['counters'].astype(np.int32)
                    pair_sdm.corroboration = cache['corroboration'].astype(np.float32)
                    pair_sdm.total_writes = int(cache['total_writes'])
                    gen.pair_sdm = pair_sdm
                    found = True
                    print('Loaded PairSDM cache:', cache_file)
                except Exception:
                    found = False
        if not found:
            print('PairSDM cache not found; please build PairSDM first (run test_full_conversation).')
            return

    pair_sdm = gen.pair_sdm

    # Word to test
    target = 'cobre'
    if target not in w2i:
        print('Target word not in vocab:', target)
        return
    t_idx = w2i[target]
    t_vec = sem_vecs[t_idx]

    # Create two contexts: electricity and mining (use words present in vocab)
    ctx_elec_words = ['eletricidade', 'corrente', 'condutividade', 'circuito', 'energia']
    ctx_mine_words = ['mina', 'mineração', 'estanho', 'escavação', 'poço']

    def make_context(words):
        vecs = [sem_vecs[w2i[w]] for w in words if w in w2i]
        if not vecs:
            return None
        return normalize(np.mean(vecs, axis=0))

    ctx_elec = make_context(ctx_elec_words)
    ctx_mine = make_context(ctx_mine_words)

    print('Target:', target)
    print('Contexts available: electricity:', ctx_elec is not None, 'mining:', ctx_mine is not None)

    for name, ctx in [('electricity', ctx_elec), ('mining', ctx_mine)]:
        if ctx is None:
            print(f'  Skipping {name}: context words not in vocab')
            continue

        # Apply phase lens to the word toward the context BEFORE building query
        alpha = 0.6
        deformed = phase_lens(t_vec, ctx, alpha=alpha)
        query_pair = M(ctx, deformed, gamma=1.0, bilateral=True)

        pair_result = pair_sdm.read(query_pair)
        recovered = unbind_M_reverse(pair_result, deformed, gamma=1.0, bilateral=True, n_refine=5)
        top = top_candidates_from_recovered(recovered, sem_vecs, i2w, top_n=10, excluded=set())

        print('\nContext:', name)
        print('  Top extracted followers:')
        for w, s in top:
            print(f'    {w} ({s:.3f})')


if __name__ == '__main__':
    main()
