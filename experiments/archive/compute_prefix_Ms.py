#!/usr/bin/env python3
"""
Compute projected M (VSA) features for every token position in corpus_final.txt.

Outputs saved to experiments/no_prop_data.npz containing:
 - X: float32 array (N_examples, Xf) projected M features for prefix before token
 - Y: int32 array (N_examples,) token indices in dataset vocab
 - w2i mapping saved as a small json next to the npz
 - P_embed and P_M projection matrices (saved inside npz)
 - vocab list saved inside npz

This script uses celn.core.projective_resonance and celn_full_vectors.npz
for token semantic vectors. Tokens not present in semantic vocab use zero vector.
"""

import os
import json
import numpy as np
try:
    # when run as part of package
    from experiments.test_trigram_backoff import tokenize, load_corpus
except Exception:
    # fallback: load module by path
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "test_trigram_backoff",
        os.path.join(os.path.dirname(__file__), "test_trigram_backoff.py"),
    )
    tt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tt)
    tokenize = tt.tokenize
    load_corpus = tt.load_corpus

try:
    from celn.core import projective_resonance
except Exception:
    # local import fallback
    import importlib.util
    spec_c = importlib.util.spec_from_file_location(
        "celn.core",
        os.path.join(os.path.dirname(__file__), '..', 'celn', 'core.py'),
    )
    cmod = importlib.util.module_from_spec(spec_c)
    spec_c.loader.exec_module(cmod)
    projective_resonance = cmod.projective_resonance


def load_semantic_vectors(path='data/celn_full_vectors.npz'):
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), '..', path)
    z = np.load(path)
    vectors = z['vectors'].astype(np.float32)
    vocab = [str(w) for w in z['vocab']]
    w2idx = {w: i for i, w in enumerate(vocab)}
    return vectors, vocab, w2idx


def build_dataset(corpus_path='corpus_final.txt', Xf=128, d=512, seed=42):
    print('Loading corpus...')
    if not os.path.exists(corpus_path):
        corpus_path = os.path.join(os.path.dirname(__file__), '..', corpus_path)
    sents = load_corpus(corpus_path)

    print('Building dataset vocab...')
    # build dataset vocabulary from corpus tokens
    from collections import Counter
    counts = Counter()
    for s in sents:
        counts.update(s)
    vocab = sorted(counts.keys())
    w2i = {w: i for i, w in enumerate(vocab)}

    print('Loading semantic vectors (SVD)...')
    sem_vectors, sem_vocab, sem_w2idx = load_semantic_vectors()
    D = sem_vectors.shape[1]

    rng = np.random.RandomState(seed)
    print('Creating random projection matrices P_M and P_embed')
    P_M = (rng.randn(D, Xf).astype(np.float32) / np.sqrt(D)).astype(np.float32)
    P_embed = (rng.randn(D, d).astype(np.float32) / np.sqrt(D)).astype(np.float32)

    # build WEmbed for dataset vocab by projecting sem_vectors when available
    V = len(vocab)
    WEmbed = np.zeros((V, d), dtype=np.float32)
    for token, idx in w2i.items():
        if token in sem_w2idx:
            WEmbed[idx] = sem_vectors[sem_w2idx[token]].dot(P_embed)
        else:
            # random fallback
            WEmbed[idx] = rng.randn(d).astype(np.float32) / np.sqrt(d)

    # iterate sentences and compute prefix M features
    X_list = []
    Y_list = []
    total_tokens = 0
    print('Scanning sentences and computing projected prefix M features...')
    for si, toks in enumerate(sents):
        # initialize state as zero vector
        state = np.zeros(D, dtype=np.float32)
        # prefix before first token is zero
        for j, tok in enumerate(toks):
            # project current prefix state
            mproj = state.dot(P_M)
            X_list.append(mproj)
            # label is current token
            if tok in w2i:
                Y_list.append(w2i[tok])
            else:
                # skip token if not in vocab (shouldn't happen)
                Y_list.append(0)
            total_tokens += 1

            # update state by composing with token vector
            if tok in sem_w2idx:
                wv = sem_vectors[sem_w2idx[tok]]
            else:
                wv = np.zeros(D, dtype=np.float32)
            # projective_resonance normalizes output by default
            try:
                state = projective_resonance(state, wv, gamma=1.0, bilateral=True)
            except Exception:
                # fallback to simple addition if projective fails
                state = state + wv

        if (si + 1) % 100 == 0:
            print(f'  processed {si+1}/{len(sents)} sentences, tokens so far: {total_tokens}')

    X = np.asarray(X_list, dtype=np.float32)
    Y = np.asarray(Y_list, dtype=np.int32)

    print('Saving dataset to experiments/no_prop_data.npz')
    out_path = os.path.join(os.path.dirname(__file__), 'no_prop_data.npz')
    np.savez_compressed(out_path, X=X, Y=Y, WEmbed=WEmbed, P_M=P_M, P_embed=P_embed, vocab=np.array(vocab))

    # also save w2i mapping
    with open(os.path.join(os.path.dirname(__file__), 'no_prop_w2i.json'), 'w', encoding='utf-8') as f:
        json.dump(w2i, f, ensure_ascii=False)

    print('Done. Examples:', X.shape[0], 'Xf=', X.shape[1], 'Vocab size=', V)
    return out_path


if __name__ == '__main__':
    build_dataset(corpus_path='corpus_final.txt', Xf=128, d=512, seed=42)
