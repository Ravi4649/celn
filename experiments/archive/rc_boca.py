#!/usr/bin/env python3
"""
Reservoir Computing (ESN) as 'boca' for CELN.

This script builds a small ESN, trains output weights with ridge regression
on corpus_final.txt (word-level tokens), and generates text conditioned on
an injected thematic vector (VSA M-state) by seeding the input and adding
thematic perturbation to the reservoir state.

Fallback: uses pure-Python esn (celn_rc.esn_py) if Cython extension not compiled.
"""

import os
import sys
import time
import numpy as np
import random
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from celn_rc import esn_core as esn_impl
    has_cython = True
except Exception:
    from celn_rc import esn_py as esn_impl
    has_cython = False

from experiments.test_trigram_backoff import tokenize, load_corpus
from celn.core import encode_sequence


def load_semantic_vectors(path=None):
    # Load precomputed semantic vectors (default: celn_full_vectors.npz)
    if path is None:
        candidates = [
            'celn_full_vectors.npz',
            'celn_vectors_3007.npz',
            'celn_native_vectors.npz',
        ]
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
    if path is None or not os.path.exists(path):
        raise FileNotFoundError('Semantic vectors NPZ not found')
    z = np.load(path)
    vectors = z['vectors'].astype(np.float32)
    vocab = [w for w in z['vocab']]
    w2idx = {w: i for i, w in enumerate(vocab)}
    return vectors, vocab, w2idx


def build_vocab(sentences):
    counts = Counter()
    for s in sentences:
        counts.update(s)
    vocab = sorted(counts.keys())
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}
    return vocab, w2i, i2w


def build_training_stream(sentences, w2i):
    # flatten sentences into token ids and record sentence index for each token
    stream = []
    stream_sent_idx = []
    for si, s in enumerate(sentences):
        for t in s:
            if t in w2i:
                stream.append(w2i[t])
                stream_sent_idx.append(si)
    return stream, stream_sent_idx


def make_reservoir(N, sr=0.9, density=0.01, seed=42):
    rng = np.random.RandomState(seed)
    W = rng.randn(N, N).astype(np.float32)
    mask = rng.rand(N, N) < density
    W *= mask
    # scale to spectral radius approx
    try:
        vals = np.linalg.eigvals(W)
        rho = max(abs(vals))
        if rho > 1e-12:
            W = W * (sr / rho)
    except Exception:
        pass
    return W


def train_readout(stream_ids, stream_sent_idx, sentence_Ms, N, V, Win_scale=0.2, leak=0.3, m_proj_dim=None, seed=42):
    rng = np.random.RandomState(seed)
    W = make_reservoir(N, seed=seed)
    Win = (rng.rand(N, V).astype(np.float32) * 2 - 1) * Win_scale
    bias = (rng.rand(N).astype(np.float32) * 2 - 1) * 0.01
    # projection dimension: default to reservoir size
    if m_proj_dim is None:
        m_proj_dim = N

    D = sentence_Ms.shape[1]
    # random projection matrix: D -> m_proj_dim
    P = rng.randn(D, m_proj_dim).astype(np.float32) / np.sqrt(D)

    X = []
    Y = []
    x = np.zeros(N, dtype=np.float32)
    for t in range(len(stream_ids) - 1):
        u = stream_ids[t]
        s_idx = stream_sent_idx[t]
        # step reservoir
        x = esn_impl.reservoir_step(x, W, Win, bias, u, leak=leak)
        # project sentence M into low-dim
        mvec = sentence_Ms[s_idx]
        mproj = mvec.astype(np.float32).dot(P)
        # append [x; mproj; 1]
        feat = np.concatenate([x, mproj, np.array([1.0], dtype=np.float32)])
        X.append(feat)
        y = stream_ids[t + 1]
        yvec = np.zeros(V, dtype=np.float32)
        yvec[y] = 1.0
        Y.append(yvec)

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    # ridge regression: Wout = (X^T X + reg I)^-1 X^T Y
    reg = 1e-6
    XtX = X.T.dot(X)
    n_feat = XtX.shape[0]
    A = XtX + reg * np.eye(n_feat, dtype=np.float64)
    XtY = X.T.dot(Y)
    Wout = np.linalg.solve(A, XtY).astype(np.float32)

    model = {'W': W, 'Win': Win, 'bias': bias, 'Wout': Wout, 'N': N, 'V': V, 'leak': leak, 'P': P, 'm_proj_dim': m_proj_dim}
    return model


def generate_from_model(model, w2i, i2w, prefix_words, thematic_vec=None, max_len=50, temp=1.0, seed=None):
    rng = np.random.RandomState(seed)
    N = model['N']
    W = model['W']
    Win = model['Win']
    bias = model['bias']
    Wout = model['Wout']
    leak = model['leak']
    P = model.get('P', None)
    m_proj_dim = model.get('m_proj_dim', 0)

    x = np.zeros(N, dtype=np.float32)
    out = []
    # feed prefix words to initialize state
    for w in prefix_words:
        if w in w2i:
            u = w2i[w]
        else:
            u = 0
        x = esn_impl.reservoir_step(x, W, Win, bias, u, leak=leak)

    # optionally include thematic vector projection in the readout features
    if thematic_vec is not None and P is not None and P.shape[0] == thematic_vec.shape[0]:
        mproj = thematic_vec.astype(np.float32).dot(P)
    elif thematic_vec is not None and P is not None and P.shape[0] != thematic_vec.shape[0]:
        # fallback: random project to m_proj_dim
        rngp = np.random.RandomState(123)
        P2 = rngp.randn(thematic_vec.shape[0], m_proj_dim).astype(np.float32) / np.sqrt(thematic_vec.shape[0])
        mproj = thematic_vec.astype(np.float32).dot(P2)
    else:
        mproj = None

    for i in range(max_len):
        if mproj is not None:
            svec = np.concatenate([x, mproj, np.array([1.0], dtype=np.float32)])
        else:
            svec = np.concatenate([x, np.array([1.0], dtype=np.float32)])
        logits = svec.dot(Wout)
        # softmax with temp
        l = logits.astype(np.float64) / max(1e-12, temp)
        l = l - l.max()
        probs = np.exp(l)
        probs = probs / probs.sum()
        idx = rng.choice(len(probs), p=probs)
        out.append(i2w[idx])
        # feed chosen token into reservoir
        x = esn_impl.reservoir_step(x, W, Win, bias, idx, leak=leak)

    return ' '.join(out)


def main():
    corpus_path = 'corpus_final.txt'
    if not os.path.exists(corpus_path):
        corpus_path = '/home/ravizin/celn-v3/corpus_final.txt'

    sents = load_corpus(corpus_path)
    # load semantic vectors to compute M-states per sentence
    sem_vectors, sem_vocab, sem_w2idx = load_semantic_vectors()
    # ensure vocab overlap mapping
    # build per-sentence semantic vectors list for encode_sequence
    # convert token strings to semantic vectors if present else zeros
    word_vecs_map = {w: sem_vectors[i] for i, w in enumerate(sem_vocab)}

    # build sentence word vectors and compute M for each sentence
    sentence_word_vecs = []
    for toks in sents:
        vecs = []
        for t in toks:
            if t in word_vecs_map:
                vecs.append(word_vecs_map[t])
        if not vecs:
            # fallback zero vector
            vecs = [np.zeros(sem_vectors.shape[1], dtype=np.float32)]
        sentence_word_vecs.append(vecs)

    # compute M-state per sentence using projective_resonance scan
    sentence_Ms = []
    from celn.core import projective_resonance
    for vecs in sentence_word_vecs:
        if not vecs:
            sentence_Ms.append(np.zeros(sem_vectors.shape[1], dtype=np.float32))
            continue
        st = vecs[0].astype(np.float32).copy()
        for wv in vecs[1:]:
            st = projective_resonance(st, wv.astype(np.float32), gamma=1.0, bilateral=True)
        sentence_Ms.append(st.astype(np.float32))
    sentence_Ms = np.asarray(sentence_Ms, dtype=np.float32)

    vocab, w2i, i2w = build_vocab(sents)
    stream, stream_sent_idx = build_training_stream(sents, w2i)

    print('Vocab size:', len(vocab))
    print('Training stream length:', len(stream))

    # Train a small ESN for speed
    # Train reservoir readout with conditional M projections
    model = train_readout(stream, stream_sent_idx, sentence_Ms, N=200, V=len(vocab), Win_scale=0.2, leak=0.3, m_proj_dim=200, seed=42)

    # Test generation conditioned on simple 'thematic vector' approximations
    questions = [
        'o cobre conduz eletricidade?'
        , 'quem comeu o rato?'
        , 'cobre está para metal como onça está para o quê?'
    ]

    # build naive thematic vectors as centroid of topic words from question
    for q in questions:
        toks = tokenize(q)
        # pick words that are in vocab
        kws = [t for t in toks if t in w2i]
        thematic = None
        if kws:
            # mean one-hot of keywords projected into semantic-like random vector
            dim = 300
            rng = np.random.RandomState(123)
            sem = rng.randn(dim).astype(np.float32)
            thematic = sem / (np.linalg.norm(sem) + 1e-12)
        gen = generate_from_model(model, w2i, i2w, kws[:3], thematic_vec=thematic, max_len=30, temp=0.8, seed=42)
        print('\nQ:', q)
        print('Generated:', gen)


if __name__ == '__main__':
    main()
