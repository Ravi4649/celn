#!/usr/bin/env python3
"""
Minimal Echo State Network (Reservoir Computing) test for CELN corpus.

Objective: train an ESN (no backprop) to predict next word (word-level)
and generate text autoregressively from prefixes. Compare qualitatively
with VSA baseline.

Principles: no backprop, ridge regression readout, CPU-only, simple code.
"""

import re
import time
import os
import numpy as np
from collections import Counter
import math


def tokenize(text):
    # simple unicode words tokenizer
    toks = re.findall(r"\w+|[^\\s\w]", text.lower(), flags=re.UNICODE)
    return toks


def load_corpus(path):
    sents = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = tokenize(line)
            sents.append(toks)
    return sents


def build_vocab(sentences, max_vocab=None, min_count=1):
    ctr = Counter()
    for s in sentences:
        ctr.update(s)
    if min_count > 1:
        items = [(w, c) for w, c in ctr.items() if c >= min_count]
        items.sort(key=lambda x: -x[1])
    else:
        items = ctr.most_common()
    if max_vocab is not None:
        items = items[:max_vocab]
    vocab = [w for w, _ in items]
    w2i = {w: i for i, w in enumerate(vocab)}
    w2i['<UNK>'] = len(w2i)
    i2w = {i: w for w, i in w2i.items()}
    return w2i, i2w


def build_training_sequences(sentences, w2i):
    seqs = []
    unk = w2i.get('<UNK>')
    for s in sentences:
        idxs = [w2i.get(w, unk) for w in s]
        if len(idxs) >= 2:
            seqs.append(idxs)
    return seqs


def make_reservoir(N, density=0.01, spectral_radius=0.9, seed=42):
    rng = np.random.RandomState(seed)
    W = rng.randn(N, N).astype(np.float32)
    # sparsify
    mask = rng.rand(N, N) < density
    W *= mask

    # scale to spectral radius via power iteration
    def power_it(A, n_it=50):
        v = rng.randn(A.shape[0]).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        for _ in range(n_it):
            v = A.dot(v)
            norm = np.linalg.norm(v)
            if norm < 1e-12:
                break
            v /= norm
        # Rayleigh quotient
        Av = A.dot(v)
        return float(np.dot(v, Av))

    approx_rho = abs(power_it(W, n_it=50))
    if approx_rho <= 0:
        approx_rho = 1.0
    W = W * (spectral_radius / approx_rho)
    return W


def softmax(x, temp=1.0):
    x = x.astype(np.float64)
    x = x / max(1e-12, temp)
    x = x - x.max()
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


def train_esn(seqs, V, N=2000, input_scale=0.1, leak=0.3, density=0.01, sr=0.9, reg=1e-6, seed=123):
    rng = np.random.RandomState(seed)
    print('Building reservoir...')
    W = make_reservoir(N, density=density, spectral_radius=sr, seed=seed)
    # input weight: columns correspond to token indices
    Win = (rng.rand(N, V).astype(np.float32) * 2 - 1) * input_scale
    bias = (rng.rand(N).astype(np.float32) * 2 - 1) * 0.01

    # collect states X and targets Y
    # estimate total timesteps
    T_est = sum(len(s) - 1 for s in seqs)
    print(f'Estimated training steps: {T_est}')

    n_features = N + 1  # include bias term in readout
    X = np.zeros((T_est, n_features), dtype=np.float32)
    Y = np.zeros((T_est, V), dtype=np.float32)

    row = 0
    x = np.zeros(N, dtype=np.float32)
    for s in seqs:
        x[:] = 0.0
        for t in range(len(s) - 1):
            u = s[t]
            target = s[t + 1]
            # reservoir update
            # input is one-hot select Win[:, u]
            pre = W.dot(x) + Win[:, u] + bias
            x = (1 - leak) * x + leak * np.tanh(pre)
            X[row, :N] = x
            X[row, N] = 1.0
            Y[row, target] = 1.0
            row += 1
    # truncate in case overestimated
    X = X[:row]
    Y = Y[:row]

    print('Computing ridge regression (closed-form)...')
    # cast to float64 for numeric stability in solve
    X64 = X.astype(np.float64)
    Y64 = Y.astype(np.float64)
    XtX = X64.T.dot(X64)
    XtY = X64.T.dot(Y64)
    regI = reg * np.eye(n_features, dtype=np.float64)
    A = XtX + regI
    # solve A Wout = XtY  => Wout shape (n_features, V)
    Wout = np.linalg.solve(A, XtY)

    # convert weights back to float32 for generation
    Wout = Wout.astype(np.float32)

    model = {
        'W': W,
        'Win': Win,
        'bias': bias,
        'Wout': Wout,
        'N': N,
        'V': V,
        'leak': leak,
        'w2i': None,
        'i2w': None,
    }
    return model


def generate(model, w2i, i2w, prefix, max_len=30, temp=1.0, seed=0):
    rng = np.random.RandomState(seed)
    N = model['N']
    W = model['W']
    Win = model['Win']
    bias = model['bias']
    Wout = model['Wout']
    leak = model['leak']
    V = model['V']

    x = np.zeros(N, dtype=np.float32)
    toks = tokenize(prefix)
    words = []
    # feed prefix
    for w in toks:
        idx = w2i.get(w, w2i.get('<UNK>'))
        pre = W.dot(x) + Win[:, idx] + bias
        x = (1 - leak) * x + leak * np.tanh(pre)
        words.append(w)

    # generate
    for _ in range(max_len):
        s = np.empty(N + 1, dtype=np.float32)
        s[:N] = x
        s[N] = 1.0
        logits = s.dot(Wout)  # shape (V,)
        probs = softmax(logits, temp=temp)
        # sample
        idx = rng.choice(V, p=probs)
        w = i2w.get(idx, '<UNK>')
        words.append(w)
        # update reservoir with chosen token as input
        pre = W.dot(x) + Win[:, idx] + bias
        x = (1 - leak) * x + leak * np.tanh(pre)

    return ' '.join(words)


def main():
    corpus_path = 'corpus_final.txt'
    if not os.path.exists(corpus_path):
        corpus_path = '/home/ravizin/celn-v3/corpus_final.txt'
    sents = load_corpus(corpus_path)
    print(f'Loaded {len(sents)} sentences')

    # build vocab (limit vocabulary to top 3000)
    w2i, i2w = build_vocab(sents, max_vocab=3000)
    V = len(w2i)
    print('Vocab size (with UNK):', V)

    seqs = build_training_sequences(sents, w2i)
    print('Training sequences:', len(seqs))

    # Train ESN
    N = 2000
    model = train_esn(seqs, V, N=N, input_scale=0.1, leak=0.3, density=0.01, sr=0.9, reg=1e-6, seed=42)
    model['w2i'] = w2i
    model['i2w'] = {i: w for w, i in w2i.items()}

    prefixes = ['o cobre', 'a eletricidade', 'o brasil', 'o gato']
    func_words = set([
        'o','a','os','as','um','uma','uns','umas',
        'de','do','da','dos','das','em','no','na','nos','nas',
        'e','ou','mas','que','se','nem','pois','é','foi','era',
        'são','está','ser','não','sim','como','quando','onde',
        'porque','para','com','por','pelo','pela','pelas','sem','sob','sobre',
    ])

    print('\nGenerated texts:')
    all_func_counts = []
    for p in prefixes:
        gen = generate(model, w2i, model['i2w'], p, max_len=30, temp=1.0, seed=123)
        toks = tokenize(gen)
        func_frac = sum(1 for t in toks if t in func_words) / max(1, len(toks)) * 100.0
        all_func_counts.append(func_frac)
        # domain keywords
        domain = p.split()[-1]
        domain_count = sum(1 for t in toks if t == domain)
        print(f"Prefix: '{p}' -> {gen}")
        print(f"  func words: {func_frac:.1f}%  domain_count('{domain}')={domain_count}\n")

    mean_func = sum(all_func_counts) / len(all_func_counts)
    print(f'Average function words across prefixes: {mean_func:.1f}%')
    print('Baseline VSA (provided): 38% func words, 0 domain keywords')


if __name__ == '__main__':
    main()
