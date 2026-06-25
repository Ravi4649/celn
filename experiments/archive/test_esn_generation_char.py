#!/usr/bin/env python3
"""
Char-level ESN generation test on CELN corpus (no backprop).

Train ESN to predict next character, generate text from prefixes.

Streaming ridge regression accumulators used to avoid storing full state matrix.
"""

import os
import re
import time
import numpy as np


def load_corpus_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.rstrip('\n') for ln in f]
    return lines


def build_charset(lines):
    s = '\n'.join(lines).lower()
    chars = sorted(set(s))
    # ensure space is early
    if ' ' in chars:
        chars.remove(' ')
        chars.insert(0, ' ')
    return chars


def make_reservoir(N, density=0.01, spectral_radius=0.9, seed=42):
    rng = np.random.RandomState(seed)
    W = rng.randn(N, N).astype(np.float32)
    mask = rng.rand(N, N) < density
    W *= mask

    # power iteration to estimate spectral radius
    def power_it(A, n_it=10):
        v = rng.randn(A.shape[0]).astype(np.float64)
        v /= (np.linalg.norm(v) + 1e-12)
        for _ in range(n_it):
            v = A.dot(v)
            norm = np.linalg.norm(v)
            if norm < 1e-12:
                break
            v /= norm
        Av = A.dot(v)
        return float(np.dot(v, Av))

    approx_rho = abs(power_it(W, n_it=40))
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


def train_esn_stream(char_ids, V, N=200, input_scale=0.2, leak=0.3, density=0.01, sr=0.9, reg=1e-6, seed=123):
    rng = np.random.RandomState(seed)
    print('Building reservoir... N=', N)
    W = make_reservoir(N, density=density, spectral_radius=sr, seed=seed)
    Win = (rng.rand(N, V).astype(np.float32) * 2 - 1) * input_scale
    bias = (rng.rand(N).astype(np.float32) * 2 - 1) * 0.01

    n_features = N + 1
    XtX = np.zeros((n_features, n_features), dtype=np.float64)
    XtY = np.zeros((n_features, V), dtype=np.float64)

    x = np.zeros(N, dtype=np.float32)
    total_steps = 0

    print('Streaming through data to accumulate XtX and XtY...')
    step_print = max(1, len(char_ids) // 10)
    for t in range(len(char_ids) - 1):
        u = char_ids[t]
        y = char_ids[t + 1]
        pre = W.dot(x) + Win[:, u] + bias
        x = (1 - leak) * x + leak * np.tanh(pre)

        s = np.empty(n_features, dtype=np.float64)
        s[:N] = x.astype(np.float64)
        s[N] = 1.0

        XtX += np.outer(s, s)
        XtY[:, y] += s
        total_steps += 1
        if (t & 0x3FFF) == 0:  # progress every ~16384 steps or so
            print(f'  step {t}/{len(char_ids)}')

    print('Total training timesteps:', total_steps)
    # solve ridge: (XtX + reg I) Wout = XtY
    A = XtX + reg * np.eye(n_features, dtype=np.float64)
    print('Solving linear system for Wout... (this may take a moment)')
    Wout = np.linalg.solve(A, XtY)
    Wout = Wout.astype(np.float32)

    model = {'W': W, 'Win': Win, 'bias': bias, 'Wout': Wout, 'N': N, 'V': V, 'leak': leak}
    return model


def generate_char(model, char2i, i2char, prefix, max_len=400, temp=1.0, seed=0):
    rng = np.random.RandomState(seed)
    N = model['N']
    W = model['W']
    Win = model['Win']
    bias = model['bias']
    Wout = model['Wout']
    leak = model['leak']

    x = np.zeros(N, dtype=np.float32)
    out = []
    s = prefix.lower()
    for ch in s:
        if ch not in char2i:
            idx = char2i.get(' ', 0)
        else:
            idx = char2i[ch]
        pre = W.dot(x) + Win[:, idx] + bias
        x = (1 - leak) * x + leak * np.tanh(pre)
        out.append(ch)

    for _ in range(max_len):
        svec = np.empty(N + 1, dtype=np.float32)
        svec[:N] = x
        svec[N] = 1.0
        logits = svec.dot(model['Wout'])
        probs = softmax(logits, temp=temp)
        idx = rng.choice(model['V'], p=probs)
        ch = i2char[idx]
        out.append(ch)
        pre = W.dot(x) + Win[:, idx] + bias
        x = (1 - leak) * x + leak * np.tanh(pre)

    return ''.join(out)


def main():
    corpus_path = 'corpus_final.txt'
    if not os.path.exists(corpus_path):
        corpus_path = '/home/ravizin/celn-v3/corpus_final.txt'
    lines = load_lines = load = None
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = [ln.rstrip('\n') for ln in f]

    print('Loaded', len(lines), 'lines')

    chars = build_charset(lines)
    print('Charset size:', len(chars))
    char2i = {ch: i for i, ch in enumerate(chars)}
    i2char = {i: ch for ch, i in char2i.items()}
    V = len(chars)

    # Build long character id stream (join with newline)
    text = '\n'.join(lines).lower()
    char_ids_full = [char2i.get(ch, char2i[chars[0]]) for ch in text]
    # limit training length for quick runs (can increase for final runs)
    max_chars = 80000
    if len(char_ids_full) > max_chars:
        print(f'Truncating training stream to first {max_chars} chars (from {len(char_ids_full)}) for speed')
    char_ids = char_ids_full[:max_chars]

    # Train ESN with streaming accumulators
    N = 1000  # reservoir size — reduced to keep runtime reasonable
    model = train_esn_stream(char_ids, V, N=N, input_scale=0.2, leak=0.3, density=0.01, sr=0.9, reg=1e-6, seed=42)

    prefixes = ['o cobre', 'a eletricidade', 'o brasil', 'o gato']
    func_words = set([
        'o','a','os','as','um','uma','uns','umas',
        'de','do','da','dos','das','em','no','na','nos','nas',
        'e','ou','mas','que','se','nem','pois','é','foi','era',
        'são','está','ser','não','sim','como','quando','onde',
        'porque','para','com','por','pelo','pela','pelas','sem','sob','sobre',
    ])

    print('\nGenerated texts (char-level ESN):')
    all_func_counts = []
    for p in prefixes:
        gen = generate_char(model, char2i, i2char, p, max_len=300, temp=0.8, seed=123)
        # compute function word fraction in tokenized words
        toks = re.findall(r"\w+|[^\s\w]", gen.lower(), flags=re.UNICODE)
        func_frac = sum(1 for t in toks if t in func_words) / max(1, len(toks)) * 100.0
        all_func_counts.append(func_frac)
        domain = p.split()[-1]
        domain_count = sum(1 for t in toks if t == domain)
        print(f"Prefix: '{p}' -> {gen[:300]}...")
        print(f"  func words: {func_frac:.1f}%  domain_count('{domain}')={domain_count}\n")

    mean_func = sum(all_func_counts) / len(all_func_counts)
    print(f'Average function words across prefixes: {mean_func:.1f}%')
    print('Baseline VSA (provided): 38% func words, 0 domain keywords')


if __name__ == '__main__':
    main()
