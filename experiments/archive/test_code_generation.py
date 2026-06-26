#!/usr/bin/env python3
"""
Test: Can CELN architecture generate syntactically-correct Python code?

Pipeline (self-contained experiment):
 1. Collect real Python functions under /home/ravizin (AST extraction)
 2. Tokenize preserving Python structure using tokenize module
 3. Build co-occurrence → PPMI → SVD semantic vectors (reduced dim)
 4. Extract Type Field via TruncatedSVD on PPMI (type_dim)
 5. Build a lightweight Pair store of M(src, fol) pairs
 6. Generator: given prefix token(s) try to continue using
    Type Field + Pair transport. Baseline: plain cosine similarity
 7. Evaluate syntactic validity (ast.parse) and simple heuristics

Principles: no backprop, CPU-only, no templates. Works with small corpus
of real functions (200-500 targeted). This is an experimental probe.
"""

import ast
import io
import os
import re
import sys
import tokenize as py_tokenize
from collections import defaultdict

import numpy as np
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from celn.train import build_cooccurrence, compute_ppmi
from celn.dual_channel import extract_type_vectors
from celn.core import normalize, projective_resonance as M, inverse_projective_resonance


def find_python_files(root='/home/ravizin', max_files=2000):
    py_files = []
    for dirpath, dirs, files in os.walk(root):
        # skip hidden directories
        parts = dirpath.split(os.sep)
        if any(p.startswith('.') for p in parts):
            continue
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(dirpath, f)
                py_files.append(path)
                if len(py_files) >= max_files:
                    return py_files
    return py_files


def extract_functions_from_file(path):
    try:
        src = open(path, 'r', encoding='utf-8').read()
    except Exception:
        return []
    try:
        tree = ast.parse(src)
    except Exception:
        return []
    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                start = node.lineno - 1
                end = node.end_lineno
                lines = src.splitlines()[start:end]
                funcs.append('\n'.join(lines))
            except Exception:
                continue
    return funcs


def collect_functions(target=300):
    files = find_python_files()
    functions = []
    for p in files:
        fs = extract_functions_from_file(p)
        for f in fs:
            # reject tiny one-line functions
            if len(f.splitlines()) < 2:
                continue
            functions.append(f)
            if len(functions) >= target:
                return functions
    return functions


def tokenize_python_code(src):
    # Use Python tokenizer to produce a list of string tokens preserving
    # structural tokens (INDENT/DEDENT, NEWLINE, NAME, OP, NUMBER, STRING)
    toks = []
    try:
        g = py_tokenize.generate_tokens(io.StringIO(src).readline)
        for toknum, tokval, _, _, _ in g:
            if toknum == py_tokenize.INDENT:
                toks.append('<INDENT>')
            elif toknum == py_tokenize.DEDENT:
                toks.append('<DEDENT>')
            elif toknum == py_tokenize.NEWLINE or toknum == py_tokenize.NL:
                toks.append('<NEWLINE>')
            else:
                v = tokval.strip()
                if v == '':
                    continue
                toks.append(v)
    except Exception:
        # fallback: simple split by non-word but keep punctuation
        toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\(|\)|:|\.|,|\[|\]|\{|\}|\+|\-|\*|/|==|!=|<=|>=|<|>|=|\"[^\"]*\"|'[^']*'", src)
    return toks


def build_vocab(token_lists, min_count=2):
    counts = defaultdict(int)
    for sent in token_lists:
        for t in sent:
            counts[t] += 1
    vocab = [w for w, c in counts.items() if c >= min_count]
    w2i = {w: i for i, w in enumerate(sorted(vocab))}
    i2w = {i: w for w, i in w2i.items()}
    return w2i, i2w


def build_sentences_token_indices(token_lists, w2i):
    sentences = []
    for toks in token_lists:
        idxs = [w2i[t] for t in toks if t in w2i]
        if len(idxs) >= 2:
            # treat as sentence of tokens
            sentences.append([str(i) for i in idxs])
    return sentences


def simple_join_tokens(tokens):
    # join tokens into plausible code string with spacing rules
    out = ''
    no_space_before = {')', ',', ':', '.', ']', '}', '%', '==', '!=', '>=', '<=', '+', '-', '*', '/', '='}
    no_space_after = {'(', '[', '{', '.', ''}
    for i, t in enumerate(tokens):
        if i == 0:
            out += t
            continue
        if t in no_space_before:
            out += t
        elif tokens[i-1] in no_space_after:
            out += t
        else:
            out += ' ' + t
    return out


def train_vectors_from_token_lists(token_lists, w2i, dim_sem=2048, svd_comp=512):
    # Build cooccurrence on token indices (using integer strings expected by build_cooccurrence)
    # We adapt build_cooccurrence to accept token lists where tokens are strings. We'll call it directly.
    # Reuse build_cooccurrence and compute_ppmi from train.py but those expect word strings; we will pass tokens as strings (their indices) which is fine.
    sentences = [[t for t in toks] for toks in token_lists]
    word_counts, cooc_counts, word2idx, idx2word = build_cooccurrence(sentences, window_size=5)
    ppmi = compute_ppmi(word_counts, cooc_counts, word2idx)

    V = len(word2idx)
    nc = min(svd_comp, max(2, V - 1))
    svd = TruncatedSVD(n_components=nc, random_state=42)
    vr = svd.fit_transform(ppmi)
    sv = svd.singular_values_
    var = sv**2 / (sv**2).sum()
    vr = vr * (var / var.max())[None, :]

    if nc < dim_sem:
        R = np.random.RandomState(42).randn(nc, dim_sem).astype(np.float32) / np.sqrt(nc)
        sem_vecs = vr.astype(np.float32) @ R
    else:
        sem_vecs = vr.astype(np.float32)

    # Normalize
    norms = np.linalg.norm(sem_vecs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    sem_vecs = sem_vecs / norms

    # Map back to canonical w2i ordering: word2idx keys are token-strings (indices as strings)
    # Build mapping from token string -> row in sem_vecs
    return sem_vecs, word2idx, idx2word, ppmi


def build_pair_store(sem_vecs, token_idx_pairs):
    # token_idx_pairs: list of (src_idx, fol_idx)
    pair_vecs = []
    srcs = []
    fols = []
    for s, f in token_idx_pairs:
        v = M(sem_vecs[s], sem_vecs[f], gamma=1.0, bilateral=True)
        pair_vecs.append(v)
        srcs.append(s)
        fols.append(f)
    pair_vecs = np.vstack(pair_vecs).astype(np.float32)
    # normalize rows
    norms = np.linalg.norm(pair_vecs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    pair_vecs = pair_vecs / norms
    return pair_vecs, np.array(srcs, dtype=np.int32), np.array(fols, dtype=np.int32)


def find_topk_pair_matches(pair_vecs, query_pair, k=5):
    sims = pair_vecs @ query_pair
    topk = np.argpartition(sims, -k)[-k:]
    # sort descending
    topk = topk[np.argsort(sims[topk])[::-1]]
    return topk, sims[topk]


def run_experiment():
    print('\n[1] Collecting functions...')
    funcs = collect_functions(target=400)
    print(f'  Collected {len(funcs)} functions')

    token_lists = [tokenize_python_code(f) for f in funcs]
    # Filter out extremely short token lists
    token_lists = [t for t in token_lists if len(t) >= 5]
    print(f'  Tokenized into {len(token_lists)} examples')

    # Build vocabulary over tokens
    w2i, i2w = build_vocab(token_lists, min_count=2)
    print(f'  Vocab size: {len(w2i)} tokens')

    # Replace tokens with their string indices (for build_cooccurrence compatibility)
    token_lists_idx = [[str(w2i[t]) for t in toks if t in w2i] for toks in token_lists]
    sem_vecs, word2idx, idx2word, ppmi = train_vectors_from_token_lists(token_lists_idx, w2i, dim_sem=2048, svd_comp=256)
    # sem_vecs rows correspond to indices in word2idx (string token -> row index)

    # Build mapping from original token string to sem vector row
    token_to_row = {tok: word2idx[str(w2i[tok])] for tok in w2i}

    # Build type vectors from ppmi with desired type dimension
    type_dim = 512
    type_vecs = extract_type_vectors(ppmi, type_dim=type_dim, seed=42)

    # Build pair index list from consecutive tokens across token_lists
    pair_list = []
    for toks in token_lists:
        for i in range(len(toks) - 1):
            a, b = toks[i], toks[i+1]
            if a in w2i and b in w2i:
                ai = word2idx[str(w2i[a])]
                bi = word2idx[str(w2i[b])]
                pair_list.append((ai, bi))

    if not pair_list:
        print('No pairs found; aborting')
        return

    pair_vecs, srcs, fols = build_pair_store(sem_vecs, pair_list)
    print(f'  Pair store: {pair_vecs.shape[0]} pairs, sem dim={sem_vecs.shape[1]}')

    # Build Type Field: for each token index i, centroid of type vectors of its followers
    V = type_vecs.shape[0]
    type_field = np.zeros((V, type_dim), dtype=np.float32)
    counts = np.zeros(V, dtype=np.int32)
    for s, f in pair_list:
        type_field[s] += type_vecs[f]
        counts[s] += 1
    for i in range(V):
        if counts[i] > 0:
            type_field[i] = normalize(type_field[i])
        else:
            type_field[i] = type_vecs[i]

    # Generator: given prefix (token list), generate up to N tokens
    def generate(prefix_tokens, max_len=20, use_transport=True):
        generated = list(prefix_tokens)
        for step in range(max_len):
            # current word = last token if available else <NEWLINE>
            last = generated[-1] if generated else '<NEWLINE>'
            if last not in w2i:
                # fallback: choose frequent token
                candidates = sorted(w2i.items(), key=lambda x: -len(x[0]))
                break
            last_row = word2idx[str(w2i[last])]

            # context centroid: mean of sem_vecs of recent tokens
            recent = [t for t in generated if t in w2i]
            rows = [word2idx[str(w2i[t])] for t in recent]
            if rows:
                ctx = normalize(np.mean(sem_vecs[rows], axis=0))
            else:
                ctx = np.zeros(sem_vecs.shape[1], dtype=np.float32)

            # Type scores
            type_scores = type_vecs @ type_field[last_row]

            # Transport (PairSDM-like) candidate extraction
            transport_scores = None
            if use_transport:
                query_pair = M(ctx if np.linalg.norm(ctx)>1e-12 else sem_vecs[last_row], sem_vecs[last_row], gamma=1.0, bilateral=True)
                qn = normalize(query_pair)
                topk, sims = find_topk_pair_matches(pair_vecs, qn, k=8)
                # For each top pair, recover follower via inverse_projective_resonance
                accum_scores = np.zeros(V, dtype=np.float32)
                for idx in topk:
                    sidx = int(srcs[idx])
                    try:
                        rec = inverse_projective_resonance(pair_vecs[idx], sem_vecs[sidx], gamma=1.0, bilateral=True, n_iter=10)
                    except Exception:
                        rec = sem_vecs[fols[idx]]
                    rec = normalize(rec)
                    word_scores = sem_vecs @ rec
                    accum_scores += np.maximum(word_scores, 0.0)
                transport_scores = accum_scores

            # Baseline semantic (cosine to context)
            sem_scores = sem_vecs @ ctx

            # Combine: type_maestro_blend simple version — weighted sum with type priority
            if transport_scores is not None:
                # normalize channels
                tchan = transport_scores / (transport_scores.max() + 1e-12)
                schan = sem_scores / (np.abs(sem_scores).max() + 1e-12)
                ts = type_scores / (np.abs(type_scores).max() + 1e-12)
                # type weight biased to preserve syntax
                type_weight = 0.4
                sem_weight = 0.2
                trans_weight = 0.4
                combined = type_weight * ts + sem_weight * schan + trans_weight * tchan
            else:
                ts = type_scores / (np.abs(type_scores).max() + 1e-12)
                schan = sem_scores / (np.abs(sem_scores).max() + 1e-12)
                combined = 0.6 * ts + 0.4 * schan

            # mask recent duplicates lightly
            for t in set(generated[-3:]):
                if t in w2i:
                    ri = word2idx[str(w2i[t])]
                    combined[ri] *= 0.3

            # pick best index
            best = int(np.argmax(combined))
            next_tok = idx2word[best]
            generated.append(next_tok)

            # stop if generated NEWLINE and last token was DEDENT or colon
            if next_tok == '<NEWLINE>' and len(generated) > 1:
                break

        return generated

    prefixes = [
        ['def'],
        ['class'],
        ['async', 'def'],
        ['for'],
        ['if'],
        ['with'],
        ['try',':'],
        ['return'],
        ['import'],
        ['@']
    ]

    results = []
    print('\n[2] Generation tests (Type+Pair Transport vs Baseline)')
    for p in prefixes:
        out_transport = generate(p, max_len=30, use_transport=True)
        out_baseline = generate(p, max_len=30, use_transport=False)

        code_t = simple_join_tokens(out_transport)
        code_b = simple_join_tokens(out_baseline)

        # Try parse
        def check_syntax(code):
            try:
                ast.parse(code)
                return True, ''
            except Exception as e:
                return False, str(e)

        ok_t, err_t = check_syntax(code_t)
        ok_b, err_b = check_syntax(code_b)

        print('\nPREFIX:', ' '.join(p))
        print('  TRANSPORT:', code_t[:200].replace('\n','\\n'))
        print('    syntax_ok=', ok_t, 'err=', err_t)
        print('  BASELINE :', code_b[:200].replace('\n','\\n'))
        print('    syntax_ok=', ok_b, 'err=', err_b)

        results.append(((p, code_t, ok_t, err_t), (p, code_b, ok_b, err_b)))

    # Summary
    t_ok = sum(1 for a, b in results if a[2])
    b_ok = sum(1 for a, b in results if b[2])
    print('\nSummary:')
    print(f'  Transport syntactically valid: {t_ok}/{len(prefixes)}')
    print(f'  Baseline syntactically valid : {b_ok}/{len(prefixes)}')


if __name__ == '__main__':
    run_experiment()
