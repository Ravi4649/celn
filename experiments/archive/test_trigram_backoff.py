#!/usr/bin/env python3
"""
Simple trigram language model with stupid backoff for generation.

Train on corpus_final.txt (word-level). Generate from prefixes and
report function-word fraction and domain keyword counts.
"""

import re
import os
import random
from collections import Counter, defaultdict


def tokenize(sentence: str):
    # simple tokenization: words and punctuation as separate tokens
    return re.findall(r"\w+|[^\s\w]", sentence.lower(), flags=re.UNICODE)


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


def build_ngram_counts(sents):
    unigram = Counter()
    bigram = Counter()
    trigram = Counter()

    for toks in sents:
        # add two start tokens
        seq = ['<s>', '<s>'] + toks + ['</s>']
        for i in range(len(seq)):
            if i >= 2:
                trigram[(seq[i-2], seq[i-1], seq[i])] += 1
            if i >= 1:
                bigram[(seq[i-1], seq[i])] += 1
            unigram[seq[i]] += 1

    return unigram, bigram, trigram


def build_vocab(unigram, min_count=1):
    vocab = [w for w, c in unigram.items() if c >= min_count]
    # preserve special tokens
    if '<s>' not in vocab:
        vocab.insert(0, '<s>')
    if '</s>' not in vocab:
        vocab.append('</s>')
    return vocab


def next_word_stupid_backoff(w1, w2, vocab, unigram, bigram, trigram, alpha=0.4):
    # Compute scores for all candidates in vocab using stupid backoff
    scores = []
    total_unigrams = sum(unigram.values())
    bigram_count_context = bigram.get((w1, w2), 0)
    for w in vocab:
        # trigram
        tri = trigram.get((w1, w2, w), 0)
        if tri > 0 and bigram_count_context > 0:
            # use conditional probability estimate
            score = tri / bigram_count_context
        else:
            # backoff to bigram
            bi = bigram.get((w2, w), 0)
            if bi > 0:
                score = alpha * (bi / max(1, unigram.get(w2, 1)))
            else:
                # unigram fallback (smoothed by alpha^2)
                score = (alpha ** 2) * (unigram.get(w, 0) / max(1, total_unigrams))
        scores.append(score)

    # normalize to probabilities
    ssum = sum(scores)
    if ssum <= 0:
        # fallback uniform
        probs = [1.0 / len(vocab)] * len(vocab)
    else:
        probs = [s / ssum for s in scores]
    # sample next word
    return random.choices(vocab, probs, k=1)[0]


def generate(prefix, vocab, unigram, bigram, trigram, max_len=25):
    toks = tokenize(prefix)
    # ensure at least two context tokens
    if len(toks) == 0:
        ctx = ['<s>', '<s>']
    elif len(toks) == 1:
        ctx = ['<s>', toks[-1]]
    else:
        ctx = toks[-2:]

    out = toks.copy()
    for _ in range(max_len):
        w1, w2 = ctx[-2], ctx[-1]
        w = next_word_stupid_backoff(w1, w2, vocab, unigram, bigram, trigram)
        if w == '</s>':
            break
        out.append(w)
        ctx.append(w)
    return ' '.join(out)


def generate_from_structure(struct, vocab, unigram, bigram, trigram, max_len=25, seed=None):
    """Generate a fluent sentence conditioned on a structured dict.

    struct: dict with keys 'sujeito', 'predicado', 'objeto', optional 'target' (for analogy)
    The function composes a reasonable prefix from the structure and calls
    the standard generate() function to produce fluent text.
    """
    # Defensive copy and normalization
    s = (struct.get('sujeito') or '').strip().lower()
    p = (struct.get('predicado') or '').strip().lower()
    o = (struct.get('objeto') or '').strip().lower()
    target = (struct.get('target') or '').strip().lower()

    # Build prefix heuristically
    if target:
        # Analogy: "s está para o as como target está para"
        # Ensure predicate expresses 'está para' form
        pred_phrase = p if p else 'está para'
        # subject and object likely nouns; add articles for fluency
        subj_token = s
        obj_token = o
        tgt_token = target
        prefix = f"{subj_token} {pred_phrase} {obj_token} como {tgt_token} {pred_phrase}"
    elif s and p and o:
        # Declarative conditioning: prefer adding an article before subject
        articles = {'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas'}
        subj_first = s.split()[0]
        if subj_first in articles:
            prefix = f"{s} {p} {o}"
        else:
            prefix = f"o {s} {p} {o}"
    elif p and o:
        prefix = f"{p} {o}"
    else:
        # Fallback: empty prefix
        prefix = ''

    # Ensure prefix tokens are in vocab; else fallback to shorter prefix
    toks = re.findall(r"\w+|[^\s\w]", prefix.lower(), flags=re.UNICODE)
    # If none of the prefix tokens are in vocab, fall back to first two words of object
    if toks and not any(t in vocab for t in toks):
        if o:
            prefix = o
        else:
            prefix = (s or p or '')

    # If a seed is provided, seed the random module used by generate
    if seed is not None:
        random.seed(seed)

    return generate(prefix, vocab, unigram, bigram, trigram, max_len=max_len)


def evaluate_generated(text, func_words, domain):
    toks = tokenize(text)
    if len(toks) == 0:
        return 0.0, 0
    func_frac = sum(1 for t in toks if t in func_words) / len(toks) * 100.0
    domain_count = sum(1 for t in toks if t == domain)
    return func_frac, domain_count


def main():
    corpus_path = 'corpus_final.txt'
    if not os.path.exists(corpus_path):
        corpus_path = '/home/ravizin/celn-v3/corpus_final.txt'

    sents = load_corpus(corpus_path)
    print('Loaded sentences:', len(sents))

    unigram, bigram, trigram = build_ngram_counts(sents)
    vocab = build_vocab(unigram, min_count=1)
    print('Vocab size:', len(vocab))

    prefixes = ['o cobre', 'a eletricidade', 'o brasil', 'o gato']
    func_words = set([
        'o','a','os','as','um','uma','uns','umas',
        'de','do','da','dos','das','em','no','na','nos','nas',
        'e','ou','mas','que','se','nem','pois','é','foi','era',
        'são','está','ser','não','sim','como','quando','onde',
        'porque','para','com','por','pelo','pela','pelas','sem','sob','sobre',
    ])

    print('\nGenerated outputs:')
    results = []
    for p in prefixes:
        gen = generate(p, vocab, unigram, bigram, trigram, max_len=30)
        func_frac, domain_count = evaluate_generated(gen, func_words, p.split()[-1])
        results.append((p, gen, func_frac, domain_count))
        print('Prefix:', p)
        print('Output:', gen)
        print(f"  func words: {func_frac:.1f}%  domain_count('{p.split()[-1]}')={domain_count}\n")

    mean_func = sum(r[2] for r in results) / len(results)
    print('Average function words across prefixes: {:.1f}%'.format(mean_func))
    print('Baseline VSA (provided): 38% func words, 0 domain keywords')


if __name__ == '__main__':
    main()
