#!/usr/bin/env python3
"""
Integration test: VSA structured output -> trigram backoff generator

1. Use a lightweight VSA structured parser (dual_channel_structured.vsa_generate_structure)
2. Use experiments/test_trigram_backoff.generate_from_structure to produce fluent output
3. Run tests for 3 questions and print results alongside baseline
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celn_v3.dual_channel_structured import vsa_generate_structure
import experiments.test_trigram_backoff as tri


QUESTIONS = [
    "o cobre conduz eletricidade?",
    "quem comeu o rato?",
    "cobre está para metal como onça está para o quê?",
]


def main():
    corpus_path = 'corpus_final.txt'
    if not os.path.exists(corpus_path):
        corpus_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'corpus_final.txt')

    sents = tri.load_corpus(corpus_path)
    unigram, bigram, trigram = tri.build_ngram_counts(sents)
    vocab = tri.build_vocab(unigram, min_count=1)

    print('Loaded trigram LM with vocab size', len(vocab))

    results = []
    for q in QUESTIONS:
        struct = vsa_generate_structure(q, pipeline=None)
        print('\nQuestion:', q)
        print('VSA structure:', struct)
        gen = tri.generate_from_structure(struct, vocab, unigram, bigram, trigram, max_len=25, seed=42)
        # compute simple metrics
        func_words = set([
            'o','a','os','as','um','uma','uns','umas',
            'de','do','da','dos','das','em','no','na','nos','nas',
            'e','ou','mas','que','se','nem','pois','é','foi','era',
            'são','está','ser','não','sim','como','quando','onde',
            'porque','para','com','por','pelo','pela','pelas','sem','sob','sobre',
        ])
        toks = tri.tokenize(gen)
        func_frac = sum(1 for t in toks if t in func_words) / max(1, len(toks)) * 100.0
        domain = struct.get('sujeito') or struct.get('objeto') or ''
        domain_count = sum(1 for t in toks if t == domain)
        print('Generated:', gen)
        print(f"  func words: {func_frac:.1f}%  domain_count('{domain}')={domain_count}")
        results.append({'question': q, 'structure': struct, 'generation': gen, 'func_frac': func_frac, 'domain_count': domain_count})

    # Summary
    mean_func = sum(r['func_frac'] for r in results) / len(results)
    print('\nAverage function words across questions: {:.1f}%'.format(mean_func))
    print('Baseline VSA (provided): 38% func words, 0 domain keywords')


if __name__ == '__main__':
    main()
