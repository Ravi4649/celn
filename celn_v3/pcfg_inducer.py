"""
PCFG Inducer (unsupervised)

Implements an unsupervised PCFG induction pipeline based on contiguous
PPMI cohesion. Spans with high internal cohesion (relative to their
context) are replaced by anonymous nonterminals X0, X1, ... iteratively.

Outputs a binarized PCFG (counts + probabilities) as JSON.

Principles respected: no supervision, nonterminals are anonymous,
no hard thresholds (cutoff is chosen by knee detection on the score
distribution, fallback to a high percentile), no classifiers, CPU-first.
"""

from __future__ import annotations

import os
import json
import math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Iterable

import numpy as np

from .train import load_corpus


def _join_span(span: Tuple[str, ...]) -> str:
    return ' '.join(span)


class PCFGInducer:
    def __init__(self, max_span: int = 5, max_iter: int = 5, verbose: bool = True):
        self.max_span = max_span
        self.max_iter = max_iter
        self.verbose = verbose

        # mappings created during induction
        self.span_to_nonterm: Dict[Tuple[str, ...], str] = {}
        self.nonterm_expansions: Dict[str, Tuple[str, ...]] = {}
        self.next_nonterm_id = 0

        # rule counts collected during finalization
        self.binary_rule_counts: Counter = Counter()
        self.unary_rule_counts: Counter = Counter()

    # ----------------------------- IO -----------------------------------
    def _default_corpus_path(self) -> str:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, 'corpus_final.txt')

    # ------------------------- counts / PPMI -----------------------------
    def _collect_contiguous_counts(self, sentences: List[List[str]]):
        word_counts = Counter()
        bigram_counts = Counter()
        trigram_counts = Counter()

        for sent in sentences:
            word_counts.update(sent)
            for i in range(len(sent) - 1):
                bigram_counts[(sent[i], sent[i + 1])] += 1
            for i in range(len(sent) - 2):
                trigram_counts[(sent[i], sent[i + 1], sent[i + 2])] += 1

        return word_counts, bigram_counts, trigram_counts

    def _compute_ppmi_from_counts(self, word_counts: Counter, pair_counts: Counter):
        total_tokens = sum(word_counts.values())
        total_pairs = sum(pair_counts.values())
        ppmi = {}
        if total_pairs == 0 or total_tokens == 0:
            return ppmi

        # unigram probabilities
        word_probs = {w: (word_counts[w] / total_tokens) for w in word_counts}

        # pair_counts keys may be tuples of length 2 (bigrams) or 3 (trigrams)
        for key, count in pair_counts.items():
            p_joint = count / total_pairs
            # product of unigram probabilities for tokens in key
            p_prod = 1.0
            ok = True
            for tok in key:
                p_tok = word_probs.get(tok, 0.0)
                if p_tok <= 0.0:
                    ok = False
                    break
                p_prod *= p_tok

            if not ok or p_joint <= 0 or p_prod <= 0:
                continue

            pmi = math.log(p_joint / (p_prod) + 1e-12)
            ppmi[key] = max(0.0, pmi)

        return ppmi

    # ------------------------- span scoring ------------------------------
    def _collect_span_stats(self, sentences: List[List[str]], bigram_ppmi: Dict[Tuple[str, str], float], trigram_ppmi: Dict[Tuple[str, str, str], float]):
        # accumulate list of (score) per span text and occurrence counts
        span_scores = defaultdict(list)
        span_counts = Counter()

        for sent in sentences:
            L = len(sent)
            for i in range(L):
                for l in range(2, min(self.max_span, L - i) + 1):
                    span = tuple(sent[i:i + l])
                    # internal: mean bigram ppmi inside span
                    internal_vals = []
                    for k in range(len(span) - 1):
                        pair = (span[k], span[k + 1])
                        if pair in bigram_ppmi:
                            internal_vals.append(bigram_ppmi[pair])
                    # include trigram info for spans >=3
                    tri_vals = []
                    if len(span) >= 3:
                        for k in range(len(span) - 2):
                            tri = (span[k], span[k + 1], span[k + 2])
                            if tri in trigram_ppmi:
                                tri_vals.append(trigram_ppmi[tri])

                    if not internal_vals and not tri_vals:
                        continue

                    internal_score = 0.0
                    if internal_vals:
                        internal_score += sum(internal_vals) / len(internal_vals)
                    if tri_vals:
                        internal_score += sum(tri_vals) / len(tri_vals)
                        internal_score /= 2.0  # average bigram/trigram contribution

                    # external: neighbors on left/right (if present)
                    ext_vals = []
                    if i - 1 >= 0:
                        left_pair = (sent[i - 1], span[0])
                        if left_pair in bigram_ppmi:
                            ext_vals.append(bigram_ppmi[left_pair])
                    if i + l < L:
                        right_pair = (span[-1], sent[i + l])
                        if right_pair in bigram_ppmi:
                            ext_vals.append(bigram_ppmi[right_pair])

                    external_score = sum(ext_vals) / len(ext_vals) if ext_vals else 0.0

                    cohesion = internal_score - external_score
                    span_scores[span].append(cohesion)
                    span_counts[span] += 1

        # produce aggregated stats per unique span
        span_stats = []
        for span, scores in span_scores.items():
            avg_score = float(sum(scores) / len(scores))
            span_stats.append((span, avg_score, span_counts[span]))

        return span_stats

    # --------------------- knee detection (auto-cutoff) ------------------
    def _find_knee_percentile(self, values: List[float]) -> float:
        # values: list of scores (one per unique span), unsorted
        if not values:
            return 1.0
        arr = np.array(sorted(values, reverse=True))
        n = len(arr)
        if n <= 2 or (arr.max() - arr.min()) < 1e-6:
            return 0.95

        # normalize x and y to [0,1]
        x = np.linspace(0.0, 1.0, n)
        y = (arr - arr.min()) / (arr.max() - arr.min())

        # line from first to last point
        line_y = x * (y[-1] - y[0]) + y[0]
        distances = np.abs(y - line_y)
        knee_idx = int(np.argmax(distances))
        percentile = (knee_idx + 1) / n
        # clip to [0.5, 0.999] to avoid extremes; fallback if too small
        if percentile < 0.01 or percentile > 0.999:
            return 0.95
        return float(percentile)

    # -------------------- span replacement (greedy) ----------------------
    def _replace_spans(self, sentences: List[List[str]], candidate_spans: Iterable[Tuple[str, ...]]):
        # candidate_spans: set of span tuples to replace (strings tokens)
        # map length -> set for quick matching
        by_len = defaultdict(set)
        for s in candidate_spans:
            by_len[len(s)].add(s)

        new_sentences = []
        replacements = Counter()

        for sent in sentences:
            i = 0
            out = []
            L = len(sent)
            while i < L:
                matched = False
                # try longest first
                for l in range(min(self.max_span, L - i), 1, -1):
                    if l in by_len:
                        tup = tuple(sent[i:i + l])
                        if tup in by_len[l]:
                            # obtain or create nonterminal
                            if tup not in self.span_to_nonterm:
                                nid = self.next_nonterm_id
                                name = f"X{nid}"
                                self.next_nonterm_id += 1
                                self.span_to_nonterm[tup] = name
                                self.nonterm_expansions[name] = tup

                            name = self.span_to_nonterm[tup]
                            out.append(name)
                            replacements[tup] += 1
                            i += l
                            matched = True
                            break
                if not matched:
                    out.append(sent[i])
                    i += 1

            new_sentences.append(out)

        return new_sentences, replacements

    # ------------------------ binarize & finalize ------------------------
    def _binarize_and_count_rules(self, sentences: List[List[str]]):
        # Clear counters
        self.binary_rule_counts = Counter()
        self.unary_rule_counts = Counter()

        # First record nonterminal expansions we created during induction
        for nt, expansion in self.nonterm_expansions.items():
            rhs = list(expansion)
            # record as one rule occurrence per replacement count? We don't
            # have exact counts here; instead we will count occurrences in final corpus
            # by scanning sentences for nt occurrences and mapping back to expansion
            pass

        # Count S -> sentence sequences (start symbol)
        START = 'S'
        for sent in sentences:
            # record start -> sent (we'll binarize)
            self._binarize_rule_and_count(START, sent)

        # Count occurrences of nonterminal expansions by scanning sentences
        for sent in sentences:
            for token in sent:
                if token in self.nonterm_expansions:
                    expansion = list(self.nonterm_expansions[token])
                    self._binarize_rule_and_count(token, expansion)

    def _binarize_rule_and_count(self, lhs: str, rhs: List[str]):
        # rhs: list of symbols (terminals or nonterminals)
        if len(rhs) == 0:
            return
        if len(rhs) == 1:
            # unary rule: A -> w
            self.unary_rule_counts[(lhs, rhs[0])] += 1
            return

        # left-factoring binarization: A -> r0 B1 ; B1 -> r1 B2 ; ... ; Bn -> r_{n-2} r_{n-1}
        cur_lhs = lhs
        for i in range(len(rhs) - 2):
            r0 = rhs[i]
            bin_nt = f"@BIN_{lhs}_{i}_{r0}"
            self.binary_rule_counts[(cur_lhs, (r0, bin_nt))] += 1
            cur_lhs = bin_nt

        # final binary
        self.binary_rule_counts[(cur_lhs, (rhs[-2], rhs[-1]))] += 1

    # ----------------------------- main ---------------------------------
    def induce(self, sentences: List[List[str]]):
        # operate in-place on sentences copy
        sents = [list(s) for s in sentences]

        for it in range(self.max_iter):
            if self.verbose:
                print(f"PCFG induction iter {it+1}/{self.max_iter}: {len(sents)} sentences, current nonterms={len(self.nonterm_expansions)}")

            word_counts, bigram_counts, trigram_counts = self._collect_contiguous_counts(sents)
            bigram_ppmi = self._compute_ppmi_from_counts(word_counts, bigram_counts)
            trigram_ppmi = self._compute_ppmi_from_counts(word_counts, trigram_counts)

            span_stats = self._collect_span_stats(sents, bigram_ppmi, trigram_ppmi)
            if not span_stats:
                if self.verbose:
                    print("  No candidate spans found. Stopping.")
                break

            # filter extremely rare spans (occurrence >=2)
            span_stats = [t for t in span_stats if t[2] >= 2]
            if not span_stats:
                if self.verbose:
                    print("  No spans with enough support. Stopping.")
                break

            scores = [t[1] for t in span_stats]
            pct = self._find_knee_percentile(scores)
            # select spans with rank <= knee index
            sorted_spans = sorted(span_stats, key=lambda x: x[1], reverse=True)
            cutoff_idx = max(1, int(len(sorted_spans) * pct))
            selected = [t[0] for t in sorted_spans[:cutoff_idx]]

            if self.verbose:
                print(f"  Candidates: {len(span_stats)}, selected {len(selected)} (pct={pct:.3f})")

            if not selected:
                if self.verbose:
                    print("  Knee selection returned empty. Stopping.")
                break

            # Replace selected spans greedily in sentences
            sents, replacements = self._replace_spans(sents, selected)

            if self.verbose:
                total_repl = sum(replacements.values())
                print(f"  Replaced {total_repl} span occurrences, new nonterms={len(self.nonterm_expansions)}")

            # stop early if no replacements
            if total_repl == 0:
                break

        # finalize: binarize and count rules from final sentences
        self._binarize_and_count_rules(sents)

        # compute probabilities
        pcfg = self._compute_pcfg()
        return pcfg

    def _compute_pcfg(self):
        # aggregate counts per LHS
        lhs_counts = defaultdict(int)
        for (A, rhs), c in list(self.binary_rule_counts.items()):
            lhs_counts[A] += c
        for (A, w), c in list(self.unary_rule_counts.items()):
            lhs_counts[A] += c

        rules = {}
        for (A, rhs), c in self.binary_rule_counts.items():
            rhs_list = list(rhs)
            prob = c / lhs_counts[A] if lhs_counts[A] > 0 else 0.0
            rules.setdefault(A, []).append({'rhs': rhs_list, 'count': int(c), 'prob': float(prob)})

        for (A, w), c in self.unary_rule_counts.items():
            prob = c / lhs_counts[A] if lhs_counts[A] > 0 else 0.0
            rules.setdefault(A, []).append({'rhs': [w], 'count': int(c), 'prob': float(prob)})

        # include expansions mapping for the induced Xn symbols
        nonterms = {nt: list(exp) for nt, exp in self.nonterm_expansions.items()}

        return {'rules': rules, 'nonterm_expansions': nonterms, 'meta': {'num_nonterms': len(nonterms)}}

    def save_pcfg(self, pcfg: dict, outpath: str):
        with open(outpath, 'w', encoding='utf-8') as f:
            json.dump(pcfg, f, ensure_ascii=False, indent=2)
        if self.verbose:
            print(f"Saved PCFG to {outpath}")


def main(corpus_path: str | None = None, outpath: str | None = None):
    if corpus_path is None:
        inducer = PCFGInducer()
        corpus_path = inducer._default_corpus_path()
    else:
        inducer = PCFGInducer()

    if outpath is None:
        outpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pcfg_induced.json')

    sentences = load_corpus(corpus_path, min_len=2)
    if inducer.verbose:
        print(f"Loaded {len(sentences)} sentences from {corpus_path}")

    pcfg = inducer.induce(sentences)
    inducer.save_pcfg(pcfg, outpath)

    # report brief summary
    rules = pcfg.get('rules', {})
    n_rules = sum(len(v) for v in rules.values())
    print(f"PCFG induction complete: nonterms={pcfg['meta']['num_nonterms']}, rules={n_rules}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Induce PCFG from corpus (unsupervised)')
    parser.add_argument('--corpus', default=None, help='Path to corpus_final.txt')
    parser.add_argument('--out', default=None, help='Output JSON path')
    args = parser.parse_args()
    main(corpus_path=args.corpus, outpath=args.out)
