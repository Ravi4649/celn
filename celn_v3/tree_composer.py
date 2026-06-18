"""
Tree Composer: convert PCFG trees into VSA vectors (D=10k)

Features:
- Load induced PCFG JSON
- Load lexical vectors from a .npz (expects arrays 'vectors', 'word2idx', 'idx2word' or similar)
- Compute centroid vectors for anonymous nonterminals (Xn and @BIN_...)
- Compose a tree bottom-up using core.projective_resonance with type vectors
- Memoize subtree compositions by hash
- Precompute spectra for terminal/nonterminal vectors via decoder.precompute_spectra

Assumptions:
- lexical vectors are available in a .npz file under keys 'vectors' and 'idx2word' or via train.train_vectors
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from .core import projective_resonance, normalize
from .train import precompute_spectra
from .train import load_corpus, train_vectors
from .hdc_types import train_hdc_type_vectors


class TreeComposer:
    def __init__(self,
                 pcfg_path: str = None,
                 vectors_path: str = None,
                 type_vectors: np.ndarray | None = None,
                 state_path: str | None = None,
                 verbose: bool = True):
        self.verbose = verbose

        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if pcfg_path is None:
            pcfg_path = os.path.join(base, 'pcfg_induced.json')
        self.pcfg = self._load_pcfg(pcfg_path)

        # load lexical vectors
        if vectors_path is None:
            vectors_path = os.path.join(base, 'celn_v3_full_vectors.npz')
        self.vectors, self.word2idx, self.idx2word = self._load_vectors(vectors_path)

        # state persistence path
        if state_path is None:
            state_path = os.path.join(base, 'tree_composer_state.npz')
        self.state_path = state_path

        # compute or load nonterminal centroids and bin-nodes
        self.nonterm_vectors: Dict[str, np.ndarray] = {}
        if os.path.exists(self.state_path):
            try:
                self._load_state(self.state_path)
                if self.verbose:
                    print(f"Loaded TreeComposer state from {self.state_path}")
            except Exception:
                if self.verbose:
                    print("Failed to load state file; recomputing")
                self._compute_nonterm_centroids()
                self._build_all_nonterm_vectors()
                self._precompute_all_spectra()
                self._save_state(self.state_path)
        else:
            self._compute_nonterm_centroids()
            self._build_all_nonterm_vectors()
            self._precompute_all_spectra()
            self._save_state(self.state_path)

        # optional type vectors for roles (if provided)
        self.type_vectors = type_vectors

        # memoization cache for subtree compositions
        self._cache: Dict[str, np.ndarray] = {}

        # precompute spectra for all available vectors (terminals + nonterms)
        self.spectra = None
        try:
            self._precompute_all_spectra()
        except Exception:
            # non-fatal: precomputation optional
            if self.verbose:
                print("Warning: precompute_spectra failed; continue without spectra.")

    # ---------------- IO -----------------
    def _load_pcfg(self, path: str) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_vectors(self, path: str) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
        if not os.path.exists(path):
            # fallback: train quick vectors from corpus
            if self.verbose:
                print(f"Vectors file {path} not found — training quick vectors from corpus (slow).")
            sentences = load_corpus(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'corpus_final.txt'), min_len=2)
            vecs, w2i, i2w, ppmi = train_vectors(sentences, epochs=5, verbose=self.verbose)
            return vecs, w2i, i2w

        data = np.load(path, allow_pickle=True)
        # try common key names
        if 'vectors' in data:
            vectors = data['vectors']
        elif 'word_vectors' in data:
            vectors = data['word_vectors']
        else:
            # try first array
            arrays = [k for k in data.files if isinstance(data[k], np.ndarray) and data[k].ndim == 2]
            if arrays:
                vectors = data[arrays[0]]
            else:
                raise ValueError(f"No suitable vectors array found in {path}")

        # word2idx / idx2word handling
        if 'word2idx' in data:
            word2idx = dict(data['word2idx'].item()) if data['word2idx'].dtype == object else dict(data['word2idx'])
            idx2word = {int(k): v for k, v in data.get('idx2word', {}).item()} if 'idx2word' in data else {i: w for w, i in word2idx.items()}
        elif 'idx2word' in data:
            idx2word = dict(data['idx2word'].item()) if data['idx2word'].dtype == object else dict(data['idx2word'])
            word2idx = {v: int(k) for k, v in idx2word.items()}
        else:
            # try to find a mapping stored as object
            word2idx = {}
            idx2word = {}

        return vectors, word2idx, idx2word

    # ---------------- nonterm centroids ---------------
    def _compute_nonterm_centroids(self):
        nonterms = self.pcfg.get('nonterm_expansions', {})
        for nt, expansion in nonterms.items():
            # expansion is list of tokens
            vecs = []
            for tok in expansion:
                if tok in self.word2idx:
                    vecs.append(self.vectors[self.word2idx[tok]])
            if vecs:
                centroid = np.mean(np.stack(vecs, axis=0), axis=0)
                centroid = normalize(centroid)
                self.nonterm_vectors[nt] = centroid
            else:
                # fallback: random small vector
                self.nonterm_vectors[nt] = normalize(np.random.randn(self.vectors.shape[1]).astype(np.float32))

        if self.verbose:
            print(f"Computed centroids for {len(self.nonterm_vectors)} nonterminals")

    # ---------------- build nonterm vectors for @BIN_* and other LHS ------
    def _build_all_nonterm_vectors(self):
        # Build vectors for all LHS in rules (including @BIN_*) by
        # selecting the most frequent RHS and composing its children.
        rules = self.pcfg.get('rules', {})
        memo: Dict[str, np.ndarray] = dict(self.nonterm_vectors)  # seed with Xn centroids

        def resolve(sym: str, visiting: set[str], depth: int = 0) -> np.ndarray:
            # stop recursion if too deep
            if sym in memo:
                return memo[sym]
            if depth > 50:
                # fallback: random vector
                v = normalize(np.random.randn(self.vectors.shape[1]).astype(np.float32))
                memo[sym] = v
                return v
            if sym in visiting:
                # cycle detected: fallback to random
                v = normalize(np.random.randn(self.vectors.shape[1]).astype(np.float32))
                memo[sym] = v
                return v

            # terminal word
            if sym in self.word2idx:
                v = self.vectors[self.word2idx[sym]]
                memo[sym] = v
                return v

            # if sym has explicit expansion recorded (Xn), use it
            if sym in self.pcfg.get('nonterm_expansions', {}):
                # centroid already in memo? check
                if sym in memo:
                    return memo[sym]
                toks = self.pcfg['nonterm_expansions'][sym]
                child_vecs = [resolve(t, visiting, depth + 1) for t in toks if isinstance(t, str)]
                if child_vecs:
                    v = normalize(np.mean(np.stack(child_vecs, axis=0), axis=0))
                else:
                    v = normalize(np.random.randn(self.vectors.shape[1]).astype(np.float32))
                memo[sym] = v
                return v

            # if we have rules for sym, pick the most frequent RHS
            if sym in rules:
                # select rhs with highest count
                rhss = rules[sym]
                if not rhss:
                    v = normalize(np.random.randn(self.vectors.shape[1]).astype(np.float32))
                    memo[sym] = v
                    return v
                best = max(rhss, key=lambda r: r.get('count', 0))
                rhs = best.get('rhs', [])

                # resolve children recursively
                visiting.add(sym)
                child_vecs = [resolve(r, visiting, depth + 1) for r in rhs]
                visiting.remove(sym)

                # compose children by projective_resonance left-to-right
                if child_vecs:
                    state = child_vecs[0]
                    for w in child_vecs[1:]:
                        state = projective_resonance(state, w)
                    state = normalize(state)
                else:
                    state = normalize(np.random.randn(self.vectors.shape[1]).astype(np.float32))

                memo[sym] = state
                return state

            # unknown symbol fallback
            v = normalize(np.random.randn(self.vectors.shape[1]).astype(np.float32))
            memo[sym] = v
            return v

        # resolve for all rule LHS
        for lhs in list(rules.keys()):
            _ = resolve(lhs, set())

        # update nonterm_vectors with memo entries that are nonterminals (not raw terminals)
        # consider nonterm keys as those appearing in rules keys
        for k, v in memo.items():
            if k not in self.word2idx:
                self.nonterm_vectors[k] = v

        if self.verbose:
            print(f"Built vectors for {len(self.nonterm_vectors)} nonterminals (including bin nodes)")

    # ---------------- spectra -----------------
    def _precompute_all_spectra(self):
        # build symbol list and corresponding vectors for persistence
        symbol_list: List[str] = []
        vec_list: List[np.ndarray] = []

        # terminals: prefer idx2word ordering if available
        if self.idx2word:
            vocab_size = max(self.idx2word.keys()) + 1
            for i in range(vocab_size):
                word = self.idx2word.get(i)
                if word and word in self.word2idx:
                    symbol_list.append(word)
                    vec_list.append(self.vectors[self.word2idx[word]])
        else:
            # fallback: include all vectors without names
            for i in range(self.vectors.shape[0]):
                symbol_list.append(f"__TERM_{i}")
                vec_list.append(self.vectors[i])

        # nonterms (including @BIN_* and Xn)
        for nt in sorted(self.nonterm_vectors.keys()):
            symbol_list.append(nt)
            vec_list.append(self.nonterm_vectors[nt])

        all_vecs = np.stack(vec_list, axis=0).astype(np.float32)
        self.spectra = precompute_spectra(all_vecs)
        self._symbol_list = symbol_list
        self._vectors_stack = all_vecs
        if self.verbose:
            print(f"Precomputed spectra for {all_vecs.shape[0]} vectors (terminals+nonterms)")

    # ---------------- state persistence -----------------
    def _save_state(self, path: str):
        # save symbol list (object), vectors stack and spectra
        np.savez(path,
                 symbols=np.array(self._symbol_list, dtype=object),
                 vectors=self._vectors_stack.astype(np.float32),
                 spectra=self.spectra.astype(np.float32))
        if self.verbose:
            print(f"Saved TreeComposer state to {path}")

    def _load_state(self, path: str):
        data = np.load(path, allow_pickle=True)
        symbols = list(data['symbols'])
        vectors = data['vectors']
        spectra = data['spectra']
        # reconstruct mappings
        self._symbol_list = symbols
        self._vectors_stack = vectors
        self.spectra = spectra

        # build nonterm_vectors from symbol list entries that are not terminals
        self.nonterm_vectors = {}
        for sym, vec in zip(symbols, vectors):
            if sym in self.word2idx:
                # terminal, skip
                continue
            self.nonterm_vectors[sym] = vec

    # ---------------- composition -----------------
    def compose(self, tree: Any, gamma: float = 1.0, bilateral: bool = False) -> np.ndarray:
        """Compose a tree into a D-dimensional VSA vector.

        Tree representation expected:
          - terminal: string token
          - nonterminal/internal: [label, child1, child2] or [child1, child2]
          - binary nodes preferred for this composer

        Returns normalized vector (numpy array)
        """
        # canonicalize to string key for memoization
        key = self._tree_hash(tree)
        if key in self._cache:
            return self._cache[key]

        # leaf
        if isinstance(tree, str):
            if tree in self.word2idx:
                v = self.vectors[self.word2idx[tree]]
            elif tree in self.nonterm_vectors:
                v = self.nonterm_vectors[tree]
            else:
                # unknown terminal: random vector
                v = normalize(np.random.randn(self.vectors.shape[1]).astype(np.float32))
            self._cache[key] = v
            return v

        # list/tuple: children
        # accept forms: [A, B] or ['X123', child] or ['A', B, C] (fold left-to-right)
        if isinstance(tree, (list, tuple)) and tree:
            # compute child vectors
            child_vecs = [self.compose(child, gamma=gamma, bilateral=bilateral) for child in tree]
            # fold left-to-right via projective_resonance
            state = child_vecs[0]
            for w in child_vecs[1:]:
                state = projective_resonance(state, w, gamma=gamma, bilateral=bilateral)
            state = normalize(state)
            self._cache[key] = state
            return state

        raise ValueError("Unsupported tree node type for compose")

    def _tree_hash(self, tree: Any) -> str:
        if isinstance(tree, str):
            return tree
        if isinstance(tree, (list, tuple)):
            return '(' + ' '.join(self._tree_hash(t) for t in tree) + ')'
        return str(tree)


def _quick_test():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pcfg_path = os.path.join(base, 'pcfg_induced.json')
    composer = TreeComposer(pcfg_path=pcfg_path)

    # pick one induced nonterm and compose its expansion
    nonterms = composer.pcfg.get('nonterm_expansions', {})
    some = next(iter(nonterms.items()))
    nt, expansion = some
    print(f"Sample nonterm: {nt} -> {' '.join(expansion)}")
    tree_vec = composer.compose(expansion)
    print(f"Composed vector norm: {float(np.linalg.norm(tree_vec))}")


if __name__ == '__main__':
    _quick_test()
