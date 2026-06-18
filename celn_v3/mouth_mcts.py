"""
MCTS engine that searches PCFG trees guided by a target VSA state M.

Algorithm (implemented):
- Nodes represent partial trees with pending nonterminals to expand
- Selection uses UCT: Q + C * P * sqrt(N_parent) / (1 + N_child)
- Expansion uses progressive widening: top-K actions where K ~ sqrt(N_parent)
- Rollouts: two-stage evaluation
  1) fast: random projection to d=1024 and approximate composition
  2) exact: full D=10000 composition via TreeComposer for top candidates
- Reward: cosine similarity between composed vector and target M
- Backup: average reward (W/N), visits, max

This is a CPU-first, no-backprop approach compatible with CELN principles.
"""

from __future__ import annotations

import math
import time
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .core import similarity, D, encode_sequence, projective_resonance
from .tree_composer import TreeComposer


@dataclass
class MCTSNode:
    tree: Any  # partial tree representation (list/tuple/str)
    pending: List[Any]  # list of nonterminals to expand (in-order)
    parent: Optional['MCTSNode'] = None
    children: Dict[Any, 'MCTSNode'] = field(default_factory=dict)
    visits: int = 0
    total_value: float = 0.0
    max_value: float = -float('inf')
    prior: float = 1.0

    def q(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0


class MouthMCTS:
    def __init__(self, pcfg: dict, composer: TreeComposer, verbose: bool = True, proj_d: int = 1024, calib_interval: int = 20):
        self.pcfg = pcfg
        self.composer = composer
        self.verbose = verbose

        # Precompute action lists for nonterminals from PCFG rules
        self.actions: Dict[str, List[List[str]]] = {}
        for lhs, rules in pcfg.get('rules', {}).items():
            # sort by prob descending
            sorted_rules = sorted(rules, key=lambda r: r.get('prob', 0.0), reverse=True)
            self.actions[lhs] = [r['rhs'] for r in sorted_rules]

        # Random projection for fast rollout (reduced-dim composition)
        self.proj_d = proj_d
        self._proj = np.random.randn(self.proj_d, D).astype(np.float32) / math.sqrt(self.proj_d)

        # dynamic calibration of exploration constant C
        self.rewards_history: List[float] = []
        self.calib_interval = calib_interval
        self._iter_since_calib = 0
        self._C = 1.0

    # ---------------- utilities ----------------
    def _project(self, v: np.ndarray) -> np.ndarray:
        return (self._proj @ v).astype(np.float32)

    def _approx_similarity(self, v_proj: np.ndarray, m_proj: np.ndarray) -> float:
        # cosine in projected space
        denom = (np.linalg.norm(v_proj) * np.linalg.norm(m_proj) + 1e-12)
        return float(np.dot(v_proj, m_proj) / denom)

    # ---------------- tree helpers ----------------
    def _is_complete(self, node: MCTSNode) -> bool:
        return len(node.pending) == 0

    def _expand_actions(self, lhs: str, limit: int) -> List[List[str]]:
        # progressive widening: return top-k actions where k = max(1, int(sqrt(limit)))
        all_actions = self.actions.get(lhs, [])
        if not all_actions:
            return []
        k = max(1, int(math.sqrt(limit)))
        return all_actions[:k]

    # ---------------- selection ----------------
    def _select(self, root: MCTSNode) -> MCTSNode:
        node = root
        while node.children and not self._is_complete(node):
            total_N = sum(child.visits for child in node.children.values())
            best_score = -float('inf')
            best_child = None
            # use dynamically calibrated C
            C = self._C
            for a, child in node.children.items():
                q = child.q()
                u = C * child.prior * math.sqrt(total_N) / (1 + child.visits)
                score = q + u
                if score > best_score:
                    best_score = score
                    best_child = child
            if best_child is None:
                break
            node = best_child
        return node

    # ---------------- expansion ----------------
    def _expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        if self._is_complete(node):
            return None
        # pick first pending nonterminal to expand
        lhs = node.pending[0]
        count_parent = node.visits if node.visits > 0 else 1
        candidate_rhs = self._expand_actions(lhs, count_parent)
        # create children for new actions not yet in node.children
        for rhs in candidate_rhs:
            rhs_key = tuple(rhs)
            if rhs_key in node.children:
                continue
            # build new partial tree: replace first pending with rhs tokens
            new_tree = self._replace_first_pending(node.tree, lhs, rhs)
            # build new pending list: replace lhs with rhs tokens that are nonterminals
            new_pending = list(rhs) + node.pending[1:]
            # child node
            child = MCTSNode(tree=new_tree, pending=new_pending, parent=node, prior=1.0)
            node.children[rhs_key] = child
            return child
        # if all candidate actions already expanded, return a random child
        if node.children:
            return random.choice(list(node.children.values()))
        return None

    def _replace_first_pending(self, tree: Any, lhs: str, rhs: List[str]) -> Any:
        # tree is only used as identity in this implementation; we keep tree as a flat list of tokens
        # build a new list replacing the first occurrence of lhs with rhs
        if isinstance(tree, list):
            new = []
            replaced = False
            for tok in tree:
                if not replaced and tok == lhs:
                    new.extend(rhs)
                    replaced = True
                else:
                    new.append(tok)
            return new
        # otherwise, fallback
        return tree

    # ---------------- rollout ----------------
    def _rollout_once(self, node: MCTSNode, m_state: np.ndarray, m_proj: np.ndarray, rollout_depth_limit: int = 50) -> Tuple[float, Any]:
        # simulate by fully expanding pending nonterminals probabilistically using PCFG probs
        # start from a shallow copy
        pending = list(node.pending)
        tokens = list(node.tree) if isinstance(node.tree, list) else [node.tree]

        steps = 0
        # simple probabilistic sampling from actions for each nonterminal encountered
        while pending and steps < rollout_depth_limit:
            sym = pending.pop(0)
            actions = self.actions.get(sym, [])
            if not actions:
                # treat as terminal
                tokens.append(sym)
                continue
            # sample according to prob if available
            # pcfg rules stored with probs; map actions back
            rules = self.pcfg['rules'].get(sym, [])
            if rules:
                probs = np.array([r.get('prob', 0.0) for r in rules], dtype=np.float32)
                if probs.sum() <= 0:
                    idx = 0
                else:
                    probs = probs / probs.sum()
                    idx = int(np.random.choice(len(probs), p=probs))
                chosen_rhs = rules[idx]['rhs']
            else:
                chosen_rhs = actions[0]

            # append RHS tokens to tokens list and pending (nonterminals)
            for tok in chosen_rhs:
                tokens.append(tok)
                if tok in self.actions:  # nonterminal
                    pending.insert(0, tok)
            steps += 1

        # Stage 1: improved fast rollout using reduced-dim projective_resonance
        # Project each vector down, then compose them via projective_resonance in reduced space
        approx_vecs = []
        for t in tokens:
            if t in self.composer.nonterm_vectors:
                v_full = self.composer.nonterm_vectors[t]
            elif t in self.composer.word2idx:
                v_full = self.composer.vectors[self.composer.word2idx[t]]
            else:
                v_full = np.random.randn(D).astype(np.float32)
            approx_vecs.append(self._project(v_full))

        # compose left-to-right in projected space using projective_resonance approximation
        # we emulate M by applying projective_resonance on projected vectors padded to D via inverse projection
        # Simpler approach: perform pairwise circular convolution in reduced dim via FFT on projected vectors
        # Use np.fft.fft/ifft on projected vectors (real) to get an approximate composed vector
        def compose_reduced(vecs_proj: List[np.ndarray]) -> np.ndarray:
            if not vecs_proj:
                return np.zeros(self.proj_d, dtype=np.float32)
            state = vecs_proj[0].astype(np.float32)
            for w in vecs_proj[1:]:
                # circular convolution approximation in reduced dim
                S = np.fft.fft(state)
                W = np.fft.fft(w.astype(np.float32))
                state = np.fft.ifft(S * W).real.astype(np.float32)
                # normalize
                norm = np.linalg.norm(state)
                if norm > 0:
                    state = state / norm
            return state

        v_proj_composed = compose_reduced(approx_vecs)
        sim_fast = self._approx_similarity(v_proj_composed, m_proj)

        # return both fast score and tokens for exact stage if selected later
        return sim_fast, tokens

    def _evaluate_exact(self, tokens: List[str], m_state: np.ndarray) -> float:
        # compose using TreeComposer (treat tokens as flat left-to-right sequence)
        # composer.compose expects trees; we use flat list
        vec = self.composer.compose(tokens)
        return similarity(vec, m_state)

    # ---------------- backup ----------------
    def _backup(self, node: MCTSNode, value: float):
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.total_value += value
            cur.max_value = max(cur.max_value, value)
            cur = cur.parent
        # record reward for calibration
        self.rewards_history.append(value)
        self._iter_since_calib += 1
        if self._iter_since_calib >= self.calib_interval:
            self._calibrate_C()
            self._iter_since_calib = 0

    def _calibrate_C(self):
        # set C to interquartile range (IQR) of last rewards (or fallback)
        if not self.rewards_history:
            self._C = 1.0
            return
        arr = np.array(self.rewards_history, dtype=np.float32)
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = float(q75 - q25)
        # scale C to be proportional to IQR, with fallback
        self._C = max(0.1, min(10.0, iqr * 10.0))
        if self.verbose:
            print(f"Calibrated C to {self._C:.4f} using IQR={iqr:.6f}")

    # ---------------- main generate API ----------------
    def generate(self, m_state: np.ndarray, budget: int = 200, top_k_exact: int = 5) -> Tuple[Any, float]:
        """Run MCTS guided by m_state and return best found tree and its similarity.

        budget: number of rollouts (fast stage). For each rollout we may run exact eval
                for a small number of top candidates.
        top_k_exact: number of candidates from fast stage to evaluate exactly per rollout.
        """
        # root: initial tree is start symbol expansion simple list with 'S' pending
        root = MCTSNode(tree=['S'], pending=['S'])

        # precompute projected target
        m_proj = self._project(m_state)

        best_tree = None
        best_score = -1.0

        # store candidate tokens from fast stage to later exact-evaluate top ones
        fast_pool: List[Tuple[float, List[str]]] = []

        for it in range(budget):
            # selection
            node = self._select(root)
            # expansion
            child = self._expand(node)
            if child is None:
                child = node

            # rollout (fast)
            sim_fast, tokens = self._rollout_once(child, m_state, m_proj)
            fast_pool.append((sim_fast, tokens))

            # occasional exact evaluation for top fast candidates so far
            if (it + 1) % max(1, budget // top_k_exact) == 0:
                # pick top candidates from fast_pool
                fast_pool.sort(key=lambda x: x[0], reverse=True)
                for sim_f, toks in fast_pool[:top_k_exact]:
                    # ensure bins resolved before exact evaluation
                    toks_resolved = self._resolve_bins_in_tokens(toks)
                    sim_exact = self._evaluate_exact(toks_resolved, m_state)
                    # backup using exact value
                    self._backup(child, sim_exact)
                    if sim_exact > best_score:
                        best_score = sim_exact
                        best_tree = toks_resolved
                # keep pool bounded
                fast_pool = fast_pool[:top_k_exact * 2]
            else:
                # backup using fast approx (cheap)
                self._backup(child, sim_fast)

        # final exact evaluation of best candidates
        if fast_pool:
            fast_pool.sort(key=lambda x: x[0], reverse=True)
            for sim_f, toks in fast_pool[:top_k_exact]:
                toks_resolved = self._resolve_bins_in_tokens(toks)
                sim_exact = self._evaluate_exact(toks_resolved, m_state)
                if sim_exact > best_score:
                    best_score = sim_exact
                    best_tree = toks_resolved

        return best_tree, best_score

    def _resolve_bins_in_tokens(self, tokens: List[str]) -> List[str]:
        # Recursively expand any @BIN_* or nonterminal tokens using the most probable PCFG rule
        resolved = []
        stack = list(tokens)
        depth = 0
        max_depth = 200
        while stack and depth < max_depth:
            tok = stack.pop(0)
            if tok in self.actions:
                # expand using most probable rule
                rules = self.pcfg['rules'].get(tok, [])
                if rules:
                    best = max(rules, key=lambda r: r.get('prob', 0.0))
                    rhs = best.get('rhs', [])
                    # place rhs at front to resolve depth-first
                    stack = list(rhs) + stack
                else:
                    # no rule: treat as terminal
                    resolved.append(tok)
            else:
                resolved.append(tok)
            depth += 1

        # if exceeded depth, append remaining tokens as-is
        if stack:
            resolved.extend(stack)
        return resolved
