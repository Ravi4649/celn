"""
GFLinearizer — Bottom-up Tree Linearizer inspirado em Grammatical Framework
=============================================================================

Three phases:
  1. Build Edges: para cada par de content words, edge_score via
     PairGraph (transição existe?) × TypeAlign (compatibilidade sintática)
  2. Tree Build: combina bottom-up os pares com maior edge_score
     usando projective_resonance
  3. Linearize: percorre a árvore in-order → plano textual ordenado

O GF usa regras manuais de linearização (escritas por linguistas).
O GFLinearizer usa dados do corpus (PairGraph + Type Field) para
que a ordem EMERJA das transições canônicas, não de regras fixas.

Princípios: sem backprop, sem transformer, sem similaridade em 10k,
sem lista fixa, sem template, sem threshold mágico. Universal.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .core import normalize, projective_resonance as M


@dataclass
class TreeNode:
    """Nó da árvore de conteúdo construída bottom-up."""
    word: str | None = None
    idx: int = -1
    state: np.ndarray | None = None
    left: TreeNode | None = None
    right: TreeNode | None = None
    is_leaf: bool = True


class GFLinearizer:
    """Linearizador bottom-up ao estilo GF, emergente dos dados.

    Dado um conjunto de palavras de conteúdo, o linearizador:
      1. Constrói arestas entre pares via PairGraph (transições atestadas)
         + Type Field (compatibilidade sintática)
      2. Constrói árvore binária bottom-up combinando pares com maior score
      3. Lineariza a árvore (in-order traversal) → plano textual ordenado

    A ordem EMERGE do PairGraph e Type Field, não de regras manuais.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        w2i: dict[str, int],
        pair_graph: Any,
        type_field: np.ndarray | None = None,
    ):
        self.vectors = vectors.astype(np.float32)
        self.w2i = w2i
        self.i2w = {i: w for w, i in w2i.items()}
        self.pg = pair_graph
        self.type_field = type_field

    # ------------------------------------------------------------------
    # Phase 1: Build Edges
    # ------------------------------------------------------------------

    def _edge_score(self, w_a: str, w_b: str) -> float:
        """Score da transição w_a → w_b via PairGraph + TypeField.

        Retorna 0 se a transição NÃO existe no PairGraph.
        Se existe, score = TypeAlign (compatibilidade sintática).
        """
        idx_a = self.w2i.get(w_a)
        idx_b = self.w2i.get(w_b)
        if idx_a is None or idx_b is None:
            return 0.0

        followers = self.pg.get_followers(idx_a)
        if idx_b not in followers:
            return 0.0

        if self.type_field is not None:
            tf_a = self.type_field[idx_a]
            tf_b = self.type_field[idx_b]
            an = np.linalg.norm(tf_a)
            bn = np.linalg.norm(tf_b)
            if an > 1e-12 and bn > 1e-12:
                align = float(np.dot(tf_b / bn, tf_a / an))
                return float(np.tanh(max(align, 0.0)))
        return 1.0

    def build_edges(self, words: list[str]) -> dict[tuple[int, int], float]:
        """Calcula edge_score para cada par (i,j) onde i < j.

        Returns:
            edges: {(i, j): score} — score > 0 se PairGraph atesta transição i→j ou j→i
        """
        n = len(words)
        edges: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                # Check both directions
                s_ij = self._edge_score(words[i], words[j])
                s_ji = self._edge_score(words[j], words[i])
                if s_ij > 0:
                    edges[(i, j)] = s_ij
                if s_ji > 0:
                    edges[(j, i)] = s_ji
        return edges

    # ------------------------------------------------------------------
    # Phase 2: Tree Build (bottom-up)
    # ------------------------------------------------------------------

    def build_tree(
        self, words: list[str], edges: dict[tuple[int, int], float]
    ) -> list[TreeNode]:
        """Constrói árvore binária bottom-up.

        A cada iteração, encontra o par (i,j) com maior edge_score,
        combina os dois nós via projective_resonance, e substitui
        os dois nós pelo nó combinado.

        Ao final, retorna TODAS as raízes (nós que não foram combinados).
        Se nenhum par teve aresta, cada palavra vira uma raiz folha isolada.

        Args:
            words: Lista de palavras de conteúdo.
            edges: Dict {(i,j): score} com transições atestadas.

        Returns:
            Lista de raízes (árvores binárias), uma para cada cluster.
        """
        n = len(words)
        nodes: list[TreeNode | None] = [
            TreeNode(word=w, idx=i, state=normalize(self.vectors[self.w2i[w]].copy()))
            if w in self.w2i else None
            for i, w in enumerate(words)
        ]

        active = set(range(n))
        remaining = dict(edges)

        while len(active) > 1 and remaining:
            best_pair = max(
                ((i, j) for (i, j) in remaining if i in active and j in active),
                key=lambda ij: remaining[ij],
                default=None,
            )
            if best_pair is None:
                break

            i, j = best_pair
            if remaining[best_pair] <= 0:
                break

            left_node = nodes[i]
            right_node = nodes[j]
            if left_node is None or right_node is None:
                break
            if left_node.state is None or right_node.state is None:
                break

            combined_state = M(
                left_node.state,
                self.vectors[self.w2i[words[j]]].astype(np.float32),
                gamma=1.0, bilateral=True,
            )

            merged = TreeNode(
                word=None, idx=-1,
                state=normalize(combined_state),
                left=left_node, right=right_node,
                is_leaf=False,
            )

            nodes[i] = merged
            nodes[j] = None
            active.remove(j)

            for (a, b) in list(remaining.keys()):
                if a == j or b == j:
                    del remaining[(a, b)]

        return [nodes[i] for i in active if nodes[i] is not None]

    # ------------------------------------------------------------------
    # Phase 3: Linearize (in-order traversal)
    # ------------------------------------------------------------------

    def _inorder(self, node: TreeNode | None) -> list[str]:
        """Percorre a árvore in-order: left → current → right.

        In-order produz uma sequência linear de palavras de conteúdo
        na ordem canônica do corpus (preservando a estrutura da PairGraph).
        """
        if node is None:
            return []
        if node.is_leaf and node.word is not None:
            return [node.word]
        return self._inorder(node.left) + self._inorder(node.right)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def linearize(self, words: list[str]) -> list[str]:
        """Pipeline completo: edges → tree → inorder plan.

        Args:
            words: Lista de palavras de conteúdo (não ordenada).

        Returns:
            Lista ordenada de palavras de conteúdo.
            - Se há arestas, as palavras são reordenadas por ordem canônica
              (transições atestadas pelo PairGraph)
            - Se não há arestas, preserva a ordem original
            - Nós sem conexão são preservados na ordem original
        """
        if len(words) <= 1:
            return words

        # Phase 1: Build edges
        edges = self.build_edges(words)

        if not edges:
            return words

        # Phase 2: Build tree → multiple roots (one per cluster)
        roots = self.build_tree(words, edges)

        if not roots:
            return words

        # Phase 3: In-order traversal of each root, preserving original
        # indices to maintain relative order of disconnected clusters
        # Build (position, inorder_words) for each root
        root_indices = []
        for root in roots:
            seq = self._inorder(root)
            if seq:
                # Find position of first word in original list
                first_w = seq[0]
                pos = words.index(first_w) if first_w in words else len(words)
                root_indices.append((pos, seq))

        # Sort by position and flatten
        root_indices.sort(key=lambda x: x[0])
        result = []
        seen = set()
        for _, seq in root_indices:
            for w in seq:
                if w not in seen:
                    result.append(w)
                    seen.add(w)

        return result

    @staticmethod
    def _index_of(lst: list[str], item: str) -> int:
        """Índice de item na lista (para depuração)."""
        try:
            return lst.index(item)
        except ValueError:
            return -1
