"""
CELN v3 — Lexicalizer (Holographic Beam Search)
=================================================
Transforma vetores compostos em sequências de palavras via beam search
guiado por decomposição reversa.

Pipeline:
  1. decode_rule → (role, ant, cons) — conteúdo semântico
  2. Constrói TARGET como encoding sequencial do pensamento
  3. Beam search: cada candidato é unbind-M do residual
     - Norma alta do unbind = candidato está no pensamento
     - PairGraph + Type Field filtram (não geram)
  4. Múltiplos beams mantêm caminhos alternativos
  5. Melhor frase = maior similaridade ao target

Sem backprop, sem templates, sem listas fixas.
Ordem emerge da decomposição, não de S→V→O.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

from .core import D, normalize, similarity, encode_sequence
from .logic_encoder import LogicRoles, decode_rule, encode_rule
from .pair_graph import PairGraph


@dataclass
class Beam:
    path_indices: List[int] = field(default_factory=list)
    path_words: List[str] = field(default_factory=list)
    residual: Optional[np.ndarray] = None
    score: float = 0.0
    norms: List[float] = field(default_factory=list)

    def copy(self) -> 'Beam':
        return Beam(
            path_indices=list(self.path_indices),
            path_words=list(self.path_words),
            residual=self.residual.copy() if self.residual is not None else None,
            score=self.score,
            norms=list(self.norms),
        )


@dataclass
class GenerationStep:
    word: str
    idx: int
    score: float
    norm_after_unbind: float
    transition_valid: bool


class Lexicalizer:
    """
    Holographic Beam Search: gera frases decompondo o vetor composto.

    A cada passo:
      1. Para cada candidato do PairGraph, unbind_M_forward(residual, vec)
      2. Mede a norma do unbind — alta = candidato explicita o pensamento
      3. Mantém top-K beams
      4. Para quando residual ≈ zero (toda informação foi extraída)

    Args:
        codebook: Matriz (V, D) de vetores normalizados
        w2i, i2w: mapeamentos
        pair_graph: PairGraph para restrições de transição (opcional)
        type_field: Type Field (V, H) HDC (opcional)
        type_vecs: Type vectors (V, H) HDC (opcional)
        roles: LogicRoles (opcional)
    """

    def __init__(
        self,
        codebook: np.ndarray,
        w2i: Dict[str, int],
        i2w: Dict[int, str],
        pair_graph: Optional[PairGraph] = None,
        type_field: Optional[np.ndarray] = None,
        type_vecs: Optional[np.ndarray] = None,
        roles: Optional[LogicRoles] = None,
    ):
        self.codebook = codebook.astype(np.float32)
        self.w2i = w2i
        self.i2w = i2w
        self.V = len(w2i)
        self.pair_graph = pair_graph
        self.type_field = type_field.astype(np.float32) if type_field is not None else None
        self.type_vecs = type_vecs.astype(np.float32) if type_vecs is not None else None
        self.roles = roles or LogicRoles(seed=42)

    # ------------------------------------------------------------------
    # Build target: encode_sequence ao contrário (reverso)
    # ------------------------------------------------------------------

    def _build_target(self, ant_idx: int, cons_idx: int,
                      role_idx: Optional[int] = None) -> np.ndarray:
        """Constrói target: encoding sequencial do pensamento."""
        v_ant = normalize(self.codebook[ant_idx])
        v_cons = normalize(self.codebook[cons_idx])
        if role_idx is not None and role_idx < len(self._role_vectors):
            v_role = self._role_vectors[role_idx]
            return encode_sequence([v_role, v_ant, v_cons])
        return encode_sequence([v_ant, v_cons])

    def _score_by_similarity(self, path_indices: List[int],
                              target: np.ndarray) -> float:
        """Score: similaridade entre encoding do caminho e target."""
        if len(path_indices) < 1:
            return 0.0
        path_vecs = [normalize(self.codebook[i]) for i in path_indices
                     if i < self.V]
        if not path_vecs:
            return 0.0
        if len(path_vecs) == 1:
            return float(path_vecs[0] @ target)
        path_encoded = encode_sequence(path_vecs)
        return float(path_encoded @ target)

    # ------------------------------------------------------------------
    # Scoring: quanto o candidato explicita o residual?
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Beam Search
    # ------------------------------------------------------------------

    def generate(
        self,
        composite: np.ndarray,
        max_steps: int = 12,
        beam_width: int = 5,
        n_candidates: int = 8,
        verbose: bool = False,
    ) -> List[Beam]:
        """
        Holographic Beam Search: gera frases guiadas pelo pensamento.

        Fase 1 (decode_rule): extrai (role, ant, cons) do composto.
        Fase 2 (target): encoding sequencial do pensamento completo.
        Fase 3 (beam): a cada passo, candidatos PairGraph são testados
          por similaridade ao target. Quanto mais similar, melhor.

        Args:
            composite: Vetor composto da dedução
            max_steps: Passos máximos
            beam_width: Beams paralelos
            n_candidates: Candidatos por passo

        Returns:
            Beams ranqueados por score
        """
        # ── Fase 1: Extrai conteúdo ──
        role_name, ant_word, cons_word, meta = decode_rule(
            composite, self.roles, self.codebook, self.w2i, self.i2w,
        )
        if role_name is None or ant_word is None or cons_word is None:
            return []

        ant_idx = self.w2i.get(ant_word)
        cons_idx = self.w2i.get(cons_word)
        if ant_idx is None or cons_idx is None:
            return []

        role_idx = None
        for i, rn in enumerate(self.roles.ROLE_NAMES):
            if rn == role_name:
                role_idx = i
                break
        self._role_vectors = [
            self.roles.get(rn) for rn in self.roles.ROLE_NAMES
        ]

        if verbose:
            print(f"[Lexicalizer] {role_name}({ant_word} → {cons_word})")

        # ── Fase 2: Target (pensamento) ──
        target = normalize(self._build_target(ant_idx, cons_idx, role_idx))

        # ── Fase 3: Beam search ──
        beams = [Beam(
            path_indices=[ant_idx],
            path_words=[ant_word],
            residual=target.copy(),
            score=self._score_by_similarity([ant_idx], target),
            norms=[1.0],
        )]

        for step in range(max_steps):
            new_beams = []

            for beam in beams:
                last_idx = beam.path_indices[-1]
                last_word = self.i2w.get(last_idx, '')

                if last_word == cons_word:
                    new_beams.append(beam)
                    continue

                # Candidatos do PairGraph
                successors = []
                if self.pair_graph is not None:
                    followers = self.pair_graph.get_followers(last_idx)
                    successors = [f for f in followers
                                  if f not in beam.path_indices]
                else:
                    v_last = normalize(self.codebook[last_idx])
                    sims = self.codebook @ v_last
                    order = np.argsort(sims)[::-1]
                    successors = [int(i) for i in order[:n_candidates * 2]
                                  if int(i) not in beam.path_indices]

                if not successors:
                    new_beams.append(beam)
                    continue

                # Expande beam com cada candidato
                for idx in successors[:n_candidates]:
                    new_idx_list = beam.path_indices + [idx]
                    sim_score = self._score_by_similarity(new_idx_list, target)

                    new_beam = beam.copy()
                    new_beam.path_indices.append(idx)
                    new_beam.path_words.append(self.i2w.get(idx, '?'))
                    new_beam.score = sim_score
                    new_beam.norms.append(sim_score)
                    new_beam.residual = target.copy()
                    new_beams.append(new_beam)

            if not new_beams:
                break

            # Top-K beams por similaridade ao target
            beams = sorted(new_beams, key=lambda b: -b.score)[:beam_width]

            # Auto-calibração: para se convergiu
            if step > 0:
                scores = [b.score for b in beams]
                improvement = max(scores) - min(scores)
                if improvement < 1e-6 and step >= 3:
                    if verbose:
                        print(f"  [beam] Convergiu em {step+1} passos")
                    break

            if verbose:
                top = beams[0]
                print(f"  [step {step+1}] score={top.score:.4f} "
                      f"'{' → '.join(top.path_words[:4])}...'")

        # Garante que o alvo está na frase do melhor beam
        top = beams[0]
        if self.i2w.get(top.path_indices[-1], '') != cons_word:
            top.path_indices.append(cons_idx)
            top.path_words.append(cons_word)
            top.score = self._score_by_similarity(top.path_indices, target)

        return beams

    # ------------------------------------------------------------------
    # decode_and_generate (retorna frase do melhor beam)
    # ------------------------------------------------------------------

    def decode_and_generate(
        self,
        composite: np.ndarray,
        max_steps: int = 12,
    ) -> Tuple[str, float, List[Beam]]:
        """Gera frase do melhor beam."""
        beams = self.generate(composite, max_steps=max_steps)
        if not beams:
            return '', 0.0, []
        top = beams[0]
        sentence = ' '.join(top.path_words)
        return sentence, top.score, beams

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    def word_to_vector(self, word: str) -> Optional[np.ndarray]:
        idx = self.w2i.get(word)
        if idx is None:
            return None
        return normalize(self.codebook[idx].copy())
