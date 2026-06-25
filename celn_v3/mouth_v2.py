"""
CELN v3 — Mouth v2 (Reconstrução Atencional)
===============================================
Geração guiada por TRÊS scores competitivos:
  syn_score  = PairGraph: fluência da transição
  sem_score  = GHRR atenção: match com conteúdo não expresso
  fidelity   = GHRR atenção: sequência ainda reflete o pensamento?

O vetor THOUGHT é CONGELADO — nunca sofre unbind.
A geração RECONSTRÓI o pensamento, não o decompõe.

Princípios:
  ZERO backprop/transformers. ZERO listas fixas. ZERO templates.
  ZERO thresholds mágicos. Tudo auto-calibrável.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .core import D, normalize, similarity
from .decomposer import Decomposer
from .ghrr_core import (
    D as GHRR_D, M as GHRR_M,
    vec_10k_to_ghrr, bulk_10k_to_ghrr,
    ghrr_attention_score, ghrr_encode_sequence,
    ghrr_similarity,
)
from .logic_encoder import LogicRoles
from .pair_graph import PairGraph
from .linearizer import linearize

ROLE_TO_WORD = {
    'ROLE_TODOS': {'todo', 'toda', 'todos', 'todas'},
    'ROLE_NENHUM': {'nenhum', 'nenhuma', 'nenhuns', 'nenhumas'},
    'ROLE_ALGUM': {'algum', 'alguma', 'alguns', 'algumas'},
    'ROLE_SE_ENTAO': {'se'},
    'ROLE_NEGACAO': {'não'},
}


@dataclass
class StepScore:
    word: str
    idx: int
    syn: float
    sem: float
    fdel: float
    total: float


@dataclass
class GenResult:
    sentence: str
    steps: List[StepScore] = field(default_factory=list)
    fidelity: float = 0.0
    n_candidates_evaluated: int = 0
    content_expressed: int = 0
    content_total: int = 0


class MouthV2:
    """
    Geração por reconstrução atencional (3 scores competitivos).

    Args:
        codebook: Matriz (V, 10000) de vetores normalizados
        w2i, i2w: Mapeamentos
        pair_graph: PairGraph para syn_score (opcional)
        roles: LogicRoles
    """

    def __init__(
        self,
        codebook: np.ndarray,
        w2i: Dict[str, int],
        i2w: Dict[int, str],
        pair_graph: Optional[PairGraph] = None,
        roles: Optional[LogicRoles] = None,
    ):
        self.codebook = codebook.astype(np.float32)
        self.w2i = w2i
        self.i2w = i2w
        self.V = len(w2i)
        self.pair_graph = pair_graph
        self.roles = roles or LogicRoles(seed=42)

        self._vocab_indices = np.arange(self.V, dtype=np.int32)
        self._all_words = [i2w.get(i, '?') for i in range(self.V)]

        # GHRR: converte sob demanda (cache LRU pequeno)
        self._ghrr_cache: Dict[int, np.ndarray] = {}

        # Decomposer
        self.decomposer = Decomposer(
            codebook, w2i, i2w, roles=self.roles,
        )

    # ────────────────────────────────────────────────────────────────
    # Content set extraction
    # ────────────────────────────────────────────────────────────────

    def _extract_content(self, composite: np.ndarray) -> Optional[dict]:
        result = self.decomposer.decompose(composite)
        if result['method'] == 'failed' or result['role'] is None:
            return None

        content = {
            'role_name': result['role'],
            'ant_word': result['ant'],
            'cons_word': result['cons'],
            'ant_vec': result['ant_vec'],
            'cons_vec': result['cons_vec'],
        }
        role_vec = self.roles.get(result['role'])
        content['role_vec'] = role_vec
        content['role_ghrr'] = vec_10k_to_ghrr(role_vec)
        return content

    # ────────────────────────────────────────────────────────────────
    # GHRR lazy
    # ────────────────────────────────────────────────────────────────

    def _ghrr(self, idx: int) -> np.ndarray:
        """Retorna GHRR (D, M, M) para word_idx, com cache."""
        if idx not in self._ghrr_cache:
            if idx >= self.V:
                return np.zeros((GHRR_D, GHRR_M, GHRR_M), dtype=np.float32)
            self._ghrr_cache[idx] = vec_10k_to_ghrr(self.codebook[idx])
        return self._ghrr_cache[idx]

    def _ghrr_seq(self, idxs: List[int]) -> np.ndarray:
        """Codifica sequência de índices em GHRR."""
        vecs = [self._ghrr(i) for i in idxs]
        return ghrr_encode_sequence(vecs)

    # ────────────────────────────────────────────────────────────────
    # Scores
    # ────────────────────────────────────────────────────────────────

    def _syn_score(self, word_idx: int, last_idx: Optional[int]) -> float:
        """syn_score: probabilidade de transição no PairGraph."""
        if last_idx is None or self.pair_graph is None:
            return 0.5  # neutro para primeira palavra
        followers = self.pair_graph.get_followers(last_idx)
        if not followers:
            return 0.3
        if word_idx in followers:
            rank = followers.index(word_idx)
            return 1.0 / (1.0 + rank)
        return 0.0

    def _sem_score(
        self, word_ghrr: np.ndarray,
        unexpressed: List[np.ndarray],
    ) -> float:
        """sem_score: GHRR atenção entre w e conteúdo não expresso."""
        if not unexpressed:
            return 0.0
        scores = [
            ghrr_attention_score(v, word_ghrr, temperature=0.3)
            for v in unexpressed
        ]
        return float(max(scores))

    def _fidelity_score(
        self, thought_ghrr: np.ndarray,
        sequence_ghrr: np.ndarray,
    ) -> float:
        """fidelity_score: GHRR atenção entre pensamento e sequência."""
        if sequence_ghrr is None:
            return 0.0
        return float(
            ghrr_attention_score(thought_ghrr, sequence_ghrr, temperature=0.3)
        )

    # ────────────────────────────────────────────────────────────────
    # Geração
    # ────────────────────────────────────────────────────────────────

    def _first_candidates(
        self, content: dict, top_k: int = 50,
    ) -> List[Tuple[int, float]]:
        """Candidatos para primeira palavra: próximos ao ant_vec + forçados."""
        v_query = normalize(content['ant_vec'])
        sims = self.codebook @ v_query
        order = np.argsort(sims)[::-1]
        result = []
        seen = set()
        # Garante que ant e cons estão no topo
        for forced in [content['ant_word'], content['cons_word']]:
            idx = self.w2i.get(forced)
            if idx is not None:
                result.append((int(idx), float(sims[idx])))
                seen.add(idx)
        # Preenche com vizinhos até top_k
        for idx in order:
            idx = int(idx)
            if idx in seen:
                continue
            result.append((idx, float(sims[idx])))
            seen.add(idx)
            if len(result) >= top_k:
                break
        return result

    def _next_candidates(
        self, last_idx: int, top_k: int = 20,
        forced_idxs: Optional[List[int]] = None,
    ) -> List[int]:
        """Candidatos: seguidores PairGraph + forçados (ant, cons)."""
        result = []
        seen = set()
        # Palavras forçadas sempre incluídas
        if forced_idxs:
            for fi in forced_idxs:
                if fi is not None and fi not in seen:
                    result.append(fi)
                    seen.add(fi)
        # Seguidores PairGraph
        if self.pair_graph is not None:
            followers = self.pair_graph.get_followers(last_idx, top_k=top_k)
            for fi in followers:
                if fi not in seen:
                    result.append(fi)
                    seen.add(fi)
                    if len(result) >= top_k:
                        break
        if not result:
            # Fallback: vizinhos
            v = normalize(self.codebook[last_idx])
            sims = self.codebook @ v
            order = np.argsort(sims)[::-1]
            result = [int(i) for i in order[1:top_k + 1] if int(i) not in seen]
        return result[:top_k]

    def _is_expressed(
        self, word: str, content: dict, expressed: set,
    ) -> bool:
        """Verifica se word expressa algum elemento do content_set."""
        if word == content.get('ant_word'):
            expressed.add('A')
            return True
        if word == content.get('cons_word'):
            expressed.add('B')
            return True
        role_words = ROLE_TO_WORD.get(content.get('role_name', ''), set())
        if word.lower() in role_words:
            expressed.add('ROLE')
            return True
        return False

    def generate(
        self,
        composite: np.ndarray,
        max_steps: int = 15,
        alpha: float = 0.25,
        beta: float = 0.25,
        gamma: float = 0.25,
        content_boost: float = 0.25,
        top_k_first: int = 50,
        top_k_next: int = 20,
        verbose: bool = False,
    ) -> GenResult:
        """
        Gera frase por reconstrução atencional.

        Args:
            composite: Vetor composto (encode_rule)
            max_steps: Passos máximos
            alpha: Peso syn_score
            beta: Peso sem_score
            gamma: Peso fidelity_score
            top_k_first: Candidatos para primeira palavra
            top_k_next: Candidatos para passos seguintes

        Returns:
            GenResult com steps e fidelidade final
        """
        # ── 1. Extrai conteúdo ──
        content = self._extract_content(composite)
        if content is None:
            return GenResult(sentence='', fidelity=0.0)

        thought_ghrr = vec_10k_to_ghrr(normalize(composite))
        content_vecs_ghrr = {
            'ROLE': content['role_ghrr'],
            'A': vec_10k_to_ghrr(content['ant_vec']),
            'B': vec_10k_to_ghrr(content['cons_vec']),
        }

        unexpressed_vecs = list(content_vecs_ghrr.values())
        expressed: set = set()
        spoken: List[int] = []
        steps: List[StepScore] = []
        total_candidates = 0

        if verbose:
            print(f"[MouthV2] Conteúdo: {content['role_name']}"
                  f"({content['ant_word']} → {content['cons_word']})")

        for step in range(max_steps):
            # Critério de parada: ant e cons expressos (conteúdo semântico completo)
            if 'A' in expressed and 'B' in expressed:
                if verbose:
                    print(f"  [parada] Conteúdo semântico completo em {step} passos ({len(expressed)}/3)")
                break

            # Candidatos
            if step == 0:
                candidates = self._first_candidates(content, top_k=top_k_first)
            else:
                forced = []
                if 'A' not in expressed:
                    forced.append(content['ant_word'])
                if 'B' not in expressed:
                    forced.append(content['cons_word'])
                forced_idxs = [self.w2i.get(w) for w in forced if w is not None]
                follower_idxs = self._next_candidates(
                    spoken[-1], top_k=top_k_next,
                    forced_idxs=[i for i in forced_idxs if i is not None],
                )
                candidates = [(idx, 0.0) for idx in follower_idxs]

            if not candidates:
                break

            if not candidates:
                if verbose:
                    print(f"  [parada] Sem candidatos no passo {step}")
                break

            # Conteúdo ainda não expresso para sem_score
            remaining = [
                content_vecs_ghrr[k] for k in ['ROLE', 'A', 'B']
                if k not in expressed
            ]

            # Avalia cada candidato
            best_word_idx = -1
            best_total = -np.inf
            best_scores = None

            forced_idxs_set = set()
            if step > 0:
                for w in [content['ant_word'], content['cons_word']]:
                    idx = self.w2i.get(w)
                    if idx is not None:
                        forced_idxs_set.add(idx)

            for cand_idx, _ in candidates:
                cand_idx = int(cand_idx)
                if cand_idx >= self.V:
                    continue

                syn = self._syn_score(cand_idx, spoken[-1] if spoken else None)
                sem = self._sem_score(self._ghrr(cand_idx), remaining)

                trial_idxs = spoken + [cand_idx]
                trial_ghrr = self._ghrr_seq(trial_idxs)
                fdel = self._fidelity_score(thought_ghrr, trial_ghrr)

                cb = 0.0
                if cand_idx in forced_idxs_set:
                    cand_word = self.i2w.get(cand_idx, '')
                    needs_boost = (
                        ('A' not in expressed and cand_word == content['ant_word']) or
                        ('B' not in expressed and cand_word == content['cons_word'])
                    )
                    if needs_boost:
                        cb = content_boost

                total = alpha * syn + beta * sem + gamma * fdel + cb
                total_candidates += 1

                if total > best_total:
                    best_total = total
                    best_word_idx = cand_idx
                    best_scores = (syn, sem, fdel)

            if best_word_idx < 0:
                if verbose:
                    print(f"  [parada] Nenhum candidato válido no passo {step}")
                break

            # Atualiza estado
            spoken.append(best_word_idx)
            best_word = self.i2w.get(best_word_idx, '?')
            syn, sem, fdel = best_scores

            steps.append(StepScore(
                word=best_word, idx=best_word_idx,
                syn=syn, sem=sem, fdel=fdel, total=best_total,
            ))

            # Marca conteúdo expresso
            self._is_expressed(best_word, content, expressed)

            if verbose:
                rem = 3 - len(expressed)
                print(f"  [{step}] '{best_word}' syn={syn:.3f} sem={sem:.3f} "
                      f"fid={fdel:.3f} total={best_total:.3f} "
                      f"restam={rem}")

        # ── Fidelidade final ──
        if spoken:
            final_ghrr = self._ghrr_seq(spoken)
            fidelity = ghrr_attention_score(thought_ghrr, final_ghrr, temperature=0.3)
        else:
            fidelity = 0.0

        # ── Lineariza ──
        words = [self.i2w.get(i, '?') for i in spoken]
        sentence = linearize(
            words, content['role_name'],
            ant=content['ant_word'], cons=content['cons_word'],
            capitalize=True, add_period=True,
        )

        return GenResult(
            sentence=sentence,
            steps=steps,
            fidelity=float(fidelity),
            n_candidates_evaluated=total_candidates,
            content_expressed=len(expressed),
            content_total=3,
        )

    def generate_from_components(
        self, role_name: str, ant_word: str, cons_word: str,
        **kwargs,
    ) -> GenResult:
        """Gera a partir de (role, ant, cons)."""
        ant_idx = self.w2i.get(ant_word)
        cons_idx = self.w2i.get(cons_word)
        if ant_idx is None or cons_idx is None:
            return GenResult(sentence='', fidelity=0.0)

        from .logic_encoder import encode_rule
        v_ant = normalize(self.codebook[ant_idx])
        v_cons = normalize(self.codebook[cons_idx])
        composite = encode_rule(self.roles.get(role_name), v_ant, v_cons)
        return self.generate(composite, **kwargs)
