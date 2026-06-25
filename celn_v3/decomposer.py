"""
CELN v3 — Decomposer
=====================
Decompõe vetores compostos em componentes estruturados.
Usa logic_encoder.decode_rule para regras FOL e resonator para
decomposição multi-fator.

Pipeline:
  1. decode_rule → (role, ant, cons)
  2. Se falhar, resonator decode_2factor ou decode_3factor
  3. Retorna estrutura hierárquica

Princípios:
  ZERO backprop/transformers. ZERO listas fixas. ZERO templates.
  ZERO thresholds mágicos. Tudo auto-calibrável.
"""

import numpy as np
from typing import Optional, Dict, Tuple, List

from .core import D, normalize, similarity
from .logic_encoder import (
    LogicRoles, decode_rule, decode_antecedent, decode_consequent,
    get_perm_ant, get_perm_cons,
)
from .resonator import ResonatorDecoder


class Decomposer:
    """
    Decompõe vetores compostos em componentes estruturados.

    Args:
        codebook: Matriz (V, D) de vetores de palavras
        w2i: Mapeamento palavra → índice
        i2w: Mapeamento índice → palavra
        roles: Instância LogicRoles (criada automaticamente se None)
    """

    def __init__(
        self,
        codebook: np.ndarray,
        w2i: Dict[str, int],
        i2w: Dict[int, str],
        roles: Optional[LogicRoles] = None,
        seed: int = 42,
    ):
        self.codebook = codebook.astype(np.float32)
        self.w2i = w2i
        self.i2w = i2w
        self.roles = roles or LogicRoles(seed=seed)
        self.V = codebook.shape[0]
        self.resonator = ResonatorDecoder(
            codebook=self.codebook, seed=seed,
        )

    def decompose(self, composite: np.ndarray) -> dict:
        """
        Decompõe um vetor composto em (role, ant, cons).

        Primeiro tenta decode_rule (regra FOL). Se falhar,
        tenta resonator 2-factor.

        Returns:
            dict com:
                'role': str or None
                'ant': str or None
                'cons': str or None
                'ant_vec': np.ndarray or None
                'cons_vec': np.ndarray or None
                'confidence': float
                'method': 'decode_rule' | 'resonator' | 'failed'
                'ant_sim': float
                'cons_sim': float
        """
        role, ant, cons, meta = decode_rule(
            composite, self.roles, self.codebook, self.w2i, self.i2w
        )

        if role is not None and ant is not None and cons is not None:
            v_ant = normalize(self._get_vec(ant))
            v_cons = normalize(self._get_vec(cons))
            return {
                'role': role,
                'ant': ant,
                'cons': cons,
                'ant_vec': v_ant,
                'cons_vec': v_cons,
                'confidence': meta.get('reconstruction_sim', 0.0),
                'ant_sim': meta.get('ant_sim', 0.0),
                'cons_sim': meta.get('conseq_sim', 0.0),
                'method': 'decode_rule',
            }

        res = self.resonator.decode_2factor(composite, binding_op='bind')
        if res and res.get('indices'):
            idx_a, idx_b = res['indices']
            sims = res.get('similarities', [0.0, 0.0])
            return {
                'role': None,
                'ant': self.i2w.get(int(idx_a)),
                'cons': self.i2w.get(int(idx_b)),
                'ant_vec': normalize(self.codebook[int(idx_a)].copy()),
                'cons_vec': normalize(self.codebook[int(idx_b)].copy()),
                'confidence': float(np.mean(sims)),
                'ant_sim': float(sims[0]),
                'cons_sim': float(sims[1]),
                'method': 'resonator',
            }

        return {
            'role': None, 'ant': None, 'cons': None,
            'ant_vec': None, 'cons_vec': None,
            'confidence': 0.0, 'ant_sim': 0.0, 'cons_sim': 0.0,
            'method': 'failed',
        }

    def decompose_deep(self, composite: np.ndarray) -> dict:
        """
        Decomposição hierárquica: tenta 3 fatores primeiro,
        depois 2 fatores, depois decode_rule.

        Returns:
            dict com:
                'role': str or None
                'ant': str or None
                'cons': str or None
                'ant_vec': np.ndarray or None
                'cons_vec': np.ndarray or None
                'inner': dict or None — sub-decomposição do consequent
                'confidence': float
                'method': str
                'n_factors': int
        """
        res_3 = self.resonator.decode_3factor(composite, binding_op='M')
        if res_3 and res_3.get('converged', False):
            indices = res_3['indices']
            sims = res_3.get('similarities', [0.0, 0.0, 0.0])
            return {
                'role': None,
                'ant': self.i2w.get(int(indices[0])),
                'cons': self.i2w.get(int(indices[1])),
                'inner': {
                    'role': None,
                    'ant': self.i2w.get(int(indices[1])),
                    'cons': self.i2w.get(int(indices[2])),
                },
                'ant_vec': normalize(self.codebook[int(indices[0])].copy()),
                'cons_vec': normalize(self.codebook[int(indices[2])].copy()),
                'confidence': float(np.mean(sims)),
                'method': 'resonator_3factor',
                'n_factors': 3,
            }

        res_2 = self.resonator.decode_2factor(composite, binding_op='M')
        if res_2 and res_2.get('indices'):
            idx_a, idx_b = res_2['indices']
            sims = res_2.get('similarities', [0.0, 0.0])
            return {
                'role': None,
                'ant': self.i2w.get(int(idx_a)),
                'cons': self.i2w.get(int(idx_b)),
                'inner': None,
                'ant_vec': normalize(self.codebook[int(idx_a)].copy()),
                'cons_vec': normalize(self.codebook[int(idx_b)].copy()),
                'confidence': float(np.mean(sims)),
                'method': 'resonator_2factor',
                'n_factors': 2,
            }

        flat = self.decompose(composite)
        flat['inner'] = None
        flat['n_factors'] = 1 if flat['method'] != 'failed' else 0
        return flat

    def decompose_batch(self, composites: np.ndarray) -> List[dict]:
        return [self.decompose(c) for c in composites]

    def _get_vec(self, word: str) -> Optional[np.ndarray]:
        idx = self.w2i.get(word)
        if idx is None or idx >= self.V:
            return None
        return self.codebook[idx].astype(np.float32)
