"""
CELN v3 — Mouth (Orquestrador com verificação de loop fechado) [LEGACY]
=======================================================================

⚠️  DEPRECATED — Use `celn_v3.mouth_v2.MouthV2` instead.
    MouthV2 usa reconstrução atencional com 3 scores competitivos
    (syn, sem, fidelity) via GHRR, que substitui o beam search
    do Mouth v1. O v1 é mantido apenas para referência.

Pipeline do v1:
  1. Decomposer → (role, ant, cons) do vetor composto
  2. Lexicalizer → beam search → múltiplas word_sequences
  3. Linearizer → string final para cada beam
  4. similarity(original, parsed) → score de fidelidade
  5. Aceita a primeira frase com score > limiar auto-calibrável

Sem templates, sem thresholds fixos, sem backprop.
Tudo auto-calibrável via percentis da distribuição real.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .core import D, normalize, similarity
from .decomposer import Decomposer
from .lexicalizer import Lexicalizer, Beam
from .linearizer import linearize
from .logic_encoder import LogicRoles
from .pair_graph import PairGraph


@dataclass
class MouthResult:
    """Resultado da geração com verificação."""
    sentence: str
    role: str
    ant: str
    cons: str
    beam: Beam
    fidelity_score: float       # similarity(orig, parsed_back)
    threshold: float            # limiar auto-calibrável usado
    accepted: bool              # passou no threshold?
    low_confidence: bool        # se nenhum beam passou
    beams_tried: int            # quantos beams foram testados


class Mouth:
    """
    Orquestrador completo: vetor composto → frase verificada.

    Pipeline:
      1. Decomposer: extrai (role, ant, cons)
      2. Lexicalizer: beam search sobre PairGraph
      3. Linearizer: formata cada candidato
      4. nl_parser: parseia de volta e mede fidelidade
      5. Aceita se passar no limiar auto-calibrável

    Args:
        codebook: Matriz (V, D) de vetores normalizados
        w2i, i2w: Mapeamentos palavra↔índice
        pair_graph: PairGraph para transições (opcional)
        roles: LogicRoles para decode_rule (opcional)
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
        self.roles = roles or LogicRoles(seed=42)

        # Submódulos
        self.decomposer = Decomposer(
            codebook, w2i, i2w, roles=self.roles,
        )
        self.lexicalizer = Lexicalizer(
            codebook, w2i, i2w, pair_graph=pair_graph, roles=self.roles,
        )

    # ------------------------------------------------------------------
    # Parser (removed — use MouthV2 for attention-based generation)
    # ------------------------------------------------------------------

    def _get_parser(self):
        return None

    # ------------------------------------------------------------------
    # Geração principal
    # ------------------------------------------------------------------

    def generate(
        self,
        composite: np.ndarray,
        max_steps: int = 12,
        beam_width: int = 5,
        n_candidates: int = 8,
        verbose: bool = False,
    ) -> MouthResult:
        """
        Gera e verifica frase a partir de vetor composto.

        Args:
            composite: Vetor composto da dedução (encode_rule)
            max_steps: Passos máximos do Lexicalizer
            beam_width: Número de beams paralelos
            n_candidates: Candidatos por passo do Lexicalizer
            verbose: Printa debug

        Returns:
            MouthResult com frase final e métricas
        """
        # ── 1. Decomposer ──
        if verbose:
            print(f"[Mouth] Decompondo vetor composto...")

        result = self.decomposer.decompose(composite)
        if result['method'] == 'failed' or result['role'] is None:
            if verbose:
                print("[Mouth] ERRO: Decomposer falhou")
            return MouthResult(
                sentence='', role='', ant='', cons='',
                beam=Beam(), fidelity_score=0.0,
                threshold=0.0, accepted=False,
                low_confidence=True, beams_tried=0,
            )

        role_name = result['role']
        ant_word = result['ant']
        cons_word = result['cons']

        if verbose:
            print(f"[Mouth] Decompose: {role_name}({ant_word} → {cons_word})")

        # ── 2. Lexicalizer (beam search) ──
        if verbose:
            print(f"[Mouth] Beam search (width={beam_width})...")

        beams = self.lexicalizer.generate(
            composite,
            max_steps=max_steps,
            beam_width=beam_width,
            n_candidates=n_candidates,
            verbose=False,
        )

        if not beams:
            if verbose:
                print("[Mouth] ERRO: Lexicalizer não gerou beams")
            return MouthResult(
                sentence='', role=role_name, ant=ant_word, cons=cons_word,
                beam=Beam(), fidelity_score=0.0,
                threshold=0.0, accepted=False,
                low_confidence=True, beams_tried=0,
            )

        # ── 3. Linearizer + verificação para cada beam ──
        parser = self._get_parser()
        all_scores = []
        candidates = []

        for bi, beam in enumerate(beams):
            words = beam.path_words
            if not words:
                continue

            # Linearizer: word_sequence + role → string
            sentence = linearize(
                words, role_name,
                ant=ant_word, cons=cons_word,
                capitalize=True, add_period=True,
            )

            if verbose and bi < 3:
                print(f"  [beam {bi}] '{sentence[:60]}'")

            # Closed-loop: parser(phrase) → vector
            fidelity = 0.0
            if parser is not None:
                try:
                    rule_vec, premise = parse_and_encode(sentence, parser)
                    if rule_vec is not None:
                        fidelity = float(similarity(
                            normalize(composite),
                            normalize(rule_vec),
                        ))
                except Exception:
                    fidelity = 0.0

            all_scores.append(fidelity)
            candidates.append({
                'sentence': sentence,
                'beam': beam,
                'fidelity': fidelity,
            })

        # ── 4. Auto-calibração do threshold ──
        if not candidates:
            return MouthResult(
                sentence='', role=role_name, ant=ant_word, cons=cons_word,
                beam=Beam(), fidelity_score=0.0,
                threshold=0.0, accepted=False,
                low_confidence=True, beams_tried=0,
            )

        scores = np.array([c['fidelity'] for c in candidates])

        # Threshold: percentil 66 das similaridades (auto-calibrável)
        if len(scores) >= 3:
            threshold = float(np.percentile(scores, 66))
        else:
            threshold = float(np.mean(scores)) if len(scores) > 0 else 0.0

        # Score mínimo absoluto (round-trip mínimo para ser válido)
        min_threshold = max(threshold, 0.15)

        # ── 5. Seleciona melhor candidato ──
        # Ordena por fidelity score (maior = melhor)
        candidates.sort(key=lambda c: -c['fidelity'])

        best = candidates[0]
        accepted = best['fidelity'] >= min_threshold

        if verbose:
            print(f"\n[Mouth] Threshold: {min_threshold:.3f} "
                  f"(p66={threshold:.3f})")
            for bi, c in enumerate(candidates[:3]):
                flag = '✓' if c['fidelity'] >= min_threshold else '✗'
                print(f"  [{bi}] {flag} fidelity={c['fidelity']:.4f} "
                      f"'{c['sentence'][:60]}'")

        return MouthResult(
            sentence=best['sentence'],
            role=role_name,
            ant=ant_word,
            cons=cons_word,
            beam=best['beam'],
            fidelity_score=best['fidelity'],
            threshold=min_threshold,
            accepted=accepted,
            low_confidence=not accepted,
            beams_tried=len(candidates),
        )

    # ------------------------------------------------------------------
    # generate_from_components (para quando role/ant/cons já são conhecidos)
    # ------------------------------------------------------------------

    def generate_from_components(
        self,
        role_name: str,
        ant_word: str,
        cons_word: str,
        max_steps: int = 12,
        beam_width: int = 5,
        n_candidates: int = 8,
        verbose: bool = False,
    ) -> MouthResult:
        """Gera a partir de role/ant/cons (sem vetor composto)."""
        ant_idx = self.w2i.get(ant_word)
        cons_idx = self.w2i.get(cons_word)
        if ant_idx is None or cons_idx is None:
            return MouthResult(
                sentence='', role=role_name, ant=ant_word, cons=cons_word,
                beam=Beam(), fidelity_score=0.0,
                threshold=0.0, accepted=False,
                low_confidence=True, beams_tried=0,
            )

        v_ant = normalize(self.codebook[ant_idx])
        v_cons = normalize(self.codebook[cons_idx])
        role_vec = self.roles.get(role_name)

        from .logic_encoder import encode_rule
        composite = encode_rule(role_vec, v_ant, v_cons)

        return self.generate(
            composite,
            max_steps=max_steps,
            beam_width=beam_width,
            n_candidates=n_candidates,
            verbose=verbose,
        )
