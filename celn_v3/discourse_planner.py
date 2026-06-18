"""
DiscoursePlanner — Plan Textual Universal para CELN v3
========================================================

Três estágios:
  1. Seeding: extrai palavras de conteúdo do prompt via magnitude resonance
  2. Expansion: beam search pelo PairGraph, guiado por TypeAlign + IntentAlign
  3. Linearization: melhor beam vira sequência de content words (plano)

O plano textual é uma lista de palavras de conteúdo que formam o esqueleto
da resposta. A GPVE (modo plan-guided) usa essas palavras como âncoras,
intercalando com function words da PCFG para produzir texto fluente.

Princípios: sem backprop, sem transformer, sem similaridade em 10k,
sem lista fixa, sem template, sem threshold mágico. Universal.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .core import (
    normalize, spectral_entropy, projective_resonance as M,
    encode_sequence,
)
from .gf_linearizer import GFLinearizer


@dataclass
class Beam:
    """Estado de um beam no beam search."""
    words: list[str] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    state: np.ndarray | None = None
    score: float = 0.0
    depth: int = 0


class DiscoursePlanner:
    """Planeja uma sequência de palavras de conteúdo para a GPVE realizar.

    O plano é uma lista de palavras de conteúdo que formam o esqueleto
    do discurso. Cada palavra é uma âncora que a GPVE deve realizar
    (inserir function words ao redor via PCFG).

    O beam search é guiado por dois scores:
      - TypeAlign: dot(type_field[candidate], type_field[current])
        Mede compatibilidade sintática via distribuição dos seguidores
      - IntentAlign: magnitude_resonance(M(state, candidate), M_intent)
        Mede coerência com o tópico da pergunta
    """

    def __init__(
        self,
        vectors: np.ndarray,
        w2i: dict[str, int],
        pair_graph: Any,
        type_field: np.ndarray | None = None,
        beam_width: int = 3,
        max_depth: int = 10,
    ):
        self.vectors = vectors.astype(np.float32)
        self.w2i = w2i
        self.i2w = {i: w for w, i in w2i.items()}
        self.pg = pair_graph
        self.type_field = type_field
        self.beam_width = beam_width
        self.max_depth = max_depth

        # Precompute FFT of each vector for magnitude resonance scoring
        self._word_ffts = np.fft.fft(self.vectors)
        self._word_mags = np.abs(self._word_ffts)
        # Normalize magnitude spectra
        self._word_mags_n = self._word_mags / (
            np.linalg.norm(self._word_mags, axis=1, keepdims=True) + 1e-12
        )

        # Auto-calibrated content-word filter:
        # Compute follower_popularity: for each word, how many source words
        # point to it? Function words are followers of MANY sources.
        # Content words are followers of FEW sources.
        # Threshold = median(follower_popularity) / 2 (auto-calibrável)
        self._content_word_mask: np.ndarray | None = None
        self._compute_content_words()

        # GFLinearizer: bottom-up tree linearizer
        self._gf = GFLinearizer(
            vectors=self.vectors,
            w2i=self.w2i,
            pair_graph=self.pg,
            type_field=self.type_field,
        )

    def _compute_content_words(self):
        """Computa máscara de palavras de conteúdo (não função).

        follower_popularity[i] = número de fontes que têm i como seguidor.
        Função: follower_popularity > threshold (auto-calibrável).
        Conteúdo: follower_popularity <= threshold.

        Duas versões:
          _content_mask: estrita (mediana/2 — só palavras muito raras como conteúdo)
          _content_mask_leve: P90 (inclui palavras razoavelmente comuns)
        """
        V = self.vectors.shape[0]
        popularity = np.zeros(V, dtype=np.int32)
        for _src, arr in self.pg.follower_map.items():
            for f in arr:
                if int(f) >= 0 and int(f) < V:
                    popularity[int(f)] += 1
        non_zero = popularity[popularity > 0]
        median = float(np.median(non_zero)) if len(non_zero) > 0 else 5
        p90 = float(np.percentile(non_zero, 90)) if len(non_zero) > 0 else 20
        threshold_strict = max(median / 2.0, 2.0)
        threshold_leve = max(p90, 10.0)
        self._content_mask = popularity <= int(threshold_strict)
        self._content_leve = popularity <= int(threshold_leve)

    # ------------------------------------------------------------------
    # Magnitude resonance (same as ch_mag channel)
    # ------------------------------------------------------------------

    def _magnitude_resonance(self, state: np.ndarray, target: np.ndarray) -> float:
        """Magnitude spectrum resonance (mesmo do ch_mag)."""
        sm = np.abs(np.fft.fft(state.astype(np.float32)))
        tm = np.abs(np.fft.fft(target.astype(np.float32)))
        sn = np.linalg.norm(sm)
        tn = np.linalg.norm(tm)
        if sn < 1e-12 or tn < 1e-12:
            return 0.0
        return float(np.dot(sm / sn, tm / tn))

    # ------------------------------------------------------------------
    # Stage 1: Seeding — extract content words as M_intent seeds
    # ------------------------------------------------------------------

    def _seed_from_intent(
        self,
        M_intent: np.ndarray,
        exclude_indices: set[int] | None = None,
        top_k: int = 3,
    ) -> list[int]:
        """Extrai seeds de palavras de conteúdo de M_intent.

        Usa magnitude spectrum resonance (não cosine similarity em 10k).
        Apenas palavras de conteúdo (auto-calibradas por follower_popularity).
        Palavras do prompt são excluídas.
        Retorna índices das palavras no vocabulário.
        """
        if M_intent is None:
            return []
        intent_mag = np.abs(np.fft.fft(normalize(M_intent).astype(np.float32)))
        inrm = np.linalg.norm(intent_mag)
        if inrm < 1e-12:
            return []
        intent_mag_n = intent_mag / inrm

        scores = self._word_mags_n @ intent_mag_n

        # Filter: only content words not in prompt
        if exclude_indices is None:
            exclude_indices = set()
        valid_indices = np.arange(len(scores))
        mask = self._content_mask.copy()
        for ex in exclude_indices:
            if 0 <= ex < len(mask):
                mask[ex] = False
        valid_indices = valid_indices[mask]

        if len(valid_indices) == 0:
            return []

        valid_scores = scores[valid_indices]
        k = min(top_k, len(valid_scores))
        top_pos = np.argpartition(valid_scores, -k)[-k:]
        top_idx = valid_indices[top_pos[np.argsort(valid_scores[top_pos])[::-1]]]
        return [int(idx) for idx in top_idx]

    # ------------------------------------------------------------------
    # Stage 2: Expansion — beam search pelo PairGraph
    # ------------------------------------------------------------------

    def _expand_beam(
        self,
        beams: list[Beam],
        M_intent: np.ndarray,
    ) -> list[Beam]:
        """Expande cada beam um passo pelo PairGraph (TODAS as palavras, sem filtro).

        Navigation through ALL words (content + function) ensures dense connectivity.
        The content filter is applied AFTER the best trajectory is chosen.

        Para cada beam:
          1. Obtém TODOS os seguidores do PairGraph para a última palavra
          2. Para cada seguidor:
             state = M(beam.state, vec[follower])
             step_score = magnitude_resonance(state, M_intent)
             beam_score += step_score (cumulativo)
          3. Mantém top-K beams por SCORE CUMULATIVO / depth

        Returns:
            Lista expandida de beams (até beam_width beams).
        """
        candidates: list[Beam] = []

        for beam in beams:
            last_idx = beam.indices[-1] if beam.indices else -1
            if last_idx < 0:
                continue

            followers = self.pg.get_followers(last_idx, top_k=5)
            if not followers:
                continue

            for f_idx in followers:
                if f_idx in beam.indices:
                    continue  # avoid repetition

                f_vec = self.vectors[f_idx].astype(np.float32)
                state = M(beam.state, f_vec, gamma=1.0, bilateral=True) if beam.state is not None else normalize(f_vec)

                step_score = self._magnitude_resonance(state, M_intent)

                new_beam = Beam(
                    words=beam.words + [self.i2w[f_idx]],
                    indices=beam.indices + [f_idx],
                    state=state,
                    score=beam.score + step_score,
                    depth=beam.depth + 1,
                )
                candidates.append(new_beam)

        if not candidates:
            return beams
        # Sort by average score (cumulative / depth)
        candidates.sort(key=lambda b: -b.score / max(b.depth, 1))
        return candidates[:self.beam_width]

    # ------------------------------------------------------------------
    # Stage 3: Linearization — converter beam em plano
    # ------------------------------------------------------------------

    def _linearize(self, best_beam: Beam) -> list[str]:
        """Converte o melhor beam em sequência linear de palavras."""
        return best_beam.words

    # ------------------------------------------------------------------
    # Stop criterion: spectral entropy divergence
    # ------------------------------------------------------------------

    def _should_stop(self, beam: Beam, M_intent: np.ndarray) -> bool:
        """Auto-calibrável: para quando o estado diverge do tópico.

        Critério: spectral_entropy(estado) > spectral_entropy(M_intent) * 1.5
        Quando o estado fica significativamente mais disperso que o pensamento
        original, a trajetória divergiu do tópico.
        """
        if beam.state is None:
            return False
        e_state = spectral_entropy(beam.state)
        e_intent = spectral_entropy(M_intent)
        if e_intent < 1e-12:
            return False
        return e_state > e_intent * 1.5

    # ------------------------------------------------------------------
    # Plan — orquestra os 3 estágios
    # ------------------------------------------------------------------

    def plan(self, prompt_tokens: list[str]) -> list[str]:
        """Gera plano textual via beam search profundo no PairGraph.

        Navega por TODAS as palavras (content + função) para máxima
        conectividade. Filtra content words APÓS escolher a melhor
        trajetória.

        A coerência global é medida por magnitude_resonance cumulativa
        com M_intent — a trajetória que mais se mantém alinhada ao
        tópico vence.

        Args:
            prompt_tokens: Tokens conhecidos do prompt.

        Returns:
            Lista de palavras de conteúdo (plano textual).
        """
        known = [t for t in prompt_tokens if t in self.w2i]
        if not known:
            return []

        # Encode prompt into M_intent
        word_vecs = [self.vectors[self.w2i[t]] for t in known]
        M_intent = normalize(encode_sequence(word_vecs, gamma=1.0, bilateral=True))

        # Stage 1: Seeding — first followers from last prompt word
        last_idx = self.w2i[known[-1]]
        followers = self.pg.get_followers(last_idx, top_k=5)
        if not followers:
            return []

        # Initialize beams from followers
        beams: list[Beam] = []
        for f_idx in followers:
            state = normalize(self.vectors[f_idx].copy())
            step_score = self._magnitude_resonance(state, M_intent)
            beams.append(Beam(
                words=[self.i2w[f_idx]],
                indices=[f_idx],
                state=state,
                score=step_score,
                depth=1,
            ))

        if len(beams) < 2:
            return []

        # Stage 2: Deep beam search (ALL words, no content filter)
        for _step in range(self.max_depth - 1):
            beams = self._expand_beam(beams, M_intent)

            # Stop if ALL beams diverged from topic
            if all(self._should_stop(b, M_intent) for b in beams):
                break

            # Stop if no beam could expand
            if all(
                not self.pg.get_followers(b.indices[-1]) if b.indices else True
                for b in beams
            ):
                break

        # Stage 3: Pick best trajectory by average score
        best = max(beams, key=lambda b: b.score / max(b.depth, 1))
        trajectory = best.words

        # Stage 4: Extract content words (threshold leve: P90), dedup consecutive
        plan_words = []
        for w in trajectory:
            idx = self.w2i.get(w)
            if idx is not None and self._content_leve[idx]:
                if not plan_words or w != plan_words[-1]:
                    plan_words.append(w)

        # If too few content words, fall back to GFLinearizer ordering of seeds
        if len(plan_words) < 2:
            n_seeds = max(5, min(8, len(known)))
            exclude = {self.w2i[t] for t in known if t in self.w2i}
            seed_indices = self._seed_from_intent(
                M_intent, exclude_indices=exclude, top_k=n_seeds,
            )
            if seed_indices and len(seed_indices) >= 2:
                raw_words = [self.i2w[idx] for idx in seed_indices]
                ordered_words = self._gf.linearize(raw_words)
                deduped = []
                for w in ordered_words:
                    if not deduped or w != deduped[-1]:
                        deduped.append(w)
                return deduped

        return plan_words
