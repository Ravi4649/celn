"""
Dual-Space Generator for CELN-v3
==================================
Arquitetura de Dois Espaços — SEM PCFG.

Espaço spaCy 300d: busca semântica de candidatos
Espaço 10k VSA: raciocínio, binding, memória, canais de votação

Fluxo:
  1. M_pr (10k) projetado para 300d via matriz aprendida
  2. Busca em 300d por cosine similarity (brute-force ou LSH)
  3. Type Field filtra candidatos com tipo sintático compatível
  4. PairGraph filtra candidatos com transição canônica
  5. 6 canais VSA (10k) votam nos sobreviventes (SEM ch_pcfg)
  6. IQR-weighted gating escolhe o vencedor
  7. M_pr atualizado via projective_resonance
  8. Oja's Rule atualiza vetores online

Princípios:
  ZERO backprop/transformers. ZERO listas fixas. ZERO templates.
  ZERO thresholds mágicos. Tudo auto-calibrável.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft, ifft
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .core import (
    D, normalize, bind, similarity, phi, projective_resonance,
    spectral_entropy, encode_sequence,
)


class DualSpaceGenerator:

    def __init__(
        self,
        vectors_10k: np.ndarray,
        word2idx: dict[str, int],
        spacy_words: np.ndarray,
        spacy_vectors: np.ndarray,
        type_field_array: np.ndarray,
        type_word2idx: dict[str, int],
        pair_graph: "PairGraph",
        projection_matrix: Optional[np.ndarray] = None,
        n_lsh_tables: int = 20,
        n_lsh_bits: int = 14,
        seed: int = 42,
    ):
        self.vectors_10k = vectors_10k.astype(np.float32)
        self.word2idx = word2idx
        self.idx2word = {i: w for w, i in word2idx.items()}
        self.n_vocab = len(word2idx)

        # spaCy 300d — otimizar inicialização
        self.spacy_words = spacy_words
        self.spacy_vectors = spacy_vectors.astype(np.float32)

        # Interseção rápida: word2idx tem ~20k, spacy_words tem ~415k
        # Converter spacy_words para set uma vez
        spacy_set = set(str(w) for w in spacy_words)
        self.common_words = sorted(set(word2idx.keys()) & spacy_set)
        self.n_common = len(self.common_words)

        # Build mappings apenas para palavras comuns
        self.cw_to_10k = {w: word2idx[w] for w in self.common_words}
        # spacy_w2i apenas para common words (não 415k!)
        self.spacy_w2i = {str(w): i for i, w in enumerate(spacy_words) if str(w) in self.cw_to_10k}
        self.cw_to_300d = {w: self.spacy_w2i[w] for w in self.common_words}

        # Pre-computar índices e vetores normalizados da interseção
        self._common_10k_idxs = np.array([self.cw_to_10k[w] for w in self.common_words], dtype=np.int32)
        self._common_300d_idxs = np.array([self.cw_to_300d[w] for w in self.common_words], dtype=np.int32)
        _cv = self.spacy_vectors[self._common_300d_idxs]
        _cv_norms = np.linalg.norm(_cv, axis=1, keepdims=True)
        _cv_norms[_cv_norms < 1e-12] = 1.0
        self._common_300d_normalized = (_cv / _cv_norms).astype(np.float32)
        self._common_word_arr = np.array(self.common_words)

        # Tipo sintático
        self.type_field = type_field_array.astype(np.float32)
        self.type_word2idx = type_word2idx

        # PairGraph
        self.pair_graph = pair_graph

        # Matriz de projeção 10k → 300d — SEM renormalização destrutiva
        if projection_matrix is None:
            rng = np.random.RandomState(seed)
            self.projection_matrix = rng.randn(300, D).astype(np.float32) * 0.01
        else:
            self.projection_matrix = projection_matrix.astype(np.float32)

        # LSH (construído depois, após projeção aprender)
        self.n_lsh_tables = n_lsh_tables
        self.n_lsh_bits = n_lsh_bits
        self.lsh_tables: dict[tuple, list[str]] = defaultdict(list)
        self._lsh_built = False

        # NOT precomputing all FFTs - compute on demand to avoid 20k*10k FFT slowness

        # Precompute type field normals para context type
        _tf_norms = np.linalg.norm(self.type_field, axis=1, keepdims=True)
        _tf_norms[_tf_norms < 1e-12] = 1.0
        self.type_field_normalized = self.type_field / _tf_norms

        # NOT precomputing all FFTs - too slow (20k x 10k). Compute on demand.

    # ------------------------------------------------------------------
    # LSH (build depois da projeção aprender)
    # ------------------------------------------------------------------

    def build_lsh_index(self, seed: int = 42):
        rng = np.random.RandomState(seed + 7)
        n_proj = self.n_lsh_tables * self.n_lsh_bits
        self._lsh_rand = rng.randn(n_proj, 300).astype(np.float32)
        col_norms = np.linalg.norm(self._lsh_rand, axis=0, keepdims=True)
        col_norms[col_norms < 1e-12] = 1.0
        self._lsh_rand = self._lsh_rand / col_norms

        self.lsh_tables = defaultdict(list)
        for i, w in enumerate(self.common_words):
            vec_n = self._common_300d_normalized[i]
            h = self._simhash(vec_n)
            self.lsh_tables[tuple(h)].append(w)

        self._lsh_built = True

    def _simhash(self, vec_300d: np.ndarray) -> np.ndarray:
        proj = vec_300d @ self._lsh_rand.T
        return (proj > 0).astype(np.int8)

    def _hamming(self, h1: np.ndarray, h2: np.ndarray) -> int:
        return int(np.sum(h1 != h2))

    def lsh_query(self, query_300d: np.ndarray, top_k: int = 500, max_hd: int = 3) -> list[str]:
        if not self._lsh_built:
            return self._brute_force_search(query_300d, top_k)
        q_hash = self._simhash(query_300d)
        candidates = set()
        for stored_hash, words in self.lsh_tables.items():
            if self._hamming(q_hash, np.array(stored_hash, dtype=np.int8)) <= max_hd:
                candidates.update(words)
        result = list(candidates)
        if len(result) > top_k:
            result = result[:top_k]
        return result

    # ------------------------------------------------------------------
    # Busca brute-force em 300d (fallback / inicial)
    # ------------------------------------------------------------------

    def _brute_force_search(self, query_300d: np.ndarray, top_k: int = 500) -> list[str]:
        q_norm = np.linalg.norm(query_300d)
        if q_norm < 1e-12:
            return [self.common_words[i] for i in np.random.choice(self.n_common, min(top_k, self.n_common), replace=False)]
        q_n = query_300d / q_norm
        sims = self._common_300d_normalized @ q_n
        top_indices = np.argsort(sims)[-top_k:][::-1]
        return [self.common_words[i] for i in top_indices]

    # ------------------------------------------------------------------
    # Projeção 10k → 300d
    # ------------------------------------------------------------------

    def project_10k_to_300d(self, M_pr: np.ndarray) -> np.ndarray:
        projected = self.projection_matrix @ M_pr
        pn = np.linalg.norm(projected)
        if pn > 1e-12:
            return projected / pn
        return projected

    # ------------------------------------------------------------------
    # Filtros
    # ------------------------------------------------------------------

    def type_field_filter(self, candidates: list[str], recent_tokens: list[str], keep_percentile: float = 50.0) -> list[str]:
        if len(candidates) <= 10:
            return candidates

        # Context type = soma dos type vectors dos últimos tokens
        ctx_vec = np.zeros(self.type_field.shape[1], dtype=np.float32)
        n_ctx = 0
        for w in recent_tokens[-3:]:
            if w in self.type_word2idx:
                ctx_vec += self.type_field_normalized[self.type_word2idx[w]]
                n_ctx += 1
        if n_ctx == 0:
            return candidates
        cn = np.linalg.norm(ctx_vec)
        if cn < 1e-12:
            return candidates
        ctx_vec = ctx_vec / cn

        # Score cada candidato por cosine similarity com context type
        scored = []
        for w in candidates:
            if w not in self.type_word2idx:
                scored.append((w, 0.0))
                continue
            t_idx = self.type_word2idx[w]
            t_vec = self.type_field_normalized[t_idx]
            sim = float(np.dot(t_vec, ctx_vec))
            scored.append((w, sim))

        # Filtrar por percentil: manter só acima do threshold
        all_sims = np.array([s for _, s in scored])
        threshold = float(np.percentile(all_sims, keep_percentile))
        filtered = [w for w, s in scored if s >= threshold]
        return filtered if len(filtered) >= 5 else candidates

    def pairgraph_filter(self, candidates: list[str], last_word: Optional[str], top_k: int = 200) -> list[str]:
        if last_word is None or last_word not in self.word2idx:
            return candidates

        src_idx = self.word2idx[last_word]
        followers = self.pair_graph.get_followers(src_idx, top_k=20)
        if not followers:
            return candidates

        follower_words = set()
        for f_idx in followers:
            if f_idx in self.idx2word:
                follower_words.add(self.idx2word[f_idx])

        # 2-hop
        two_hop = set()
        for f_idx in followers[:5]:
            for ff_idx in self.pair_graph.get_followers(f_idx, top_k=5):
                if ff_idx in self.idx2word:
                    two_hop.add(self.idx2word[ff_idx])

        scored = []
        for w in candidates:
            if w in follower_words:
                scored.append((w, 2.0))
            elif w in two_hop:
                scored.append((w, 1.0))
            else:
                scored.append((w, 0.0))

        positive = [w for w, s in scored if s > 0]
        if len(positive) < max(10, len(candidates) // 5):
            sorted_by_score = sorted(scored, key=lambda x: x[1], reverse=True)
            return [w for w, _ in sorted_by_score[:top_k]]
        return positive[:top_k]

    # ------------------------------------------------------------------
    # Canais VSA (6 canais, SEM PCFG)
    # ------------------------------------------------------------------

    def compute_channel_scores(
        self,
        M_pr: np.ndarray,
        candidates: list[str],
        recent_tokens: list[str],
    ) -> np.ndarray:
        """Returns (n_candidates, 6) scores.
        Ch0: anchor, Ch1: magnitude, Ch2: phase, Ch3: type, Ch4: sdm, Ch5: trajectory
        """
        n = len(candidates)
        if n == 0:
            return np.zeros((0, 6), dtype=np.float32)

        scores = np.zeros((n, 6), dtype=np.float32)
        M_pr_f = M_pr.astype(np.float32)

        # Pre-compute target para magnitude channel
        target_phi = phi(M_pr_f)
        target_mag = np.abs(fft(target_phi))
        tn = np.linalg.norm(target_mag)
        target_mag_n = target_mag / tn if tn > 1e-12 else target_mag

        # Pre-compute spaCy context para phase channel
        spacy_context = np.zeros(300, dtype=np.float32)
        n_ctx = 0
        for w in recent_tokens:
            if w in self.cw_to_300d:
                spacy_context += self.spacy_vectors[self.cw_to_300d[w]]
                n_ctx += 1
        if n_ctx > 0:
            sn = np.linalg.norm(spacy_context)
            if sn > 1e-12:
                spacy_context = spacy_context / sn

        # Pre-compute context type para type channel
        ctx_type = np.zeros(self.type_field.shape[1], dtype=np.float32)
        n_t = 0
        for rw in recent_tokens[-3:]:
            if rw in self.type_word2idx:
                ctx_type += self.type_field_normalized[self.type_word2idx[rw]]
                n_t += 1
        if n_t > 0:
            cn = np.linalg.norm(ctx_type)
            if cn > 1e-12:
                ctx_type = ctx_type / cn
        else:
            ctx_type = None

        # Pre-compute M_pr projected para SDM channel
        mp_300d = self.project_10k_to_300d(M_pr_f)

        # Pre-compute anchor magnitudes from recent tokens (compute on demand, not pre-stored)
        anchor_mags = []
        for w in recent_tokens[-3:]:
            if w not in self.word2idx:
                continue
            idx = self.word2idx[w]
            vec = self.vectors_10k[idx].astype(np.float32)
            tok_fft = fft(vec)
            mag = np.abs(tok_fft)
            nrm = np.linalg.norm(mag)
            if nrm > 1e-12:
                anchor_mags.append((mag / nrm).astype(np.float32))

        for i, w in enumerate(candidates):
            if w not in self.word2idx:
                continue
            idx = self.word2idx[w]
            vec = self.vectors_10k[idx]
            tok_fft = fft(vec.astype(np.float32))

            # --- Ch 0: Anchor (magnitude resonance with recent tokens) ---
            anchor = 0.0
            cand_mag = np.abs(tok_fft)
            cn_mag = np.linalg.norm(cand_mag)
            if cn_mag > 1e-12 and anchor_mags:
                cand_mag_n = cand_mag / cn_mag
                anchor = max(float(np.dot(cand_mag_n, a)) for a in anchor_mags)
            scores[i, 0] = float(np.clip(anchor, -1.0, 1.0))

            # --- Ch 1: Magnitude (hypothetical state resonance with target) ---
            M_hyp = projective_resonance(M_pr_f, vec, gamma=1.0, bilateral=True)
            m_hyp_mag = np.abs(fft(M_hyp))
            mhn = np.linalg.norm(m_hyp_mag)
            scores[i, 1] = float(np.dot(m_hyp_mag / mhn, target_mag_n)) if mhn > 1e-12 else 0.0

            # --- Ch 2: Phase (cosine sim in 300d with context) ---
            if w in self.cw_to_300d and n_ctx > 0:
                cand_spacy = self.spacy_vectors[self.cw_to_300d[w]]
                csn = np.linalg.norm(cand_spacy)
                scores[i, 2] = float(np.dot(cand_spacy / csn, spacy_context)) if csn > 1e-12 else 0.0

            # --- Ch 3: Type (HDC type alignment) ---
            if ctx_type is not None and w in self.type_word2idx:
                t_idx = self.type_word2idx[w]
                t_vec = self.type_field_normalized[t_idx]
                scores[i, 3] = float(np.dot(t_vec, ctx_type))

            # --- Ch 4: SDM (proxied: cosine sim of candidate spaCy with M_pr projected) ---
            if w in self.cw_to_300d:
                cand_spacy = self.spacy_vectors[self.cw_to_300d[w]]
                csn = np.linalg.norm(cand_spacy)
                if csn > 1e-12:
                    scores[i, 4] = float(np.dot(cand_spacy / csn, mp_300d))

            # --- Ch 5: Trajectory (PairGraph lookahead) ---
            if self.pair_graph is not None and len(recent_tokens) > 0:
                last_w = recent_tokens[-1]
                if last_w in self.word2idx:
                    m_intent = normalize(M_pr_f)
                    try:
                        scores[i, 5] = self.pair_graph.lookahead_coherence(
                            idx, self.vectors_10k, m_intent, depth=2, width=2
                        )
                    except Exception:
                        scores[i, 5] = 0.0

        return scores

    # ------------------------------------------------------------------
    # Gating
    # ------------------------------------------------------------------

    def competitive_gate(self, scores: np.ndarray) -> np.ndarray:
        """IQR-weighted gating: cada canal pesado pelo IQR da sua distribuição."""
        n_ch = scores.shape[1]
        iqrs = np.zeros(n_ch, dtype=np.float32)
        for j in range(n_ch):
            ch = scores[:, j]
            if len(ch) > 1:
                q75, q25 = np.percentile(ch, [75, 25])
                iqrs[j] = max(q75 - q25, 0.0)
        total = float(iqrs.sum())
        if total > 1e-12:
            weights = iqrs / total
        else:
            weights = np.ones(n_ch, dtype=np.float32) / n_ch

        combined = scores @ weights
        return combined

    # ------------------------------------------------------------------
    # Geração
    # ------------------------------------------------------------------

    def generate_token(
        self,
        M_pr: np.ndarray,
        recent_tokens: list[str],
        temperature: float = 0.8,
        top_k_search: int = 500,
        top_k_rerank: int = 50,
    ) -> tuple[str, dict]:

        info = {}

        # 1. Projetar M_pr para 300d
        M_pr_300d = self.project_10k_to_300d(M_pr)

        # 2. Buscar candidatos semanticamente relevantes (sempre brute-force)
        search_candidates = self._brute_force_search(M_pr_300d, top_k=top_k_search)
        info['n_search'] = len(search_candidates)

        if self._lsh_built:
            lsh_candidates = self.lsh_query(M_pr_300d, top_k=top_k_search)
            if lsh_candidates:
                search_candidates = lsh_candidates
                info['lsh_hit'] = len(lsh_candidates)

        if not search_candidates:
            w = self.common_words[np.random.randint(self.n_common)]
            info['fallback'] = 'search_empty'
            return w, info

        # 3. Type Field filtra
        type_filtered = self.type_field_filter(search_candidates, recent_tokens)
        info['n_type'] = len(type_filtered)

        # 4. PairGraph filtra
        last_word = recent_tokens[-1] if recent_tokens else None
        traj_filtered = self.pairgraph_filter(type_filtered, last_word, top_k=100)
        info['n_traj'] = len(traj_filtered)

        # 5. spaCy reranking em 300d
        reranked = self._brute_force_search(M_pr_300d, top_k=top_k_rerank)
        # Intersect with trajectory-filtered to keep only candidates that passed
        traj_set = set(traj_filtered)
        reranked = [w for w in reranked if w in traj_set]
        if len(reranked) < 5:
            reranked = traj_filtered[:top_k_rerank]
        info['n_reranked'] = len(reranked)

        if not reranked:
            reranked = traj_filtered[:top_k_rerank]
        if not reranked:
            reranked = search_candidates[:top_k_rerank]

        # 6. 6 canais VSA competem
        channel_scores = self.compute_channel_scores(M_pr, reranked, recent_tokens)
        combined = self.competitive_gate(channel_scores)

        # 7. Sampling com temperatura
        if temperature > 0 and len(combined) > 1:
            scale = float(np.std(combined)) if np.std(combined) > 1e-12 else 1.0
            centered = combined - np.max(combined)
            logits = centered / (scale * temperature)
            logits = np.clip(logits, -20, 20)
            probs = np.exp(logits)
            prob_sum = probs.sum()
            if prob_sum > 1e-12:
                probs = probs / prob_sum
            else:
                probs = np.ones_like(probs) / len(probs)
            chosen_idx = int(np.random.choice(len(combined), p=probs))
        else:
            chosen_idx = int(np.argmax(combined))

        chosen_word = reranked[chosen_idx]
        info['chosen_word'] = chosen_word
        info['combined_score'] = float(combined[chosen_idx])
        info['channel_scores'] = channel_scores[chosen_idx].tolist()

        return chosen_word, info

    def generate(
        self,
        prompt: str,
        max_tokens: int = 20,
        temperature: float = 0.8,
    ) -> tuple[str, list[dict]]:

        tokens = prompt.lower().split()
        tokens = [t for t in tokens if t in self.word2idx]

        if not tokens:
            return "", []

        vecs = [self.vectors_10k[self.word2idx[t]] for t in tokens]
        M_pr = encode_sequence(vecs, gamma=1.0, bilateral=True).astype(np.float32)

        generated = []
        recent = tokens[-3:]
        all_info = []

        for step in range(max_tokens):
            word, info = self.generate_token(
                M_pr=M_pr,
                recent_tokens=recent,
                temperature=temperature,
            )
            info['step'] = step
            all_info.append(info)

            generated.append(word)
            recent = (recent + [word])[-3:]

            if word in self.word2idx:
                vec = self.vectors_10k[self.word2idx[word]].astype(np.float32)
                M_pr = projective_resonance(M_pr, vec, gamma=1.0, bilateral=True)

        full_text = prompt + " " + " ".join(generated)
        return full_text, all_info

    # ------------------------------------------------------------------
    # Aprendizado da matriz de projeção
    # ------------------------------------------------------------------

    def learn_projection_matrix(
        self,
        sentences: list[list[str]],
        n_iterations: int = 100,
        learning_rate: float = 0.01,
    ):
        """Aprender projeção 10k → 300d via regressão linear iterativa."""
        print(f"Learning projection matrix on {len(sentences)} sentences...")

        X = []
        Y = []
        used = 0
        for sentence in sentences:
            valid_10k = [w for w in sentence if w in self.word2idx]
            valid_300d = [w for w in sentence if w in self.cw_to_300d]
            if len(valid_10k) < 2 or len(valid_300d) < 1:
                continue

            vecs_10k = [self.vectors_10k[self.word2idx[w]] for w in valid_10k]
            M_pr = encode_sequence(vecs_10k, gamma=1.0, bilateral=True)

            vecs_300d = [self.spacy_vectors[self.cw_to_300d[w]] for w in valid_300d if w in self.cw_to_300d]
            centroid_300d = normalize(np.mean(vecs_300d, axis=0))

            X.append(M_pr.astype(np.float32))
            Y.append(centroid_300d.astype(np.float32))
            used += 1

            if used >= 2000:
                break

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        print(f"  Training pairs: {used}")

        lam = 1.0
        XtX = X.T @ X + lam * np.eye(D, dtype=np.float32)
        XtX_inv = np.linalg.inv(XtX)
        P_opt = Y.T @ X @ XtX_inv
        self.projection_matrix = P_opt.astype(np.float32)
        pred = self.projection_matrix @ X.T
        loss = float(np.mean((pred - Y.T) ** 2))
        print(f"  Final loss: {loss:.6f}")
        print("  Projection matrix learned.")

    # ------------------------------------------------------------------
    # Oja's Rule (aprendizado online)
    # ------------------------------------------------------------------

    def oja_update(self, M_pr: np.ndarray, chosen_idx: int, lr: float = 1e-5):
        w = self.vectors_10k[chosen_idx]
        y = float(np.dot(w, M_pr))
        delta = lr * (y * M_pr - y * y * w)
        self.vectors_10k[chosen_idx] = normalize(w + delta).astype(np.float32)

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def save(self, path: str = "dual_space_generator.npz"):
        np.savez_compressed(
            path,
            projection_matrix=self.projection_matrix,
        )
        print(f"Saved DualSpaceGenerator to {path}")

    @classmethod
    def load(cls, path: str = "dual_space_generator.npz", **kwargs):
        data = np.load(path)
        return cls(projection_matrix=data['projection_matrix'], **kwargs)
