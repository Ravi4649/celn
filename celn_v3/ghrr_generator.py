"""
GHRR Generator for CELN-v3 — BUNDLING Architecture
====================================================
Arquitetura alinhada com o paper 2405.09689:

  - Cada palavra é um hipervetor GHRR ℝ^{D×M×M} (binding Q*Λ interno)
  - O estado da sequência é um BUNDLE (soma) dos vetores: S = Σ w_i
  - Similaridade GHRR S·c^T implementa atenção nativa via Q·K^T nas fatias
  - Busca semântica no espaço spaCy 300d (brute-force cosine)

FLUXO:
  1. Bundle das palavras do prompt → estado GHRR
  2. Para cada passo de geração:
     a. Busca candidatos no espaço 300d (contexto ponderado dos tokens recentes)
     b. Type Field filtra por coerência sintática
     c. PairGraph filtra por transições canônicas
     d. Score GHRR: similaridade trace-based + type + trajectory
     e. IQR-weighted gating → sampling com temperatura → vencedor
     f. Adiciona palavra ao bundle (estado += word)

PRINCÍPIOS:
  ZERO backprop/transformers. ZERO listas fixas. ZERO templates.
  ZERO thresholds mágicos. Tudo auto-calibrável.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional

from .ghrr_core import (
    D, M, DTYPE, EFFECTIVE_DIM,
    vec_10k_to_ghrr, ghrr_to_10k,
    normalize_slices_batch, normalize_slices,
    ghrr_similarity, ghrr_similarity_batch,
    ghrr_attention, ghrr_attention_score,
    auto_threshold,
)


class GHRRGenerator:

    def __init__(
        self,
        vectors_10k: np.ndarray,
        word2idx: dict[str, int],
        spacy_words: np.ndarray,
        spacy_vectors: np.ndarray,
        type_field_array: np.ndarray,
        type_word2idx: dict[str, int],
        pair_graph: "PairGraph",
        seed: int = 42,
    ):
        self.word2idx = word2idx
        self.idx2word = {int(i): str(w) for w, i in word2idx.items()}
        self.n_vocab = len(word2idx)

        self.vectors_10k = vectors_10k.astype(np.float32)

        # ── spaCy 300d ──
        self.spacy_words = spacy_words
        self.spacy_vectors = spacy_vectors.astype(np.float32)

        spacy_set = set(str(w) for w in spacy_words)
        self.common_words = sorted(set(word2idx.keys()) & spacy_set)
        self.n_common = len(self.common_words)

        self.spacy_w2i = {
            str(w): i for i, w in enumerate(spacy_words)
            if str(w) in set(self.common_words)
        }
        self.cw_to_300d = {w: self.spacy_w2i[w] for w in self.common_words}

        self._common_300d_idxs = np.array(
            [self.cw_to_300d[w] for w in self.common_words], dtype=np.int32
        )
        _cv = self.spacy_vectors[self._common_300d_idxs]
        _cv_norms = np.linalg.norm(_cv, axis=1, keepdims=True)
        _cv_norms[_cv_norms < 1e-12] = 1.0
        self._common_300d_normalized = (_cv / _cv_norms).astype(np.float32)

        # ── GHRR vectors para palavras comuns (conversão lazy) ──
        self._common_10k_idxs = np.array(
            [int(word2idx[w]) for w in self.common_words], dtype=np.int32
        )
        self._common_ghrr: np.ndarray | None = None

        # ── Type Field ──
        self.type_field = type_field_array.astype(np.float32)
        self.type_word2idx = type_word2idx
        _tf_norms = np.linalg.norm(self.type_field, axis=1, keepdims=True)
        _tf_norms[_tf_norms < 1e-12] = 1.0
        self.type_field_normalized = self.type_field / _tf_norms

        # ── PairGraph ──
        self.pair_graph = pair_graph

        self._precomputed: dict[int, np.ndarray] = {}  # word_idx → ghrr vector cache

    # ------------------------------------------------------------------
    # Conversão GHRR (lazy)
    # ------------------------------------------------------------------

    def _ensure_ghrr(self):
        """Converte vetores 10k para GHRR na primeira chamada."""
        if self._common_ghrr is not None:
            return
        print(f"  Converting {self.n_common} words to GHRR...")
        import time
        t0 = time.time()
        raw = self.vectors_10k[self._common_10k_idxs]
        h = raw.reshape(self.n_common, D, M, M).astype(DTYPE)
        self._common_ghrr = normalize_slices_batch(h)
        t1 = time.time()
        print(f"  Done in {t1 - t0:.1f}s")

    def _get_ghrr(self, word: str) -> Optional[np.ndarray]:
        """Obtém vetor GHRR para uma palavra (com cache)."""
        if word not in self.word2idx:
            return None
        idx = int(self.word2idx[word])
        if idx not in self._precomputed:
            vec = self.vectors_10k[idx].reshape(D, M, M).astype(DTYPE)
            self._precomputed[idx] = normalize_slices(vec)
        return self._precomputed[idx]

    def _get_ghrr_batch(self, words: list[str]) -> np.ndarray:
        """Obtém vetores GHRR para uma lista de palavras (batched)."""
        idxs = np.array([int(self.word2idx[w]) if w in self.word2idx else -1 for w in words], dtype=np.int32)
        valid_mask = idxs >= 0
        valid_idxs = idxs[valid_mask]
        valid_words = [w for w, m in zip(words, valid_mask) if m]

        result = np.zeros((len(words), D, M, M), dtype=DTYPE)
        for i, (w, vidx) in enumerate(zip(valid_words, valid_idxs)):
            matched = False
            for j, w_orig in enumerate(words):
                if w_orig == w and not matched:
                    result[j] = self._get_ghrr(w)
                    matched = True
                    break
            if not matched:
                pass
        return result

    # ------------------------------------------------------------------
    # Busca 300d
    # ------------------------------------------------------------------

    def _make_context_300d(self, recent_tokens: list[str]) -> np.ndarray:
        """Média ponderada dos embeddings spaCy dos tokens recentes.
        
        Pesos: tokens mais recentes têm peso maior (decaimento linear).
        """
        ctx = np.zeros(300, dtype=np.float32)
        n = len(recent_tokens)
        if n == 0:
            return ctx
        weights = np.linspace(0.3, 1.0, n)
        weight_sum = 0.0
        for w, weight in zip(recent_tokens, weights):
            if w in self.cw_to_300d:
                ctx += weight * self.spacy_vectors[self.cw_to_300d[w]]
                weight_sum += weight
        if weight_sum > 0:
            ctx /= weight_sum
        norm = np.linalg.norm(ctx)
        if norm > 1e-12:
            ctx /= norm
        return ctx

    def _brute_force_search(self, query_300d: np.ndarray, top_k: int = 500) -> list[str]:
        """Busca brute-force em 300d via cosine similarity."""
        q_norm = np.linalg.norm(query_300d)
        if q_norm < 1e-12:
            indices = np.random.choice(self.n_common, min(top_k, self.n_common), replace=False)
            return [self.common_words[i] for i in indices]
        q_n = query_300d / q_norm
        sims = self._common_300d_normalized @ q_n
        top_indices = np.argsort(sims)[-top_k:][::-1]
        return [self.common_words[i] for i in top_indices]

    # ------------------------------------------------------------------
    # Filtros
    # ------------------------------------------------------------------

    def type_field_filter(self, candidates: list[str], recent_tokens: list[str],
                          keep_percentile: float = 50.0) -> list[str]:
        if len(candidates) <= 10:
            return candidates
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
        ctx_vec /= cn

        scored = []
        for w in candidates:
            if w not in self.type_word2idx:
                scored.append((w, 0.0))
                continue
            t_vec = self.type_field_normalized[self.type_word2idx[w]]
            scored.append((w, float(np.dot(t_vec, ctx_vec))))

        all_sims = np.array([s for _, s in scored])
        threshold = float(np.percentile(all_sims, keep_percentile))
        filtered = [w for w, s in scored if s >= threshold]
        return filtered if len(filtered) >= 5 else candidates

    def pairgraph_filter(self, candidates: list[str], last_word: Optional[str],
                         top_k: int = 200) -> list[str]:
        if last_word is None or last_word not in self.word2idx:
            return candidates
        src_idx = int(self.word2idx[last_word])
        followers = self.pair_graph.get_followers(src_idx, top_k=20)
        if not followers:
            return candidates

        follower_words = set()
        for f_idx in followers:
            if f_idx in self.idx2word:
                follower_words.add(self.idx2word[f_idx])

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
    # Canais de scoring (GHRR + Type + Trajectory)
    # ------------------------------------------------------------------

    def compute_channel_scores(
        self,
        ghrr_state: np.ndarray,
        candidates: list[str],
        recent_tokens: list[str],
    ) -> np.ndarray:
        """4 canais:
          Ch0: GHRR similarity — similaridade direta estado-candidato
          Ch1: GHRR attention — concentração da atenção Q·K^T
          Ch2: Type field — coerência sintática
          Ch3: Trajectory — PairGraph lookahead
        """
        n = len(candidates)
        if n == 0:
            return np.zeros((0, 4), dtype=np.float32)

        scores = np.zeros((n, 4), dtype=np.float32)

        # ── Ch0 & Ch1: GHRR scoring ──
        cand_ghrr_list = []
        for w in candidates:
            gh = self._get_ghrr(w)
            if gh is not None:
                cand_ghrr_list.append(gh)
            else:
                cand_ghrr_list.append(np.zeros((D, M, M), dtype=DTYPE))
        cand_ghrr = np.stack(cand_ghrr_list) if cand_ghrr_list else np.zeros((0, D, M, M), dtype=DTYPE)

        # Ch0: GHRR similarity
        scores[:, 0] = ghrr_similarity_batch(ghrr_state, cand_ghrr)

        # Ch1: GHRR attention score (concentração)
        for i in range(n):
            if cand_ghrr_list[i] is not None and np.linalg.norm(cand_ghrr_list[i]) > 1e-12:
                scores[i, 1] = ghrr_attention_score(ghrr_state, cand_ghrr_list[i])

        # ── Ch2: Type field ──
        ctx_type = np.zeros(self.type_field.shape[1], dtype=np.float32)
        n_t = 0
        for rw in recent_tokens[-3:]:
            if rw in self.type_word2idx:
                ctx_type += self.type_field_normalized[self.type_word2idx[rw]]
                n_t += 1
        if n_t > 0:
            cn = np.linalg.norm(ctx_type)
            if cn > 1e-12:
                ctx_type /= cn
            for i, w in enumerate(candidates):
                if w in self.type_word2idx:
                    scores[i, 2] = float(np.dot(
                        self.type_field_normalized[self.type_word2idx[w]], ctx_type
                    ))

        # ── Ch3: PairGraph trajectory ──
        if len(recent_tokens) > 0:
            last_w = recent_tokens[-1]
            if last_w in self.word2idx:
                for i, w in enumerate(candidates):
                    if w in self.word2idx:
                        try:
                            scores[i, 3] = self.pair_graph.lookahead_coherence(
                                int(self.word2idx[w]),
                                self.vectors_10k,
                                ghrr_to_10k(ghrr_state),
                                depth=2, width=2,
                            )
                        except Exception:
                            pass

        return scores

    # ------------------------------------------------------------------
    # Gating
    # ------------------------------------------------------------------

    def competitive_gate(self, scores: np.ndarray) -> np.ndarray:
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
        return scores @ weights

    # ------------------------------------------------------------------
    # Geração
    # ------------------------------------------------------------------

    def generate_token(
        self,
        ghrr_state: np.ndarray,
        recent_tokens: list[str],
        excluded_tokens: set[str] | None = None,
        temperature: float = 0.8,
        top_k_search: int = 500,
        top_k_rerank: int = 50,
    ) -> tuple[str, dict]:

        info = {}

        # 1. Busca em 300d
        ctx_300d = self._make_context_300d(recent_tokens)
        search_candidates = self._brute_force_search(ctx_300d, top_k=top_k_search)
        info['n_search'] = len(search_candidates)

        if not search_candidates:
            w = self.common_words[np.random.randint(self.n_common)]
            info['fallback'] = 'search_empty'
            return w, info

        # 1b. Excluir tokens já gerados recentemente da busca
        if excluded_tokens:
            search_candidates = [w for w in search_candidates if w not in excluded_tokens]
            info['n_search'] = len(search_candidates)
            if not search_candidates:
                w = self.common_words[np.random.randint(self.n_common)]
                info['fallback'] = 'all_excluded'
                return w, info

        # 2. Type Field filter
        type_filtered = self.type_field_filter(search_candidates, recent_tokens)
        info['n_type'] = len(type_filtered)

        # 3. PairGraph filter
        last_word = recent_tokens[-1] if recent_tokens else None
        traj_filtered = self.pairgraph_filter(type_filtered, last_word, top_k=100)
        info['n_traj'] = len(traj_filtered)

        # 4. Rerank: interseção com busca mais restrita
        reranked = self._brute_force_search(ctx_300d, top_k=top_k_rerank)
        traj_set = set(traj_filtered)
        reranked = [w for w in reranked if w in traj_set]
        if len(reranked) < 5:
            reranked = traj_filtered[:top_k_rerank]
        info['n_reranked'] = len(reranked)

        if not reranked:
            reranked = traj_filtered[:top_k_rerank]
        if not reranked:
            reranked = search_candidates[:top_k_rerank]

        # 5. GHRR channel scoring
        channel_scores = self.compute_channel_scores(ghrr_state, reranked, recent_tokens)
        combined = self.competitive_gate(channel_scores)

        # 6. Temperature sampling
        if temperature > 0 and len(combined) > 1:
            scale = float(np.std(combined)) if np.std(combined) > 1e-12 else 1.0
            centered = combined - np.max(combined)
            logits = centered / (scale * temperature)
            logits = np.clip(logits, -20, 20)
            probs = np.exp(logits)
            prob_sum = probs.sum()
            if prob_sum > 1e-12:
                probs /= prob_sum
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
        ema_alpha: float = 0.4,
        exclude_window: int = 3,
        verbose: bool = True,
    ) -> tuple[str, list[dict]]:

        self._ensure_ghrr()

        tokens = prompt.lower().split()
        tokens = [t for t in tokens if t in self.word2idx]

        if not tokens:
            return prompt, []

        # BUNDLE: estado = soma dos vetores GHRR das palavras do prompt
        ghrr_state = np.zeros((D, M, M), dtype=DTYPE)
        n_state = 0
        for t in tokens:
            gh = self._get_ghrr(t)
            if gh is not None:
                ghrr_state += gh
                n_state += 1

        if n_state == 0:
            return prompt, []

        ghrr_state = normalize_slices(ghrr_state)

        generated = []
        recent = tokens[-3:]
        excluded = set(tokens)  # não repetir palavras do prompt
        all_info = []

        for step in range(max_tokens):
            word, info = self.generate_token(
                ghrr_state=ghrr_state,
                recent_tokens=recent,
                excluded_tokens=excluded if len(excluded) > 0 else None,
                temperature=temperature,
            )
            info['step'] = step
            all_info.append(info)

            if verbose:
                ch = info.get('channel_scores', [0, 0, 0, 0])
                print(f"  [{step+1}] {word} "
                      f"(sim={ch[0]:.3f} attn={ch[1]:.3f} type={ch[2]:.3f} traj={ch[3]:.3f} | "
                      f"combined={info['combined_score']:.4f})")

            generated.append(word)
            recent = (recent + [word])[-3:]

            # Janela de exclusão: últimas N palavras não podem ser repetidas
            excluded = set(generated[-(exclude_window):]) if exclude_window > 0 else set()

            # EMA update: estado = (1-alpha)*estado + alpha*palavra
            word_ghrr = self._get_ghrr(word)
            if word_ghrr is not None:
                ghrr_state = normalize_slices((1.0 - ema_alpha) * ghrr_state + ema_alpha * word_ghrr)

        full_text = prompt + " " + " ".join(generated)
        return full_text, all_info
