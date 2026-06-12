"""
RadicalPathGenerator — ZERO cosine similarity for content selection.
Uses only: real corpus edges, Type Field, and M-path consistency.
"""
import numpy as np
from celn_v3.core import normalize, projective_resonance as M


class RadicalPathGenerator:
    """Generate text without measuring any vector-to-vector angle.

    Content comes from three non-cosine sources:
      1. Graph edges: real corpus transitions (source→follower pairs)
      2. Type Field: grammatical role restriction
      3. M-Path: continuation concentration via inverse entropy
    """

    def __init__(
        self,
        sem_vecs: np.ndarray,
        type_vecs: np.ndarray,
        w2i: dict[str, int],
        i2w: dict[int, str],
        pair_source_indices: np.ndarray,
        pair_follower_indices: np.ndarray,
        window_size: int = 5,
    ):
        self.sem_vecs = sem_vecs.astype(np.float32)
        self.type_vecs = type_vecs.astype(np.float32)
        self.w2i = w2i
        self.i2w = i2w
        self.vocab_size = sem_vecs.shape[0]
        self.window_size = window_size
        self.pair_src = pair_source_indices
        self.pair_fol = pair_follower_indices

        # Learn type_field from pairs
        type_dim = type_vecs.shape[1]
        self.type_field = np.zeros((self.vocab_size, type_dim), dtype=np.float32)

    def learn_type_field(self, sentences: list[list[str]]):
        accum = np.zeros((self.vocab_size, self.type_vecs.shape[1]), dtype=np.float32)
        counts = np.zeros(self.vocab_size, dtype=np.int32)
        for tokens in sentences:
            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i + 1]
                if w1 not in self.w2i or w2 not in self.w2i:
                    continue
                i1, i2 = self.w2i[w1], self.w2i[w2]
                accum[i1] += self.type_vecs[i2]
                counts[i1] += 1
        for i in range(self.vocab_size):
            if counts[i] >= 1:
                self.type_field[i] = normalize(accum[i] / counts[i])

    def _acceleration_scores(self, recent: list[int]) -> np.ndarray:
        """Diferença finita de 2ª ordem — penaliza guinadas bruscas.

        Para cada palavra do vocabulário, computa:
            accel = || v_w - 2*v_{previo} + v_{antes_previo} ||²
            score = 1 / (1 + sqrt(accel))  → inverso da penalização
        
        Trajetórias suaves mantêm aceleração baixa → score alto.
        """
        if len(recent) < 2:
            # sem histórico suficiente para aceleração → uniforme
            return np.ones(self.vocab_size, dtype=np.float32)
        
        prev = recent[-1]       # v_{previo}
        bprev = recent[-2]      # v_{antes_previo}
        
        v_prev = self.sem_vecs[prev]
        v_bprev = self.sem_vecs[bprev]
        
        # aceleração para CADA palavra do vocabulário (vectorizada)
        # shape: (vocab_size, D)
        accel = self.sem_vecs - 2.0 * v_prev + v_bprev  # broadcasting
        # penalização: norma L2 ao quadrado
        penalty = np.sum(accel ** 2, axis=1)  # (vocab_size,)
        # score: inverso da penalização (suavizado)
        scores = 1.0 / (1.0 + np.sqrt(penalty))
        return scores.astype(np.float32)
    def _graph_scores(self, recent: list[int], n_depth: int = 2) -> np.ndarray:
        """Count real corpus paths of length n_depth reaching each word."""
        scores = np.zeros(self.vocab_size, dtype=np.float32)
        for start_idx in recent:
            # depth-1 followers
            d1_mask = self.pair_src == start_idx
            d1_fols = self.pair_fol[d1_mask]
            if n_depth == 1:
                for fol in d1_fols:
                    scores[fol] += 1.0
                continue
            # depth-2 (follower of depth-1 follower)
            for fol in d1_fols:
                d2_mask = self.pair_src == fol
                if d2_mask.any():
                    for fol2 in self.pair_fol[d2_mask]:
                        scores[fol2] += 1.0
        return scores

    def generate(
        self,
        prefix_words: list[str],
        max_len: int = 15,
        temperature: float = 0.8,
        inhibition_window: int = 5,
        seed: int | None = None,
        n_depth: int = 2,
    ) -> list[str]:
        rng = np.random.RandomState(seed)

        prefix_indices = [self.w2i[w] for w in prefix_words if w in self.w2i]
        if not prefix_indices:
            idx = rng.randint(0, self.vocab_size)
            prefix_indices = [idx]

        generated: list[str] = []
        recent: list[int] = list(prefix_indices)

        # M-state
        m_state = self.sem_vecs[prefix_indices[0]].copy()
        for idx in prefix_indices[1:]:
            m_state = M(m_state, self.sem_vecs[idx], gamma=1.0, bilateral=True)

        for _ in range(max_len):
            excluded = set(recent[-inhibition_window:] if recent else [])
            last_idx = recent[-1]

            # ── Type channel ──
            type_target = self.type_field[last_idx]
            if np.linalg.norm(type_target) > 1e-12:
                type_scores = self.type_vecs @ type_target.astype(np.float32)
            else:
                type_scores = np.zeros(self.vocab_size, dtype=np.float32)
            for idx in excluded:
                type_scores[idx] = -1.0
            # normalize to [0,1]
            t_min = type_scores.min()
            if t_min < 0:
                type_scores = type_scores - t_min
            t_max = type_scores.max()
            if t_max > 1e-12:
                type_scores = type_scores / t_max

            # ── Acceleration channel (diferença finita de 2ª ordem) ──
            # score(w) = 1/(1 + ||v_w - 2*v_prev + v_before_prev||)
            # penaliza guinadas bruscas; favorece trajetórias suaves
            accel_scores = self._acceleration_scores(recent)
            for idx in excluded:
                accel_scores[idx] = -1.0
            a_min = accel_scores.min()
            if a_min < 0:
                accel_scores = accel_scores - a_min
            a_max = accel_scores.max()
            if a_max > 1e-12:
                accel_scores = accel_scores / a_max

            # ── M-Path consistency (inverse entropy of continuations) ──
            m_scores = np.zeros(self.vocab_size, dtype=np.float32)
            if m_state is not None:
                candidate_mask = (accel_scores > 0.0) | (type_scores > 0.3)
                cands = np.where(candidate_mask)[0]
                for c in cands:
                    if c in excluded:
                        continue
                    mask = self.pair_src == c
                    fols = self.pair_fol[mask]
                    if len(fols) > 0:
                        _, counts = np.unique(fols, return_counts=True)
                        probs = counts / counts.sum()
                        ent = -np.sum(probs * np.log(probs + 1e-12))
                        max_ent = np.log(len(counts) + 1e-12)
                        m_scores[c] = max(0.0, 1.0 - (ent / (max_ent + 1e-12)))

            m_min = m_scores.min()
            if m_min < 0:
                m_scores = m_scores - m_min
            m_max = m_scores.max()
            if m_max > 1e-12:
                m_scores = m_scores / m_max

            # ── Final: product of three channels ──
            # Type (structure) × Acceleration (smoothness) × M-Path (consistency)
            scores = type_scores * accel_scores * (m_scores + 0.1)
            for idx in excluded:
                scores[idx] = -1.0

            score_std = np.std(scores)
            eff_temp = temperature * max(score_std, 1e-6)
            centered = scores - scores.max()
            exp_s = np.exp(centered / eff_temp)
            probs = exp_s / (exp_s.sum() + 1e-12)

            idx = rng.choice(self.vocab_size, p=probs)
            next_word = self.i2w[idx]
            generated.append(next_word)

            m_state = M(m_state, self.sem_vecs[idx], gamma=1.0, bilateral=True)
            recent.append(idx)

        return generated
