"""
CELN v3 - GPVE Mouth
====================

Grammar Probabilistica Vetorialmente Enderecada.

This mouth generates text from an opaque projective-resonance thought state
without cosine similarity, nearest-neighbor search, or distance metrics in
10k dimensions. M_pr is converted to M_ctrl by PortAdapter; the mouth reads
known registers from M_ctrl and uses empirical rule/register associations to
reweight an induced PCFG.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import hashlib

from .core import phase_lens, phi, projective_resonance
from .port_adapter import PortAdapter, load_word_vectors, sentence_state
from .train import load_corpus, tokenize
from .pair_graph import PairGraph


@dataclass(frozen=True)
class GPVEConfig:
    start_symbol: str = "S"
    max_tokens: int = 32
    seed: int = 31415
    token_min_len: int = 2
    use_phase_lens: bool = False
    phase_lens_alpha: float = 0.5
    structure_only: bool = False


class PCFGRuleView:
    """Access a single PCFG rule's RHS and prob by index.

    Provides dict-like .get() for backward compatibility with
    methods like _first_surface_token and _rule_evidence_tokens.
    """
    __slots__ = ('_rhs', '_prob', '_count')
    def __init__(self, rhs: list, prob: float):
        self._rhs = rhs
        self._prob = prob
        self._count = prob  # count ≈ prob for PCFG magra
    def get(self, key, default=None):
        if key == 'rhs': return self._rhs
        if key == 'prob': return self._prob
        if key == 'count': return self._count
        return default
    def __getitem__(self, key):
        return self.get(key)


class PCFGIndex:
    """Binary PCFG index built from numpy arrays. O(1) access to rules."""

    def __init__(self, pcfg_binary_path: str | Path = "pcfg_binary.npz"):
        data = np.load(pcfg_binary_path, allow_pickle=True)
        self.lhs_names: np.ndarray = data['lhs_names']  # (n_lhs,) object
        self.lhs_index: np.ndarray = data['lhs_index']  # (n_lhs, 2) int32
        self.rhs_flat: np.ndarray = data['rhs_flat']    # (total_tokens,) object
        self.rhs_offsets: np.ndarray = data['rhs_offsets']  # (n_rules, 2) int32
        self.rule_probs: np.ndarray = data['rule_probs']  # (n_rules,) float32
        # Nonterm expansions
        self.ne_keys: np.ndarray = data['ne_keys']
        self.ne_vals: np.ndarray = data['ne_values_flat']
        self.ne_offsets: np.ndarray = data['ne_offsets']

        # Build name → index mapping
        self._lhs_map: dict[str, int] = {str(self.lhs_names[i]): i for i in range(len(self.lhs_names))}
        self._ne_map: dict[str, int] = {str(self.ne_keys[i]): i for i in range(len(self.ne_keys))}

        # Build dict-like nonterm_expansions for backward compat
        self.nonterm_expansions: dict[str, list[str]] = {}
        for i in range(len(self.ne_keys)):
            key = str(self.ne_keys[i])
            s, e = int(self.ne_offsets[i, 0]), int(self.ne_offsets[i, 1])
            self.nonterm_expansions[key] = [str(self.ne_vals[j]) for j in range(s, e)]

    def __getitem__(self, lhs: str) -> list[PCFGRuleView]:
        """Return list of rules for an LHS (backward compatible)."""
        idx = self._lhs_map.get(lhs)
        if idx is None:
            return []
        start = int(self.lhs_index[idx, 0])
        end = int(self.lhs_index[idx, 1])
        rules = []
        for r in range(start, end):
            s, e = int(self.rhs_offsets[r, 0]), int(self.rhs_offsets[r, 1])
            rhs = [str(self.rhs_flat[j]) for j in range(s, e)]
            rules.append(PCFGRuleView(rhs, float(self.rule_probs[r])))
        return rules

    def keys(self):
        return [str(n) for n in self.lhs_names]

    # ── Fast numpy access for scoring hot path ──

    def get_rule_info(self, lhs: str) -> tuple[int, int, np.ndarray]:
        """Return (n_rules, rule_start, probs_slice). O(1)."""
        idx = self._lhs_map.get(lhs)
        if idx is None:
            return 0, 0, np.array([], dtype=np.float32)
        start = int(self.lhs_index[idx, 0])
        end = int(self.lhs_index[idx, 1])
        return end - start, start, self.rule_probs[start:end].copy()

    def get_first_tokens(self, rule_start: int, n_rules: int) -> list[str | None]:
        """Return first surface token for each rule. O(n_rules)."""
        tokens = []
        for r in range(rule_start, rule_start + n_rules):
            s, e = int(self.rhs_offsets[r, 0]), int(self.rhs_offsets[r, 1])
            tokens.append(str(self.rhs_flat[s]) if e > s else None)
        return tokens

    def get_first_token(self, rule_idx: int) -> str | None:
        s, e = int(self.rhs_offsets[rule_idx, 0]), int(self.rhs_offsets[rule_idx, 1])
        return str(self.rhs_flat[s]) if e > s else None

    def get_rhs(self, rule_idx: int) -> list[str]:
        s, e = int(self.rhs_offsets[rule_idx, 0]), int(self.rhs_offsets[rule_idx, 1])
        return [str(self.rhs_flat[j]) for j in range(s, e)]

    def __len__(self):
        return len(self.lhs_names)

    def __contains__(self, item: str) -> bool:
        return item in self._lhs_map


class GPVEMouth:
    """PCFG generator controlled by PortAdapter registers."""

    def __init__(
        self,
        pcfg_binary: PCFGIndex,
        adapter: PortAdapter,
        vectors: np.ndarray,
        word2idx: dict[str, int],
        config: GPVEConfig | None = None,
        n_bins: int | None = None,
    ):
        self.pcfg_idx = pcfg_binary
        self.rules = pcfg_binary  # dict-like access (backward compat)
        self.nonterm_expansions = pcfg_binary.nonterm_expansions
        self.adapter = adapter
        self.vectors = vectors.astype(np.float32)
        self.word2idx = word2idx
        self.i2w = {i: w for w, i in word2idx.items()}
        self.config = config or GPVEConfig()

        rule_total = len(pcfg_binary.rule_probs)
        self.n_bins = int(n_bins or self._sturges_bins(max(2, rule_total)))
        self.bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1, dtype=np.float32)
        self.rule_counts: dict[str, np.ndarray] = {}
        self.lhs_bin_counts: dict[str, np.ndarray] = {}
        self.rule_bin_counts: dict[str, np.ndarray] = {}
        self.rule_surface_bins: dict[str, np.ndarray] = {}
        self._calibrated = False
        self._token_ffts: np.ndarray | None = None
        self._word2idx_cache: dict[str, int] | None = None
        self._plan_queue: list[str] | None = None
        self.debug_pcfg_enabled: bool = False
        self._debug_pcfg_log: list[dict] = []

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        pcfg_path: str | Path = "pcfg_pruned.json",
        vectors_path: str | Path = "celn_v3_full_vectors.npz",
        corpus_path: str | Path = "corpus_final.txt",
        adapter: PortAdapter | None = None,
        adapter_max_sentences: int | None = 512,
        n_ports: int = 64,
        seed: int = 31415,
        max_tokens: int = 32,
        token_min_len: int = 2,
        rule_calibration_corpus: str | Path | None = None,
        use_phase_lens: bool = False,
        phase_lens_alpha: float = 0.5,
        use_intent_distiller: bool = False,
        pair_graph_path: str | Path | None = None,
        structure_only: bool = False,
    ) -> "GPVEMouth":
        pcfg_idx = PCFGIndex()  # loads pcfg_binary.npz
        vectors, word2idx = load_word_vectors(vectors_path)
        state_transform = None
        if use_phase_lens:
            state_transform = lambda state: _phase_lensed_state(state, alpha=phase_lens_alpha)
        if adapter is None:
            adapter = PortAdapter.calibrate_from_corpus(
                corpus_path=corpus_path,
                vectors_path=vectors_path,
                n_ports=n_ports,
                max_sentences=adapter_max_sentences,
                seed=seed,
                min_token_len=token_min_len,
                state_transform=state_transform,
            )
        # Optional IntentDistiller (lightweight build)
        distiller = None
        if use_intent_distiller:
            try:
                # Import lazily to avoid top-level dependency issues
                from .intent_distiller import IntentDistiller
                distiller = IntentDistiller(vectors_path=vectors_path, corpus_path=corpus_path, sample_sentences=256, seed=seed)
                # Log successful init
                try:
                    log_d = Path('/tmp/opencode')
                    log_d.mkdir(parents=True, exist_ok=True)
                    with (log_d / 'intent_distiller_init.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps({'timestamp': time.time(), 'status': 'ok', 'sample_sentences': 256, 'seed': seed}) + '\n')
                except Exception:
                    pass
            except Exception as e:
                distiller = None
                try:
                    log_d = Path('/tmp/opencode')
                    log_d.mkdir(parents=True, exist_ok=True)
                    with (log_d / 'intent_distiller_init.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps({'timestamp': time.time(), 'status': 'failed', 'sample_sentences': 256, 'seed': seed, 'error': str(e)}) + '\n')
                except Exception:
                    pass
        mouth = cls(
            pcfg_binary=pcfg_idx,
            adapter=adapter,
            vectors=vectors,
            word2idx=word2idx,
            config=GPVEConfig(
                max_tokens=max_tokens,
                seed=seed,
                token_min_len=token_min_len,
                use_phase_lens=use_phase_lens,
                phase_lens_alpha=phase_lens_alpha,
                structure_only=structure_only,
            ),
        )
        # Build token sensor cache for semantic alignment scoring
        try:
            from .sensor_cache import TokenSensorCache
            mouth._sensor_cache = TokenSensorCache.build(vectors_path=vectors_path, adapter=adapter)
        except Exception:
            mouth._sensor_cache = None
        # Precompute FFTs for VSA-guided scoring
        try:
            mouth._token_ffts = np.fft.fft(mouth.vectors.astype(np.float32))
            mouth._word2idx_cache = dict(mouth.word2idx)
        except Exception:
            mouth._token_ffts = None
            mouth._word2idx_cache = None
        # Load Type Field for enriched M_pr generation
        try:
            tf_data = np.load(Path(vectors_path).parent / "celn_v3_type_field.npz", allow_pickle=True)
            mouth._type_field = tf_data["type_field"].astype(np.float32)
            tf_w2i = dict(tf_data["word2idx"].item())
            mouth._type_word2idx = {str(k): int(v) for k, v in tf_w2i.items()}
        except Exception:
            mouth._type_field = None
            mouth._type_word2idx = None
        # Load PairDict for hybrid candidate generation
        try:
            pair_path = Path("/tmp/opencode/pair_dict.json")
            if pair_path.exists():
                import json as _json
                pd_data = _json.load(open(pair_path))
                mouth._pair_fwd = {int(k): v for k, v in pd_data.get("fwd", {}).items()}
                mouth._pair_total = pd_data.get("total", 0)
            else:
                mouth._pair_fwd = None
                mouth._pair_total = 0
        except Exception:
            mouth._pair_fwd = None
            mouth._pair_total = 0

        # Precompute Type Field sensor readings for all vocab words
        try:
            tf_sensors = []
            for idx in range(len(mouth.vectors)):
                w = mouth._word2idx_cache and list(mouth._word2idx_cache.keys())[list(mouth._word2idx_cache.values()).index(idx)] \
                    if mouth._word2idx_cache and idx in list(mouth._word2idx_cache.values()) else None
                if w is not None and mouth._type_word2idx and w in mouth._type_word2idx:
                    t_idx = mouth._type_word2idx[w]
                    tv = mouth._type_field[t_idx]
                    if np.linalg.norm(tv) > 1e-12:
                        reading = adapter.sense(tv.astype(np.float32))
                        tf_sensors.append((w, reading.astype(np.float32)))
            mouth._type_sensor_cache = dict(tf_sensors)
        except Exception:
            mouth._type_sensor_cache = None

        mouth._intent_distiller = distiller
        # Load Knowledge Channel for factual scoring (5th channel in _vsa_scores)
        try:
            from .knowledge_channel import KnowledgeChannel
            mouth._knowledge_channel = KnowledgeChannel(
                Path(vectors_path).parent / "sentence_centroids.npz"
            )
        except Exception:
            mouth._knowledge_channel = None
        # Load spaCy vectors for comparison in 300d semantic space (7th channel)
        try:
            import spacy
            _nlp = spacy.load("pt_core_news_lg")
            _dim = 300
            _spacy_map = np.zeros((len(word2idx), _dim), dtype=np.float32)
            for w, idx in word2idx.items():
                if _nlp.vocab.has_vector(w):
                    _spacy_map[idx] = _nlp.vocab.get_vector(w).astype(np.float32)
            # Normalize rows
            _norms = np.linalg.norm(_spacy_map, axis=1, keepdims=True)
            _norms[_norms < 1e-12] = 1.0
            mouth._spacy_vecs = _spacy_map / _norms
        except Exception:
            mouth._spacy_vecs = None
        # Load PairGraph for trajectory coherence (6th channel in _vsa_scores)
        try:
            from .pair_graph import PairGraph
            pg_path = pair_graph_path or (Path(vectors_path).parent / "pair_graph.npz")
            mouth._pair_graph = PairGraph(pg_path)
        except Exception:
            mouth._pair_graph = None
        mouth.calibrate_rule_statistics(corpus_path=rule_calibration_corpus or corpus_path)
        return mouth

    @staticmethod
    def _sturges_bins(n: int) -> int:
        return int(np.ceil(np.log2(n) + 1.0))

    # ------------------------------------------------------------------
    # Calibration (minimal: PCFG magra, sem estatística)
    # ------------------------------------------------------------------
    def calibrate_rule_statistics(self, corpus_path: str | Path | None = None) -> None:
        """PCFG magra: sem calibração estatística.

        O VSA-Guided Scoring + RES substituem todos os canais de controle
        (PMI, bins, dinâmico). A PCFG fornece apenas a estrutura sintática
        (RHS das regras) e probabilidades base da indução.
        """
        self._calibrated = True

    def _derive_rule_choices(self, tokens: list[str]) -> list[tuple[str, int]]:
        lhs = self.config.start_symbol
        pos = 0
        choices: list[tuple[str, int]] = []
        lhs = self.config.start_symbol
        pos = 0
        choices: list[tuple[str, int]] = []
        for _ in range(max(self.config.max_tokens * 4, 64)):
            rlist = self.rules.get(lhs)
            if not rlist:
                return choices
            best: tuple[tuple[float, float, int], int, int, str | None] | None = None
            for ridx, rule in enumerate(rlist):
                pos2 = pos
                next_lhs: str | None = None
                consumed = 0
                ok = True
                for sym in rule.get("rhs", []):
                    surface = self._symbol_literal_surface(sym)
                    if surface is None:
                        if next_lhs is not None:
                            ok = False
                            break
                        next_lhs = sym
                        continue
                    n = len(surface)
                    if tokens[pos2:pos2 + n] != surface:
                        ok = False
                        break
                    pos2 += n
                    consumed += n
                if not ok:
                    continue
                key = (float(consumed), float(rule.get("prob", 0.0)), -ridx)
                if best is None or key > best[0]:
                    best = (key, ridx, pos2, next_lhs)
            if best is None:
                return choices
            _, ridx, pos, next_lhs = best
            choices.append((lhs, ridx))
            if next_lhs is None:
                return choices if pos == len(tokens) else []
            lhs = next_lhs
        return []

    def _symbol_literal_surface(self, sym: str) -> list[str] | None:
        if sym in self.nonterm_expansions:
            return list(self.nonterm_expansions[sym])
        if sym in self.rules:
            return None
        return [sym]

    def _rule_evidence_tokens(self, rule: dict[str, Any], max_tokens: int) -> list[str]:
        """Return a bounded PCFG surface for one rule.

        A rule such as `S -> token @BIN...` only exposes its first token
        directly, but the continuation is part of the same generation path.
        Expanding continuations through their highest-probability rules gives
        the rule a richer empirical register signature without templates or
        manually named linguistic categories.
        """
        tokens: list[str] = []
        for sym in rule.get("rhs", []):
            tokens.extend(self._symbol_surface(sym, max_tokens - len(tokens), depth=0))
            if len(tokens) >= max_tokens:
                break
        return tokens[:max_tokens]

    def _symbol_surface(self, sym: str, budget: int, depth: int) -> list[str]:
        if budget <= 0 or depth > self.config.max_tokens:
            return []
        if sym in self.nonterm_expansions:
            tokens: list[str] = []
            for child in self.nonterm_expansions[sym]:
                tokens.extend(self._symbol_surface(child, budget - len(tokens), depth + 1))
                if len(tokens) >= budget:
                    break
            return tokens[:budget]
        if sym not in self.rules:
            return [sym]

        rlist = self.rules[sym]
        if not rlist:
            return []
        best = max(rlist, key=lambda r: (float(r.get("prob", 0.0)), float(r.get("count", 0.0))))
        tokens: list[str] = []
        for child in best.get("rhs", []):
            tokens.extend(self._symbol_surface(child, budget - len(tokens), depth + 1))
            if len(tokens) >= budget:
                break
        return tokens

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate_from_text(
        self,
        text: str,
        sample: bool = True,
        max_tokens: int | None = None,
        beam_width: int = 1,
    ) -> str:
        tokens = tokenize(text, min_len=self.config.token_min_len)
        tf = getattr(self, '_type_field', None)
        tf_w2i = getattr(self, '_type_word2idx', None)
        if tf is not None and tf_w2i is not None:
            from .port_adapter import sentence_state_positional
            state_dict = sentence_state_positional(
                tokens, self.vectors, self.word2idx,
                type_field=tf, type_word2idx=tf_w2i,
                gamma=self.adapter.config.gamma,
                bilateral=self.adapter.config.bilateral,
            )
            m_pr = state_dict['m_pr']
        else:
            from .port_adapter import sentence_state
            state_dict = None
            m_pr = sentence_state(
                tokens, self.vectors, self.word2idx,
                gamma=self.adapter.config.gamma,
                bilateral=self.adapter.config.bilateral,
            )
        if beam_width > 1:
            return self.generate_beam(
                m_pr, beam_width=beam_width, max_tokens=max_tokens, prompt_tokens=tokens
            )
        return self.generate(m_pr, sample=sample, max_tokens=max_tokens, prompt_tokens=tokens, position_targets=state_dict)

    def generate(
        self,
        m_pr: np.ndarray,
        sample: bool = True,
        max_tokens: int | None = None,
        prompt_tokens: list[str] | None = None,
        position_targets: dict | None = None,
    ) -> str:
        if not self._calibrated:
            self.calibrate_rule_statistics()

        # If an IntentDistiller is attached, attempt to distill the intent with token window
        distiller = getattr(self, '_intent_distiller', None)
        m_for_adapter = m_pr
        if distiller is not None:
            pkt = None
            exc_info = None
            try:
                pkt = distiller.distill(m_pr, tokens=prompt_tokens)
                m_for_adapter = pkt.m_intent
            except Exception as e:
                # Record exception and fall back to raw m_pr
                exc_info = str(e)
                m_for_adapter = m_pr

            # Best-effort log entry collected here (do not raise on failure)
            try:
                log_dir = Path('/tmp/opencode')
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / 'intent_distiller_logs.jsonl'
                entry: dict[str, Any] = {}
                entry['timestamp'] = time.time()
                entry['n_tokens'] = len(prompt_tokens) if prompt_tokens else 0
                entry['n_ports'] = int(self.adapter.n_ports)
                entry['phase_lens'] = bool(self.config.use_phase_lens)
                if pkt is not None:
                    entry['n_tokens_used'] = pkt.n_tokens_used
                    entry['alphas_used'] = str(pkt.alphas_used[:6]) if pkt.alphas_used else []
                    entry['min_alpha'] = pkt.min_alpha
                    entry['max_alpha'] = pkt.max_alpha
                    entry['similarity_shift'] = pkt.similarity_shift
                    entry['capl_conf'] = float(pkt.confidences.get('similarity_shift', 0.0))
                else:
                    entry['n_tokens_used'] = 0
                    entry['alphas_used'] = []
                    entry['min_alpha'] = 0.0
                    entry['max_alpha'] = 0.0
                    entry['similarity_shift'] = 0.0
                    entry['capl_conf'] = 0.0
                if exc_info is not None:
                    entry['distill_exception'] = exc_info
                # append to JSONL
                with log_path.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(entry) + '\n')
            except Exception:
                pass

        # Store prompt tokens for KnowledgeChannel + PairGraph
        self._current_prompt_tokens = prompt_tokens
        self._last_word = prompt_tokens[-1] if prompt_tokens else None

        m_ctrl = self.adapter.to_control_state(self._state_for_adapter(m_for_adapter))
        registers = self.adapter.read_registers(m_ctrl)
        target_bins = self._bins(registers)
        rng = np.random.RandomState(self.config.seed)

        output: list[str] = []
        pending = [self.config.start_symbol]
        limit = int(max_tokens or self.config.max_tokens)
        current_vsa_state = m_for_adapter.copy()

        from .core import phi, encode_sequence, normalize
        # Precompute positional target states via phi destillation
        pos_targets = []
        n_parts = 0
        if position_targets is not None:
            targets = position_targets.get('m_targets')
            n_parts = position_targets.get('n_parts', 0)
            if targets and n_parts > 0:
                for v in targets:
                    if v is not None:
                        v_phi = phi(v)
                        pos_targets.append(v_phi.astype(np.float32))
                    else:
                        pos_targets.append(None)

        # Precompute Type Field trajectory for global coherence
        self._type_traj_targets = []
        type_tf = getattr(self, '_type_field', None)
        type_w2i = getattr(self, '_type_word2idx', None)
        if type_tf is not None and type_w2i is not None and prompt_tokens and n_parts > 0:
            n_tok = len(prompt_tokens)
            for i in range(n_parts):
                s = (i * n_tok) // n_parts
                e = ((i + 1) * n_tok) // n_parts
                type_vecs = []
                for j in range(s, e):
                    idx = type_w2i.get(prompt_tokens[j])
                    if idx is not None and np.linalg.norm(type_tf[idx]) > 1e-12:
                        type_vecs.append(type_tf[idx])
                if type_vecs:
                    type_state = normalize(encode_sequence(type_vecs, gamma=1.0, bilateral=True))
                    self._type_traj_targets.append(type_state.astype(np.float32))
                else:
                    self._type_traj_targets.append(None)
        else:
            self._type_traj_targets = []

        # Precompute anchor magnitude spectra from prompt tokens
        self._anchor_mags = []
        if prompt_tokens and hasattr(self, '_token_ffts'):
            for tok in prompt_tokens:
                idx = self.word2idx.get(tok)
                if idx is not None:
                    mag = np.abs(self._token_ffts[idx])
                    nrm = np.linalg.norm(mag)
                    if nrm > 1e-12:
                        self._anchor_mags.append((mag / nrm).astype(np.float32))
        if not self._anchor_mags:
            self._anchor_mags = None

        while pending and len(output) < limit:
            sym = pending.pop(0)
            if sym in self.nonterm_expansions:
                pending = list(self.nonterm_expansions[sym]) + pending
                continue
            if sym not in self.rules:
                output.append(sym)
                self._last_word = sym
                # Consume plan queue if plan word generated
                pq = getattr(self, '_plan_queue', None)
                if pq and len(pq) > 0 and sym == pq[0]:
                    self._plan_queue = pq[1:]
                idx = self.word2idx.get(sym)
                if idx is not None:
                    tok_vec = self.vectors[idx].astype(np.float32)
                    current_vsa_state = projective_resonance(current_vsa_state, tok_vec, gamma=1.0, bilateral=True)
                continue

            # Choose position-appropriate target (phi-destilled state)
            progress = len(output) / max(1, limit)
            if n_parts > 0 and len(pos_targets) == n_parts:
                part_idx = min(int(progress * n_parts), n_parts - 1)
                current_target = pos_targets[part_idx] if pos_targets[part_idx] is not None else phi(m_for_adapter)
            else:
                current_target = phi(m_for_adapter)

            # Type Field trajectory target for this position
            current_type_target = None
            if self._type_traj_targets and n_parts > 0:
                part_idx = min(int(progress * n_parts), n_parts - 1)
                if part_idx < len(self._type_traj_targets) and self._type_traj_targets[part_idx] is not None:
                    current_type_target = self.adapter.sense(self._type_traj_targets[part_idx]).astype(np.float32)

            ridx = self._choose_rule(sym, target_bins, rng, sample=sample, prefix_tokens=output, target_readings=current_target, current_state=current_vsa_state, type_target=current_type_target, structure_only=self.config.structure_only)

            # ── Debug PCFG ──
            if self.debug_pcfg_enabled:
                vsa_d = getattr(self, '_debug_last_vsa', None)
                rule_d = getattr(self, '_debug_last_rule', None)
                step_d = {'step': len(output), 'lhs': sym, 'ridx': int(ridx)}
                if vsa_d is not None:
                    ft = vsa_d.get('first_tokens', [])
                    step_d['first_tokens'] = list(ft)
                    step_d['winner_token'] = ft[ridx] if 0 <= ridx < len(ft) else None
                    npcfg = vsa_d.get('n_rules', 0)
                    if vsa_d.get('knowledge_primary', False):
                        step_d['source'] = 'PairGraph (knowledge_primary)'
                    elif ridx >= npcfg and vsa_d.get('knowledge_tokens'):
                        step_d['source'] = 'Knowledge (extended)'
                    elif ridx < npcfg:
                        step_d['source'] = 'PCFG'
                    else:
                        step_d['source'] = 'TypeField (extension)'
                    step_d['n_rules_pcfg'] = npcfg
                    wtok = step_d['winner_token']
                    ct = vsa_d.get('ch_tokens', [])
                    if wtok is not None and wtok in ct:
                        i = ct.index(wtok)
                        w = vsa_d.get('weights', [])
                        step_d['ch_anchor'] = float(vsa_d['ch_anchor'][i])
                        step_d['ch_mag'] = float(vsa_d['ch_mag'][i])
                        step_d['ch_phase'] = float(vsa_d['ch_phase'][i])
                        step_d['ch_type'] = float(vsa_d['ch_type'][i])
                        step_d['ch_sdm'] = float(vsa_d['ch_sdm'][i])
                        step_d['ch_traj'] = float(vsa_d['ch_traj'][i])
                        step_d['ch_spacy'] = float(vsa_d['ch_spacy'][i])
                        ch_raw = {'Anchor': step_d['ch_anchor'], 'Magnitude': step_d['ch_mag'], 'Phase': step_d['ch_phase'], 'Type': step_d['ch_type'], 'SDM': step_d['ch_sdm'], 'Trajectory': step_d['ch_traj'], 'spaCy': step_d['ch_spacy']}
                        if rule_d is not None and ridx < len(rule_d.get('base_norm', [])):
                            ch_raw['PCFG'] = float(rule_d['base_norm'][ridx])
                        else:
                            ch_raw['PCFG'] = 0.0
                        if len(w) == 7 and wtok in ct:
                            i2 = ct.index(wtok)
                            wtd = {k: round(float(ch_raw[k]) * float(w[j]) if k != 'PCFG' else float(ch_raw[k]), 4) for j, k in enumerate(['Anchor','Magnitude','Phase','Type','SDM','Trajectory','spaCy'])}
                            wtd['PCFG'] = ch_raw['PCFG']
                            step_d['weighted_ch'] = wtd
                            dom = max(wtd, key=wtd.get)
                            step_d['dominant_channel'] = dom
                            step_d['dominant_val'] = wtd[dom]
                self._debug_pcfg_log.append(step_d)

            # Check if knowledge is primary (PairGraph replace PCFG)
            kc_tokens = getattr(self, '_last_kc_tokens', None)
            if getattr(self, '_knowledge_primary', False) and kc_tokens:
                if 0 <= ridx < len(kc_tokens):
                    first_tok = kc_tokens[ridx]
                    output.append(first_tok)
                    self._last_word = first_tok
                    pq = getattr(self, '_plan_queue', None)
                    if pq and len(pq) > 0 and first_tok == pq[0]:
                        self._plan_queue = pq[1:]
                    idx = self.word2idx.get(first_tok)
                    if idx is not None:
                        tok_vec = self.vectors[idx].astype(np.float32)
                        current_vsa_state = projective_resonance(current_vsa_state, tok_vec, gamma=1.0, bilateral=True)
                    # Restart PCFG from root for next iteration
                    pending = [self.config.start_symbol]
                    continue
            # Extended mode: ridx beyond PCFG rules
            n_rules_sym, _, _ = self.pcfg_idx.get_rule_info(sym)
            if n_rules_sym > 0 and ridx >= n_rules_sym and kc_tokens:
                kc_idx = ridx - n_rules_sym
                if kc_idx < len(kc_tokens):
                    first_tok = kc_tokens[kc_idx]
                    output.append(first_tok)
                    self._last_word = first_tok
                    pq = getattr(self, '_plan_queue', None)
                    if pq and len(pq) > 0 and first_tok == pq[0]:
                        self._plan_queue = pq[1:]
                    idx = self.word2idx.get(first_tok)
                    if idx is not None:
                        tok_vec = self.vectors[idx].astype(np.float32)
                        current_vsa_state = projective_resonance(current_vsa_state, tok_vec, gamma=1.0, bilateral=True)
                    continue  # No rule expansion needed (single token)
            rhs = list(self.rules[sym][ridx].get("rhs", []))
            pending = rhs + pending
            selected_rule = self.rules[sym][ridx]
            first_tok = self._first_surface_token(selected_rule)
            if first_tok is not None:
                self._last_word = first_tok
                pq = getattr(self, '_plan_queue', None)
                if pq and len(pq) > 0 and first_tok == pq[0]:
                    self._plan_queue = pq[1:]
                idx = self.word2idx.get(first_tok)
                if idx is not None:
                    tok_vec = self.vectors[idx].astype(np.float32)
                    current_vsa_state = projective_resonance(current_vsa_state, tok_vec, gamma=1.0, bilateral=True)

        return self._surface(output)

    def generate_beam(
        self,
        m_pr: np.ndarray,
        beam_width: int = 5,
        max_tokens: int = 24,
        prompt_tokens: list[str] | None = None,
    ) -> str:
        """Generate beam_width samples with sampling, rerank by hybrid score.

        Gera K amostras com sample=True, depois reordena cada frase completa
        pelo score híbrido (magnitude + phase + anchor):

          1. magnitude: resonance_score(estado_frase, target_phi)
          2. phase: pearsonr(sense(phi(estado_frase)), sense(phi(target)))
          3. anchor: média do max resonance_score de cada token da frase
             com as palavras-âncora do prompt

        A amostra com maior score híbrido é selecionada.
        Isso dá VISÃO GLOBAL — avalia a frase completa contra o pensamento.
        """
        if not self._calibrated:
            self.calibrate_rule_statistics()
        if beam_width <= 1:
            return self.generate(m_pr, sample=False, max_tokens=max_tokens, prompt_tokens=prompt_tokens)

        # Compute target once (CAPL-transformed)
        distiller = getattr(self, '_intent_distiller', None)
        m_target = m_pr
        if distiller is not None:
            try:
                pkt = distiller.distill(m_pr, tokens=prompt_tokens)
                m_target = pkt.m_intent
            except Exception:
                pass

        from .core import phi
        target_phi = phi(m_target)
        target_phi_reading = self.adapter.sense(target_phi).astype(np.float32)
        target_mag = np.abs(np.fft.fft(target_phi.astype(np.float32)))
        tn = np.linalg.norm(target_mag)
        if tn > 1e-12:
            target_mag = target_mag / tn

        # Precompute anchor magnitude spectra from prompt tokens
        anchor_mags = []
        if prompt_tokens and hasattr(self, '_token_ffts'):
            for tok in prompt_tokens:
                idx = self.word2idx.get(tok)
                if idx is not None:
                    mag = np.abs(self._token_ffts[idx])
                    nrm = np.linalg.norm(mag)
                    if nrm > 1e-12:
                        anchor_mags.append((mag / nrm).astype(np.float32))

        from .port_adapter import sentence_state
        cache = getattr(self, '_sensor_cache', None)
        if cache is None:
            return self.generate(m_pr, sample=False, max_tokens=max_tokens, prompt_tokens=prompt_tokens)

        best_text = ""
        best_score = -1.0
        for trial in range(beam_width):
            text = self.generate(m_pr, sample=True, max_tokens=max_tokens, prompt_tokens=prompt_tokens)
            try:
                from .train import tokenize
                toks = tokenize(text, min_len=self.config.token_min_len)
                if not toks:
                    continue
                sent_state = sentence_state(toks, self.vectors, self.word2idx, gamma=1.0, bilateral=True)

                # 1. Magnitude: resonance_score(sent_state, target_phi)
                sent_mag = np.abs(np.fft.fft(sent_state.astype(np.float32)))
                sn = np.linalg.norm(sent_mag)
                magnitude = float(np.dot(sent_mag / sn, target_mag)) if sn > 1e-12 else 0.0

                # 2. Phase: pearsonr(sense(phi(sent_state)), sense(phi(target)))
                sent_phi = phi(sent_state)
                sent_phi_reading = self.adapter.sense(sent_phi).astype(np.float32)
                phase = cache.correlate(sent_phi_reading, target_phi_reading)

                # 3. Anchor: mean per-token resonance with prompt words
                anchor = 0.0
                if anchor_mags:
                    n_anchored = 0
                    for tok in toks:
                        idx = self.word2idx.get(tok)
                        if idx is not None and hasattr(self, '_token_ffts'):
                            cand_mag = np.abs(np.fft.fft(self.vectors[idx].astype(np.float32)))
                            cn = np.linalg.norm(cand_mag)
                            if cn > 1e-12:
                                cand_mag_norm = cand_mag / cn
                                best_a = max(float(np.dot(cand_mag_norm, a)) for a in anchor_mags)
                                anchor += best_a
                                n_anchored += 1
                    if n_anchored > 0:
                        anchor = float(np.clip(anchor / n_anchored, -1.0, 1.0))

                score = magnitude + phase + anchor
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_text = text
        return best_text if best_text else self.generate(m_pr, sample=False, max_tokens=max_tokens, prompt_tokens=prompt_tokens)

    def _state_for_adapter(self, state: np.ndarray) -> np.ndarray:
        if not self.config.use_phase_lens:
            return state
        return _phase_lensed_state(state, alpha=self.config.phase_lens_alpha)

    def _choose_rule(
        self,
        lhs: str,
        target_bins: np.ndarray,
        rng: np.random.RandomState,
        sample: bool,
        prefix_tokens: list[str],
        target_readings: np.ndarray | None = None,
        current_state: np.ndarray | None = None,
        type_target: np.ndarray | None = None,
        structure_only: bool = False,
    ) -> int:
        scores = self._rule_scores(lhs, target_bins, prefix_tokens, target_readings=target_readings, current_state=current_state, type_target=type_target, structure_only=structure_only)
        return self._sample_from_scores(scores, rng, sample)

    def _sample_from_scores(
        self,
        scores: np.ndarray,
        rng: np.random.RandomState,
        sample: bool,
    ) -> int:
        """Temperature-based selection from scores array."""
        if not sample or len(scores) <= 1:
            return int(np.argmax(scores))

        scale = self._score_scale(scores)
        centered = scores - np.max(scores)
        weights = np.exp(centered / scale)
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total):
            return int(np.argmax(scores))
        probs = weights / total
        return int(rng.choice(len(scores), p=probs))

    def _first_surface_token(self, rule: dict[str, Any]) -> str | None:
        """Extract the first surface token from a rule's RHS."""
        rhs = rule.get("rhs", [])
        if not rhs:
            return None
        first = rhs[0]
        if first in self.nonterm_expansions:
            children = self.nonterm_expansions[first]
            return children[0] if children else None
        if first in self.rules:
            sub_rules = self.rules[first]
            if not sub_rules:
                return None
            best = max(sub_rules, key=lambda r: (float(r.get("prob", 0.0)), float(r.get("count", 0.0))))
            return self._first_surface_token(best)
        return first

    def _first_surface_token_fast(self, rule_idx: int) -> str | None:
        """Fast first surface token via binary index. O(depth)."""
        rhs = self.pcfg_idx.get_rhs(rule_idx)
        if not rhs:
            return None
        first = rhs[0]
        if first in self.nonterm_expansions:
            children = self.nonterm_expansions[first]
            return children[0] if children else None
        if first in self.rules:
            sub_n, sub_start, sub_probs = self.pcfg_idx.get_rule_info(first)
            if sub_n > 0:
                best_idx = sub_start + int(np.argmax(sub_probs))
                return self._first_surface_token_fast(best_idx)
            return None
        return first

    def _fast_project_resonance(self, curr_fft: np.ndarray, tok_fft: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Fast projective_resonance using precomputed FFTs. Returns the resulting state."""
        mag_x = np.abs(curr_fft)
        mag_y = np.abs(tok_fft)
        ratio = mag_y / (mag_x + 1e-12)
        median_ratio = np.median(ratio)
        rel_weight = ratio / max(median_ratio, 1e-12)
        weight_mag = np.tanh(rel_weight ** gamma)
        result_spectrum = curr_fft * tok_fft * weight_mag
        result = np.fft.ifft(result_spectrum).real
        from .core import normalize
        return normalize(result.astype(np.float32))

    @staticmethod
    def _band_energy(v: np.ndarray, n_bands: int = 10) -> np.ndarray:
        """Compute band energy signature in the Fourier domain.

        Divide the magnitude spectrum into n_bands log-spaced bands,
        compute energy in each band, normalize.

        Returns: (n_bands,) signature — energy distribution across bands.
        Self-calibrating: band boundaries adapt to D (log-spaced).
        No magic thresholds: bands cover the frequency range naturally.
        """
        mag = np.abs(np.fft.fft(v))
        D = len(mag)
        sig = np.zeros(n_bands, dtype=np.float32)
        half = D // 2
        mag_half = mag[:half]
        for i in range(n_bands):
            lo = max(0, int(half * (2 ** (-n_bands + i))))
            hi = min(half, int(half * (2 ** (-n_bands + i + 1))))
            if hi > lo:
                sig[i] = float(np.sum(mag_half[lo:hi]))
        total = float(np.sum(sig))
        if total > 1e-12:
            sig /= total
        return sig

    def _get_type_field_candidates(self, lhs: str, pcfg_tokens: list[str], last_word: str | None = None) -> list[str]:
        """Extend candidates with Type Field-validated words (beyond PCFG).

        Hybrid architecture:
          1. Type Field: find top-50 words whose syntactic role matches expected type
             (pearsonr in 128-dim sensor space, NOT 10k similarity)
          2. PairDict: rank by transition confidence (how often does last_word → candidate?)
          3. PCFG validation: keep only candidates that a rule for this LHS could produce
          4. Return extended list (PCFG tokens + top Type Field candidates)

        This allows generating words that NEVER appeared together in the corpus,
        as long as their syntactic roles match and the transition is plausible.
        """
        if not self._type_sensor_cache or not self.pcfg_idx:
            return pcfg_tokens

        # Get the expected type reading (from current position in type trajectory)
        tf_target = getattr(self, '_last_type_target', None)
        if tf_target is None:
            return pcfg_tokens

        # Score all words by type alignment in sensor space
        scores: list[tuple[float, str]] = []
        for w, reading in self._type_sensor_cache.items():
            # pearsonr in 128-dim sensor space (NOT 10k similarity)
            r = self._sensor_cache.correlate(reading, tf_target)
            if r > 0.3:  # reasonable type match
                scores.append((r, w))

        scores.sort(key=lambda x: -x[0])
        type_candidates = [w for _, w in scores[:50]]

        # Cross-reference: which PCFG-valid tokens are already covered?
        pcfg_set = set(t for t in pcfg_tokens if t is not None)

        # Add Type Field candidates that pass PCFG validation
        # (check if the word appears as a rule's first token for this LHS)
        n_rules, rule_start, _ = self.pcfg_idx.get_rule_info(lhs)
        pcfg_valid_tokens = set()
        for r in range(n_rules):
            tok = self._first_surface_token_fast(rule_start + r)
            if tok is not None:
                pcfg_valid_tokens.add(tok)

        new_tokens = []
        for w in type_candidates:
            if w not in pcfg_set and w in pcfg_valid_tokens:
                new_tokens.append(w)
            elif w not in pcfg_set and w in self.word2idx:
                # Accept non-PCFG candidates too (novel transitions)
                # They'll get base_prob=0 but might win via type/vsa channels
                new_tokens.append(w)

        # Deduplicate and return extended list
        extended = list(dict.fromkeys(pcfg_tokens + new_tokens))
        return extended

    def set_plan(self, plan: list[str] | None):
        """Define o plano textual para modo plan-guided."""
        self._plan_queue = list(plan) if plan else None

    def _knowledge_candidates(self, last_word: str | None) -> list[str]:
        """Gera candidatos do Transporte Paralelo (PairGraph).

        Único sinal: contagem de transições no PairGraph.
        Se a transição last_word → follower é frequente no corpus,
        a transição é sintaticamente válida — não precisa de validação
        adicional por Type Field ou PCFG.

        Retorna lista de palavras candidatas, ou [] se não houver.
        """
        pg = getattr(self, '_pair_graph', None)
        if pg is None or not last_word:
            return []
        w2i = getattr(self, '_word2idx_cache', None)
        if w2i is None:
            return []
        idx = w2i.get(last_word)
        if idx is None:
            return []
        followers = pg.get_followers(idx, top_k=5)
        if not followers:
            return []
        i2w = {i: w for w, i in w2i.items()}
        tokens = []
        for f_idx in followers:
            w = i2w.get(int(f_idx))
            if w and w not in tokens:
                tokens.append(w)
        return tokens

    def _vsa_scores(self, lhs: str, target_readings: np.ndarray, current_state: np.ndarray, type_target: np.ndarray | None = None) -> np.ndarray:
        """Hybrid score via fast binary PCFG index + Type Field extension.

        Gera candidatos de duas fontes:
          1. PCFG: regras sintáticas (5-10 candidatos)
          2. Type Field: palavras com papel sintático similar (top-50 por pearsonr em 128 dims)

        Para candidatos do Type Field não presentes na PCFG, cria "regras virtuais"
        com prob=0. A pontuação por 4 canais decide a melhor escolha.
        Isso permite gerar palavras que nunca apareceram em regras PCFG juntas.
        """
        n_rules, rule_start, _ = self.pcfg_idx.get_rule_info(lhs)
        n_base = max(n_rules, 1)
        w2i = self._word2idx_cache
        from .core import phi, normalize

        # PCFG candidates (fallback syntactic source)
        pcfg_tokens = []
        for r in range(n_rules):
            tok = self._first_surface_token_fast(rule_start + r)
            pcfg_tokens.append(tok)

        # Knowledge candidates: PairGraph get_followers (plan queue > PairGraph)
        last_word = getattr(self, '_last_word', None)
        plan_q = getattr(self, '_plan_queue', None)
        if plan_q and len(plan_q) > 0:
            knowledge_tokens = [plan_q[0]]
        else:
            knowledge_tokens = self._knowledge_candidates(last_word)

        # PRIMARY: if knowledge produces >= 2 candidates, REPLACE PCFG
        if knowledge_tokens and len(knowledge_tokens) >= 2:
            first_tokens = list(knowledge_tokens)  # no dedup needed (single token per concept)
            unique_tokens = list(dict.fromkeys(first_tokens))
            self._last_kc_tokens = knowledge_tokens
            self._knowledge_primary = True
        else:
            # FALLBACK: PCFG (one per rule, not deduplicated) + knowledge extras
            first_tokens = list(pcfg_tokens)  # full list, NOT deduplicated
            if knowledge_tokens:
                seen = set(t for t in first_tokens if t is not None)
                for kt in knowledge_tokens:
                    if kt not in seen:
                        first_tokens.append(kt)
                        seen.add(kt)
                self._last_kc_tokens = knowledge_tokens
            else:
                self._last_kc_tokens = None
            self._knowledge_primary = False
            unique_tokens = list(dict.fromkeys(first_tokens))

        if self._token_ffts is None or self._sensor_cache is None:
            return np.zeros(len(first_tokens), dtype=np.float32)

        curr_fft = np.fft.fft(current_state.astype(np.float32))

        target_mag = np.abs(np.fft.fft(target_readings.astype(np.float32)))
        tn = np.linalg.norm(target_mag)
        if tn > 1e-12: target_mag = target_mag / tn
        target_phi = phi(target_readings.astype(np.float32))
        target_phi_reading = self.adapter.sense(target_phi).astype(np.float32)

        anchor_mags = getattr(self, '_anchor_mags', None)

        # Precompute KnowledgeChannel vector for 5th channel (factual relevance)
        kc = getattr(self, '_knowledge_channel', None)
        kc_knowledge_vec = None
        pt = getattr(self, '_current_prompt_tokens', None)
        if kc is not None and pt:
            known = [t for t in pt if t in w2i]
            if known:
                kc_knowledge_vec = normalize(
                    kc.query_and_read(known, self.vectors, w2i)
                )

        # Collect per-channel scores for each unique token
        ch_anchor: list[float] = []
        ch_mag: list[float] = []
        ch_phase: list[float] = []
        ch_type: list[float] = []
        ch_sdm: list[float] = []
        ch_traj: list[float] = []
        ch_spacy: list[float] = []
        ch_tokens: list[str] = []

        # Precompute spaCy context vector (300d semantic space for 7th channel)
        spacy_map = getattr(self, '_spacy_vecs', None)
        spacy_context = None
        if spacy_map is not None and pt:
            known = [t for t in pt if t in w2i]
            if known:
                ctx_spacy = np.mean([spacy_map[w2i[t]] for t in known], axis=0).astype(np.float32)
                cn = np.linalg.norm(ctx_spacy)
                if cn > 1e-12:
                    spacy_context = ctx_spacy / cn

        # Precompute PairGraph for 6th channel (trajectory coherence)
        pg = getattr(self, '_pair_graph', None)
        m_intent = normalize(target_readings) if pg is not None else None

        for tok in unique_tokens:
            if tok is None or tok not in w2i:
                ch_anchor.append(-1.0); ch_mag.append(0.0); ch_phase.append(0.0); ch_type.append(0.0); ch_sdm.append(0.0); ch_traj.append(0.0); ch_spacy.append(0.0)
                ch_tokens.append(tok)
                continue
            idx = w2i[tok]
            tok_fft = self._token_ffts[idx]

            # Anchor
            anchor = 0.0
            if anchor_mags is not None:
                cand_mag = np.abs(tok_fft)
                cn = np.linalg.norm(cand_mag)
                if cn > 1e-12:
                    cand_mag_norm = cand_mag / cn
                    for a_mag in anchor_mags:
                        s = float(np.dot(cand_mag_norm, a_mag))
                        if s > anchor: anchor = s
            anchor = float(np.clip(anchor, -1.0, 1.0))

            M_hyp = self._fast_project_resonance(curr_fft, tok_fft)

            # Magnitude
            m = np.abs(np.fft.fft(M_hyp))
            nm = np.linalg.norm(m)
            magnitude = float(np.dot(m / nm, target_mag)) if nm > 1e-12 else 0.0

            # Phase
            M_phi = phi(M_hyp)
            phi_reading = self.adapter.sense(M_phi).astype(np.float32)
            phase = self._sensor_cache.correlate(phi_reading, target_phi_reading)

            # Type
            type_score = 0.0
            if type_target is not None and self._type_field is not None:
                t_idx = self._type_word2idx.get(tok)
                if t_idx is not None:
                    cand_type = self._type_field[t_idx]
                    if np.linalg.norm(cand_type) > 1e-12:
                        cand_type_reading = self.adapter.sense(cand_type).astype(np.float32)
                        type_score = self._sensor_cache.correlate(cand_type_reading, type_target)

            # SDM Knowledge (5th channel): factual relevance from corpus
            if kc is not None and kc_knowledge_vec is not None:
                sdm = float(self.vectors[idx] @ kc_knowledge_vec)
            else:
                sdm = 0.0

            # spaCy 300d space (7th channel): semantic comparison in concentrated space
            if spacy_context is not None and spacy_map is not None:
                cand_spacy = spacy_map[idx]
                spacy_score = float(cand_spacy @ spacy_context)
            else:
                spacy_score = 0.0

            # Trajectory placeholder (filled after rank-norm)
            ch_traj.append(0.0)

            ch_anchor.append(anchor); ch_mag.append(magnitude); ch_phase.append(phase); ch_type.append(type_score); ch_sdm.append(sdm); ch_spacy.append(spacy_score)
            ch_tokens.append(tok)

        # ── Rank-norm trajectory scores for top-20 candidates (efficient) ──
        def _iqr_of(arr):
            if len(arr) < 4: return 0.0
            a = np.array(arr, dtype=np.float32)
            q75, q25 = np.percentile(a, [75, 25])
            return float(max(q75 - q25, 0.0))

        n_ut = len(ch_tokens)

        if pg is not None and m_intent is not None:
            # Preliminary 5-channel combined score to find top candidates
            p_ch = 5
            p_iqrs = np.array([_iqr_of(ch_anchor), _iqr_of(ch_mag), _iqr_of(ch_phase), _iqr_of(ch_type), _iqr_of(ch_sdm)])
            p_total = float(np.sum(p_iqrs))
            if p_total > 1e-12:
                p_w = p_iqrs / p_total
            else:
                p_w = np.ones(p_ch, dtype=np.float32) / p_ch
            prelim = np.zeros(n_ut, dtype=np.float32)
            for i in range(n_ut):
                prelim[i] = p_w[0]*ch_anchor[i] + p_w[1]*ch_mag[i] + p_w[2]*ch_phase[i] + p_w[3]*ch_type[i] + p_w[4]*ch_sdm[i]
            # Top-20 candidates get trajectory scores
            n_traj = min(20, n_ut)
            traj_indices = np.argpartition(prelim, -n_traj)[-n_traj:]
            for i in traj_indices:
                tok = ch_tokens[i]
                if tok is not None and tok in w2i:
                    traj = pg.lookahead_coherence(
                        w2i[tok], self.vectors, m_intent, depth=5, width=2,
                    )
                    ch_traj[i] = traj
            # Rank-normalize trajectory scores to [0, 1]
            traj_arr = np.array(ch_traj, dtype=np.float32)
            t_min, t_max = float(traj_arr.min()), float(traj_arr.max())
            if t_max > t_min + 1e-12:
                ch_traj = ((traj_arr - t_min) / (t_max - t_min)).tolist()

        # Auto-calibrated weighting by IQR (higher IQR = more informative channel)
        if n_ut == 0:
            return np.zeros(len(first_tokens), dtype=np.float32)

        n_ch = 7
        iqrs = np.array([_iqr_of(ch_anchor), _iqr_of(ch_mag), _iqr_of(ch_phase), _iqr_of(ch_type), _iqr_of(ch_sdm), _iqr_of(ch_traj), _iqr_of(ch_spacy)], dtype=np.float32)
        total_iqr = float(np.sum(iqrs))
        if total_iqr > 1e-12:
            weights = np.array([iqr / total_iqr for iqr in iqrs], dtype=np.float32)
        else:
            weights = np.ones(n_ch, dtype=np.float32) / n_ch

        # Weighted combination per unique token
        combined = {}
        for i in range(n_ut):
            tok = ch_tokens[i]
            combined[tok] = float(
                weights[0] * ch_anchor[i] + weights[1] * ch_mag[i] +
                weights[2] * ch_phase[i] + weights[3] * ch_type[i] +
                weights[4] * ch_sdm[i] + weights[5] * ch_traj[i] +
                weights[6] * ch_spacy[i]
            )

        # Store first_tokens for generate loop to handle Type Field candidates
        self._vsa_first_tokens = first_tokens

        n_scores = len(first_tokens)
        scores = np.zeros(n_scores, dtype=np.float32)
        for idx, tok in enumerate(first_tokens):
            scores[idx] = combined.get(tok, -1.0)

        if self.debug_pcfg_enabled:
            self._debug_last_vsa = {
                'lhs': lhs,
                'first_tokens': list(first_tokens),
                'unique_tokens': list(unique_tokens),
                'ch_anchor': list(ch_anchor),
                'ch_mag': list(ch_mag),
                'ch_phase': list(ch_phase),
                'ch_type': list(ch_type),
                'ch_sdm': list(ch_sdm),
                'ch_traj': list(ch_traj),
                'ch_spacy': list(ch_spacy),
                'ch_tokens': list(ch_tokens),
                'combined': dict(combined),
                'weights': np.array(weights, dtype=np.float32),
                'pcfg_tokens': list(pcfg_tokens),
                'knowledge_tokens': list(knowledge_tokens) if knowledge_tokens else [],
                'knowledge_primary': getattr(self, '_knowledge_primary', False),
                'n_rules': n_rules,
            }
        return scores

    def _rule_scores(self, lhs: str, target_bins: np.ndarray, prefix_tokens: list[str], target_readings: np.ndarray | None = None, current_state: np.ndarray | None = None, type_target: np.ndarray | None = None, structure_only: bool = False) -> np.ndarray:
        """VSA scores with PCFG as optional syntactic tiebreaker.

        structure_only=True:
            PCFG fornece o conjunto de candidatos (via _vsa_scores → first_tokens),
            mas NÃO vota no ranking. VSA escolhe puramente pelo significado.

        structure_only=False (default):
            Comportamento original: rank_norm(PCFG) + rank_norm(VSA).
        """
        n_rules, _, probs = self.pcfg_idx.get_rule_info(lhs)
        if n_rules == 0:
            return np.array([], dtype=np.float32)
        base_scores = np.log(np.maximum(probs, np.finfo(np.float32).tiny))
        base_norm = self._rank_norm(base_scores)

        if target_readings is not None and current_state is not None and self._token_ffts is not None:
            vsa = self._vsa_scores(lhs, target_readings, current_state, type_target=type_target)
            if getattr(self, '_knowledge_primary', False):
                if self.debug_pcfg_enabled:
                    self._debug_last_rule = {'base_norm': base_norm.copy() if len(base_norm) > 0 else np.array([]), 'n_rules': n_rules, 'knowledge_primary': True}
                return vsa
            vsa_rank = self._rank_norm(vsa)
            if structure_only:
                # PCFG only provides syntactic candidate set (via _vsa_scores → first_tokens).
                # No PCFG probability vote. VSA chooses purely by meaning.
                result = vsa_rank
            else:
                # Original behavior: PCFG + VSA both vote (additive)
                if len(vsa) > n_rules:
                    pad = np.zeros(len(vsa) - n_rules, dtype=np.float32)
                    extended_base = np.concatenate([base_norm, pad])
                    result = self._rank_norm(extended_base) + vsa_rank
                else:
                    result = base_norm + vsa_rank
            if self.debug_pcfg_enabled:
                self._debug_last_rule = {'base_norm': base_norm.copy() if len(base_norm) > 0 else np.array([]), 'n_rules': n_rules, 'knowledge_primary': False, 'extended': len(vsa) > n_rules if not structure_only else False, 'structure_only': structure_only}
            return result
        if self.debug_pcfg_enabled:
            self._debug_last_rule = {'base_norm': base_norm.copy() if len(base_norm) > 0 else np.array([]), 'n_rules': n_rules, 'knowledge_primary': False, 'extended': False}
        return self._rank_norm(base_scores)

    def _bins(self, registers: np.ndarray) -> np.ndarray:
        clipped = np.clip(registers, 0.0, 1.0)
        bins = np.searchsorted(self.bin_edges, clipped, side="right") - 1
        return np.clip(bins, 0, self.n_bins - 1).astype(np.int32)

    @staticmethod
    def _iqr(values: np.ndarray) -> float:
        q75, q25 = np.percentile(values, [75, 25])
        return float(q75 - q25)

    @staticmethod
    def _rank_norm(values: np.ndarray) -> np.ndarray:
        """Convert values to normalized ranks in [0, 1]."""
        n = len(values)
        if n <= 1:
            return np.zeros_like(values, dtype=np.float32)
        ranks = np.argsort(np.argsort(values)).astype(np.float32)
        return ranks / float(n - 1)

    def _score_scale(self, scores: np.ndarray) -> float:
        iqr = self._iqr(scores)
        if iqr > 1e-12:
            return iqr
        std = float(np.std(scores))
        if std > 1e-12:
            return std
        return 1.0

    @staticmethod
    def _surface(tokens: list[str]) -> str:
        text = " ".join(tokens).strip()
        if not text:
            return text
        return text[0].upper() + text[1:]


def _phase_lensed_state(state: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    context = phi(state)
    return phase_lens(state, context, alpha=alpha).astype(np.float32)
