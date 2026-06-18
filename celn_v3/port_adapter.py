"""
CELN v3 - PortAdapter for GPVE
==============================

Converts an opaque projective-resonance state (M_pr) into an addressable
control state (M_ctrl) without cosine similarity, nearest-neighbor search,
or any distance metric in 10k dimensions.

Pipeline:
  M_pr -> signed coordinate sensors -> empirical percentiles -> VSA ports

M_ctrl is a superposition of bound registers:
  M_ctrl = sum_i bind(PORT_i, (2*p_i - 1) * CARRIER_i)

The GPVE mouth reads registers by known-address unbinding:
  p_i = read(unbind(M_ctrl, PORT_i), CARRIER_i)

The read step is a fixed linear contraction with the known carrier, not a
comparison against words, phrases, rules, memories, or candidate vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import hashlib
import os
from multiprocessing import Pool, cpu_count

import numpy as np

from .core import D, bind, normalize, projective_resonance, unbind, phi
from .train import load_corpus


@dataclass(frozen=True)
class PortAdapterConfig:
    """Configuration stored with a calibrated PortAdapter."""

    dim: int = D
    n_ports: int = 64
    sensor_width: int = 0
    seed: int = 1729
    gamma: float = 1.0
    bilateral: bool = True


class PortAdapter:
    """Non-metric bridge from opaque M_pr states to addressable M_ctrl states."""

    def __init__(
        self,
        config: PortAdapterConfig,
        ports: np.ndarray,
        carriers: np.ndarray,
        sensor_indices: np.ndarray,
        sensor_signs: np.ndarray,
        sensor_ecdf: np.ndarray,
        read_gains: np.ndarray | None = None,
    ):
        self.config = config
        self.dim = config.dim
        self.n_ports = config.n_ports
        self.sensor_width = config.sensor_width
        self.ports = ports.astype(np.float32)
        self.carriers = carriers.astype(np.float32)
        self.sensor_indices = sensor_indices.astype(np.int32)
        self.sensor_signs = sensor_signs.astype(np.float32)
        self.sensor_ecdf = np.sort(sensor_ecdf.astype(np.float32), axis=0)

        self._port_spectra = np.fft.fft(self.ports, axis=1)
        self._carrier_spectra = np.fft.fft(self.carriers, axis=1)

        if read_gains is None:
            read_gains = self._compute_read_gains()
        self.read_gains = read_gains.astype(np.float32)

    # ------------------------------------------------------------------
    # Construction and calibration
    # ------------------------------------------------------------------
    @classmethod
    def calibrate_from_corpus(
        cls,
        corpus_path: str | Path = "corpus_final.txt",
        vectors_path: str | Path = "celn_v3_full_vectors.npz",
        n_ports: int = 64,
        sensor_width: int | None = None,
        max_sentences: int | None = None,
        seed: int = 1729,
        gamma: float = 1.0,
        bilateral: bool = True,
        min_token_len: int = 1,
        state_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> "PortAdapter":
        """Calibrate sensor ECDFs from corpus M_pr states.

        This learns only empirical distributions of non-metric sensor readings.
        It does not fit a classifier, does not backpropagate, and does not
        compare M_pr with vocabulary vectors.
        """
        vectors, word2idx = load_word_vectors(vectors_path)
        dim = int(vectors.shape[1])
        width = int(sensor_width or np.ceil(np.sqrt(dim)))
        # Corpus + vectors signature for cache keys
        def file_sig(path: str | Path) -> str:
            p = Path(path)
            try:
                st = p.stat()
                return f"{p}:{st.st_mtime}:{st.st_size}"
            except Exception:
                return str(p)

        sig = f"{file_sig(vectors_path)}|{file_sig(corpus_path)}|{n_ports}|{seed}|{gamma}|{bilateral}|{min_token_len}|{sensor_width}|{bool(state_transform)}"
        key = hashlib.md5(sig.encode("utf-8")).hexdigest()
        adapter_cache = Path(f"/tmp/opencode/port_adapter_{key}.npz")

        # If a cached adapter exists for this exact config, load it.
        if adapter_cache.exists():
            try:
                return cls.load(adapter_cache)
            except Exception:
                # Fall through to (re)calibration on error
                pass

        # Precompute or load sentence states to avoid repeated projective_resonance scans.
        states_cache = Path(f"/tmp/opencode/sentence_states_{hashlib.md5((file_sig(vectors_path)+file_sig(corpus_path)).encode('utf-8')).hexdigest()}.npz")

        if states_cache.exists():
            data = np.load(states_cache, allow_pickle=True)
            sentences = list(data["tokens_list"].tolist())
            states = data["states"].astype(np.float32)
        else:
            sentences = load_corpus(str(corpus_path), max_sentences=max_sentences, min_len=min_token_len)
            # Compute states in parallel using ThreadPoolExecutor to avoid
            # pickling issues with multiprocessing on some platforms.
            from concurrent.futures import ThreadPoolExecutor

            def _compute_state(tok_list):
                return sentence_state(tok_list, vectors, word2idx, gamma=gamma, bilateral=bilateral)

            workers = max(1, min(cpu_count() - 1, 8))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_compute_state, s) for s in sentences]
                states_list = [f.result() for f in futures]

            states = np.stack(states_list).astype(np.float32)
            # Save token list and states for reuse
            try:
                np.savez_compressed(states_cache, tokens_list=np.array(sentences, dtype=object), states=states)
            except Exception:
                pass

            dim = int(vectors.shape[1])
            width = int(sensor_width or np.ceil(np.sqrt(dim)))

        config = PortAdapterConfig(
            dim=dim,
            n_ports=int(n_ports),
            sensor_width=width,
            seed=int(seed),
            gamma=float(gamma),
            bilateral=bool(bilateral),
        )
        ports, carriers, sensor_indices, sensor_signs = cls._make_banks(config)

        probe = cls(
            config=config,
            ports=ports,
            carriers=carriers,
            sensor_indices=sensor_indices,
            sensor_signs=sensor_signs,
            sensor_ecdf=np.zeros((1, n_ports), dtype=np.float32),
        )

        # If state_transform provided (e.g. Phase Lens), apply it to precomputed states
        if state_transform is not None:
            states_used = np.stack([state_transform(s) for s in states])
        else:
            states_used = states

        # Vectorized sensing for all states -> faster than Python loop
        # states_used: (N, D)
        N = int(states_used.shape[0])
        if probe.sensor_width <= 0:
            raise ValueError("Invalid sensor_width in PortAdapter")
        flat_idx = probe.sensor_indices.ravel()
        # gathered: (N, n_ports * sensor_width)
        gathered = np.take(states_used, flat_idx, axis=1)
        gathered = gathered.reshape(N, probe.n_ports, probe.sensor_width)
        # sensor_signs: (n_ports, sensor_width)
        values = gathered * probe.sensor_signs[None, :, :]
        readings = values.sum(axis=2).astype(np.float32)  # (N, n_ports)

        if readings.size == 0:
            raise ValueError("No valid corpus states were available for PortAdapter calibration")

        # sensor_ecdf should be sorted values per port (rows = sentences)
        sensor_ecdf = np.sort(readings, axis=0).astype(np.float32)
        adapter = cls(
            config=config,
            ports=ports,
            carriers=carriers,
            sensor_indices=sensor_indices,
            sensor_signs=sensor_signs,
            sensor_ecdf=sensor_ecdf,
        )

        try:
            adapter.save(adapter_cache)
        except Exception:
            pass

        return adapter

    @staticmethod
    def _make_banks(config: PortAdapterConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.RandomState(config.seed)
        ports = np.stack([
            make_unitary_vector(config.dim, rng) for _ in range(config.n_ports)
        ]).astype(np.float32)

        carriers = rng.choice(
            [-1.0, 1.0], size=(config.n_ports, config.dim)
        ).astype(np.float32) / np.sqrt(config.dim)

        sensor_indices = np.zeros((config.n_ports, config.sensor_width), dtype=np.int32)
        sensor_signs = np.zeros((config.n_ports, config.sensor_width), dtype=np.float32)
        scale = 1.0 / np.sqrt(config.sensor_width)
        for i in range(config.n_ports):
            sensor_indices[i] = rng.choice(config.dim, size=config.sensor_width, replace=False)
            sensor_signs[i] = rng.choice([-scale, scale], size=config.sensor_width).astype(np.float32)

        return ports, carriers, sensor_indices, sensor_signs

    # ------------------------------------------------------------------
    # Non-metric M_pr readings
    # ------------------------------------------------------------------
    def sense(self, m_pr: np.ndarray) -> np.ndarray:
        """Read raw scalar sensors from M_pr by signed coordinate blocks."""
        state = np.asarray(m_pr, dtype=np.float32)
        values = state[self.sensor_indices] * self.sensor_signs
        return values.sum(axis=1).astype(np.float32)

    def percentilize(self, raw: np.ndarray) -> np.ndarray:
        """Map raw sensor readings to empirical percentiles learned from corpus."""
        raw = np.asarray(raw, dtype=np.float32)
        n = self.sensor_ecdf.shape[0]
        pct = np.empty(self.n_ports, dtype=np.float32)
        for i in range(self.n_ports):
            pct[i] = np.searchsorted(self.sensor_ecdf[:, i], raw[i], side="right") / n
        return pct

    def registers_from_m_pr(self, m_pr: np.ndarray) -> np.ndarray:
        """Return the GPVE control register vector p in [0, 1]^n_ports."""
        return self.percentilize(self.sense(m_pr))

    def registers_from_states(self, states: np.ndarray) -> np.ndarray:
        """Vectorized: compute percentiles for multiple states.

        Args:
            states: array shape (N, D)

        Returns:
            percentiles: array shape (N, n_ports)
        """
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != self.dim:
            raise ValueError(f"Expected states shape (N, {self.dim}), got {states.shape}")

        N = states.shape[0]
        # Gather sensor values vectorized: use sensor_indices to index along axis 1
        flat_idx = self.sensor_indices.ravel()
        gathered = np.take(states, flat_idx, axis=1)
        gathered = gathered.reshape(N, self.n_ports, self.sensor_width)
        values = (gathered * self.sensor_signs[None, :, :]).sum(axis=2)

        # Percentilize per port using searchsorted against sensor_ecdf[:, port]
        n_ecdf = self.sensor_ecdf.shape[0]
        pct = np.empty((N, self.n_ports), dtype=np.float32)
        for i in range(self.n_ports):
            pct[:, i] = np.searchsorted(self.sensor_ecdf[:, i], values[:, i], side="right") / float(max(1, n_ecdf))
        return np.clip(pct, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # M_ctrl writing and reading
    # ------------------------------------------------------------------
    def to_control_state(self, m_pr: np.ndarray) -> np.ndarray:
        """Convert opaque M_pr into addressable M_ctrl."""
        return self.write_registers(self.registers_from_m_pr(m_pr))

    def write_registers(self, percentiles: np.ndarray) -> np.ndarray:
        """Write percentiles into anonymous VSA ports."""
        p = np.asarray(percentiles, dtype=np.float32)
        if p.shape != (self.n_ports,):
            raise ValueError(f"Expected shape ({self.n_ports},), got {p.shape}")

        amplitudes = 2.0 * np.clip(p, 0.0, 1.0) - 1.0
        spectrum = np.sum(
            amplitudes[:, None] * self._port_spectra * self._carrier_spectra,
            axis=0,
        )
        return np.fft.ifft(spectrum).real.astype(np.float32)

    def read_registers(self, m_ctrl: np.ndarray) -> np.ndarray:
        """Read GPVE registers from M_ctrl by known-address unbinding.

        This performs no candidate comparison. Each register address is known in
        advance, and the recovered value is a scalar contraction with its own
        carrier.
        """
        ctrl_spectrum = np.fft.fft(np.asarray(m_ctrl, dtype=np.float32))
        recovered_spectra = ctrl_spectrum[None, :] * np.conj(self._port_spectra)
        recovered = np.fft.ifft(recovered_spectra, axis=1).real.astype(np.float32)
        amplitudes = np.sum(recovered * self.carriers, axis=1) / (self.read_gains + 1e-12)
        return np.clip((amplitudes + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)

    def encode_and_read(self, m_pr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convenience helper returning (M_ctrl, written_p, read_p)."""
        written = self.registers_from_m_pr(m_pr)
        ctrl = self.write_registers(written)
        read = self.read_registers(ctrl)
        return ctrl, written, read

    def _compute_read_gains(self) -> np.ndarray:
        gains = np.empty(self.n_ports, dtype=np.float32)
        for i in range(self.n_ports):
            recovered = unbind(bind(self.ports[i], self.carriers[i]), self.ports[i])
            gains[i] = float(np.sum(recovered * self.carriers[i]))
        gains[np.abs(gains) < 1e-12] = 1.0
        return gains

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.savez_compressed(
            path,
            dim=np.array(self.config.dim, dtype=np.int32),
            n_ports=np.array(self.config.n_ports, dtype=np.int32),
            sensor_width=np.array(self.config.sensor_width, dtype=np.int32),
            seed=np.array(self.config.seed, dtype=np.int32),
            gamma=np.array(self.config.gamma, dtype=np.float32),
            bilateral=np.array(self.config.bilateral, dtype=np.bool_),
            ports=self.ports,
            carriers=self.carriers,
            sensor_indices=self.sensor_indices,
            sensor_signs=self.sensor_signs,
            sensor_ecdf=self.sensor_ecdf,
            read_gains=self.read_gains,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PortAdapter":
        data = np.load(path, allow_pickle=False)
        config = PortAdapterConfig(
            dim=int(data["dim"]),
            n_ports=int(data["n_ports"]),
            sensor_width=int(data["sensor_width"]),
            seed=int(data["seed"]),
            gamma=float(data["gamma"]),
            bilateral=bool(data["bilateral"]),
        )
        return cls(
            config=config,
            ports=data["ports"],
            carriers=data["carriers"],
            sensor_indices=data["sensor_indices"],
            sensor_signs=data["sensor_signs"],
            sensor_ecdf=data["sensor_ecdf"],
            read_gains=data["read_gains"],
        )


def make_unitary_vector(dim: int = D, rng: np.random.RandomState | None = None) -> np.ndarray:
    """Create a real vector whose FFT has unit magnitude in every bin."""
    rng = rng or np.random.RandomState()
    spectrum = np.empty(dim, dtype=np.complex128)
    spectrum[0] = rng.choice([-1.0, 1.0])
    half = dim // 2
    for k in range(1, half + 1):
        if dim % 2 == 0 and k == half:
            spectrum[k] = rng.choice([-1.0, 1.0])
        else:
            phase = rng.uniform(0.0, 2.0 * np.pi)
            value = np.cos(phase) + 1j * np.sin(phase)
            spectrum[k] = value
            spectrum[-k] = np.conj(value)
    return np.fft.ifft(spectrum).real.astype(np.float32)


def load_word_vectors(path: str | Path) -> tuple[np.ndarray, dict[str, int]]:
    """Load CELN word vectors from the formats used in this repository."""
    data = np.load(path, allow_pickle=True)
    if "vectors" in data:
        vectors = data["vectors"].astype(np.float32)
    elif "word_vectors" in data:
        vectors = data["word_vectors"].astype(np.float32)
    else:
        raise ValueError(f"No vector matrix found in {path}")

    if "vocab" in data:
        vocab = [str(w) for w in data["vocab"]]
        word2idx = {w: i for i, w in enumerate(vocab)}
    elif "word2idx" in data:
        word2idx = dict(data["word2idx"].item())
    elif "idx2word" in data:
        idx2word = dict(data["idx2word"].item())
        word2idx = {str(w): int(i) for i, w in idx2word.items()}
    else:
        raise ValueError(f"No vocabulary mapping found in {path}")

    return vectors, word2idx


def sentence_state(
    tokens: Iterable[str],
    vectors: np.ndarray,
    word2idx: dict[str, int],
    gamma: float = 1.0,
    bilateral: bool = True,
) -> np.ndarray:
    """Encode a token sequence with the existing projective-resonance scan."""
    state: np.ndarray | None = None
    for tok in tokens:
        idx = word2idx.get(tok)
        if idx is None:
            continue
        word_vec = vectors[idx]
        if state is None:
            state = word_vec.copy()
        else:
            state = projective_resonance(state, word_vec, gamma=gamma, bilateral=bilateral)

    if state is None:
        return np.zeros(vectors.shape[1], dtype=np.float32)
    return normalize(state).astype(np.float32)


def sentence_state_enriched(
    tokens: Iterable[str],
    vectors: np.ndarray,
    word2idx: dict[str, int],
    type_field: np.ndarray | None = None,
    type_word2idx: dict[str, int] | None = None,
    gamma: float = 1.0,
    bilateral: bool = True,
    alpha_chain: float | None = None,
    alpha_type: float | None = None,
) -> np.ndarray:
    """Encode a token sequence with MULTI-CHANNEL M_pr.

    Channels:
      1. M_chain: word vectors chain-encoded (existing projective_resonance scan)
      2. M_type: Type Field vectors chain-encoded (syntactic structure)

    Blend weights are auto-calibrated by spectral entropy:
      weight ∝ 1 / entropy(channel)
      Lower entropy = more focused = higher weight

    If alpha_chain/alpha_type are provided, use them as fixed blend weights.
    If not, auto-calibrate from spectral entropy.
    """
    from .core import spectral_entropy, encode_sequence

    tokens_list = list(tokens)
    n_dims = int(vectors.shape[1])
    zero = np.zeros(n_dims, dtype=np.float32)

    # --- Channel 1: word vector chain ---
    word_vecs = []
    for tok in tokens_list:
        idx = word2idx.get(tok)
        if idx is not None:
            word_vecs.append(vectors[idx])

    M_chain = encode_sequence(word_vecs, gamma=gamma, bilateral=bilateral) if word_vecs else zero

    # --- Channel 2: Type Field chain ---
    M_type = zero.copy()
    if type_field is not None and type_word2idx is not None:
        type_vecs = []
        for tok in tokens_list:
            idx = type_word2idx.get(tok)
            if idx is not None and np.linalg.norm(type_field[idx]) > 1e-12:
                type_vecs.append(type_field[idx])
        if type_vecs:
            M_type = encode_sequence(type_vecs, gamma=gamma, bilateral=bilateral)

    # --- Auto-calibrated blend ---
    if alpha_chain is not None and alpha_type is not None:
        # Fixed weights provided
        M_pr = normalize(alpha_chain * M_chain + alpha_type * M_type)
    else:
        # Auto-calibrate from spectral entropy
        e_chain = max(spectral_entropy(M_chain), 1e-12) if M_chain is not zero else 1.0
        e_type = max(spectral_entropy(M_type), 1e-12) if M_type is not zero else 1.0
        w_chain = 1.0 / e_chain
        w_type = 1.0 / e_type
        total = w_chain + w_type
        M_pr = normalize((w_chain / total) * M_chain + (w_type / total) * M_type)

    return M_pr.astype(np.float32)


def sentence_state_positional(
    tokens: Iterable[str],
    vectors: np.ndarray,
    word2idx: dict[str, int],
    type_field: np.ndarray | None = None,
    type_word2idx: dict[str, int] | None = None,
    gamma: float = 1.0,
    bilateral: bool = True,
) -> dict:
    """Encode token sequence with MULTI-CHANNEL M_pr AND positional sub-targets.

    Partições dinâmicas auto-calibráveis:
      n_parts = max(3, min(7, len(tokens) // 2))

    Cada partição tem ~n_tokens/n_parts tokens. A geração em cada estágio
    compara candidatos com o sub-alvo da posição correspondente, reduzindo
    a assimetria de escala entre estado parcial e alvo.

    Returns dict with:
      'm_pr': full blended M_pr (para CAPL)
      'm_targets': list of ~n_parts sub-target vectors
      'n_parts': number of partitions
      'n_tokens': total number of prompt tokens encoded
    """
    from .core import spectral_entropy, encode_sequence

    tokens_list = list(tokens)
    n_dims = int(vectors.shape[1])
    zero = np.zeros(n_dims, dtype=np.float32)
    n = len(tokens_list)

    # --- Full M_pr (blend chain + type) ---
    word_vecs = []
    for tok in tokens_list:
        idx = word2idx.get(tok)
        if idx is not None:
            word_vecs.append(vectors[idx])
    M_chain = encode_sequence(word_vecs, gamma=gamma, bilateral=bilateral) if word_vecs else zero

    M_type = zero.copy()
    if type_field is not None and type_word2idx is not None:
        t_vecs = []
        for tok in tokens_list:
            idx = type_word2idx.get(tok)
            if idx is not None and np.linalg.norm(type_field[idx]) > 1e-12:
                t_vecs.append(type_field[idx])
        if t_vecs:
            M_type = encode_sequence(t_vecs, gamma=gamma, bilateral=bilateral)

    e_chain = max(spectral_entropy(M_chain), 1e-12) if M_chain is not zero else 1.0
    e_type = max(spectral_entropy(M_type), 1e-12) if M_type is not zero else 1.0
    w_chain, w_type = 1.0 / e_chain, 1.0 / e_type
    total_w = w_chain + w_type
    M_pr = normalize((w_chain / total_w) * M_chain + (w_type / total_w) * M_type)

    # --- Dynamic positional sub-targets ---
    n_parts = max(3, min(7, n // 2))
    m_targets = []

    if n < n_parts:
        # Too short: use full M_pr for all positions
        m_targets = [M_pr.copy() for _ in range(n_parts)]
    else:
        def _encode_range(start, end):
            subset = word_vecs[start:end]
            if not subset:
                return zero.copy()
            return normalize(encode_sequence(subset, gamma=gamma, bilateral=bilateral))

        for i in range(n_parts):
            s = (i * n) // n_parts
            e = ((i + 1) * n) // n_parts
            m_targets.append(_encode_range(s, e))

    return {
        'm_pr': M_pr.astype(np.float32),
        'm_targets': [t.astype(np.float32) for t in m_targets],
        'n_parts': n_parts,
        'n_tokens': n,
    }
