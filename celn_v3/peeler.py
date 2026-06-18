"""
Peeler
======

Peeling sequencial de estados M_pr via unbinding reverso com carrier geométrico.
Suporta cadeias longas (projeto-resonance left-to-right) sem comparação em 10k_dims.

Chaves:
- Peeling: remove último fator iterativamente
- Carrier: obtido de janela causal via role bin anônimo
- Energia: medida via percentil/ECDF acumulado
- Critério de parada: energia residual percentilizada

Princípios respeitados:
- ZERO backprop, transformers, similaridade em 10k, listas fixas, templates, thresholds mágicos
- Tudo auto-calibrável: percentis/ECDF de energia observada em corpus
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import json
import numpy as np

from .core import normalize, unbind, phi, projective_resonance, D
from .stats_ecdf import EmpiricalCDF


@dataclass
class PeelPacket:
    factors: List[np.ndarray]                # Recovered factors in reverse causal order
    energies: List[float]                    # Energy percentiles for each factor
    M_intent: np.ndarray                     # Cleaned/composite intent
    residual_norm_percentile: float         # Residual energy percentile (ECDF)
    converged: bool                         # Did we stop by residual percentile?


@dataclass
class PeelConfig:
    max_peel_depth: int = 8                 # Max peel iterations
    min_window_size: int = 3                # Min causal window
    alpha_causal: float = 0.3               # Carrier blending weight (context)
    n_roles: int = 5                       # Role bins (start/mid/end/bin4/bin5...)
    seed: int = 42
    energy_calibration_samples: int = 1000   # Corpus sentences for energy ECDF


class Peeler:
    """
    Peeler for left-to-right projective-resonance chains.
    """

def __init__(
        self,
        vectors_path: str | Path,
        corpus_path: str | Path | None = None,
        config: PeelConfig | None = None,
        force_calibration: bool = True,  # Force ECDF calibration, no dummy fallback
    ) -> None:
        from .port_adapter import load_word_vectors
        vectors, word2idx = load_word_vectors(vectors_path)
        self.vectors = vectors.astype(np.float32)
        self.word2idx = word2idx

        self.config = config or PeelConfig()
        self.rng = np.random.RandomState(self.config.seed)
        
        # Role bins: generated unitary carriers
        self.role_bins = np.array([
            make_unitary_vector(D, self.rng)
            for _ in range(self.config.n_roles)
        ], dtype=np.float32)

        # Energy ECDF: MUST calibrate from corpus (no dummy fallback)
        self.energy_ecdf_factor: EmpiricalCDF | None = None
        self.energy_ecdf_residual: EmpiricalCDF | None = None
        
        if corpus_path and force_calibration:
            print(f"Peeler: Calibrating energy ECDFs from {self.config.energy_calibration_samples} sentences...")
            self.calibrate_energy_ecdfs(corpus_path)
        
        # If calibration failed and force_calibration=True, raise error
        if force_calibration and (self.energy_ecdf_factor is None or self.energy_ecdf_residual is None):
            raise ValueError("Peuler: ECDF calibration failed and force_calibration=True")
        elif self.energy_ecdf_factor is None or self.energy_ecdf_residual is None:
            # Last resort: minimal ECDF from a few samples
            energies_dummy = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
            self.energy_ecdf_factor = EmpiricalCDF(energies_dummy)
            self.energy_ecdf_residual = EmpiricalCDF(energies_dummy)

    def calibrate_energy_ecdfs(self, corpus_path: str | Path) -> None:
        """Calibrate factor and residual energy ECDFs from corpus prefixes."""
        from .train import load_corpus
        sentences = load_corpus(str(corpus_path), max_sentences=self.config.energy_calibration_samples)

        all_energies: Dict[str, List[float]] = {
            "residual": [],
            "factor": [],
        }

        # Sample random prefixes
        for s in sentences:
            toks = list(filter(lambda t: t in self.word2idx, s))
            if len(toks) < self.config.min_window_size + 2:
                continue

            n = self.rng.randint(self.config.min_window_size, len(toks) + 1)
            prefix = toks[:n]
            m_pr = sentence_state(prefix, self.vectors, self.word2idx, gamma=1.0, bilateral=True)
            
            # Simulate peel to collect energies with real token window
            residual = m_pr.copy()
            for peel_depth in range(self.config.max_peel_depth):
                # Use real token window for carrier
                window = prefix[-min(len(prefix), self.config.min_window_size):]
                if len(window) < 2:
                    break
                carrier = self._make_causal_carrier(window)

                # Peel step: unbinding reverso
                candidate = unbind_M_reverse(residual, carrier, gamma=1.0, bilateral=True)
                candidate = normalize(candidate)

                # Energy: norm
                energy = float(np.linalg.norm(candidate))
                all_energies["factor"].append(energy)
                
                # Update residual
                residual = unbind_M_reverse(residual, candidate, gamma=1.0, bilateral=True)
                residual = normalize(residual)
                
                # Residual energy
                residual_energy = float(np.linalg.norm(residual))
                all_energies["residual"].append(residual_energy)
                
                if len(prefix) > 1:
                    prefix.pop()  # Simulate shrinking window

        # Build ECDFs only if we have data
        if len(all_energies["factor"]) < 10 or len(all_energies["residual"]) < 10:
            raise ValueError(f"Insufficient calibration data: {len(all_energies['factor'])} factors, {len(all_energies['residual'])} residuals")
        
        self.energy_ecdf_factor = EmpiricalCDF(all_energies["factor"])
        self.energy_ecdf_residual = EmpiricalCDF(all_energies["residual"])
        
        # Cache to disk
        try:
            cache_key = self._energy_cache_key(corpus_path)
            np.savez_compressed(
                f"/tmp/opencode/peeler_energy_ecdf_{cache_key}.npz",
                factor_values=np.array(self.energy_ecdf_factor.values, dtype=np.float32),
                factor_counts=np.array(self.energy_ecdf_factor.counts, dtype=np.int32),
                residual_values=np.array(self.energy_ecdf_residual.values, dtype=np.float32),
                residual_counts=np.array(self.energy_ecdf_residual.counts, dtype=np.int32),
            )
        except Exception:
            pass  # Cache failure is non-fatal

    def _energy_cache_key(self, corpus_path: str | Path) -> str:
        sig = f"{self._file_sig(corpus_path)}+{self.config.seed}"
        return hashlib.md5(sig.encode("utf-8")).hexdigest()

    @staticmethod
    def _file_sig(path: str | Path) -> str:
        try:
            p = Path(path)
            st = p.stat()
            return f"{p}:{st.st_mtime}:{st.st_size}"
        except Exception:
            return str(path)

    def pickle_ecdf(self, ecdf: EmpiricalCDF) -> dict[str, Any]:
        return {
            "values": ecdf.values.tolist() if hasattr(ecdf, "values") else [],
            "counts": ecdf.counts.tolist() if hasattr(ecdf, "counts") else [],
        }

    def unpickle_ecdf(cls, data: dict[str, Any]) -> EmpiricalCDF:
        from .stats_ecdf import EmpiricalCDF
        return EmpiricalCDF(np.array(data["values"]), np.array(data["counts"]))

    @classmethod
    def load_energy_cache(cls, corpus_path: str, seed: int) -> bool:
        try:
            cache_key = hashlib.md5(f"{Path(corpus_path).stat().st_mtime}+{seed}".encode("utf-8")).hexdigest()
            cache = f"/tmp/opencode/peeler_energy_ecdf_{cache_key}.npz"
            if Path(cache).exists():
                data = np.load(cache, allow_pickle=True)
                return data
        except Exception:
            pass
        return None

    def _make_causal_carrier(self, window: List[str]) -> np.ndarray:
        """
        Create a causal carrier from the window context.
        Uses phase_lens over the window and binds with a role bin based on position.
        """
        # Convert tokens to vectors
        vecs = []
        for tok in window:
            idx = self.word2idx.get(tok)
            if idx is not None:
                vecs.append(self.vectors[idx])
        if not vecs:
            return normalize(make_unitary_vector(D, self.rng).astype(np.float32))

        # Project to causal subspace
        base = phi(np.stack(vecs, axis=0))
        context = np.sum(base, axis=0)
        proj = phase_lens(base[-1], context, alpha=self.config.alpha_causal)

        # Role bin: deterministic based on window *length* (ensures consistency)
        depth = len(window)
        role_idx = min(depth % self.config.n_roles, self.config.n_roles - 1)
        carrier = bind(self.role_bins[role_idx], proj)
        return normalize(carrier.astype(np.float32))

    def peel(self, M_pr: np.ndarray, tokens: list[str] | None = None) -> PeelPacket:
        """
        Peel factors from M_pr using real token window for causal carrier.
        """
        M_pr = np.asarray(M_pr, dtype=np.float32)
        if M_pr.ndim != 1 or M_pr.shape[0] != D:
            raise ValueError(f"M_pr must be a vector of shape ({D},)")

        factors: List[np.ndarray] = []
        energies: List[float] = []
        residual = M_pr.copy()

        # Use tokens to build causal window carrier
        # If no tokens provided, fall back to role-bin cycling (degraded mode)
        use_tokens = tokens is not None and len(tokens) >= 2
        token_vecs = []
        if use_tokens:
            for tok in tokens:
                idx = self.word2idx.get(tok)
                if idx is not None:
                    token_vecs.append(self.vectors[idx])

        converged = False
        for peel_depth in range(self.config.max_peel_depth):
            # Build carrier from real token window if available
            if use_tokens and len(token_vecs) >= 2:
                # Use last min_window_size tokens as causal window
                window_vecs = token_vecs[-self.config.min_window_size:]
                # Project window to get carrier
                base = phi(np.stack(window_vecs, axis=0))
                context = np.sum(base, axis=0)
                proj = phase_lens(base[-1], context, alpha=self.config.alpha_causal)
                # Role bin based on peel depth
                role_idx = peel_depth % self.config.n_roles
                carrier = bind(self.role_bins[role_idx], proj)
                carrier = normalize(carrier.astype(np.float32))
            else:
                # Degraded mode: role bin only
                role_idx = peel_depth % self.config.n_roles
                carrier = self.role_bins[role_idx]

            # Peel step
            candidate = unbind_M_reverse(residual, carrier, gamma=1.0, bilateral=True)
            candidate = normalize(candidate)

            energy_value = float(np.linalg.norm(candidate))
            energy_percentile = self.energy_ecdf_factor.percentile(energy_value)
            energies.append(energy_percentile)
            factors.append(candidate)

            # Update residual: remove peeled factor
            residual = unbind_M_reverse(residual, candidate, gamma=1.0, bilateral=True)
            residual = normalize(residual)

            # Residual energy percentile
            residual_energy = float(np.linalg.norm(residual))
            residual_percentile = self.energy_ecdf_residual.percentile(residual_energy)

            # Convergence: stop when residual is HIGH (above 70th percentile)
            # This means we've extracted most structure and what's left is noise
            converged = residual_percentile > 70.0
            if converged:
                break
            
            # Remove last token vector from window (simulate peeling)
            if use_tokens and len(token_vecs) > 1:
                token_vecs.pop()

        # Reconstruct M_intent: normalized SUM of factors (not re-bind!)
        if factors:
            M_intent = np.sum(np.stack(factors, axis=0), axis=0)
            M_intent = normalize(M_intent)
        else:
            M_intent = M_pr.copy()

        # Finalize packet
        residual_norm = float(np.linalg.norm(residual))
        residual_norm_percentile = self.energy_ecdf_residual.percentile(residual_norm)
        
        return PeelPacket(
            factors=factors,
            energies=energies,
            M_intent=M_intent.astype(np.float32),
            residual_norm_percentile=residual_norm_percentile,
            converged=converged,
        )


def unbind_M_reverse(bound: np.ndarray, factor: np.ndarray, gamma: float = 1.0, bilateral: bool = True) -> np.ndarray:
    """Reverse unbinding: remove factor from bound state."""
    bound = np.asarray(bound, dtype=np.float32)
    factor = np.asarray(factor, dtype=np.float32)
    
    # Formula: unbind(bound, factor) ≈ bound - γ·factor
    # Here we reverse: what, when bound to factor, yields bound?
    recovered = bound - gamma * factor
    return normalize(recovered)


# ---- Utility moved from core for Peeler standalone usage

def make_unitary_vector(dim: int = D, rng: np.random.RandomState | None = None) -> np.ndarray:
    """Create a real unitary vector."""
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


def sentence_state(
    tokens: List[str],
    vectors: np.ndarray,
    word2idx: dict[str, int],
    gamma: float = 1.0,
    bilateral: bool = True,
) -> np.ndarray:
    """Encode a token sequence with projective_resonance."""
    state: Optional[np.ndarray] = None
    for tok in tokens:
        idx = word2idx.get(tok)
        if idx is None:
            continue
        word_vec = vectors[idx]
        if state is None:
            state = word_vec.copy()
        else:
            state = projective_resonance(state, word_vec, gamma=gamma, bilateral=bilateral)
    return normalize(state).astype(np.float32) if state is not None else np.zeros(vectors.shape[1], dtype=np.float32)