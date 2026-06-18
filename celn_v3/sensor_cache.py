"""
TokenSensorCache
================

Precomputa leituras raw dos sensores do PortAdapter para cada token
do vocabulário. Permite comparar tokens individuais com M_intent
no ESPAÇO DOS SENSORES (n_ports dimensões) via correlação de Pearson.

Princípios:
- Sem backprop, transformers, similaridade em 10k
- Espaço de comparação: n_ports (64-128), não 10k
- Auto-calibrável: médias e stds para z-normalização
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import numpy as np

from .core import normalize
from .port_adapter import PortAdapter, load_word_vectors


class TokenSensorCache:
    """Cache de leituras raw dos sensores para cada token do vocabulário."""

    def __init__(
        self,
        token_readings: Dict[str, np.ndarray],
        port_median: np.ndarray,
        port_std: np.ndarray,
        n_ports: int,
        adapter: PortAdapter,
    ):
        self.token_readings = token_readings
        self.port_median = port_median.astype(np.float32)
        self.port_std = port_std.astype(np.float32)
        self.port_std[self.port_std < 1e-12] = 1.0
        self.n_ports = n_ports
        self._adapter = adapter

    @classmethod
    def build(
        cls,
        vectors_path: str | Path,
        adapter: PortAdapter,
    ) -> "TokenSensorCache":
        """Build sensor cache for all vocabulary tokens."""
        vectors, word2idx = load_word_vectors(vectors_path)
        vectors = vectors.astype(np.float32)
        n_ports = adapter.n_ports

        all_readings = []
        token_readings: Dict[str, np.ndarray] = {}

        for word, idx in word2idx.items():
            v = normalize(vectors[idx].astype(np.float32))
            reading = adapter.sense(v)
            token_readings[word] = reading.astype(np.float32)
            all_readings.append(reading)

        all_arr = np.stack(all_readings, axis=0)
        port_median = np.median(all_arr, axis=0).astype(np.float32)
        port_std = np.std(all_arr, axis=0).astype(np.float32)

        return cls(
            token_readings=token_readings,
            port_median=port_median,
            port_std=port_std,
            n_ports=n_ports,
            adapter=adapter,
        )

    def lookup(self, token: str) -> Optional[np.ndarray]:
        """Return raw sensor readings for a token, or None if unknown."""
        return self.token_readings.get(token)

    def sense_target(self, m_pr: np.ndarray) -> np.ndarray:
        """Raw sensor readings from M_pr (no percentilize)."""
        return self._adapter.sense(m_pr).astype(np.float32)

    def correlate(self, a: np.ndarray, b: np.ndarray) -> float:
        """Pearson correlation between two (n_ports,) vectors (z-normalized)."""
        a_z = (a - self.port_median) / self.port_std
        b_z = (b - self.port_median) / self.port_std
        n = float(self.n_ports)
        r = float(np.sum(a_z * b_z) / ((n - 1.0) * np.std(a_z, ddof=1) * np.std(b_z, ddof=1) + 1e-12))
        return float(np.clip(r, -1.0, 1.0))
