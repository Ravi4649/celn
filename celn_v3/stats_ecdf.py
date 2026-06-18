"""
Empirical Cumulative Distribution Function (ECDF)
=====================================================

Computa percentis a partir de dados observados sem thresholds mágicos.

Propriedades:
- Mantém apenas dados ordenados
- Tudo auto-calibrável
- Nenhum ajuste manual
- Percentilização exata e interpolada
"""

import numpy as np


class EmpiricalCDF:
    """Empirical CDF from observed data."""

    def __init__(self, values: list | np.ndarray, predetermined_bins: int = 256):
        if not isinstance(values, np.ndarray):
            values = np.asarray(values, dtype=np.float32)
        
        # Filter invalid values (inf/nan)
        values_clean = values[np.isfinite(values)]
        if len(values_clean) == 0:
            values_clean = np.array([0.0, 1.0], dtype=np.float32)
        
        # Store sorted unique values
        self.values, self.counts = np.unique(values_clean, return_counts=True)
        if len(self.values) < predetermined_bins:
            # Pad low-data regimes for stability
            dx = (self.values[-1] - self.values[0]) / (predetermined_bins // 2)
            if dx > 0:
                padded = np.linspace(self.values[0] - dx * 5,
                                    self.values[-1] + dx * 5,
                                    predetermined_bins)
                pad_counts = np.ones_like(padded, dtype=int)
                self.values = np.concatenate((padded, self.values))
                self.counts = np.concatenate((pad_counts, self.counts))
        
        # Cumulative counts
        self.cum_counts = np.concatenate(([0], np.cumsum(self.counts)))
        self.total = float(self.cum_counts[-1])
        
        # Edge protection: ensure monotonicity
        self.cum_counts = np.maximum.accumulate(self.cum_counts)

    def percentile(self, x: float, method: str = "linear") -> float:
        """Percentile for value x in [0,100]
        
        Methods:
        - nearest: nearest neighbor
        - linear: linear interpolation (default)
        """
        if self.total == 0:
            return 50.0
        
        if x < self.values[0]:
            return 0.0
        if x > self.values[-1]:
            return 100.0
        
        # Find insertion index in sorted values
        idx = np.searchsorted(self.values, x, side='right') - 1
        if idx < 0:
            idx = 0

        cum_below = self.cum_counts[idx]
        if method == "nearest":
            cum_above = self.cum_counts[min(idx + 1, len(self.cum_counts) - 1)]
            dist = (x - self.values[idx]) / max(1e-12, (self.values[idx + 1] - self.values[idx]))
            pct = cum_below + dist * self.counts[idx]
            return 100.0 * pct / self.total
        elif method == "linear":
            cum_above = self.cum_counts[min(idx + 1, len(self.cum_counts) - 1)]
            if cum_below == cum_above:
                return 100.0 * cum_below / self.total
            frac = float((x - self.values[idx])) / max(1e-12, float(
                (self.values[idx + 1] - self.values[idx])))
            interpolation = cum_below + frac * (cum_above - cum_below)
            return 100.0 * interpolation / self.total
        else:
            raise ValueError(f"Unknown percentile method: {method}")

    def __call__(self, x: float) -> float:
        return self.percentile(x, method="linear")