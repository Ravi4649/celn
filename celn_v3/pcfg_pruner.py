"""
PCFG Pruner

Loads an induced PCFG (pcfg_induced.json), prunes low-count rules per LHS using
automatic knee detection (percentile), and writes pcfg_pruned.json.

Principles: no fixed thresholds — knee detection determines the percentile per-LHS.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np


def _find_knee_percentile(values: List[float]) -> float:
    if not values:
        return 1.0
    arr = np.array(sorted(values, reverse=True), dtype=float)
    n = len(arr)
    if n <= 2 or (arr.max() - arr.min()) < 1e-12:
        return 0.95

    x = np.linspace(0.0, 1.0, n)
    y = (arr - arr.min()) / (arr.max() - arr.min())
    line_y = x * (y[-1] - y[0]) + y[0]
    distances = np.abs(y - line_y)
    knee_idx = int(np.argmax(distances))
    percentile = (knee_idx + 1) / n
    if percentile < 0.01 or percentile > 0.999:
        return 0.95
    return float(percentile)


def prune_pcfg(pcfg: Dict, outpath: str, verbose: bool = True) -> Dict:
    rules = pcfg.get('rules', {})
    pruned_rules = {}
    total_before = 0
    total_after = 0

    for lhs, rlist in rules.items():
        if not rlist:
            pruned_rules[lhs] = []
            continue

        counts = [float(r.get('count', 0)) for r in rlist]
        total_before += len(rlist)

        # determine percentile via knee detection
        pct = _find_knee_percentile(counts)
        # select top-k by count according to pct
        sorted_rules = sorted(rlist, key=lambda r: r.get('count', 0), reverse=True)
        cutoff_idx = max(1, int(len(sorted_rules) * pct))
        selected = sorted_rules[:cutoff_idx]

        # normalize probabilities within selected
        sum_counts = sum(r.get('count', 0) for r in selected) or 1
        new_rules = []
        for r in selected:
            new_r = dict(r)
            new_r['prob'] = float(r.get('count', 0) / sum_counts)
            new_rules.append(new_r)

        pruned_rules[lhs] = new_rules
        total_after += len(new_rules)

        if verbose:
            print(f"LHS={lhs}: rules_before={len(rlist)}, kept={len(new_rules)}, pct={pct:.3f}")

    new_pcfg = dict(pcfg)
    new_pcfg['rules'] = pruned_rules

    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(new_pcfg, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"Pruned PCFG: total_rules_before={total_before}, total_rules_after={total_after}")
        print(f"Saved pruned PCFG to {outpath}")

    return new_pcfg


def main(inpath: str | None = None, outpath: str | None = None):
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if inpath is None:
        inpath = os.path.join(base, 'pcfg_induced.json')
    if outpath is None:
        outpath = os.path.join(base, 'pcfg_pruned.json')

    with open(inpath, 'r', encoding='utf-8') as f:
        pcfg = json.load(f)

    prune_pcfg(pcfg, outpath)


if __name__ == '__main__':
    main()
