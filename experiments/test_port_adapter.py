#!/usr/bin/env python3
"""Functional check for the CELN v3 PortAdapter.

The check verifies that an opaque M_pr state can be converted into an
addressable M_ctrl state and read back through known ports without cosine
similarity or nearest-neighbor search in 10k dimensions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from celn_v3.port_adapter import PortAdapter, load_word_vectors, sentence_state
from celn_v3.train import load_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Test non-metric PortAdapter readback")
    parser.add_argument("--corpus", default="corpus_final.txt")
    parser.add_argument("--vectors", default="celn_v3_full_vectors.npz")
    parser.add_argument("--calib-sentences", type=int, default=256)
    parser.add_argument("--eval-sentences", type=int, default=64)
    parser.add_argument("--ports", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--save", default="")
    args = parser.parse_args()

    adapter = PortAdapter.calibrate_from_corpus(
        corpus_path=args.corpus,
        vectors_path=args.vectors,
        n_ports=args.ports,
        max_sentences=args.calib_sentences,
        seed=args.seed,
    )

    vectors, word2idx = load_word_vectors(args.vectors)
    sentences = load_corpus(args.corpus, max_sentences=args.eval_sentences, min_len=1)

    real_errors = []
    null_errors = []
    ctrl_norms = []
    for tokens in sentences:
        m_pr = sentence_state(tokens, vectors, word2idx)
        if np.linalg.norm(m_pr) <= 1e-12:
            continue
        m_ctrl, written, read = adapter.encode_and_read(m_pr)
        real_errors.append(np.abs(written - read))
        null_errors.append(np.abs(written - np.roll(read, 1)))
        ctrl_norms.append(np.linalg.norm(m_ctrl))

    if not real_errors:
        raise SystemExit("No valid evaluation states")

    real = np.vstack(real_errors)
    null = np.vstack(null_errors)
    real_mae_by_sentence = real.mean(axis=1)
    null_mae_by_sentence = null.mean(axis=1)

    median_real = float(np.median(real_mae_by_sentence))
    null_p10 = float(np.percentile(null_mae_by_sentence, 10))
    passed = median_real < null_p10

    print("PortAdapter functional check")
    print(f"  ports: {adapter.n_ports}")
    print(f"  sensor_width: {adapter.sensor_width}")
    print(f"  calibration_states: {adapter.sensor_ecdf.shape[0]}")
    print(f"  evaluation_states: {real.shape[0]}")
    print(f"  mean_abs_error: {float(real.mean()):.6f}")
    print(f"  max_abs_error: {float(real.max()):.6f}")
    print(f"  median_sentence_mae: {median_real:.6f}")
    print(f"  null_sentence_mae_p10: {null_p10:.6f}")
    print(f"  mean_m_ctrl_norm: {float(np.mean(ctrl_norms)):.6f}")
    print("  read_mechanism: known-address unbind + scalar carrier contraction")
    print("  cosine_10k_generation: not used")

    if args.save:
        out = Path(args.save)
        adapter.save(out)
        print(f"  saved: {out}")

    if not passed:
        raise SystemExit("PortAdapter readback was not better than null register alignment")


if __name__ == "__main__":
    main()
