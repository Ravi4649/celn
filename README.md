# CELN: Deterministic Logical Reasoning on CPU Without Backpropagation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20836283.svg)](https://doi.org/10.5281/zenodo.20836283)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**CELN** (*C. Elegans Learning Network*) is a deterministic reasoning engine built on Vector Symbolic Architectures (VSA). It achieves **100% accuracy on the ProofWriter benchmark** — including the "Unknown" class where Transformers fail — without backpropagation, without GPUs, and without hallucinations.

📄 **Paper:** [github.com/Ravi4649/celn-paper](https://github.com/Ravi4649/celn-paper)

---

## Quick Start

```bash
git clone https://github.com/Ravi4649/celn.git
cd celn
pip install -r requirements.txt
python examples/step_by_step_en.py
```

No downloads, no GPU, no model files. The demo encodes English rules ("Rex is a dog", "every dog is a mammal") into 10k-dimensional vectors using deterministic hash-based word vectors, then walks through each deduction step.

```bash
python examples/step_by_step.py      # Portuguese version
python experiments/benchmark_proofwriter_real.py  # Full benchmark: 500 examples, ~5 minutes
```

---

## Why CELN

LLMs predict the next token statistically. This makes them fluent, but introduces structural flaws: they hallucinate, cannot admit ignorance, and require expensive GPU clusters.

CELN treats reasoning as reversible linear algebra.

- **Zero backpropagation** — Attention ($Q \cdot K^T$) emerges natively from matrix binding (GHRR). No gradients, no training.
- **Zero GPU required** — Runs on a consumer CPU. Peak memory: 493 MB.
- **Zero hallucination** — Deduction is deterministic. If a conclusion cannot be derived, the system outputs "Unknown".
- **Zero fixed thresholds** — Everything self-calibrates via percentiles of the empirical distribution.
- **Deterministic by construction** — Same input always produces same output, with full audit trail.

---

Results

Tested on an AMD Ryzen 2600 (CPU) with 16 GB RAM. No GPU used.

| Benchmark | Examples | Accuracy | Latency (p50) | Latency (p95) | RAM Peak |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ProofWriter** | 500 | **100%** | 34.7 ms | 115 ms | 493 MB |
| **Stress Test** | 5,000 | **100%** | 34.7 ms | 115 ms | 493 MB |

Both benchmarks achieve 100% across all three classes (True, False, Unknown). Where Transformers drop 20–44 points on "Unknown" due to statistical bias, CELN maintains perfect accuracy because abstention is deterministic.

---

## Data Files

The pre-trained vector matrices (`.npz` files, ~1.5 GB total) are **not included** in the repository to keep the clone lightweight.

| File | Size | Required | Needed for |
| :--- | :--- | :--- | :--- |
| `data/celn_full_vectors.npz` | 709 MB | Yes | All benchmarks (word vectors) |
| `data/celn_type_field.npz` | 327 MB | Yes | Syntactic structure features |
| `data/spacy_300d_vectors.npz` | 439 MB | Yes | Vocabulary bridge (300d → 10k) |
| `data/sentence_centroids.npz` | 104 MB | Optional | SDM address initialization |
| `data/pair_graph.npz` | 36 KB | Yes | Transition lookahead scoring |

> **Note:** The demo (`step_by_step_en.py`) does **not** require any of these files. It uses deterministic hash-based vectors.

---

## Architecture

```text
corpus → train.py (PPMI + Hebbian) → word vectors (10k-D)
                                          │
            ┌──────────────────────────────┤
            │              │               │
       logic_encoder     memory         pair_graph
       (FOL rules)    (DenseSDM)      (transitions)
            │              │               │
            └────── forward_chainer ────────┘
                        (deduction)
                            │
                     mouth / mouth_v2
                     (vector → sentence)
```

20 modules in `celn/`:

| Module | Role |
| :--- | :--- |
| `core.py` | Projective Resonance M(x,y), bind/unbind via FFT, Phase Lens |
| `ghrr_core.py` | GHRR matrix binding and native Q·K^T attention |
| `train.py` | Word vector learning (PPMI + Hebbian / Random Projection) |
| `logic_encoder.py` | FOL rule encoding via permutation-tagged superposition |
| `memory.py` | Dense SDM with corroboration tracking and algebraic contradiction detection |
| `forward_chainer.py` | Forward chaining deduction over SDM-stored rules |
| `resonator.py` | Resonator Network decoder for factorization |
| `pair_graph.py` | Canonical transition graph for lookahead scoring |
| `vocab_bridge.py` | Aligned 300d → 10k projection (Procrustes) |
| `port_adapter.py` | Non-metric bridge from opaque states to addressable ports |
| `generate.py` | Context-window generation with PMI boosting |
| `evaluate.py` | Fluency and diversity evaluation framework |
| `mouth_v2.py` | Attention-based generation orchestrator (3 scores: syn, sem, fidelity) |
| `decomposer.py` | Composite vector decomposition into slot representations |
| `lexicalizer.py` | Holographic beam search |
| `linearizer.py` | Morphological inflection and sentence assembly |
| `content_lens.py` | Phase Lens with IDF-weighted alphas |
| `hdc_types.py` | HDC type vectors (distributional Hebbian) |
| `intent_distiller.py` | Auto-calibrated CAPL for semantic intent |
| `mouth.py` | Legacy orchestrator (deprecated — use mouth_v2.py) |

---

## Limitations

- **Rule extraction is manual** — The Inductive Chainer (learning new rules from observed patterns) is work in progress. Rules are currently hand-crafted.
- **Mouth v2 is functional but under active stabilization** — The attention-based orchestrator produces fluent output, but edge cases in long-form generation are still being resolved.
- **PairGraph trained on Portuguese corpus** — Cross-lingual generalization is structural (dependency patterns), not lexical. English parsing works via language-agnostic SVO patterns, but performance on non-Indo-European languages is untested.

---

## What's Next

- **Inductive Chainer** — Learn new rules from observed patterns without backprop.
- **Mouth v2 stabilization** — The GHRR attention-based orchestrator produces more fluent output.
- **Cross-lingual support** — VSA operations are language-agnostic; Portuguese is the initial target.
- **Continuous ingestion** — Real-time learning from text streams without catastrophic forgetting.

---

Cite

```bibtex
@article{venturini2026celn,
title={CELN: Deterministic Logical Reasoning on CPU Without Backpropagation},
author={Venturini, Flavio Oliveira},
year={2026},
doi={10.5281/zenodo.20836283}
}
```

---

License

CC BY-NC-SA 4.0 — Attribution-NonCommercial-ShareAlike 4.0 International.

---

## FAQ / Anticipated Criticisms

**Q: The code is just NumPy and Numba. Where is the deep learning framework?**
A: Exactly. CELN proves that complex logical reasoning does not require PyTorch, TensorFlow, or gradient descent. Linear algebra over high-dimensional spaces is sufficient and far more efficient.

**Q: The PairGraph was trained on a Portuguese corpus. How does it parse English?**
A: The PairGraph uses language-agnostic dependency patterns (Subject → Verb → Object). The ProofWriter benchmark uses hash-generated vectors for unknown words, proving the engine generalizes structurally without lexical memorization.

**Q: Was this code written by an AI?**
A: The architectural design, mathematical formulation, and analysis are human-authored. The Python implementation was generated via iterative prompting with AI assistants. The AI acted as a compiler for the mathematical blueprint, not the conceptualizer.

**Q: Why should I trust a 100% accuracy result?**
A: Because the ProofWriter benchmark has ground truth derived from formal logic. CELN's deduction is a deterministic matrix operation. If the math holds, 100% is expected — not surprising. You can run the code yourself to verify.