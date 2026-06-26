# CELN — Deterministic Logical Reasoning on CPU Without Backpropagation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20836283.svg)](https://doi.org/10.5281/zenodo.20836283)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**CELN** (Códigos com Estrutura Lógica Natural) is a deterministic reasoning engine that operates entirely through vector symbolic architectures (VSA) in 10,000-dimensional space — **no backpropagation, no GPUs, no statistical hallucination**.

CELN scores **100% on ProofWriter** (500/500) and **100% on PrOntoQA** (100/100), runs on a Ryzen 2600 CPU at 34.7 ms per query using 493 MB of RAM. No transformers. No attention layers. No softmax at inference time. Just vector algebra.

---

## Quick start

```bash
pip install -r requirements.txt
python examples/step_by_step.py
```

The demo creates rules in Portuguese (e.g., "every dog is an animal", "Rex is a dog"), encodes them as 10k-dimensional vectors, stores them in associative memory, then walks through each deduction step with vector snapshots. No GPU, no model download — it generates random word vectors if pre-trained ones aren't available.

```bash
# Full benchmark: 500 ProofWriter examples (~5 minutes)
python experiments/benchmark_proofwriter_real.py
```

---

## Why

Every LLM in production today hallucinates, costs GPU time, and cannot guarantee logical consistency. CELN takes a different path:

- **Zero backpropagation** — learning and reasoning are purely algebraic (Hebbian updates, projective resonance, permutation tagging)
- **Zero GPU required** — runs on a $50 CPU (Ryzen 2600), 493 MB RAM
- **Zero hallucination** — output is determined by the algebra of the bound state, not by token probability
- **Zero fixed thresholds** — everything self-calibrates via percentiles of the empirical distribution
- **100% vector algebra** — one operation (`M(x,y)` = projective resonance) unifies binding, attention, and sequence encoding

---

## Results

| Benchmark | Accuracy | Notes |
|-----------|----------|-------|
| **ProofWriter** | **100%** (5000/5000) | Forward chaining, ~50 rules, stress-tested |
| **PrOntoQA** | **100%** (100/100) | Fictional ontology, 20+ rules each |

| Metric | Value |
|--------|-------|
| Inference latency (p50) | 26.4 ms per query |
| Inference latency (p95) | 79.1 ms per query |
| RAM (peak) | 493 MB |
| RAM growth | 1.7 MB per 1k examples |
| CPU | AMD Ryzen 2600 (no GPU) |
| Vector dimensionality | 10,000 |

All benchmarks run on CPU using only the modules in this repository — no transformers, no attention layers, no softmax sampling for inference.

---

## Install

```bash
pip install -r requirements.txt
```

That's it. The step-by-step demo needs nothing else — it generates random 10k-d vectors if pre-trained ones aren't present.

```bash
# Full benchmark (requires pre-trained vectors)
python experiments/benchmark_proofwriter_real.py   # 500 examples, ~5 min
```

### Data files

These `.npz` files are excluded from the repository (generated via `celn_v3.train.train_vectors()` or available in the companion data release):

| File | Size | Needed for |
|------|------|------------|
| `celn_v3_full_vectors.npz` | 709 MB | All benchmarks (word vectors) |
| `celn_v3_type_field.npz` | 327 MB | Syntactic structure features |
| `spacy_300d_vectors.npz` | 439 MB | Vocabulary bridge (300d→10k) |
| `sentence_centroids.npz` | 104 MB | SDM address initialization |
| `pair_graph.npz` | 36 KB | Transition lookahead scoring |

### Optional dependencies

| Package | Used by | Purpose |
|---------|---------|---------|
| `spacy` + `pt_core_news_lg` | `linearizer.py` | Morphological inflection (lazy import, graceful fallback) |
| `psutil` | stress test | RAM profiling |
| `scikit-learn` | ablation experiments | PCA baselines |

---

## Architecture overview

```
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

**20 modules** in `celn_v3/`:

| Module | Role |
|--------|------|
| `core.py` | Projective Resonance `M(x,y)`, bind/unbind via FFT, Phase Lens |
| `ghrr_core.py` | GHRR matrix binding and native attention |
| `train.py` | Word vector learning (PPMI + Hebbian / Random Projection) |
| `logic_encoder.py` | FOL rule encoding via permutation-tagged superposition |
| `memory.py` | Dense SDM with corroboration tracking and conflict isolation |
| `forward_chainer.py` | Forward chaining deduction over SDM-stored rules |
| `resonator.py` | Resonator Network decoder for factorization |
| `pair_graph.py` | Canonical transition graph for lookahead scoring |
| `vocab_bridge.py` | Aligned 300d→10k projection (Procrustes) |
| `port_adapter.py` | Non-metric bridge from opaque states to addressable ports |
| `generate.py` | Context-window generation with PMI boosting |
| `evaluate.py` | Fluency and diversity evaluation framework |
| `mouth_v2.py` | Attention-based generation orchestrator (3 scores: syn, sem, fidelity) |
| `decomposer.py` | Composite vector decomposition |
| `lexicalizer.py` | Holographic beam search |
| `linearizer.py` | Morphological inflection and sentence assembly |
| `content_lens.py` | Phase Lens with IDF-weighted alphas |
| `hdc_types.py` | HDC type vectors (distributional Hebbian) |
| `intent_distiller.py` | Auto-calibrated CAPL for semantic intent |
| `mouth.py` | Legacy orchestrator (deprecated — use `mouth_v2.py`) |

---

## What's next

- **Inductive Chainer** — learn new rules from observed patterns without backprop
- **Mouth v2 stabilization** — the GHRR attention-based orchestrator is working and produces more fluent output
- **Cross-lingual support** — the VSA operations are language-agnostic; Portuguese is the initial target
- **Continuous ingestion** — real-time learning from text streams without catastrophic forgetting

---

## Cite

```bibtex
@software{celn_v3,
  author       = {Flavio Oliveira Venturini},
  title        = {{CELN v3}: Deterministic Logical Reasoning on CPU
                   Without Backpropagation},
  month        = jun,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20836283},
  url          = {https://doi.org/10.5281/zenodo.20836283}
}
```

---

## License

CC BY-NC-SA 4.0 — Attribution-NonCommercial-ShareAlike 4.0 International.
