"""
CELN v3 — Projective Resonance Architecture
=============================================
Raciocínio lógico determinístico via Vector Symbolic Architectures (VSA),
sem backpropagation. 100% álgebra vetorial, 100% CPU.
"""

# ── Core vector operations ─────────────────────────────────────
from .core import (
    D,
    normalize,
    batch_normalize,
    bind,
    unbind,
    phi,
    phi_weights,
    projective_resonance,
    inverse_projective_resonance,
    encode_sequence,
    encode_sequence_plain,
    resonance_score,
    resonance_scores_batch,
    decode_next_candidates,
    phase_lens,
    phase_lens_scores_batch,
    precompute_word_spectra,
    similarity,
    spectral_entropy,
    make_random_vector,
    auto_threshold,
    competitive_filter,
)

# ── GHRR (Matrix binding & attention) ──────────────────────────
from .ghrr_core import (
    vec_10k_to_ghrr,
    ghrr_to_10k,
    ghrr_bind,
    ghrr_unbind,
    ghrr_similarity,
    ghrr_attention,
    ghrr_attention_score,
    ghrr_encode_sequence,
    make_random_ghrr_vector,
)

# ── Word vector training ───────────────────────────────────────
from .train import (
    tokenize,
    load_corpus,
    train_vectors,
    train_vectors_rp,
    build_cooccurrence,
    compute_ppmi,
    hebbian_update,
    precompute_spectra,
)

# ── Logic encoding & deduction ─────────────────────────────────
from .logic_encoder import (
    LogicRoles,
    encode_rule,
    decode_rule,
    decode_antecedent,
    decode_consequent,
    negate,
)
from .forward_chainer import ForwardChainer, DeductionResult, DeductionStep

# ── Associative memory ─────────────────────────────────────────
from .memory import DenseSDM, sentence_to_centroid

# ── Resonator decoder (factorization) ──────────────────────────
from .resonator import ResonatorDecoder, bind_vec, unbind_vec, top_k_accuracy

# ── Pair graph (transition scoring) ────────────────────────────
from .pair_graph import PairGraph

# ── Vocabulary bridge (300d → 10k projection) ──────────────────
from .vocab_bridge import VocabBridge, get_bridge

# ── Port adapter (opaque → addressable control state) ──────────
from .port_adapter import PortAdapter, PortAdapterConfig, sentence_state

# ── Content-Aware Phase Lens ───────────────────────────────────
from .content_lens import ContentAwareLens, ContentLensPacket
from .intent_distiller import IntentDistiller, IntentPacket

# ── HDC type vectors ───────────────────────────────────────────
from .hdc_types import train_hdc_type_vectors, learn_type_field, analyze_type_clusters

# ── Text generation ────────────────────────────────────────────
from .generate import (
    ContextWindow,
    generate,
    generate_baseline,
    generate_from_prefix,
    beam_search,
)

# ── Lexicalizer & Linearizer ───────────────────────────────────
from .lexicalizer import Lexicalizer, Beam
from .linearizer import linearize

# ── Decomposer ─────────────────────────────────────────────────
from .decomposer import Decomposer

# ── Mouth (closed-loop generation orchestrator) ────────────────
from .mouth import Mouth, MouthResult
from .mouth_v2 import MouthV2, GenResult

# ── Evaluation ─────────────────────────────────────────────────
from .evaluate import run_experiment, print_report
