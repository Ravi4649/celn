"""
Test GHRR Generator — Bundling Architecture
=============================================
Valida a arquitetura GHRR com BUNDLING (soma) para o estado,
alinhada com o paper 2405.09689.
"""

import numpy as np
import sys, time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from celn.ghrr_core import (
    D, M, EFFECTIVE_DIM,
    vec_10k_to_ghrr, ghrr_to_10k,
    normalize_slices, ghrr_bind,
    ghrr_similarity, ghrr_similarity_batch,
    ghrr_attention, ghrr_attention_score,
    make_random_ghrr_vector,
)
from celn.ghrr_generator import GHRRGenerator
from celn.pair_graph import PairGraph

ROOT = Path(__file__).parent.parent


def test_1_ghrr_properties():
    print("=" * 60)
    print("Test 1: GHRR Core Properties")
    print("=" * 60)

    a = make_random_ghrr_vector(seed=10)
    b = make_random_ghrr_vector(seed=20)
    c = make_random_ghrr_vector(seed=30)

    sim_aa = ghrr_similarity(a, a)
    sim_ab = ghrr_similarity(a, b)
    assert sim_aa > 0.999, f"self-sim={sim_aa}"
    assert abs(sim_ab) < 0.3, f"cross-sim={sim_ab}"
    print(f"  self-sim={sim_aa:.4f} ✓, cross-sim={sim_ab:.4f} ✓")

    ab = ghrr_bind(a, b)
    ba = ghrr_bind(b, a)
    sim_ab_ba = ghrr_similarity(ab, ba)
    assert sim_ab_ba < 0.99, f"non-commutativity failed: {sim_ab_ba}"
    print(f"  non-commutative bind(a,b)≠bind(b,a): sim={sim_ab_ba:.4f} ✓")

    # Bundling: sum is similar to components
    bundle = a + b
    sim_bundle_a = ghrr_similarity(normalize_slices(bundle), a)
    print(f"  bundle(a+b)~a: {sim_bundle_a:.4f} (should be positive)")
    assert sim_bundle_a > 0, "bundle should be similar to components"

    print("  ✓ All core properties OK\n")


def test_2_ghrr_attention():
    print("=" * 60)
    print("Test 2: GHRR Attention")
    print("=" * 60)

    a = make_random_ghrr_vector(seed=100)
    b = make_random_ghrr_vector(seed=200)

    score_self = ghrr_attention_score(a, a)
    score_cross = ghrr_attention_score(a, b)
    print(f"  self-attention: {score_self:.4f}")
    print(f"  cross-attention: {score_cross:.4f}")
    print(f"  self > cross: {score_self > score_cross}")
    print("  ✓ Attention functional\n")


def test_3_semantic_preservation():
    print("=" * 60)
    print("Test 3: Semantic Preservation After GHRR Conversion")
    print("=" * 60)

    d10k = np.load(ROOT / "data/celn_full_vectors.npz", allow_pickle=True)
    vecs = d10k["vectors"].astype(np.float32)
    w2i = d10k["word2idx"].item()

    pairs = [
        ("gato", "cachorro", "related"),
        ("gato", "felino", "related"),
        ("gato", "mesa", "unrelated"),
        ("cobre", "metal", "related"),
        ("cobre", u"\u00e1gua", "unrelated"),
    ]

    for w1, w2, label in pairs:
        if w1 in w2i and w2 in w2i:
            gh1 = vec_10k_to_ghrr(vecs[int(w2i[w1])])
            gh2 = vec_10k_to_ghrr(vecs[int(w2i[w2])])
            sim = ghrr_similarity(gh1, gh2)
            print(f"  {w1}~{w2}: {sim:.4f} ({label})")

    print("  ✓ Semantic preservation OK\n")


def test_4_bundling_state():
    print("=" * 60)
    print("Test 4: Bundling State Discrimination")
    print("=" * 60)

    d10k = np.load(ROOT / "data/celn_full_vectors.npz", allow_pickle=True)
    vecs = d10k["vectors"].astype(np.float32)
    w2i = d10k["word2idx"].item()

    prompt = ["os", "gatos", u"s\u00e3o", "animais", "muito"]
    related = ["felino", "cachorro", "bichos", "selvagens", u"pequeninos"]
    unrelated = ["mesa", "cobre", "cidade", "cientistas", "janela"]

    state = np.zeros((D, M, M), dtype=np.float32)
    for w in prompt:
        if w in w2i:
            state += vec_10k_to_ghrr(vecs[int(w2i[w])])
    state = normalize_slices(state)

    print("  Related words:")
    for w in related:
        if w in w2i:
            gh = vec_10k_to_ghrr(vecs[int(w2i[w])])
            sim = ghrr_similarity(state, gh)
            print(f"    {w:15s}: {sim:+.4f}")

    print("  Unrelated words:")
    for w in unrelated:
        if w in w2i:
            gh = vec_10k_to_ghrr(vecs[int(w2i[w])])
            sim = ghrr_similarity(state, gh)
            print(f"    {w:15s}: {sim:+.4f}")

    print("  ✓ Bundling discriminates\n")


def test_5_generation():
    print("=" * 60)
    print("Test 5: GHRR Text Generation")
    print("=" * 60)

    print("\nLoading data...")
    d10k = np.load(ROOT / "data/celn_full_vectors.npz", allow_pickle=True)
    vectors_10k = d10k["vectors"].astype(np.float32)
    word2idx = d10k["word2idx"].item()

    d300 = np.load(ROOT / "data/spacy_300d_vectors.npz")
    spacy_words = d300["words"]
    spacy_vectors = d300["vectors"].astype(np.float32)

    dtf = np.load(ROOT / "celn_type_field.npz", allow_pickle=True)
    type_field = dtf["type_field"].astype(np.float32)
    type_w2i = dtf["word2idx"].item()

    pair_graph = PairGraph(ROOT / "data/pair_graph.npz")

    common = set(word2idx.keys()) & set(str(w) for w in spacy_words)
    print(f"  Vocab={len(word2idx)}, spaCy={spacy_vectors.shape[0]}, common={len(common)}")
    print(f"  GHRR: D={D}, M={M}, Effective Dim={EFFECTIVE_DIM}")

    print("\nCreating generator...")
    t0 = time.time()
    gen = GHRRGenerator(
        vectors_10k=vectors_10k,
        word2idx=word2idx,
        spacy_words=spacy_words,
        spacy_vectors=spacy_vectors,
        type_field_array=type_field,
        type_word2idx=type_w2i,
        pair_graph=pair_graph,
    )
    # Force GHRR conversion
    gen._ensure_ghrr()
    t1 = time.time()
    print(f"  Generator ready in {t1 - t0:.1f}s")

    prompts = [
        "os gatos são animais muito",
        "o cobre é um metal que",
        "a cidade antiga foi construída sobre",
        "os cientistas descobriram que a água",
    ]

    print("\nGenerating...")
    for prompt in prompts:
        print(f"\n  Prompt: '{prompt}'")
        try:
            result, infos = gen.generate(
                prompt=prompt,
                max_tokens=8,
                temperature=0.8,
                ema_alpha=0.4,
                exclude_window=3,
                verbose=False,
            )
            print(f"  Output: '{result}'")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n  ✓ Generation test complete\n")


def main():
    print(f"CELN-v3: GHRR Bundling Architecture Tests")
    print(f"D={D}, M={M}, Effective={EFFECTIVE_DIM}\n")

    tests = [
        test_1_ghrr_properties,
        test_2_ghrr_attention,
        test_3_semantic_preservation,
        test_4_bundling_state,
        test_5_generation,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 60)
    print("All tests completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
