"""
Train projection matrix 10k→300d and test Dual-Space Generator
"""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from celn_v3.core import D, normalize, bind, encode_sequence
from celn_v3.dual_space_generator import DualSpaceGenerator
from celn_v3.pair_graph import PairGraph

ROOT = Path(__file__).parent.parent


def load_corpus(filepath: str, max_sentences: int = 10000):
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_sentences:
                break
            line = line.strip()
            if line:
                tokens = line.lower().split()
                if len(tokens) >= 3:
                    sentences.append(tokens)
    print(f"Loaded {len(sentences)} sentences from {filepath}")
    return sentences


def main():
    print("=" * 60)
    print("Dual-Space Generator: Training and Testing")
    print("=" * 60)

    # 1. Load 10k vectors
    print("\n1. Loading 10k vectors...")
    d10k = np.load(ROOT / "celn_v3_full_vectors.npz", allow_pickle=True)
    vectors_10k = d10k["vectors"].astype(np.float32)
    word2idx = d10k["word2idx"].item()
    vocab = d10k["vocab"]
    print(f"   {vectors_10k.shape[0]} words, {vectors_10k.shape[1]}d")

    # 2. Load spaCy 300d vectors
    print("\n2. Loading spaCy 300d vectors...")
    d300 = np.load(ROOT / "spacy_300d_vectors.npz")
    spacy_words = d300["words"]
    spacy_vectors = d300["vectors"].astype(np.float32)
    print(f"   {spacy_vectors.shape[0]} words, {spacy_vectors.shape[1]}d")

    # 3. Load type field
    print("\n3. Loading Type Field...")
    dtf = np.load(ROOT / "celn_v3_type_field.npz", allow_pickle=True)
    type_field = dtf["type_field"].astype(np.float32)
    type_w2i = dtf["word2idx"].item()
    print(f"   {type_field.shape[0]} words, {type_field.shape[1]}d")

    # 4. Load PairGraph
    print("\n4. Loading PairGraph...")
    pair_graph = PairGraph(ROOT / "pair_graph.npz")
    print(f"   {len(pair_graph.follower_map)} source words")

    # 5. Count common words
    common = set(word2idx.keys()) & set(str(w) for w in spacy_words)
    print(f"\n5. Common words (10k ∩ spaCy): {len(common)}")

    # 6. Create DualSpaceGenerator
    print("\n6. Creating DualSpaceGenerator...")
    generator = DualSpaceGenerator(
        vectors_10k=vectors_10k,
        word2idx=word2idx,
        spacy_words=spacy_words,
        spacy_vectors=spacy_vectors,
        type_field_array=type_field,
        type_word2idx=type_w2i,
        pair_graph=pair_graph,
    )

    # 7. Load corpus and learn projection
    print("\n7. Learning projection matrix 10k → 300d...")
    corpus = load_corpus(str(ROOT / "corpus_final.txt"), max_sentences=2920)
    generator.learn_projection_matrix(
        sentences=corpus,
        n_iterations=50,
        learning_rate=0.001,
    )

    # 8. Save projection matrix
    np.savez(
        ROOT / "dual_space_projection.npz",
        projection_matrix=generator.projection_matrix,
    )
    print("Saved projection matrix.")

    # 9. Test generation
    print("\n9. Testing generation...")
    prompts = [
        "os gatos são animais muito",
        "o cobre é um metal que",
        "a cidade antiga foi construída sobre",
        "os cientistas descobriram que a água",
    ]

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        result, infos = generator.generate(
            prompt=prompt,
            max_tokens=10,
            temperature=0.8,
        )
        print(f"  Output: '{result}'")
        for info in infos[-1:]:
            print(f"  Last step info: {info}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
