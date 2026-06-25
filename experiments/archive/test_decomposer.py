#!/usr/bin/env python3
"""
CELN v3 — Test: Decomposer
============================
Testa decomposição de regras FOL codificadas e vetores compostos.

Pipeline:
  1. Carrega vetores CELN ou gera hash-based para palavras inventadas
  2. Codifica regras com encode_rule
  3. Decompõe com Decomposer.decompose()
  4. Verifica se role/ant/cons coincidem com o original

Princípios:
  ZERO backprop/transformers. ZERO listas fixas. ZERO templates.
  ZERO thresholds mágicos. Tudo auto-calibrável.
"""

import sys, hashlib, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from celn_v3.core import D, normalize
from celn_v3.logic_encoder import LogicRoles, encode_rule, negate
from celn_v3.decomposer import Decomposer


# =========================================================================
# Helpers
# =========================================================================

def hash_word(word: str, dim: int = D) -> np.ndarray:
    """Vetor hash-based quasi-ortogonal (PrOntoQA-style)."""
    h = hashlib.sha256(word.lower().encode()).hexdigest()
    seed = int(h[:8], 16)
    rng = np.random.RandomState(seed)
    return normalize(rng.randn(dim).astype(np.float32))


class TestVocab:
    """Vocabulário de teste com palavras reais e inventadas."""

    def __init__(self, existing_vectors=None, existing_w2i=None):
        self.vectors = {}
        self.w2i = {}
        self.i2w = {}

        if existing_vectors is not None and existing_w2i is not None:
            for w, idx in existing_w2i.items():
                if idx < len(existing_vectors):
                    self.vectors[w] = normalize(
                        existing_vectors[idx].astype(np.float32)
                    )
                    self.w2i[w] = idx
                    self.i2w[idx] = w

    def ensure(self, word: str):
        if word not in self.w2i:
            idx = len(self.w2i)
            self.w2i[word] = idx
            self.i2w[idx] = word
            self.vectors[word] = hash_word(word)

    def get(self, word: str) -> np.ndarray:
        self.ensure(word)
        return self.vectors[word]

    def build_codebook(self):
        self.ensure('')
        n = len(self.w2i)
        cb = np.zeros((n, D), dtype=np.float32)
        for w, i in self.w2i.items():
            cb[i] = self.vectors[w]
        return cb


# =========================================================================
# Testes
# =========================================================================

def test_decompose_word_pairs(decomposer, roles, vocab, pairs):
    """Codifica pares (role, ant, cons) e verifica decomposição."""
    print(f"\n{'='*60}")
    print(f"TESTE: Decomposição de pares com codebook")
    print(f"{'='*60}")

    all_passed = 0
    all_failed = 0

    for role_name in roles.ROLE_NAMES:
        role_vec = roles.get(role_name)
        for ant_word, cons_word in pairs:
            if ant_word not in vocab.w2i or cons_word not in vocab.w2i:
                continue

            v_ant = vocab.get(ant_word)
            v_cons = vocab.get(cons_word)

            composite = encode_rule(role_vec, v_ant, v_cons)
            result = decomposer.decompose(composite)

            role_ok = (result['role'] == role_name)
            ant_ok = (result['ant'] == ant_word)
            cons_ok = (result['cons'] == cons_word)

            ok = role_ok and ant_ok and cons_ok
            if ok:
                all_passed += 1
            else:
                all_failed += 1

            status = '✓' if ok else '✗'
            sim = result['confidence']
            details = ''
            if not role_ok:
                details += f' role={result["role"]}'
            if not ant_ok:
                details += f' ant={result["ant"]}'
            if not cons_ok:
                details += f' cons={result["cons"]}'

            if not ok or all_passed <= 5:
                print(f"  {status} {role_name}({ant_word} → {cons_word}): "
                      f"sim={sim:.4f}{details}")

    total = all_passed + all_failed
    print(f"\n  Resultado: {all_passed}/{total} ({100*all_passed/total:.0f}%)")
    return all_passed, all_failed


def test_invented_words():
    """Testa decomposição com palavras inventadas (PrOntoQA-style)."""
    print(f"\n{'='*60}")
    print("TESTE: Palavras inventadas (hash-based)")
    print(f"{'='*60}")

    vocab = TestVocab()
    invented_pairs = [
        ('dumpus', 'wumpus'),
        ('grimpus', 'brimpus'),
        ('fimpl', 'glorp'),
        ('zonk', 'plimp'),
        ('quux', 'blarg'),
    ]

    for ant, cons in invented_pairs:
        vocab.ensure(ant)
        vocab.ensure(cons)

    codebook = vocab.build_codebook()
    roles = LogicRoles(seed=42)
    decomposer = Decomposer(codebook, vocab.w2i, vocab.i2w, roles=roles)

    all_passed = 0
    all_failed = 0

    for role_name in ['ROLE_TODOS', 'ROLE_NENHUM', 'ROLE_ALGUM',
                       'ROLE_SE_ENTAO', 'ROLE_NEGACAO']:
        role_vec = roles.get(role_name)
        for ant_word, cons_word in invented_pairs:
            v_ant = vocab.get(ant_word)
            v_cons = vocab.get(cons_word)

            if role_name == 'ROLE_NENHUM':
                v_cons_encoded = negate(v_cons)
            else:
                v_cons_encoded = v_cons

            composite = encode_rule(role_vec, v_ant, v_cons_encoded)
            result = decomposer.decompose(composite)

            role_ok = (result['role'] == role_name)
            ant_ok = (result['ant'] == ant_word)
            cons_ok = (result['cons'] == cons_word)

            ok = role_ok and ant_ok and cons_ok
            if ok:
                all_passed += 1
            else:
                all_failed += 1

            status = '✓' if ok else '✗'
            sim = result['confidence']
            print(f"  {status} {role_name}({ant_word} → {cons_word}): "
                  f"sim={sim:.4f}")

    total = all_passed + all_failed
    print(f"\n  Resultado: {all_passed}/{total} ({100*all_passed/total:.0f}%)")
    return all_passed, all_failed


def test_decompose_deep():
    """Testa decompose_deep com sequências encadeadas."""
    print(f"\n{'='*60}")
    print("TESTE: Decomposição profunda (3 fatores)")
    print(f"{'='*60}")

    vocab = TestVocab()
    for w in ['gato', 'animal', 'ser', 'vivo']:
        vocab.ensure(w)
    codebook = vocab.build_codebook()

    from celn_v3.core import projective_resonance as M
    roles = LogicRoles(seed=42)
    decomposer = Decomposer(codebook, vocab.w2i, vocab.i2w, roles=roles)

    v_gato = vocab.get('gato')
    v_animal = vocab.get('animal')
    v_ser = vocab.get('ser')

    chain = M(v_gato, M(v_animal, v_ser))

    result = decomposer.decompose_deep(chain)
    print(f"  Result: {result['method']} n_factors={result.get('n_factors', '?')}")
    print(f"  ant={result['ant']} cons={result['cons']} "
          f"conf={result['confidence']:.4f}")

    if result.get('inner'):
        print(f"  inner: ant={result['inner']['ant']} "
              f"cons={result['inner']['cons']}")

    status = '✓' if result['confidence'] > 0.3 else '✗'
    print(f"  {status} Deep decomposition confidence: {result['confidence']:.4f}")
    return 1 if result['confidence'] > 0.3 else 0


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 60)
    print("  CELN v3 — Decomposer Test Suite")
    print("=" * 60)

    # Tenta carregar vetores reais
    vectors, w2i, i2w = None, None, None
    try:
        data = np.load('celn_v3_full_vectors.npz', allow_pickle=True)
        vocab_list = [str(w) for w in data['vocab']]
        vectors = data['vectors']
        w2i = {w: i for i, w in enumerate(vocab_list)}
        i2w = {i: w for i, w in enumerate(vocab_list)}
        print(f"\n  Vetores reais carregados: {len(vocab_list)} palavras, "
              f"{vectors.shape[1]} dims")
    except FileNotFoundError:
        print("\n  ⚠ celn_v3_full_vectors.npz não encontrado — "
              "usando apenas hash-based")

    total_passed = 0
    total_failed = 0

    # ── Teste 1: Vetores reais (se disponível) ──
    if vectors is not None:
        roles = LogicRoles(seed=42)
        decomposer = Decomposer(vectors, w2i, i2w, roles=roles)

        class RealVocab:
            def __init__(self, w2i, vectors):
                self.w2i = w2i
                self.vectors = vectors
            def ensure(self, word):
                pass
            def get(self, word):
                idx = self.w2i.get(word)
                if idx is None:
                    return None
                return normalize(self.vectors[idx].astype(np.float32))

        real_vocab = RealVocab(w2i, vectors)

        pairs = [
            ('gato', 'animal'),
            ('cachorro', 'animal'),
            ('aluno', 'escola'),
            ('professor', 'escola'),
            ('peixe', 'animal'),
            ('cobre', 'metal'),
            ('rosa', 'flor'),
            ('brasil', 'país'),
            ('sol', 'estrela'),
            ('lua', 'satélite'),
        ]

        p, f = test_decompose_word_pairs(decomposer, roles, real_vocab, pairs)
        total_passed += p
        total_failed += f

    # ── Teste 2: Palavras inventadas ──
    p, f = test_invented_words()
    total_passed += p
    total_failed += f

    # ── Teste 3: Decomposição profunda ──
    p = test_decompose_deep()
    total_passed += p

    # ── Relatório final ──
    total = total_passed + total_failed
    print(f"\n{'='*60}")
    if total_failed == 0:
        print(f"  ✓ TODOS OS TESTES PASSARAM ({total_passed}/{total})")
    else:
        print(f"  ⚠ {total_failed} FALHAS ({total_passed}/{total})")
    print(f"{'='*60}")

    return 0 if total_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
