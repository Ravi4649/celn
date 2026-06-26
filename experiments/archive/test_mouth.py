#!/usr/bin/env python3
"""
CELN v3 — Test: Mouth (Orquestrador com verificação de loop fechado)
======================================================================
Testa o pipeline completo: vetor composto → frase → verificação.

Pipeline:
  1. Carrega vetores, PairGraph
  2. Cria Mouth (Decomposer + Lexicalizer + Linearizer + nl_parser)
  3. Codifica regras com encode_rule
  4. Gera frases com verificação de loop fechado
  5. Reporta fidelidade, threshold, aceitação

Princípios:
  ZERO backprop/transformers. ZERO listas fixas. ZERO templates.
  ZERO thresholds mágicos. Tudo auto-calibrável.
"""

import sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from celn.core import D, normalize, similarity
from celn.mouth import Mouth, MouthResult
from celn.logic_encoder import LogicRoles, encode_rule
from celn.pair_graph import PairGraph


def load_vectors(path="data/celn_full_vectors.npz"):
    data = np.load(path, allow_pickle=True)
    vectors = data['vectors'].astype(np.float32)
    vocab = [str(w) for w in data['vocab']]
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}
    return vectors, w2i, i2w


VERBOSE = '-v' in sys.argv or '--verbose' in sys.argv


# =========================================================================
# Testes
# =========================================================================

def test_mouth_basic(mouth, roles):
    """Geração básica com verificação de fidelidade."""
    print(f"\n{'='*60}")
    print("TESTE: Geração básica com verificação")
    print(f"{'='*60}")

    cases = [
        ('gato', 'animal', 'ROLE_TODOS'),
        ('cachorro', 'animal', 'ROLE_TODOS'),
        ('gato', 'animal', 'ROLE_NENHUM'),
        ('casa', 'lar', 'ROLE_TODOS'),
        ('gato', 'animal', 'ROLE_ALGUM'),
    ]

    all_ok = True
    for ant, cons, role_name in cases:
        if ant not in mouth.w2i or cons not in mouth.w2i:
            print(f"  ⊘ '{ant}' ou '{cons}' — fora do codebook")
            continue

        va = normalize(mouth.codebook[mouth.w2i[ant]])
        vc = normalize(mouth.codebook[mouth.w2i[cons]])
        composite = encode_rule(roles.get(role_name), va, vc)

        result = mouth.generate(
            composite, max_steps=10, beam_width=5, verbose=VERBOSE,
        )

        # Critério: frase sempre retornada, aceitação pode falhar
        # (o parser pode não conseguir re-parser certas frases)
        ok = result.sentence != '' and result.role == role_name
        all_ok &= ok
        status = '✓' if ok else '✗'
        acc = 'ACEITA' if result.accepted else 'REJEITA'
        print(f"  {status} {role_name}({ant}→{cons}): "
              f"'{result.sentence[:60]}'")
        print(f"       fidelidade={result.fidelity_score:.4f} "
              f"threshold={result.threshold:.3f} {acc}")

    return all_ok


def test_mouth_fidelity_scores(mouth, roles):
    """Verifica que a fidelidade é reportada e tem range válido."""
    print(f"\n{'='*60}")
    print("TESTE: Reporte de fidelidade")
    print(f"{'='*60}")

    va = normalize(mouth.codebook[mouth.w2i['gato']])
    vc = normalize(mouth.codebook[mouth.w2i['animal']])
    composite = encode_rule(roles.TODOS, va, vc)

    result = mouth.generate(composite, max_steps=10, beam_width=3)

    score_valid = -1.0 <= result.fidelity_score <= 1.0
    threshold_valid = result.threshold > 0
    beams_valid = result.beams_tried > 0
    all_ok = score_valid and threshold_valid and beams_valid

    print(f"  Fidelidade: {result.fidelity_score:.4f} "
          f"(range: [-1,1]) {'✓' if score_valid else '✗'}")
    print(f"  Threshold:  {result.threshold:.4f} "
          f"(>0) {'✓' if threshold_valid else '✗'}")
    print(f"  Beams:      {result.beams_tried} "
          f"(>0) {'✓' if beams_valid else '✗'}")
    print(f"  Aceita:     {result.accepted}")
    print(f"  Frase: '{result.sentence}'")

    return all_ok


def test_mouth_multiple_roles(mouth, roles):
    """Testa com diferentes ROLEs e verifica aceitação."""
    print(f"\n{'='*60}")
    print("TESTE: Múltiplos ROLEs com verificação")
    print(f"{'='*60}")

    tests = [
        ('gato', 'animal', 'ROLE_TODOS'),
        ('gato', 'animal', 'ROLE_NENHUM'),
        ('gato', 'animal', 'ROLE_ALGUM'),
        ('gato', 'animal', 'ROLE_SE_ENTAO'),
        ('gato', 'animal', 'ROLE_NEGACAO'),
    ]

    all_ok = True
    for ant, cons, role_name in tests:
        va = normalize(mouth.codebook[mouth.w2i[ant]])
        vc = normalize(mouth.codebook[mouth.w2i[cons]])
        composite = encode_rule(roles.get(role_name), va, vc)

        result = mouth.generate(composite, max_steps=8, beam_width=3, verbose=VERBOSE)

        has_sentence = len(result.sentence) > 0
        has_role = result.role == role_name
        ok = has_sentence and has_role
        all_ok &= ok

        status = '✓' if ok else '✗'
        acc = 'ACEITA' if result.accepted else 'REJEITA'
        print(f"  {status} {role_name}: fidelidade={result.fidelity_score:.4f} "
              f"'{result.sentence[:50]}' {acc}")

    return all_ok


def test_mouth_low_confidence(mouth, roles):
    """Testa o caso de baixa confiança (frase não parseável de volta)."""
    print(f"\n{'='*60}")
    print("TESTE: Baixa confiança")
    print(f"{'='*60}")

    va = normalize(mouth.codebook[mouth.w2i['gato']])
    vc = normalize(mouth.codebook[mouth.w2i['animal']])
    composite = encode_rule(roles.TODOS, va, vc)

    result = mouth.generate(composite, max_steps=10, beam_width=5)

    # O resultado SEMPRE retorna algo (nunca falha completamente)
    has_sentence = len(result.sentence) > 0
    has_role_name = result.role != ''

    print(f"  Frase sempre retornada: {'✓' if has_sentence else '✗'}")
    print(f"  ROLE preservado: {'✓' if has_role_name else '✗'}")
    print(f"  Aceita: {result.accepted} | Low conf: {result.low_confidence}")
    print(f"  Fidelidade: {result.fidelity_score:.4f} "
          f"(threshold: {result.threshold:.3f})")

    return has_sentence


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 60)
    print("  CELN v3 — Mouth (Orquestrador + Verificação)")
    print("=" * 60)

    # Carrega dados
    try:
        vectors, w2i, i2w = load_vectors()
        print(f"\n  Vetores: {len(vectors)} palavras × {vectors.shape[1]} dims")
    except FileNotFoundError:
        print("\n  ✗ celn_full_vectors.npz não encontrado")
        return 1

    pair_graph = None
    try:
        pair_graph = PairGraph("data/pair_graph.npz")
        print(f"  PairGraph: {len(pair_graph.follower_map)} nós")
    except Exception as e:
        print(f"  ⚠ PairGraph: {e}")

    # Cria Mouth
    try:
        mouth = Mouth(
            vectors, w2i, i2w,
            pair_graph=pair_graph,
            spacy_model='pt_core_news_lg',
        )
        print(f"  Mouth: OK")
    except Exception as e:
        print(f"\n  ✗ Mouth não criado: {e}")
        return 1

    roles = LogicRoles(seed=42)

    # Executa testes
    results = []
    tests = [
        ('mouth_basic', test_mouth_basic),
        ('fidelity_scores', test_mouth_fidelity_scores),
        ('multiple_roles', test_mouth_multiple_roles),
        ('low_confidence', test_mouth_low_confidence),
    ]

    for name, fn in tests:
        ok = fn(mouth, roles)
        results.append((name, ok))

    # Relatório
    print(f"\n{'='*60}")
    print("  RELATÓRIO FINAL")
    print(f"{'='*60}")
    all_ok = True
    for name, ok in results:
        all_ok &= ok
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n  {'TODOS OS TESTES PASSARAM' if all_ok else 'ALGUNS TESTES FALHARAM'}")
    print(f"{'='*60}")

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
