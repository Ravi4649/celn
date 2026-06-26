#!/usr/bin/env python3
"""
CELN v3 — Test: Mouth v2 (Reconstrução Atencional)
=====================================================
Testa geração por 3 scores competitivos (syn, sem, fidelity)
sem unbind, sem divagações.

Princípios:
  ZERO backprop/transformers. ZERO listas fixas. ZERO templates.
  ZERO thresholds mágicos. Tudo auto-calibrável.
"""

import sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from celn.core import D, normalize
from celn.mouth_v2 import MouthV2, StepScore
from celn.logic_encoder import LogicRoles, encode_rule
from celn.pair_graph import PairGraph


VERBOSE = '-v' in sys.argv or '--verbose' in sys.argv


def load_vectors(path="data/celn_full_vectors.npz"):
    data = np.load(path, allow_pickle=True)
    vectors = data['vectors'].astype(np.float32)
    vocab = [str(w) for w in data['vocab']]
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}
    return vectors, w2i, i2w


# =========================================================================
# Teste visual
# =========================================================================

def test_visual_all_roles(mouth, roles):
    """Teste visual para todos os 5 ROLEs."""
    print(f"\n{'='*72}")
    print("  🧪 TESTE VISUAL — Mouth v2 (Reconstrução Atencional)")
    print(f"{'='*72}")

    role_tests = [
        ('ROLE_TODOS', 'TODOS'),
        ('ROLE_NENHUM', 'NENHUM'),
        ('ROLE_ALGUM', 'ALGUM'),
        ('ROLE_SE_ENTAO', 'SE-ENTAO'),
        ('ROLE_NEGACAO', 'NEGACAO'),
    ]

    all_results = []
    for role_name, label in role_tests:
        v_ant = normalize(mouth.codebook[mouth.w2i['gato']])
        v_cons = normalize(mouth.codebook[mouth.w2i['animal']])
        composite = encode_rule(roles.get(role_name), v_ant, v_cons)

        result = mouth.generate(
            composite, max_steps=12, verbose=VERBOSE,
        )

        all_results.append((label, role_name, result))

    return all_results


def print_report(all_results):
    """Imprime relatório formatado."""
    print(f"\n{'='*72}")
    print("  RELATÓRIO — Mouth v2")
    print(f"{'='*72}")

    for label, role_name, result in all_results:
        # Verifica divagações
        has_digression = any(
            bad in result.sentence.lower()
            for bad in ['corpo humano', 'rio de grande', 'argentina', 'chile']
        )

        status = '✅' if result.fidelity > 0.3 else '❌'
        dig = '⚠ DIVAGA' if has_digression else 'OK'

        print(f"\n  {status} {label:12s}  fidelidade={result.fidelity:.4f}  {dig}")
        print(f"  │ Frase: {result.sentence}")
        print(f"  │ Passos: {len(result.steps)}  "
              f"candidatos={result.n_candidates_evaluated}  "
              f"expressos={result.content_expressed}/{result.content_total}")

        # Scores por passo (compacto)
        score_line = '  │ Scores: '
        for s in result.steps[:8]:
            score_line += f"[{s.word}]↑{s.total:.2f} "
        print(score_line)

    # Sumário
    print(f"\n{'='*72}")
    accepted = sum(1 for _, _, r in all_results if r.fidelity > 0.3)
    print(f"  ACEITAS: {accepted}/{len(all_results)}  "
          f"REJEITADAS: {len(all_results) - accepted}/{len(all_results)}")
    digressions = sum(1 for _, _, r in all_results
                      if any(b in r.sentence.lower()
                             for b in ['corpo humano', 'rio de grande',
                                       'argentina', 'chile']))
    if digressions == 0:
        print("  DIVAGAÇÕES:  ✅ Nenhuma — todas as frases são focadas")
    else:
        print(f"  DIVAGAÇÕES:  ⚠ {digressions}/{len(all_results)}")
    print(f"{'='*72}")


# =========================================================================
# Testes automatizados
# =========================================================================

def test_mouth_v2_basic(mouth, roles):
    """Testa que gera frase com conteúdo expresso."""
    print(f"\n{'='*60}")
    print("TESTE: Geração básica")
    print(f"{'='*60}")

    va = normalize(mouth.codebook[mouth.w2i['gato']])
    vc = normalize(mouth.codebook[mouth.w2i['animal']])
    composite = encode_rule(roles.TODOS, va, vc)

    result = mouth.generate(composite, max_steps=12)

    ok = result.sentence != '' and result.content_expressed > 0
    status = '✓' if ok else '✗'
    print(f"  {status} Frase: {result.sentence}")
    print(f"  {status} Fidelidade: {result.fidelity:.4f}")
    print(f"  {status} Conteúdo expresso: {result.content_expressed}/{result.content_total}")
    return ok


def test_fidelity_nonzero(mouth, roles):
    """Testa que todas as ROLEs geram fidelidade > 0."""
    print(f"\n{'='*60}")
    print("TESTE: Fidelidade para todas as ROLEs")
    print(f"{'='*60}")

    all_ok = True
    for role_name, label in [
        ('ROLE_TODOS', 'TODOS'), ('ROLE_NENHUM', 'NENHUM'),
        ('ROLE_ALGUM', 'ALGUM'), ('ROLE_SE_ENTAO', 'SE-ENTAO'),
        ('ROLE_NEGACAO', 'NEGACAO'),
    ]:
        va = normalize(mouth.codebook[mouth.w2i['gato']])
        vc = normalize(mouth.codebook[mouth.w2i['animal']])
        composite = encode_rule(roles.get(role_name), va, vc)

        result = mouth.generate(composite, max_steps=12)

        ok = result.fidelity > 0.0
        all_ok &= ok
        status = '✓' if ok else '✗'
        print(f"  {status} {label}: fidelidade={result.fidelity:.4f}  '{result.sentence[:60]}'")

    return all_ok


def test_no_digressions(mouth, roles):
    """Testa que NENHUMA frase contém divagações do mouth.py antigo."""
    print(f"\n{'='*60}")
    print("TESTE: Sem divagações")
    print(f"{'='*60}")

    bad_patterns = ['corpo humano', 'rio de grande', 'argentina', 'chile']

    all_ok = True
    for role_name in ['ROLE_TODOS', 'ROLE_NENHUM', 'ROLE_ALGUM',
                       'ROLE_SE_ENTAO', 'ROLE_NEGACAO']:
        va = normalize(mouth.codebook[mouth.w2i['gato']])
        vc = normalize(mouth.codebook[mouth.w2i['animal']])
        composite = encode_rule(roles.get(role_name), va, vc)

        result = mouth.generate(composite, max_steps=12)

        has_bad = any(b in result.sentence.lower() for b in bad_patterns)
        all_ok &= not has_bad
        status = '✓' if not has_bad else '✗'
        print(f"  {status} {role_name}: '{result.sentence[:60]}'")

    return all_ok


def test_content_expressed(mouth, roles):
    """Testa que ant e cons são expressos na frase."""
    print(f"\n{'='*60}")
    print("TESTE: Conteúdo expresso na frase")
    print(f"{'='*60}")

    all_ok = True
    for role_name, ant, cons in [
        ('ROLE_TODOS', 'gato', 'animal'),
        ('ROLE_NENHUM', 'gato', 'animal'),
    ]:
        va = normalize(mouth.codebook[mouth.w2i[ant]])
        vc = normalize(mouth.codebook[mouth.w2i[cons]])
        composite = encode_rule(roles.get(role_name), va, vc)

        result = mouth.generate(composite, max_steps=12)

        has_ant = ant in result.sentence.lower()
        has_cons = cons in result.sentence.lower()
        ok = has_ant and has_cons
        all_ok &= ok
        status = '✓' if ok else '✗'
        print(f"  {status} {role_name}: ant='{ant}'✓ cons='{cons}'✓  '{result.sentence[:60]}'")

    return all_ok


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 60)
    print("  CELN v3 — Mouth v2 Test Suite")
    print("=" * 60)

    # Setup
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

    mouth = MouthV2(vectors, w2i, i2w, pair_graph=pair_graph)
    roles = LogicRoles(seed=42)

    # Testes automatizados
    results = []
    for name, fn in [
        ('basic', test_mouth_v2_basic),
        ('fidelity_nonzero', test_fidelity_nonzero),
        ('no_digressions', test_no_digressions),
        ('content_expressed', test_content_expressed),
    ]:
        ok = fn(mouth, roles)
        results.append((name, ok))

    # Teste visual
    visual_results = test_visual_all_roles(mouth, roles)
    print_report(visual_results)

    # Relatório final
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
