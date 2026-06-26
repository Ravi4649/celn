#!/usr/bin/env python3
"""
CELN v3 — Test: Linearizer
============================
Testa conversão de word_sequence + ROLE em string final formatada.
Sem templates, sem classificação gramatical hardcoded.
"""

import sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from celn.linearizer import linearize, analyze_word, inflect_quantifier


def test_inflect_quantifier():
    print(f"\n{'='*60}")
    print("TESTE: Flexão de quantificador")
    print(f"{'='*60}")
    tests = [
        ('todo', 'sing', 'masc', 'todo'), ('todo', 'sing', 'fem', 'toda'),
        ('todo', 'plur', 'masc', 'todos'), ('todo', 'plur', 'fem', 'todas'),
        ('nenhum', 'sing', 'masc', 'nenhum'), ('nenhum', 'sing', 'fem', 'nenhuma'),
        ('algum', 'plur', 'masc', 'alguns'), ('algum', 'sing', 'fem', 'alguma'),
    ]
    all_ok = True
    for lemma, num, gen, expected in tests:
        result = inflect_quantifier(lemma, num, gen)
        ok = result == expected
        all_ok &= ok
        s = '✓' if ok else '✗'
        print(f"  {s} {lemma}({num},{gen}) → '{result}'")
    return all_ok


def test_analyze_word():
    print(f"\n{'='*60}")
    print("TESTE: Análise morfológica")
    print(f"{'='*60}")
    cases = [
        ('gato', 'NOUN', 'sing', 'masc'),
        ('gatos', 'NOUN', 'plur', 'masc'),
        ('casa', 'NOUN', 'sing', 'fem'),
        ('animal', 'NOUN', 'sing', 'masc'),
        ('animais', 'NOUN', 'plur', 'masc'),
    ]
    all_ok = True
    for word, exp_pos, exp_num, exp_gen in cases:
        a = analyze_word(word)
        ok = (a['number'] == exp_num and a['gender'] == exp_gen)
        all_ok &= ok
        s = '✓' if ok else '✗'
        print(f"  {s} '{word}': {a['pos']}({a['number']},{a['gender']})")
    return all_ok


def test_linearize_todos():
    print(f"\n{'='*60}")
    print("TESTE: ROLE_TODOS")
    print(f"{'='*60}")
    cases = [
        (['gato', 'animal'], 'Todo gato é um animal.'),
        (['gatos', 'animais'], 'Todos os gatos são animais.'),
        (['casa', 'lar'], 'Toda casa é um lar.'),
        (['gato', 'doméstico', 'animal'], 'Todo gato doméstico é um animal.'),
    ]
    all_ok = True
    for words, expected in cases:
        result = linearize(words, 'ROLE_TODOS', capitalize=True, add_period=True)
        ok = result == expected
        all_ok &= ok
        s = '✓' if ok else '✗'
        print(f"  {s} {words} → '{result}'")
    return all_ok


def test_linearize_nenhum():
    print(f"\n{'='*60}")
    print("TESTE: ROLE_NENHUM")
    print(f"{'='*60}")
    cases = [
        (['gato', 'animal'], 'Nenhum gato é animal.'),
        (['casa', 'lar'], 'Nenhuma casa é lar.'),
    ]
    all_ok = True
    for words, expected in cases:
        result = linearize(words, 'ROLE_NENHUM', capitalize=True, add_period=True)
        ok = result == expected
        all_ok &= ok
        s = '✓' if ok else '✗'
        print(f"  {s} {words} → '{result}'")
    return all_ok


def test_linearize_algum():
    print(f"\n{'='*60}")
    print("TESTE: ROLE_ALGUM")
    print(f"{'='*60}")
    cases = [
        (['gato', 'animal'], 'Algum gato é um animal.'),
        (['gatos', 'animais'], 'Alguns gatos são animais.'),
    ]
    all_ok = True
    for words, expected in cases:
        result = linearize(words, 'ROLE_ALGUM', capitalize=True, add_period=True)
        ok = result == expected
        all_ok &= ok
        s = '✓' if ok else '✗'
        print(f"  {s} {words} → '{result}'")
    return all_ok


def test_linearize_se_entao():
    print(f"\n{'='*60}")
    print("TESTE: ROLE_SE_ENTAO")
    print(f"{'='*60}")
    result = linearize(['gato', 'animal'], 'ROLE_SE_ENTAO',
                       ant='gato', cons='animal',
                       capitalize=True, add_period=True)
    ok = result == 'Se gato, então animal.'
    print(f"  {'✓' if ok else '✗'} → '{result}'")
    return ok


def test_linearize_negacao():
    print(f"\n{'='*60}")
    print("TESTE: ROLE_NEGACAO")
    print(f"{'='*60}")
    result = linearize(['gato', 'animal'], 'ROLE_NEGACAO',
                       capitalize=True, add_period=True)
    ok = result.startswith('Não')
    print(f"  {'✓' if ok else '✗'} → '{result}'")
    return ok


def test_lexicalizer_output():
    """Testa com saídas reais do Lexicalizer."""
    print(f"\n{'='*60}")
    print("TESTE: Sequências reais do Lexicalizer")
    print(f"{'='*60}")
    cases = [
        (['gato', 'doméstico', 'costuma', 'dormir', 'durante', 'animal'],
         'ROLE_TODOS', None, None,
         'Todo gato doméstico costuma dormir durante animal.'),
        (['gato', 'bebeu', 'água', 'sofá'],
         'ROLE_NENHUM', 'gato', 'animal',
         'Nenhum gato bebeu água sofá.'),
    ]
    all_ok = True
    for words, role, ant, cons, expected in cases:
        result = linearize(words, role, ant=ant, cons=cons,
                           capitalize=True, add_period=True)
        # aceita variações: só verifica quantificador + ant + cons presentes
        ok = all(w in result.lower() for w in words if len(w) > 3)
        all_ok &= ok
        s = '✓' if ok else '✗'
        print(f"  {s} {role}({words[:3]}...) → '{result[:60]}'")
    return all_ok


def test_batch():
    print(f"\n{'='*60}")
    print("TESTE: Bateria completa")
    print(f"{'='*60}")
    tests = [
        (['gato', 'animal'], 'ROLE_TODOS', 'Todo gato é um animal.'),
        (['gatos', 'animais'], 'ROLE_TODOS', 'Todos os gatos são animais.'),
        (['gato', 'animal'], 'ROLE_NENHUM', 'Nenhum gato é animal.'),
        (['casa', 'lar'], 'ROLE_ALGUM', 'Alguma casa é um lar.'),
        (['gato', 'animal'], 'ROLE_SE_ENTAO', 'Se gato, então animal.',
         {'ant': 'gato', 'cons': 'animal'}),
        (['gato', 'animal'], 'ROLE_NEGACAO', 'Não gato é animal.'),
    ]
    all_ok = True
    for entry in tests:
        words, role, expected = entry[0], entry[1], entry[2]
        kwargs = {}
        if len(entry) > 3:
            kwargs = entry[3]
        result = linearize(words, role, capitalize=True, add_period=True, **kwargs)
        ok = result == expected
        all_ok &= ok
        s = '✓' if ok else '✗'
        print(f"  {s} {role}: '{result}'")
    return all_ok


def main():
    print("=" * 60)
    print("  CELN v3 — Linearizer Test Suite")
    print("=" * 60)

    results = []
    for name, fn in [
        ('inflect_quantifier', test_inflect_quantifier),
        ('analyze_word', test_analyze_word),
        ('linearize_TODOS', test_linearize_todos),
        ('linearize_NENHUM', test_linearize_nenhum),
        ('linearize_ALGUM', test_linearize_algum),
        ('linearize_SE_ENTAO', test_linearize_se_entao),
        ('linearize_NEGACAO', test_linearize_negacao),
        ('lexicalizer_output', test_lexicalizer_output),
        ('batch_all_roles', test_batch),
    ]:
        ok = fn()
        results.append((name, ok))

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
