"""
Test script for celn.nl_parser
Analyzes parsing accuracy for QMFOLBench patterns.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/ravizin/celn-v3')

from celn.nl_parser import (
    parse_premise, parse_and_encode, tokenize,
    extract_quantifier, extract_antecedent_consequent,
    ParsedPremise,
)
from celn.logic_encoder import LogicRoles, decode_rule
from celn.vocab_bridge import VocabBridge


def load_codebook():
    """Load CELN vectors for decode verification."""
    data = np.load('celn_full_vectors.npz', allow_pickle=True)
    vectors = data['vectors']
    vocab = [str(w) for w in data['vocab']]
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}
    return vectors, w2i, i2w


def test_tokenize():
    print("=" * 60)
    print("TESTE 1: Tokenização")
    print("=" * 60)
    cases = [
        "Todo gato é animal",
        "Nenhum peixe é mamífero",
        "Se chove, então a rua fica molhada",
        "Algum aluno é atleta",
        "Rex não é felino",
        "Todo gato preto é animal",
    ]
    for s in cases:
        tokens = tokenize(s)
        print(f"  '{s}' → {tokens}")


def test_extract_quantifier():
    print("\n" + "=" * 60)
    print("TESTE 2: Extração de Quantificador")
    print("=" * 60)
    cases = [
        ("Todo gato é animal", "ROLE_TODOS"),
        ("Todas as gatas são animais", "ROLE_TODOS"),
        ("Todos os gatos são animais", "ROLE_TODOS"),
        ("Nenhum peixe é mamífero", "ROLE_NENHUM"),
        ("Nenhuma baleia é peixe", "ROLE_NENHUM"),
        ("Se chove, então molha", "ROLE_SE_ENTAO"),
        ("Se chove entao molha", "ROLE_SE_ENTAO"),
        ("Algum aluno é atleta", "ROLE_ALGUM"),
        ("Alguma pessoa é médica", "ROLE_ALGUM"),
        ("Rex não é felino", "ROLE_NEGACAO"),
        ("Rex nao e felino", "ROLE_NEGACAO"),
        ("Gato é animal", "UNKNOWN"),
    ]
    for sentence, expected in cases:
        tokens = tokenize(sentence)
        role, _ = extract_quantifier(tokens, sentence)
        status = "✓" if role == expected else "✗"
        print(f"  {status} '{sentence}' → {role} (esperado: {expected})")


def test_extract_antecedent_consequent():
    print("\n" + "=" * 60)
    print("TESTE 3: Extração de Antecedente/Consequente")
    print("=" * 60)
    cases = [
        ("Todo gato é animal", "ROLE_TODOS", "gato", "animal"),
        ("Todas gatas são animais", "ROLE_TODOS", "gatas", "animais"),
        ("Nenhum peixe é mamífero", "ROLE_NENHUM", "peixe", "mamífero"),
        ("Se chove, então a rua fica molhada", "ROLE_SE_ENTAO", "chove", "a rua fica molhada"),
        ("Se chove entao molha", "ROLE_SE_ENTAO", "chove", "molha"),
        ("Algum aluno é atleta", "ROLE_ALGUM", "aluno", "atleta"),
        ("Rex não é felino", "ROLE_NEGACAO", "rex", "felino"),
        ("Rex nao e felino", "ROLE_NEGACAO", "rex", "felino"),
        ("Todo gato preto é animal", "ROLE_TODOS", "gato preto", "animal"),
    ]
    for sentence, role, exp_ant, exp_conseq in cases:
        tokens = tokenize(sentence)
        ant, conseq, conf = extract_antecedent_consequent(tokens, role, sentence)
        ant_ok = ant == exp_ant
        conseq_ok = conseq == exp_conseq
        status = "✓" if (ant_ok and conseq_ok) else "✗"
        print(f"  {status} '{sentence}'")
        print(f"     ant: '{ant}' (esperado: '{exp_ant}') {'✓' if ant_ok else '✗'}")
        print(f"     cons: '{conseq}' (esperado: '{exp_conseq}') {'✓' if conseq_ok else '✗'}")
        print(f"     conf: {conf}")


def test_full_parse():
    print("\n" + "=" * 60)
    print("TESTE 4: Parse Completo (sem encoding)")
    print("=" * 60)
    cases = [
        "Todo gato é animal",
        "Nenhum peixe é mamífero",
        "Se chove, então a rua fica molhada",
        "Algum aluno é atleta",
        "Rex não é felino",
        "Todo gato preto é animal",
        "Todos os cachorros são mamíferos",
        "Se o aluno estuda, então passa",
        "Nenhuma baleia é peixe",
        "Alguma pessoa é médica",
        "Gato é animal",  # should fail - no quantifier
    ]
    for sentence in cases:
        premise = parse_premise(sentence)
        print(f"  '{sentence}'")
        print(f"     role: {premise.role_name}")
        print(f"     ant: '{premise.antecedent}'")
        print(f"     cons: '{premise.consequent}'")
        print(f"     conf: {premise.confidence}")


def test_encode_decode():
    print("\n" + "=" * 60)
    print("TESTE 5: Encode + Decode (round-trip)")
    print("=" * 60)

    vectors, w2i, i2w = load_codebook()
    roles = LogicRoles(seed=42)
    bridge = VocabBridge()

    # Alinha bridge ao espaço CELN
    print("  Alinhando VocabBridge ao codebook CELN...")
    bridge.align_to_celn(vectors, w2i)

    test_cases = [
        "Todo gato é animal",
        "Nenhum peixe é mamífero",
        "Algum aluno é atleta",
        "Rex não é felino",
        "Todo gato preto é animal",
    ]

    for sentence in test_cases:
        rule_vec, premise = parse_and_encode(sentence, roles, bridge, vectors, w2i)

        print(f"\n  '{sentence}'")
        print(f"     Parsed: {premise.role_name}({premise.antecedent}, {premise.consequent})")

        if rule_vec is None:
            print(f"     ✗ ENCODING FALHOU - vetor não disponível")
            continue

        # Decode and verify
        decoded_role, decoded_ant, decoded_conseq, meta = decode_rule(
            rule_vec, roles, vectors, w2i, i2w
        )

        role_ok = decoded_role == premise.role_name
        ant_ok = decoded_ant == premise.antecedent
        conseq_ok = decoded_conseq == premise.consequent

        status = "✓" if (role_ok and ant_ok and conseq_ok) else "✗"
        print(f"     Decoded: {decoded_role}({decoded_ant}, {decoded_conseq})")
        print(f"     Role: {'✓' if role_ok else '✗'} | Ant: {'✓' if ant_ok else '✗'} | Cons: {'✓' if conseq_ok else '✗'}")
        print(f"     Sim:     Sim: {meta['reconstruction_sim']:.4f}")

        if not (role_ok and ant_ok and conseq_ok):
            print(f"     ⚠ FALHA NO ROUND-TRIP")


def test_multiword_composition():
    print("\n" + "=" * 60)
    print("TESTE 6: Composição de termos multi-palavra")
    print("=" * 60)

    vectors, w2i, i2w = load_codebook()
    bridge = VocabBridge()
    bridge.align_to_celn(vectors, w2i)

    from celn.nl_parser import _compose_term_vector

    # Check composed vectors for multi-word terms
    test_terms = ["gato preto", "peixe tropical", "aluno estudante", "gato"]
    for term in test_terms:
        v = _compose_term_vector(term, bridge, vectors, w2i)
        print(f"  _compose_term_vector('{term}') → {'vetor' if v is not None else 'None'}")
        if v is not None:
            # Check nearest neighbor in CELN codebook
            sims = vectors @ v.astype(np.float32)
            idx = int(np.argmax(sims))
            print(f"     NN: '{i2w[idx]}' (sim={sims[idx]:.4f})")


if __name__ == '__main__':
    test_tokenize()
    test_extract_quantifier()
    test_extract_antecedent_consequent()
    test_full_parse()
    test_encode_decode()
    test_multiword_composition()

    print("\n" + "=" * 60)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 60)