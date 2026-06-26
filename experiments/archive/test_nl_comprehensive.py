"""
Teste abrangente do parser NL com casos do estilo QMFOLBench.
"""

import sys
sys.path.insert(0, '/home/ravizin/celn-v3')

import numpy as np
from celn.nl_parser import VSAParser, parse_and_encode, parse_premise, ParsedPremise
from celn.logic_encoder import LogicRoles, decode_rule
from celn.vocab_bridge import VocabBridge


def load_codebook():
    data = np.load('celn_full_vectors.npz', allow_pickle=True)
    vectors = data['vectors']
    vocab = [str(w) for w in data['vocab']]
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}
    return vectors, w2i, i2w, vocab


# Casos de teste representativos do QMFOLBench
QMFOL_TEST_CASES = [
    # Universais afirmativas (∀)
    ("Todo gato é animal", "ROLE_TODOS", "gato", "animal"),
    ("Todos os cães são mamíferos", "ROLE_TODOS", "cães", "mamíferos"),
    ("Todas as aves voam", "ROLE_TODOS", "aves", "voam"),
    ("Todo ser humano é mortal", "ROLE_TODOS", "ser humano", "mortal"),
    
    # Universais negativas (∀¬)
    ("Nenhum peixe é mamífero", "ROLE_NENHUM", "peixe", "mamífero"),
    ("Nenhuma baleia é peixe", "ROLE_NENHUM", "baleia", "peixe"),
    ("Nenhum pássaro é mamífero", "ROLE_NENHUM", "pássaro", "mamífero"),
    
    # Existenciais (∃)
    ("Algum aluno é atleta", "ROLE_ALGUM", "aluno", "atleta"),
    ("Alguma pessoa é médica", "ROLE_ALGUM", "pessoa", "médica"),
    ("Alguns gatos são pretos", "ROLE_ALGUM", "gatos", "pretos"),
    
    # Condicionais (→)
    ("Se chove, então a rua molha", "ROLE_SE_ENTAO", "chove", "rua molha"),
    ("Se o aluno estuda, então passa", "ROLE_SE_ENTAO", "aluno estuda", "passa"),
    ("Se faz frio, então neva", "ROLE_SE_ENTAO", "faz frio", "neva"),
    
    # Negação (¬)
    ("Rex não é felino", "ROLE_NEGACAO", "Rex", "felino"),
    ("Platão não é contemporâneo", "ROLE_NEGACAO", "Platão", "contemporâneo"),
    ("Zero não é positivo", "ROLE_NEGACAO", "Zero", "positivo"),
    
    # Com modificadores (noun phrases)
    ("Todo gato preto é animal", "ROLE_TODOS", "gato preto", "animal"),
    ("Nenhum cão grande é gato", "ROLE_NENHUM", "cão grande", "gato"),
    ("Algum carro vermelho é rápido", "ROLE_ALGUM", "carro vermelho", "rápido"),
]


def run_comprehensive_test():
    print("=" * 70)
    print("TESTE ABRANGENTE: Parser NL → FOL (estilo QMFOLBench)")
    print("=" * 70)
    
    vectors, w2i, i2w, vocab = load_codebook()
    roles = LogicRoles(seed=42)
    
    # Cria parser VSA (já cria e alinha seu próprio VocabBridge)
    parser = VSAParser(vectors, w2i, i2w)
    
    print(f"\nCodebook CELN: {len(vocab)} palavras")
    print(f"Bridge alinhado: {parser.vocab_bridge._is_aligned}")
    print()
    
    results = {
        'total': 0,
        'parse_ok': 0,
        'encode_ok': 0,
        'decode_perfect': 0,
        'decode_role_ok': 0,
        'decode_ant_ok': 0,
        'decode_cons_ok': 0,
    }
    
    for sentence, exp_role, exp_ant, exp_cons in QMFOL_TEST_CASES:
        results['total'] += 1
        print(f"\n{results['total']}. '{sentence}'")
        
        # Parse
        premise = parse_premise(sentence, parser)
        parse_ok = (premise.role_name == exp_role and 
                    premise.antecedent == exp_ant and 
                    premise.consequent == exp_cons)
        
        if parse_ok:
            results['parse_ok'] += 1
            print(f"   Parse: ✓ {premise.role_name}({premise.antecedent}, {premise.consequent})")
        else:
            print(f"   Parse: ✗ Got {premise.role_name}({premise.antecedent}, {premise.consequent})")
            print(f"          Exp {exp_role}({exp_ant}, {exp_cons})")
            continue
        
        # Encode
        rule_vec, _ = parse_and_encode(sentence, parser)
        if rule_vec is None:
            print(f"   Encode: ✗ Falhou")
            continue
        results['encode_ok'] += 1
        
        # Decode
        decoded_role, decoded_ant, decoded_conseq, meta = decode_rule(
            rule_vec, roles, vectors, w2i, i2w
        )
        
        role_ok = decoded_role == exp_role
        ant_ok = decoded_ant == exp_ant
        cons_ok = decoded_conseq == exp_cons
        
        if role_ok: results['decode_role_ok'] += 1
        if ant_ok: results['decode_ant_ok'] += 1
        if cons_ok: results['decode_cons_ok'] += 1
        if role_ok and ant_ok and cons_ok: results['decode_perfect'] += 1
        
        status = "✓" if (role_ok and ant_ok and cons_ok) else "⚠"
        print(f"   Decode: {status} {decoded_role}({decoded_ant}, {decoded_conseq}) sim={meta['reconstruction_sim']:.3f}")
        if not role_ok: print(f"      Role: {decoded_role} (exp {exp_role})")
        if not ant_ok: print(f"      Ant:  {decoded_ant} (exp {exp_ant})")
        if not cons_ok: print(f"      Cons: {decoded_conseq} (exp {exp_cons})")
    
    # Resumo
    print("\n" + "=" * 70)
    print("RESUMO")
    print("=" * 70)
    r = results
    print(f"Total testados:          {r['total']}")
    print(f"Parse correto:           {r['parse_ok']}/{r['total']} ({100*r['parse_ok']/r['total']:.0f}%)")
    print(f"Encode bem-sucedido:     {r['encode_ok']}/{r['total']} ({100*r['encode_ok']/r['total']:.0f}%)")
    print(f"Decode perfeito:         {r['decode_perfect']}/{r['total']} ({100*r['decode_perfect']/r['total']:.0f}%)")
    print(f"  Role correto:          {r['decode_role_ok']}/{r['total']} ({100*r['decode_role_ok']/r['total']:.0f}%)")
    print(f"  Antecedente correto:   {r['decode_ant_ok']}/{r['total']} ({100*r['decode_ant_ok']/r['total']:.0f}%)")
    print(f"  Consequente correto:   {r['decode_cons_ok']}/{r['total']} ({100*r['decode_cons_ok']/r['total']:.0f}%)")
    
    return results


def test_chain_reasoning():
    print("\n" + "=" * 70)
    print("TESTE DE ENCADEAMENTO: Forward Chaining com Parser NL")
    print("=" * 70)
    
    vectors, w2i, i2w, _ = load_codebook()
    roles = LogicRoles(seed=42)
    
    parser = VSAParser(vectors, w2i, i2w)
    
    from celn.forward_chainer import ForwardChainer
    
    chainer = ForwardChainer(vectors, w2i, i2w, n_sdm_locations=1024, seed=42)
    
    # Premissas NL
    premises_nl = [
        "Todo gato é animal",
        "Todo animal é ser vivo",
        "Todo ser vivo respira",
    ]
    
    print("Premissas:")
    for p in premises_nl:
        rule_vec, premise = parse_and_encode(p, parser)
        if rule_vec is not None:
            ok = chainer.add_rule(premise.role_name, premise.antecedent, premise.consequent)
            print(f"  ✓ {premise.role_name}({premise.antecedent} → {premise.consequent})")
        else:
            print(f"  ✗ {p} - encoding falhou")
    
    # Dedução: gato → respira
    # strict=False permite matching aproximado entre conceitos similares
    # (ex: "ser" de palavra única ≈ composto "ser vivo")
    print("\nDedução: gato → ?")
    result = chainer.deduce(
        initial_facts=["gato"],
        conclusion="respira",
        max_depth=5,
        strict=False,
    )
    
    print(f"Label: {result.label}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Depth: {result.max_depth_reached}")
    print(f"Cadeia ({len(result.chain)} passos):")
    for i, step in enumerate(result.chain):
        print(f"  {i+1}. {step.rule_repr}: {step.fact_used} → {step.fact_derived} (conf={step.confidence:.3f})")


if __name__ == '__main__':
    run_comprehensive_test()
    test_chain_reasoning()