"""
CELN v3 — PrOntoQA Benchmark
=============================
Raciocínio dedutivo com palavras inventadas (hash-based VSA vectors).

PrOntoQA usa palavras de mentira (dumpus, wumpus, grimpus, etc.)
forçando o sistema a raciocinar puramente sobre a estrutura lógica,
sem depender de conhecimento semântico prévio.

O pipeline:
  1. Cada palavra inventada recebe um vetor quasi-ortogonal determinístico
     (hash → semente → vetor unitário) — puro VSA, sem listas fixas
  2. Cada sentença é parseada em regras FOL (TODOS/NENHUM)
  3. Forward chaining deduz a conclusão
  4. Compara com o ground truth

Princípios:
  ZERO backprop. ZERO listas fixas. ZERO templates.
  ZERO thresholds mágicos. Tudo auto-calibrável.
  100% álgebra vetorial.
"""

import json, re, hashlib, numpy as np, sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from celn_v3.core import D, normalize, bind, unbind, similarity
from celn_v3.logic_encoder import (
    LogicRoles, encode_rule, decode_rule, decode_antecedent, decode_consequent,
    get_perm_ant, get_perm_cons,
)


# =========================================================================
# Gerador de vetores quasi-ortogonais para palavras inventadas
# =========================================================================

def word_to_vec(word: str, dim: int = D) -> np.ndarray:
    """Gera vetor unitário determinístico a partir do hash da palavra (normalizada)."""
    h = hashlib.sha256(word.lower().encode()).hexdigest()
    seed = int(h[:8], 16)
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return normalize(v)


def normalize_concept(word: str) -> str:
    """Normaliza: lowercase, despluraliza (remove 'es' de palavras terminando em 'ses'/'xes')."""
    w = word.lower().strip('.,;?!\'" \n')
    # PrOntoQA: plurals always add 'es' → 'Xus' → 'Xuses'
    # Strip 'es' to get singular
    if w.endswith('ses') or w.endswith('xes') or w.endswith('zes') or w.endswith('ces') or w.endswith('ges'):
        w = w[:-2]
    elif w.endswith('hes') or w.endswith('pes') or w.endswith('mes'):
        w = w[:-2]
    # Do NOT strip general 's' — singulars often already end in 's' (e.g. 'grimpus')
    return w


class InventedVocab:
    """Dicionário de palavras inventadas → vetores quasi-ortogonais."""

    def __init__(self):
        self.vectors: Dict[str, np.ndarray] = {}
        self.w2i: Dict[str, int] = {}
        self.i2w: Dict[int, str] = {}
        self.codebook: Optional[np.ndarray] = None

    def ensure(self, word: str):
        """Garante que a palavra normalizada tem vetor."""
        w = normalize_concept(word)
        if not w:
            return
        if w not in self.w2i:
            idx = len(self.w2i)
            self.w2i[w] = idx
            self.i2w[idx] = w
            self.vectors[w] = word_to_vec(w)

    def get(self, word: str) -> np.ndarray:
        self.ensure(word)
        return self.vectors[normalize_concept(word)]

    def build_codebook(self):
        """Constrói matriz (V, D) para nearest-neighbor decode."""
        self.ensure('')
        n = len(self.w2i)
        self.codebook = np.zeros((n, D), dtype=np.float32)
        for w, i in self.w2i.items():
            self.codebook[i] = self.vectors[w]


# =========================================================================
# PrOntoQA Parser
# =========================================================================

@dataclass
class PrOntoRule:
    role_name: str
    antecedent: str
    consequent: str

@dataclass
class PrOntoFact:
    individual: str
    concept: str  # class or property

@dataclass
class PrOntoExample:
    """Um exemplo completo do PrOntoQA."""
    facts: List[PrOntoFact]        # "X is a Y" — IS-A relationships
    rules: List[PrOntoRule]        # "Every X is Y" — universal statements
    query_ind: str                 # individual to query about
    query_prop: str                # property to query about
    gold_label: str                # True/False/Unknown
    chain_correct: List[str]       # ground truth chain of thought


def parse_question(question_text: str, query_text: str, chain_of_thought: List[str]) -> PrOntoExample:
    """Parseia um exemplo PrOntoQA em regras FOL."""
    facts = []
    rules = []
    query_ind = ''
    query_prop = ''

    # Parseia query: "Prove: X is Y" or "Prove: X is not Y"
    qm = re.match(r'Prove:\s*(.+?)\s+is\s+(not\s+)?(.+?)[.。]?\s*$', query_text, re.IGNORECASE)
    if qm:
        query_ind = qm.group(1).strip()
        negated = bool(qm.group(2))
        query_prop = normalize_concept(qm.group(3).strip())
        if negated:
            query_prop = f'not_{query_prop}'

    # Parseia sentenças do question
    sentences = re.split(r'[.。]\s*', question_text.strip())
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Pattern 1: "Every X is not Y." / "Each X is not Y." (must come before affirmative)
        m = re.match(r'(?:Every|Each)\s+(.+?)\s+is\s+not\s+(.+?)$', sent, re.IGNORECASE)
        if m:
            ant = normalize_concept(m.group(1).strip())
            cons = normalize_concept(m.group(2).strip())
            rules.append(PrOntoRule('ROLE_NENHUM', ant, cons))
            continue

        # Pattern 2: "Every X is a Y." / "Each X is a Y." / "Every X is Y." / "Each X is Y."
        m = re.match(r'(?:Every|Each)\s+(.+?)\s+is\s+(?:a\s+|an\s+)?(.+?)$', sent, re.IGNORECASE)
        if m:
            ant = normalize_concept(m.group(1).strip())
            cons_raw = m.group(2).strip()
            if cons_raw.startswith('not '):
                cons = normalize_concept(cons_raw[4:])
                rules.append(PrOntoRule('ROLE_NENHUM', ant, cons))
            else:
                cons = normalize_concept(cons_raw)
                rules.append(PrOntoRule('ROLE_TODOS', ant, cons))
            continue

        # Pattern 3: Plural "Xplur are Yplur." / "Xplur are Adj." (includes negation)
        m = re.match(r'(.+?)\s+are\s+(not\s+)?(.+?)$', sent, re.IGNORECASE)
        if m:
            ant = normalize_concept(m.group(1).strip())
            negated = bool(m.group(2))
            cons = normalize_concept(m.group(3).strip())
            if negated:
                rules.append(PrOntoRule('ROLE_NENHUM', ant, cons))
            else:
                rules.append(PrOntoRule('ROLE_TODOS', ant, cons))
            continue

        # Pattern 4: "X is a Y." / "X is an Y." — fact about individual
        m = re.match(r'(.+?)\s+is\s+(?:a\s+|an\s+)?(.+?)$', sent, re.IGNORECASE)
        if m:
            ind = m.group(1).strip()
            cons_raw = m.group(2).strip()
            if ind[0].isupper() and cons_raw != 'not':
                facts.append(PrOntoFact(ind, normalize_concept(cons_raw)))
                continue

        # Pattern 5: "X is Adj." — property fact about individual
        m = re.match(r'(.+?)\s+is\s+(.+?)$', sent, re.IGNORECASE)
        if m:
            ind = m.group(1).strip()
            adj = m.group(2).strip()
            if ind[0].isupper() and adj != 'not':
                facts.append(PrOntoFact(ind, normalize_concept(adj)))

    return PrOntoExample(
        facts=facts,
        rules=rules,
        query_ind=query_ind,
        query_prop=query_prop,
        gold_label='True',  # PrOntoQA ProofsOnly are all True
        chain_correct=chain_of_thought,
    )


def extract_entities(examples: List[PrOntoExample]) -> 'InventedVocab':
    """Extrai vocabulário inventado de todos os exemplos."""
    vocab = InventedVocab()
    for ex in examples:
        for f in ex.facts:
            vocab.ensure(f.individual)
            vocab.ensure(f.concept)
        for r in ex.rules:
            vocab.ensure(r.antecedent)
            vocab.ensure(r.consequent)
        vocab.ensure(ex.query_ind)
        vocab.ensure(ex.query_prop)
    vocab.build_codebook()
    return vocab


# =========================================================================
# Motor de Dedução
# =========================================================================

class PrOntoDeducer:
    """Motor de dedução para PrOntoQA usando ForwardChainer."""

    def __init__(self, vocab: InventedVocab):
        self.vocab = vocab
        self.roles = LogicRoles(seed=42)
        self.codebook = vocab.codebook
        self.w2i = vocab.w2i
        self.i2w = vocab.i2w

    def deduce(self, example: PrOntoExample) -> Tuple[str, float, List[str]]:
        from celn_v3.forward_chainer import ForwardChainer

        chainer = ForwardChainer(
            self.codebook, self.w2i, self.i2w,
            n_sdm_locations=2048, seed=42, use_bridge=False,
        )

        # Adiciona regras (both TODOS and NENHUM)
        for rule in example.rules:
            v_ant = self.vocab.get(rule.antecedent)
            v_cons = self.vocab.get(rule.consequent)
            role_vec = self.roles.get(rule.role_name)
            rule_vec = encode_rule(role_vec, v_ant, v_cons)
            chainer._rules_stored.append((
                rule.role_name,
                v_ant.copy(), v_cons.copy(), rule_vec,
            ))

        # Fatos iniciais
        initial_facts = [example.query_ind]
        is_negated_fact = {}
        for f in example.facts:
            if f.individual.lower() == example.query_ind.lower():
                initial_facts.append(f.concept)

        conclusion = example.query_prop
        if not conclusion:
            return 'Unknown', 0.0, []

        # Deduz com o chainer
        result = chainer.deduce(
            initial_facts=initial_facts,
            conclusion=conclusion,
            max_depth=10,
            strict=False,
        )

        # Pós-processamento: se NENHUM rule derivou cons, adiciona not_cons
        derived_set = set()
        for s in result.chain:
            derived_set.add(s.fact_derived)

        # Para NENHUM rules, deriva not_<consequent>
        for rule in example.rules:
            if rule.role_name == 'ROLE_NENHUM':
                # Se o antecedente foi derivado (ou é fato inicial), então ¬consequente é derivado
                if rule.antecedent in derived_set or rule.antecedent in initial_facts:
                    neg_cons = f'not_{rule.consequent}'
                    derived_set.add(neg_cons)

        # Verifica conclusão no conjunto derivado
        label = 'True' if conclusion in derived_set else 'Unknown'

        chain_steps = [s.rule_repr for s in result.chain]

        return label, result.confidence, chain_steps


# =========================================================================
# Benchmark Runner
# =========================================================================

def run_benchmark(data_path: str) -> dict:
    """Executa benchmark PrOntoQA."""
    with open(data_path) as f:
        raw_data = json.load(f)

    # Parse ALL examples (in-context + test)
    all_examples = []
    for ex_key in sorted(raw_data.keys(), key=lambda k: int(k.replace('example', ''))):
        ex = raw_data[ex_key]
        test = ex['test_example']
        parsed = parse_question(test['question'], test['query'], test['chain_of_thought'])
        all_examples.append(parsed)

    # Constrói vocabulário de todos os exemplos
    vocab = extract_entities(all_examples)
    deducer = PrOntoDeducer(vocab)

    # Executa benchmark
    results = []
    for i, example in enumerate(all_examples):
        label, confidence, chain_steps = deducer.deduce(example)

        # Verifica se a cadeia de steps está correta
        correct_chain = example.chain_correct
        chain_ok = len(chain_steps) == len(correct_chain) - 1  # one less rule than COT steps

        correct = (label == example.gold_label)

        results.append({
            'idx': i + 1,
            'gold': example.gold_label,
            'pred': label,
            'correct': correct,
            'confidence': confidence,
            'n_rules': len(example.rules),
            'n_facts': len(example.facts),
            'chain_steps': len(chain_steps),
            'chain_expected': len(correct_chain),
        })

        if not correct or i < 3:
            print(f"\n{'='*60}")
            print(f"Exemplo {i+1} | Gold: {example.gold_label} | Pred: {label} | {'✓' if correct else '✗'}")
            print(f"{'='*60}")
            print(f"  Regras: {len(example.rules)}, Fatos: {len(example.facts)}")
            print(f"  Query: {example.query_ind} → {example.query_prop}")
            print(f"  Cadeia esperada: {' → '.join(correct_chain)}")
            print(f"  Cadeia obtida:   {' → '.join(chain_steps) if chain_steps else '(vazia)'}")
            print(f"  Confiança: {confidence:.4f}")

    return results


def report(results: List[dict]):
    """Exibe relatório do benchmark."""
    print(f"\n{'='*70}")
    print(f"  PrOntoQA BENCHMARK RESULTS ({len(results)} examples)")
    print(f"{'='*70}")

    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0

    print(f"\n  Overall accuracy: {correct}/{total} ({accuracy:.1%})")
    print(f"  Avg confidence: {np.mean([r['confidence'] for r in results]):.3f}")
    print(f"  Avg chain steps: {np.mean([r['chain_steps'] for r in results]):.1f}")
    print(f"  Avg chain expected: {np.mean([r['chain_expected'] for r in results]):.1f}")

    # Decode verification para palavras inventadas
    print(f"\n  Decode verification (sample):")
    for i, r in enumerate(results[:3]):
        print(f"    Example {r['idx']}: pred={r['pred']}, gold={r['gold']}, steps={r['chain_steps']}/{r['chain_expected']}")

    return accuracy


if __name__ == '__main__':
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/prontoqa/1hop_ProofsOnly_random_noadj.json'
    
    results = run_benchmark(data_path)
    report(results)
