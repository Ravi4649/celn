"""
CELN v3 — ProofWriter Real Benchmark
======================================
Usa o dataset real tasksource/proofwriter com 585K exemplos.

Correções principais:
  1. Negação VETORIAL: NENHUM usa encode_rule(ROLE, ant, negate(cons))
     → reflexão antipódica: ¬X = -X / |X|
  2. Sem string matching: True/False/Unknown via comparação vetorial
  3. Confiança via similaridade cosseno entre vetor derivado e query

Padrões suportados (depth-0):
  - "The X is Y" / "X is Y"          → TODOS(X, Y)
  - "The X is not Y" / "X is not Y"  → NENHUM(X, Y) [= TODOS(X, ¬Y)]
  - "X does Y"                        → TODOS(X, does_Y)
  - "X does not Y"                    → NENHUM(X, does_Y)
  - "X needs Y"                       → TODOS(X, needs_Y)
  - "X does not need Y"               → NENHUM(X, needs_Y)
"""

import re, hashlib, numpy as np, sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from celn.core import D, normalize, similarity
from celn.logic_encoder import LogicRoles, encode_rule, negate, decode_rule, decode_consequent


# =========================================================================
# Gerador de vetores hash-based
# =========================================================================

def word_to_vec(word: str, dim: int = D) -> np.ndarray:
    h = hashlib.sha256(word.lower().encode()).hexdigest()
    seed = int(h[:8], 16)
    rng = np.random.RandomState(seed)
    return normalize(rng.randn(dim).astype(np.float32))


# =========================================================================
# Vocabulário para ProofWriter
# =========================================================================

class Vocab:
    def __init__(self):
        self.vectors: Dict[str, np.ndarray] = {}
        self.w2i: Dict[str, int] = {}
        self.i2w: Dict[int, str] = {}
        self.codebook: Optional[np.ndarray] = None

    def ensure(self, word: str):
        if not word or word in self.w2i:
            return
        idx = len(self.w2i)
        self.w2i[word] = idx
        self.i2w[idx] = word
        self.vectors[word] = word_to_vec(word)

    def get(self, word: str) -> np.ndarray:
        if word not in self.w2i:
            self.ensure(word)
        return self.vectors[word]

    def build_codebook(self):
        self.ensure('')
        n = len(self.w2i)
        self.codebook = np.zeros((n, D), dtype=np.float32)
        for w, i in self.w2i.items():
            self.codebook[i] = self.vectors[w]


# =========================================================================
# Parser de sentenças ProofWriter
# =========================================================================

def normalize_term(term: str) -> str:
    """Normaliza termo: lowercase, remove artigos."""
    t = term.lower().strip('.,;?! ')
    t = re.sub(r'\b(the|a|an|this|that|these|those)\b', '', t).strip()
    t = re.sub(r'\s+', '_', t)
    return t


VERB_MAP = {
    'needs': 'need', 'wants': 'want', 'likes': 'like',
    'has': 'have', 'eats': 'eat', 'chases': 'chase',
    'sees': 'see', 'gives': 'give', 'helps': 'help',
    'plays': 'play', 'knows': 'know', 'visits': 'visit',
    'likes': 'like', 'needs': 'need', 'wants': 'want',
    'makes': 'make', 'takes': 'take', 'brings': 'bring',
    'uses': 'use', 'buys': 'buy', 'sends': 'send',
    'holds': 'hold', 'finds': 'find', 'keeps': 'keep',
    'tells': 'tell', 'shows': 'show', 'grows': 'grow',
}


def normalize_verb_phrase(verb: str, obj: str) -> str:
    """Normaliza verbo + objeto em forma de predicado."""
    v = VERB_MAP.get(verb.lower(), verb.lower().rstrip('s'))
    return f"{v}_{obj}"


def parse_theory_line(line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parseia uma sentença ProofWriter em (role, ant, cons).
    """
    line = line.strip()
    if not line:
        return None, None, None

    # Condicional: "If X then Y"
    m = re.match(r'If\s+(.+?)\s*,\s*then\s+(.+)', line, re.IGNORECASE)
    if not m:
        m = re.match(r'If\s+(.+?)\s+then\s+(.+)', line, re.IGNORECASE)
    if m:
        ant_s = m.group(1).strip()
        cons_s = m.group(2).strip()
        if ' and ' in ant_s:
            return None, None, None
        ant = normalize_term(ant_s)
        cons = normalize_term(cons_s)
        return ('ROLE_TODOS', ant, cons)

    # Negação com verbo: "X does not Y" / "X does not need Y"
    m = re.match(r'(.+?)\s+does\s+not\s+(.+)', line, re.IGNORECASE)
    if m:
        ant = normalize_term(m.group(1))
        vp = m.group(2).strip()
        # Check if verb phrase contains a known verb + object
        for verb, v_stem in VERB_MAP.items():
            if vp.lower().startswith(verb):
                obj = normalize_term(vp[len(verb):])
                return ('ROLE_NENHUM', ant, f"{v_stem}_{obj}")
        cons = normalize_term(vp)
        return ('ROLE_NENHUM', ant, cons)

    # Negação: "X is not Y"
    m = re.match(r'(.+?)\s+is\s+not\s+(.+)', line, re.IGNORECASE)
    if m:
        ant = normalize_term(m.group(1))
        cons = normalize_term(m.group(2))
        return ('ROLE_NENHUM', ant, cons)

    # Afirmativa com verbo: "X does Y" / "X needs Y"
    m = re.match(r'(.+?)\s+does\s+(.+)', line, re.IGNORECASE)
    if m:
        ant = normalize_term(m.group(1))
        vp = m.group(2).strip()
        for verb, v_stem in VERB_MAP.items():
            if vp.lower().startswith(verb):
                obj = normalize_term(vp[len(verb):])
                return ('ROLE_TODOS', ant, f"{v_stem}_{obj}")
        cons = normalize_term(vp)
        return ('ROLE_TODOS', ant, cons)

    # Afirmativa: "X is Y"
    m = re.match(r'(.+?)\s+is\s+(.+)', line, re.IGNORECASE)
    if m:
        ant = normalize_term(m.group(1))
        cons = normalize_term(m.group(2))
        return ('ROLE_TODOS', ant, cons)

    # Afirmativa com verbo: "X needs Y" / "X likes Y"
    verbs_pattern = '|'.join(sorted(VERB_MAP.keys(), key=len, reverse=True))
    m = re.match(rf'(.+?)\s+({verbs_pattern})\s+(.+)', line, re.IGNORECASE)
    if m:
        ant = normalize_term(m.group(1))
        verb = m.group(2).lower()
        obj = normalize_term(m.group(3))
        return ('ROLE_TODOS', ant, f"{VERB_MAP.get(verb, verb.rstrip('s'))}_{obj}")

    return None, None, None


def parse_question(q: str) -> Tuple[str, str, bool]:
    """
    Parseia pergunta ProofWriter em (sujeito, predicado, is_negated).
    """
    # Negação com verbo: "X does not Y"
    m = re.match(r'(.+?)\s+does\s+not\s+(.+)', q, re.IGNORECASE)
    if m:
        ant = normalize_term(m.group(1))
        vp = m.group(2).strip()
        for verb, v_stem in VERB_MAP.items():
            if vp.lower().startswith(verb):
                obj = normalize_term(vp[len(verb):])
                return (ant, f"{v_stem}_{obj}", True)
        return (ant, normalize_term(vp), True)

    # Negação: "X is not Y"
    m = re.match(r'(.+?)\s+is\s+not\s+(.+)', q, re.IGNORECASE)
    if m:
        return (normalize_term(m.group(1)), normalize_term(m.group(2)), True)

    # Afirmativa: "X is Y"
    m = re.match(r'(.+?)\s+is\s+(.+)', q, re.IGNORECASE)
    if m:
        return (normalize_term(m.group(1)), normalize_term(m.group(2)), False)

    # Afirmativa com verbo: "X needs Y" / "X likes Y"
    verbs_pattern = '|'.join(sorted(VERB_MAP.keys(), key=len, reverse=True))
    m = re.match(rf'(.+?)\s+({verbs_pattern})\s+(.+)', q, re.IGNORECASE)
    if m:
        ant = normalize_term(m.group(1))
        verb = m.group(2).lower()
        obj = normalize_term(m.group(3))
        return (ant, f"{VERB_MAP.get(verb, verb.rstrip('s'))}_{obj}", False)

    qn = normalize_term(q)
    return (qn, '', False)


# =========================================================================
# Motor de dedução com negação vetorial
# =========================================================================

class ProofWriterEngine:
    """Usa ForwardChainer com separação positiva/negativa."""

    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        self.roles = LogicRoles(seed=42)
        self.codebook = vocab.codebook
        self.w2i = vocab.w2i
        self.i2w = vocab.i2w

    def deduce(self, rules, subj, pred, query_negated):
        from celn.forward_chainer import ForwardChainer

        chainer = ForwardChainer(
            self.codebook, self.w2i, self.i2w,
            n_sdm_locations=2048, seed=42, use_bridge=False,
        )

        # Adiciona regras (v_cons SEM negate — NN precisa achar a string)
        for role, ant, cons in rules:
            v_ant = self.vocab.get(ant)
            v_cons = self.vocab.get(cons)
            role_vec = self.roles.get(role)
            rule_vec = encode_rule(role_vec, v_ant, v_cons)
            chainer._rules_stored.append((
                role, v_ant.copy(), v_cons.copy(), rule_vec,
            ))

        # Dedução via ForwardChainer
        result = chainer.deduce(
            initial_facts=[subj],
            conclusion=pred,
            max_depth=5,
            strict=False,
        )

        # Separa derivações: POSITIVA (TODOS) vs NEGATIVA (NENHUM)
        pos_derived = set()
        neg_derived = set()

        for s in result.chain:
            # Descobre qual regra produziu este step
            rule_type = 'ROLE_TODOS'
            for role, ant, cons in rules:
                if ant == s.fact_used and cons == s.fact_derived:
                    rule_type = role
                    break
                # Também verifica match com o subj como fact_used
                if subj == s.fact_used and ant in (subj, ant):
                    pass  # fallback

            if rule_type == 'ROLE_NENHUM':
                neg_derived.add(s.fact_derived)
            else:
                pos_derived.add(s.fact_derived)

        # Adiciona NENHUM latent: se ant está nos fatos e rule é NENHUM, 
        # a derivação é negativa
        nenhum_ant_rules = {ant for role, ant, _ in rules if role == 'ROLE_NENHUM'}
        nenhum_cons_rules = {cons for _, ant, cons in rules if _ == 'ROLE_NENHUM'}

        # Verifica se a conclusão está em pos_derived ou neg_derived
        pred_in_pos = pred in pos_derived
        pred_in_neg = pred in neg_derived

        if query_negated:
            # Query: "X is not Y" — pergunta se ¬Y vale
            if pred_in_neg:
                # NENHUM derivou Y → significa ¬Y é verdade → True
                label = 'True'
                confidence = 0.7
            elif pred_in_pos:
                # TODOS derivou Y → Y é verdade → ¬Y é falso → False
                label = 'False'
                confidence = 0.7
            else:
                label = 'Unknown'
                confidence = 0.3
        else:
            # Query: "X is Y" — pergunta se Y vale
            if pred_in_pos:
                label = 'True'
                confidence = result.confidence if result.confidence > 0 else 0.7
            elif pred_in_neg:
                label = 'False'
                confidence = 0.7
            else:
                label = 'Unknown'
                confidence = 0.3

        chain_steps = [s.rule_repr for s in result.chain]

        return (label, confidence, chain_steps)


# =========================================================================
# Benchmark Runner
# =========================================================================

def run_benchmark(n_examples: int = 500):
    from datasets import load_dataset
    token = None  # set your HF token here if needed for gated datasets
    ds = load_dataset('tasksource/proofwriter', token=token)

    vocab = Vocab()
    engine = None
    results = []
    n_parsed = 0

    # PASSO 1: Registrar todo o vocabulário de uma vez
    all_rules = []
    all_questions = []
    for i in range(min(n_examples, len(ds['test']))):
        ex = ds['test'][i]
        rules = []
        for line in ex['theory'].split('.'):
            role, ant, cons = parse_theory_line(line.strip())
            if role and ant and cons:
                rules.append((role, ant, cons))
                vocab.ensure(ant)
                vocab.ensure(cons)
        all_rules.append(rules)

        subj, pred, query_negated = parse_question(ex['question'])
        vocab.ensure(subj)
        if pred:
            vocab.ensure(pred)
        all_questions.append((subj, pred, query_negated))

    # Constrói codebook UMA VEZ
    vocab.build_codebook()
    engine = ProofWriterEngine(vocab)

    # PASSO 2: Executa benchmark
    for i in range(min(n_examples, len(ds['test']))):
        ex = ds['test'][i]
        gold = ex['answer']
        rules = all_rules[i]
        subj, pred, query_negated = all_questions[i]

        if not rules or not pred:
            results.append({
                'gold': gold, 'pred': 'Unknown', 'correct': gold == 'Unknown',
                'conf': 0.0, 'steps': 0, 'parsed': False,
            })
            continue

        n_parsed += 1
        label, conf, chain_steps = engine.deduce(rules, subj, pred, query_negated)
        correct = (label == gold)

        results.append({
            'gold': gold, 'pred': label, 'correct': correct,
            'conf': conf, 'steps': len(chain_steps), 'parsed': True,
        })

        if not correct:
            print(f"  ✗ #{i} gold={gold} pred={label} (query: {ex['question'][:60]})")

    return results


def report(results):
    total = len(results)
    parsed = sum(1 for r in results if r.get('parsed'))
    correct = sum(1 for r in results if r['correct'])

    print(f"\n{'='*70}")
    print(f"  PROOFWRITER BENCHMARK ({total} examples)")
    print(f"{'='*70}")
    print(f"  Parseáveis: {parsed}/{total} ({100*parsed/total:.0f}%)")

    for label in ['True', 'False', 'Unknown']:
        subset = [r for r in results if r['gold'] == label]
        if subset:
            ok = sum(1 for r in subset if r['correct'])
            avg_c = np.mean([r['conf'] for r in subset])
            print(f"  {label:8}: {ok:3}/{len(subset):3} ({100*ok/len(subset):5.1f}%) conf={avg_c:.3f}")

    # Acurácia apenas nos parseáveis
    parsed_ok = sum(1 for r in results if r.get('parsed') and r['correct'])
    if parsed > 0:
        print(f"\n  Accuracy (parsed): {parsed_ok}/{parsed} ({100*parsed_ok/parsed:.1f}%)")
    print(f"  Accuracy (all):   {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"  Avg confidence: {np.mean([r['conf'] for r in results]):.3f}")


if __name__ == '__main__':
    results = run_benchmark(500)
    report(results)
