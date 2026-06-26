"""
CELN v3 — FOLIO Benchmark (English)
======================================
Testa o pipeline NL→FOL contra o dataset FOLIO (Yale-LILY).

Adaptações para inglês:
  - spaCy en_core_web_md (UD universal)
  - ROLE_PROTOTYPES com exemplos em inglês
  - _extract_quantified: ROOT pode ser AUX (copula) ou VERB
  - _extract_conditional: 'if' e 'then' em inglês
  - VocabBridge com vetores spaCy inglês
"""

import sys, json, time, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from celn.core import normalize, similarity, auto_threshold
from celn.logic_encoder import LogicRoles, decode_rule, encode_rule
from celn.forward_chainer import ForwardChainer
from celn.vocab_bridge import VocabBridge

import spacy


# =========================================================================
# VSAParser adaptado para inglês
# =========================================================================

class FOLIOParser:
    """
    VSAParser adaptado para patterns do FOLIO em inglês.
    
    Diferenças do VSAParser original (português):
    - ROOT pode ser AUX (is/are) em vez de ADJ/NOUN
    - Verbos como predicado ("All animals breathe")
    - "No" como DET detectado via protótipo vetorial
    - Condicional "if...then"
    - Either/or, neither/nor (ignorados — simplificação)
    """

    ROLE_PROTOTYPES = {
        'ROLE_TODOS': ['all', 'every', 'any', 'each', 'todo', 'todos'],
        'ROLE_NENHUM': ['no', 'none', 'nothing', 'not', 'nenhum', 'no one'],
        'ROLE_ALGUM': ['some', 'a few', 'several', 'most', 'algum', 'alguma'],
        'ROLE_SE_ENTAO': ['if', 'then', 'se', 'então'],
        'ROLE_NEGACAO': ['not', "n't", 'no', 'never', 'não', 'nao'],
    }

    def __init__(self, celn_vectors, celn_w2i, celn_i2w, spacy_model='en_core_web_md', seed=42):
        self.celn_vectors = celn_vectors
        self.celn_w2i = celn_w2i
        self.celn_i2w = celn_i2w
        
        self.nlp = spacy.load(spacy_model)
        
        # VocabBridge para OOV
        self.bridge = VocabBridge()
        self.bridge.align_to_celn(celn_vectors, celn_w2i)
        
        self.roles = LogicRoles(seed=seed)
        
        # Protótipos ROLE
        self._role_proto_vecs = {}
        self._build_role_prototypes()

    def _build_role_prototypes(self):
        for role_name, exemplars in self.ROLE_PROTOTYPES.items():
            vecs = []
            for ex in exemplars:
                v = self._get_any_vec(ex)
                if v is not None:
                    vecs.append(v)
            if vecs:
                proto = normalize(np.mean(vecs, axis=0))
                self._role_proto_vecs[role_name] = proto

    def _get_any_vec(self, word):
        attempts = [word, word.lower(), word.capitalize()]
        for w in attempts:
            if w in self.celn_w2i:
                idx = self.celn_w2i[w]
                if idx < len(self.celn_vectors):
                    return normalize(self.celn_vectors[idx].astype(np.float32))
        for w in attempts:
            v = self.bridge.project(w)
            if v is not None:
                return v
        # Fallback: tenta cada palavra de bridge com lowercase
        v = self.bridge.project(word.lower())
        if v is not None:
            return v
        return None

    def _find_quantifier(self, doc):
        best_role = None
        best_sim = -1.0
        
        for token in doc:
            if token.pos_ not in {'DET', 'SCONJ', 'ADV', 'PART', 'AUX', 'CCONJ'}:
                continue
            v_tok = self._get_any_vec(token.lower_)
            if v_tok is None:
                continue
            for role_name, proto in self._role_proto_vecs.items():
                sim = float(v_tok @ proto)
                if sim > best_sim:
                    best_sim = sim
                    best_role = role_name
        
        return best_role

    def extract_slots(self, doc):
        """Extrai (role_name, ant, cons) de uma sentença FOLIO."""
        role_name = self._find_quantifier(doc)
        if role_name is None:
            return 'UNKNOWN', '', ''
        
        root = self._get_root(doc)
        if root is None:
            return role_name, '', ''
        
        if role_name == 'ROLE_SE_ENTAO':
            return role_name, *self._extract_conditional(doc)
        elif role_name in {'ROLE_NENHUM', 'ROLE_NEGACAO'} and self._has_negation_verb(doc):
            return role_name, *self._extract_negation(doc, root)
        else:
            return role_name, *self._extract_quantified(doc, root)

    def _get_root(self, doc):
        for t in doc:
            if t.dep_ == 'ROOT':
                return t
        return doc[-1] if doc else None

    def _has_negation_verb(self, doc):
        """Verifica se sentença tem verbo de negação (is not, are not)."""
        for t in doc:
            if t.lemma_ == 'not' and t.pos_ in {'ADV', 'PART'}:
                return True
        return False

    def _extract_conditional(self, doc):
        """Extrai de 'If X, then Y'."""
        se = None
        entao = None
        for t in doc:
            if t.lemma_ == 'if' and t.pos_ == 'SCONJ':
                se = t
            if t.lower_ in {'then', 'então', 'entao'}:
                entao = t
        
        if se is None or entao is None:
            return '', ''
        
        ant = []
        for t in doc:
            if t.i > se.i and t.i < entao.i and t.pos_ not in {'PUNCT', 'DET', 'SCONJ', 'ADV', 'CCONJ'}:
                ant.append(t.text)
        cons = []
        for t in doc:
            if t.i > entao.i and t.pos_ not in {'PUNCT', 'DET', 'ADV', 'SCONJ', 'CCONJ'}:
                cons.append(t.text)
        
        return ' '.join(ant), ' '.join(cons)

    def _extract_negation(self, doc, root):
        """Extrai de 'X is not Y' ou 'No X are Y'."""
        ant = ''
        cons = root.lemma_ if root.pos_ == 'VERB' else root.text
        
        for child in root.children:
            if child.dep_ in {'nsubj', 'nsubjpass'}:
                parts = [t.text for t in child.subtree if t.pos_ not in {'PUNCT', 'DET'}]
                ant = ' '.join(parts)
                break
        
        # Se ROOT é AUX (is/are), cons é o attr
        if root.pos_ == 'AUX':
            for child in root.children:
                if child.dep_ in {'attr', 'acomp'}:
                    cons = ' '.join(t.text for t in child.subtree if t.pos_ not in {'PUNCT', 'DET'})
                    break
        
        return ant, cons

    def _extract_quantified(self, doc, root):
        """Extrai de 'All X are Y', 'All X verb', 'No X are Y'."""
        ant = ''
        cons = ''
        
        # nsubj/csubj = antecedente
        for child in root.children:
            if child.dep_ in {'nsubj', 'nsubjpass', 'csubj'}:
                parts = [t.text for t in child.subtree if t.pos_ not in {'PUNCT', 'DET', 'ADV'}]
                if parts[0].lower() in {'any'}:
                    parts = parts[1:] if len(parts) > 1 else parts
                ant = ' '.join(parts)
                break
        
        # Verbo ou attr/acomp = consequente
        if root.pos_ == 'AUX':
            # Copula: "are Y", "is Y"
            for child in root.children:
                if child.dep_ in {'attr', 'acomp'}:
                    parts = [t.text for t in child.subtree if t.pos_ not in {'PUNCT', 'DET'}]
                    cons = ' '.join(parts)
                    break
            # Fallback: verbo + objeto
            if not cons:
                for child in root.children:
                    if child.dep_ in {'acomp', 'advmod', 'xcomp', 'dobj'}:
                        cons = child.text
                        break
        elif root.pos_ == 'VERB':
            # Verbo predicado
            cons_parts = [root.text]
            for child in root.children:
                if child.dep_ in {'dobj', 'attr', 'acomp', 'xcomp', 'advmod', 'prt'}:
                    cons_parts.append(child.text)
            cons = ' '.join(cons_parts)
        
        return ant, cons

    def compose_term(self, term):
        """Composição multi-palavra."""
        if not term:
            return None
        words = term.split()
        if len(words) == 1:
            return self._get_any_vec(term)
        
        vecs = []
        for w in words:
            v = self._get_any_vec(w)
            if v is not None:
                vecs.append(v)
        
        if not vecs:
            return None
        if len(vecs) == 1:
            return vecs[0]
        
        weights = [0.7] + [0.3/(len(vecs)-1)] * (len(vecs)-1)
        return normalize(np.average(vecs, axis=0, weights=weights))


# =========================================================================
# FOLIO Dataset
# =========================================================================

def load_folio(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_folio_benchmark(entries, max_examples=10):
    """Executa benchmark nos primeiros N exemplos FOLIO."""
    data = np.load('data/celn_full_vectors.npz', allow_pickle=True)
    vectors = data['vectors']
    vocab = [str(w) for w in data['vocab']]
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}
    
    parser = FOLIOParser(vectors, w2i, i2w)
    
    results = []
    for idx, entry in enumerate(entries[:max_examples]):
        premises = entry['premises']
        conclusion = entry['conclusion']
        gold_label = entry['label']
        
        print(f"\n{'='*70}")
        print(f"Exemplo {idx+1} | Label: {gold_label}")
        print(f"{'='*70}")
        
        # Cria chainer com CÓPIAS do codebook (para não compartilhar estado com parser)
        chainer = ForwardChainer(
            vectors.copy(), dict(w2i), dict(i2w),
            n_sdm_locations=2048, seed=42,
        )
        parse_ok = True
        for p in premises:
            role, ant, cons = parser.extract_slots(parser.nlp(p))
            if role == 'UNKNOWN' or not ant or not cons:
                print(f"  ✗ PREMISE PARSE FALHOU: {p[:80]}")
                parse_ok = False
                continue
            
            v_ant = parser.compose_term(ant)
            v_cons = parser.compose_term(cons)
            if v_ant is None or v_cons is None:
                print(f"  ✗ ENCODE FALHOU: {p[:80]}")
                parse_ok = False
                continue
            
            role_vec = parser.roles.get(role)
            rule_vec = encode_rule(role_vec, v_ant, v_cons)
            
            ok = chainer.add_rule(role, ant, cons)
            print(f"  {'✓' if ok else '✗'} {role:15}({ant:20} → {cons:20})")
            if not ok:
                parse_ok = False
        
        # Parse conclusão
        c_role, c_ant, c_cons = parser.extract_slots(parser.nlp(conclusion))
        print(f"  Conclusão: {c_role}({c_ant} → {c_cons})")
        
        # Deduz
        if parse_ok and c_ant and c_cons:
            result = chainer.deduce(
                initial_facts=[c_ant],
                conclusion=c_cons,
                max_depth=3,
                strict=False,
            )
            pred_label = result.label
            confidence = result.confidence
            steps = len(result.chain)
        else:
            pred_label = 'Unknown'
            confidence = 0.0
            steps = 0
        
        correct = (pred_label == gold_label)
        print(f"  Pred: {pred_label:8} Gold: {gold_label:8} {'✓' if correct else '✗'} (conf={confidence:.3f}, steps={steps})")
        
        results.append({
            'idx': idx,
            'gold': gold_label,
            'pred': pred_label,
            'correct': correct,
            'confidence': confidence,
            'steps': steps,
        })
    
    return results


def report(results):
    print(f"\n{'='*70}")
    print(f"  FOLIO BENCHMARK RESULTS ({len(results)} examples)")
    print(f"{'='*70}")
    
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    
    # Por label
    for label in ['True', 'False', 'Unknown']:
        subset = [r for r in results if r['gold'] == label]
        if subset:
            ok = sum(1 for r in subset if r['correct'])
            print(f"  {label:8}: {ok}/{len(subset)} ({100*ok/len(subset):.0f}%)")
    
    print(f"\n  Overall accuracy: {correct}/{total} ({accuracy:.1%})")
    print(f"  Avg confidence: {np.mean([r['confidence'] for r in results]):.3f}")
    print(f"  Avg deduction steps: {np.mean([r['steps'] for r in results]):.1f}")
    
    return accuracy


if __name__ == '__main__':
    entries = load_folio('folio_data/folio_first10.jsonl')
    print(f"Carregados {len(entries)} exemplos do FOLIO\n")
    
    results = run_folio_benchmark(entries, max_examples=10)
    report(results)
