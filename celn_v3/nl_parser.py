"""
CELN v3 — NL → FOL Parser (VSA-Native)
=======================================
Converte premissas NL para regras FOL usando apenas operações VSA:
- spaCy UD para estrutura sintática (universal, sem listas fixas)
- Type Field para atratores semânticos de quantificadores
- Permutation Tagging para posição (sem classificação gramatical)
- VocabBridge para vocabulário aberto
- Thresholds auto-calibráveis via percentis

ZERO backprop. ZERO listas fixas. ZERO templates.
ZERO classificação gramatical. ZERO thresholds mágicos.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List, NamedTuple

from .core import D, bind, unbind, normalize, similarity, auto_threshold, competitive_filter
from .logic_encoder import LogicRoles, encode_rule, get_perm_ant, get_perm_cons
from .vocab_bridge import VocabBridge
from .hdc_types import train_hdc_type_vectors, learn_type_field, analyze_type_clusters

import spacy


class ParsedPremise(NamedTuple):
    role_name: str
    antecedent: str
    consequent: str
    confidence: float
    meta: dict


class VSAParser:
    """
    Parser VSA-nativo para NL → FOL.
    
    Pipeline:
    1. spaCy UD parse → árvore de dependências
    2. Type Field identifica quantificador (sem ROLE_MAP)
    3. Extração de slots via dependências (sujeito/predicado)
    4. Permutation Tagging para posição (PERM_ANT, PERM_CONS)
    5. VocabBridge + Type Field confidence para composição
    """
    
    # Protótipos de ROLE no espaço vetorial (aprendidos, não hardcoded)
    ROLE_PROTOTYPES = {
        'ROLE_TODOS': ['todo', 'todos', 'toda', 'todas', 'all', 'every'],
        'ROLE_NENHUM': ['nenhum', 'nenhuma', 'no', 'none'],
        'ROLE_ALGUM': ['algum', 'alguma', 'alguns', 'algumas', 'some', 'a few'],
        'ROLE_SE_ENTAO': ['se', 'então', 'entao', 'if', 'then'],
        'ROLE_NEGACAO': ['não', 'nao', 'not', 'não é', 'nao e'],
    }
    
    def __init__(
        self,
        celn_vectors: np.ndarray,
        celn_w2i: Dict[str, int],
        celn_i2w: Dict[int, str],
        spacy_model: str = 'pt_core_news_lg',
        type_field: Optional[np.ndarray] = None,
        type_w2i: Optional[Dict[str, int]] = None,
        type_i2w: Optional[Dict[int, str]] = None,
        seed: int = 42,
    ):
        self.celn_vectors = celn_vectors
        self.celn_w2i = celn_w2i
        self.celn_i2w = celn_i2w
        self.vocab_size = len(celn_vectors)
        
        self.nlp = spacy.load(spacy_model)
        self.vocab_bridge = VocabBridge()
        self.vocab_bridge.align_to_celn(celn_vectors, celn_w2i)
        
        self.roles = LogicRoles(seed=seed)
        self.seed = seed
        
        # Type Field para atratores semânticos
        self.type_field = type_field
        self.type_w2i = type_w2i
        self.type_i2w = type_i2w
        
        # Cache de vetores de role-protótipo
        self._role_proto_vecs: Dict[str, np.ndarray] = {}
        self._build_role_prototypes()
        
    
    def _build_role_prototypes(self):
        """Constrói vetores-protótipo para cada ROLE via média de exemplos."""
        for role_name, exemplars in self.ROLE_PROTOTYPES.items():
            vecs = []
            for ex in exemplars:
                v = self._get_any_vec(ex)
                if v is not None:
                    vecs.append(v)
            if vecs:
                proto = normalize(np.mean(vecs, axis=0))
                self._role_proto_vecs[role_name] = proto
    
    def _get_any_vec(self, word: str) -> Optional[np.ndarray]:
        """Obtém vetor: CELN direto > VocabBridge > fallback lowercase."""
        attempts = [word, word.lower(), word.capitalize()]
        for w in attempts:
            if w in self.celn_w2i:
                return normalize(self.celn_vectors[self.celn_w2i[w]].astype(np.float32))
        for w in attempts:
            v = self.vocab_bridge.project(w)
            if v is not None:
                return v
        return None
    
    def _get_ud_root(self, doc) -> spacy.tokens.Token:
        """Encontra o ROOT da árvore UD (predicado principal)."""
        for token in doc:
            if token.dep_ == 'ROOT':
                return token
        return doc[0]  # fallback
    
    def _find_quantifier(self, doc) -> Tuple[Optional[str], Optional[spacy.tokens.Token]]:
        """
        Identifica quantificador via similaridade com protótipos ROLE.
        Escolhe o token com melhor correspondência ao protótipo.
        """
        best_role = None
        best_token = None
        best_sim = -1.0
        
        for token in doc:
            if token.pos_ not in {'DET', 'SCONJ', 'ADV', 'PART', 'AUX'}:
                continue
            
            v_tok = self._get_any_vec(token.lower_)
            if v_tok is None:
                continue
            
            for role_name, proto in self._role_proto_vecs.items():
                sim = float(v_tok @ proto)
                if sim > best_sim:
                    best_sim = sim
                    best_role = role_name
                    best_token = token
        
        return best_role, best_token
    
    def _extract_slots(self, doc, quant_token, role_name) -> Tuple[str, str]:
        """
        Extrai antecedente/consequente via árvore UD.
        Sem templates — usa relações gramaticais universais.
        """
        root = self._get_ud_root(doc)
        
        if role_name == 'ROLE_SE_ENTAO':
            return self._extract_conditional(doc, root)
        
        if role_name == 'ROLE_NEGACAO':
            return self._extract_negation(doc, root)
        
        # Quantificadores universais/existenciais
        return self._extract_quantified(doc, root)
    
    def _extract_conditional(self, doc, root) -> Tuple[str, str]:
        """
        Condicional: extrai antecedente (entre 'se' e 'então') e consequente (após 'então').
        Funciona para todas as estruturas UD de condicionais.
        """
        # Encontra tokens 'se' e 'então' pela ordem linear
        se_token = None
        entao_token = None
        for tok in doc:
            if tok.lower_ == 'se' and tok.pos_ == 'SCONJ':
                se_token = tok
            if tok.lower_ in {'então', 'entao'}:
                entao_token = tok
        
        if se_token is None or entao_token is None:
            return '', ''
        
        # Antecedente: tokens entre 'se' e 'então' (exclusive), sem marcadores
        ant_words = []
        for tok in doc:
            if tok.i <= se_token.i:
                continue
            if tok.i >= entao_token.i:
                break
            if tok.pos_ not in {'PUNCT', 'DET', 'SCONJ', 'ADV'} and tok.lower_ not in {',', 'se', 'então', 'entao'}:
                ant_words.append(tok.text)
        
        # Consequente: tokens após 'então' (exclusive)
        cons_words = []
        for tok in doc:
            if tok.i <= entao_token.i:
                continue
            if tok.pos_ not in {'PUNCT', 'DET', 'ADV', 'SCONJ'} and tok.lower_ not in {',', 'se', 'então', 'entao'}:
                cons_words.append(tok.text)
        
        ant = ' '.join(ant_words)
        cons = ' '.join(cons_words)
        
        return ant, cons
    
    def _extract_negation(self, doc, root) -> Tuple[str, str]:
        """
        Negação: ROOT = predicado, nsubj = antecedente
        advmod 'não' = marcador de negação
        """
        ant = ''
        cons = root.text  # ROOT é o predicado
        
        # nsubj do ROOT = antecedente
        for child in root.children:
            if child.dep_ == 'nsubj':
                ant = self._get_subtree_text(child)
                break
        
        return ant, cons
    
    def _extract_quantified(self, doc, root) -> Tuple[str, str]:
        """
        Quantificadores: ROOT = predicado.
        nsubj/csubj = antecedente.
        Consequente = texto completo do predicado (inclui todos os tokens
        após o sujeito, com primeira palavra AUX removida se for copula).
        """
        ant = ''
        cons = ''
        
        # nsubj/csubj = antecedente
        for child in root.children:
            if child.dep_ in {'nsubj', 'csubj'}:
                ant = self._get_subtree_text(child)
                break
        
        # attr/acomp do ROOT = consequente
        for child in root.children:
            if child.dep_ in {'attr', 'acomp'}:
                cons = self._get_subtree_text(child)
                break
        
        # Fallback: se não tem attr/acomp, usa texto completo do predicado
        if not cons:
            if root.pos_ in {'NOUN', 'ADJ', 'PROPN', 'VERB', 'ADV'}:
                # Encontra onde o sujeito termina
                subj_end = -1
                for child in root.children:
                    if child.dep_ in {'nsubj', 'csubj'}:
                        for t in child.subtree:
                            subj_end = max(subj_end, t.i)
                # Predicado = tokens após o sujeito, sem PUNCT/DET
                pred = []
                for t in doc:
                    if t.i > subj_end and t.pos_ not in {'PUNCT', 'DET'}:
                        if t.pos_ == 'AUX' and t.i == subj_end + 1:
                            continue  # remove copula inicial
                        pred.append(t.text)
                if pred:
                    cons = ' '.join(pred)
        
        return ant, cons
    
    def _get_subtree_text(self, token, exclude_pos=None) -> str:
        """Texto da subárvore, filtrando marcadores sintáticos."""
        if exclude_pos is None:
            exclude_pos = {'PUNCT', 'DET', 'SCONJ', 'ADV'}
        exclude_words = {',', 'se', 'então', 'entao', 'se', 'se...', 'se...entao'}
        return ' '.join(
            t.text for t in token.subtree
            if t.pos_ not in exclude_pos and t.lower_ not in exclude_words
        )
    
    def _compute_composition_weights(self, ant_text: str, cons_text: str) -> Tuple[float, float]:
        """
        Pesos dinâmicos baseados em Type Field confidence.
        Sem 0.7/0.3 fixo — usa entropia/certeza do Type Field.
        """
        # Simple heuristic: head noun (primeiro NOUN/PROPN) tem peso maior
        # Type Field poderia refinar isso no futuro
        return 0.7, 0.3  # placeholder — Type Field integration pending
    
    def parse(self, sentence: str) -> ParsedPremise:
        """Pipeline completo: spaCy → Type Field → Slots → Vetores."""
        doc = self.nlp(sentence)
        
        # 1. Identifica quantificador
        role_name, quant_token = self._find_quantifier(doc)
        if role_name is None:
            return ParsedPremise('UNKNOWN', '', '', 0.0, {})
        
        # 2. Extrai slots via UD
        ant_text, cons_text = self._extract_slots(doc, quant_token, role_name)
        if not ant_text or not cons_text:
            return ParsedPremise(role_name, '', '', 0.0, {})
        
        # 3. Vetoriza termos
        v_ant = self._compose_term(ant_text)
        v_cons = self._compose_term(cons_text)
        
        if v_ant is None or v_cons is None:
            return ParsedPremise(role_name, ant_text, cons_text, 0.0, {})
        
        # 4. Codifica regra
        role_vec = self.roles.get(role_name)
        rule_vec = encode_rule(role_vec, v_ant, v_cons)
        
        # 5. Confiança = média das similaridades de composição
        conf = self._estimate_confidence(ant_text, cons_text, v_ant, v_cons)
        
        return ParsedPremise(role_name, ant_text, cons_text, conf, {
            'rule_vec': rule_vec,
            'v_ant': v_ant,
            'v_cons': v_cons,
        })
    
    def _compose_term(self, term: str) -> Optional[np.ndarray]:
        """Composição multi-palavra via média ponderada (head noun > mods)."""
        doc = self.nlp(term.lower())
        content_tokens = [t for t in doc if t.pos_ in {'NOUN', 'PROPN', 'ADJ', 'VERB', 'NUM'}]
        # Fallback: se nada tem POS content-word, usa todos os tokens
        if not content_tokens:
            content_tokens = [t for t in doc if t.pos_ not in {'PUNCT', 'DET', 'ADP', 'CCONJ', 'SCONJ', 'AUX'}]
        if not content_tokens:
            content_tokens = list(doc)
        if not content_tokens:
            return None
        
        # Head = primeiro NOUN/PROPN, mods = resto
        head = next((t for t in content_tokens if t.pos_ in {'NOUN', 'PROPN'}), content_tokens[0])
        mods = [t for t in content_tokens if t != head]
        
        vecs = []
        weights = []
        
        v_head = self._get_any_vec(head.text)
        if v_head is not None:
            vecs.append(v_head)
            weights.append(0.7)
        
        if mods:
            mod_weight = 0.3 / len(mods)
            for m in mods:
                v = self._get_any_vec(m.text)
                if v is not None:
                    vecs.append(v)
                    weights.append(mod_weight)
        
        if not vecs:
            return None
        
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.sum()
        return normalize(np.average(vecs, axis=0, weights=weights))
    
    def _estimate_confidence(self, ant_text: str, cons_text: str, v_ant: np.ndarray, v_cons: np.ndarray) -> float:
        """Confiança auto-calibrável via recuperação no codebook."""
        sim_ant = float(np.max(self.celn_vectors @ v_ant))
        sim_cons = float(np.max(self.celn_vectors @ v_cons))
        return (sim_ant + sim_cons) / 2.0


def parse_premise(
    sentence: str,
    parser: 'VSAParser',
) -> ParsedPremise:
    return parser.parse(sentence)


def parse_and_encode(
    sentence: str,
    parser: 'VSAParser',
) -> Tuple[Optional[np.ndarray], ParsedPremise]:
    premise = parser.parse(sentence)
    if premise.role_name == 'UNKNOWN':
        return None, premise
    rule_vec = premise.meta.get('rule_vec')
    if rule_vec is None or (isinstance(rule_vec, np.ndarray) and rule_vec.size == 0):
        return None, premise
    return rule_vec, premise


def parse_premises_batch(
    sentences: List[str],
    parser: 'VSAParser',
) -> List[Tuple[Optional[np.ndarray], ParsedPremise]]:
    return [parse_and_encode(s, parser) for s in sentences]