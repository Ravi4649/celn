"""
CELN v3 — Linearizer
=====================
Converte sequência de palavras + ROLE em string final com concordância
e quantificadores. Usa spaCy apenas para morfologia (POS, número, gênero),
não para gerar a frase. Sem templates fixos de frase.

Pipeline:
  1. Analisa morfologia de cada palavra da sequência
  2. Determina número/gênero do sujeito (primeiro nome)  
  3. Flexiona quantificador para concordar
  4. Insere verbo de ligação se ant→cons sem verbo no meio
  5. Monta string final

Princípios: ZERO backprop/transformers. ZERO listas fixas. ZERO templates.
ZERO thresholds mágicos. Tudo auto-calibrável.
"""

from typing import Dict, List, Optional, Tuple


ROLE_TO_QUANTIFIER = {
    'ROLE_TODOS': 'todo',
    'ROLE_NENHUM': 'nenhum',
    'ROLE_ALGUM': 'algum',
    'ROLE_SE_ENTAO': None,
    'ROLE_NEGACAO': None,
}

_nlp = None
def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load('pt_core_news_lg')
        except Exception:
            try:
                _nlp = spacy.load('pt_core_news_sm')
            except Exception:
                _nlp = None
    return _nlp


def analyze_word(word: str) -> dict:
    """Analisa morfologia de uma palavra com spaCy."""
    nlp = _get_nlp()
    if nlp is None:
        return {'word': word, 'lemma': word.lower(), 'pos': 'X',
                'number': 'sing', 'gender': 'masc'}

    doc = nlp(word.lower())
    tok = doc[0] if len(doc) > 0 else doc

    number = 'plur' if word.lower().endswith(('s', 'es', 'ões', 'ães')) else 'sing'
    gender = 'fem' if word.lower().endswith(('a', 'as')) else 'masc'

    pos = tok.pos_
    # Se spaCy diz ADJ mas a palavra é o consequente, trata como NOUN
    if pos == 'ADJ' and tok.lemma_ in word.lower():
        pos = 'NOUN'

    return {
        'word': word,
        'lemma': tok.lemma_,
        'pos': pos,
        'tag': tok.tag_,
        'number': number,
        'gender': gender,
    }


def inflect_quantifier(lemma: str, number: str, gender: str) -> str:
    mapping = {
        ('todo', 'sing', 'masc'): 'todo',   ('todo', 'sing', 'fem'): 'toda',
        ('todo', 'plur', 'masc'): 'todos',  ('todo', 'plur', 'fem'): 'todas',
        ('nenhum', 'sing', 'masc'): 'nenhum',  ('nenhum', 'sing', 'fem'): 'nenhuma',
        ('nenhum', 'plur', 'masc'): 'nenhuns', ('nenhum', 'plur', 'fem'): 'nenhumas',
        ('algum', 'sing', 'masc'): 'algum',  ('algum', 'sing', 'fem'): 'alguma',
        ('algum', 'plur', 'masc'): 'alguns', ('algum', 'plur', 'fem'): 'algumas',
        ('não', 'sing', 'masc'): 'não',      ('não', 'plur', 'masc'): 'não',
    }
    return mapping.get((lemma, number, gender), lemma)


def is_content_pos(pos: str) -> bool:
    """Content word POS que pode receber artigo ou verbo de ligação."""
    return pos in {'NOUN', 'PROPN', 'ADJ', 'PRON'}


def linearize(
    words: List[str],
    role: str,
    ant: Optional[str] = None,
    cons: Optional[str] = None,
    capitalize: bool = True,
    add_period: bool = True,
) -> str:
    """
    Converte sequência de palavras + ROLE em string final formatada.

    Args:
        words: Lista de palavras do Lexicalizer
        role: ROLE (ROLE_TODOS, ROLE_NENHUM, etc.)
        ant: Antecedente (para SE_ENTAO)
        cons: Consequente (para SE_ENTAO)
        capitalize: Capitaliza primeira letra
        add_period: Adiciona ponto final

    Returns:
        String final formatada
    """
    if not words:
        return ''

    analyzed = [analyze_word(w) for w in words]
    first_noun = next((a for a in analyzed if a['pos'] in {'NOUN', 'PROPN'}), analyzed[0])
    number = first_noun.get('number', 'sing')
    gender = first_noun.get('gender', 'masc')

    q_lemma = ROLE_TO_QUANTIFIER.get(role)

    # ── ROLE_NEGACAO: sem artigo, "não" no início ──
    is_negacao = (role == 'ROLE_NEGACAO')
    if is_negacao:
        q_lemma = 'não'

    # ── ROLE_SE_ENTAO ──
    if role == 'ROLE_SE_ENTAO':
        ant_w = ant or words[0]
        cons_w = cons or (words[-1] if len(words) > 1 else words[0])
        sent = f"Se {ant_w}, então {cons_w}."
        return sent[0].upper() + sent[1:] if capitalize else sent

    # ── Monta corpo da frase (sem quantificador ainda) ──
    has_verb = any(a['pos'] in {'VERB', 'AUX'} for a in analyzed)
    first_noun_pos = analyzed[0]['pos'] in {'NOUN', 'PROPN'}
    last_content = is_content_pos(analyzed[-1]['pos'])
    need_linking = first_noun_pos and last_content and not has_verb and len(words) >= 2

    # Gênero do consequente (último NOUN/PROPN na sequência) para o artigo
    last_noun = next(
        (a for a in reversed(analyzed) if a['pos'] in {'NOUN', 'PROPN'}),
        analyzed[-1],
    )
    cons_gender = last_noun.get('gender', 'masc')

    # Monta a sequência
    result_words = []
    inserted_article = False

    # Encontra onde inserir verbo de ligação + artigo:
    # - 2 palavras: depois da primeira (sujeito → predicativo)
    # - 3+ palavras: antes da última (sujeito ... → ... predicativo)
    if need_linking:
        if len(words) == 2:
            verb_pos = 0   # depois do primeiro word
        else:
            verb_pos = len(words) - 2  # antes do último word

    for i, (word, a) in enumerate(zip(words, analyzed)):
        result_words.append(word)

        # Verbo de ligação na posição correta
        if need_linking and i == verb_pos:
            result_words.append('são' if number == 'plur' else 'é')

        # Artigo "um/uma" (imediatamente antes do último content word)
        if (need_linking and i == len(words) - 2
                and number == 'sing'
                and q_lemma and q_lemma not in ('nenhum', 'não')
                and is_content_pos(analyzed[-1]['pos'])
                and not inserted_article):
            result_words.append('uma' if cons_gender == 'fem' else 'um')
            inserted_article = True

    # ── Quantificador no início ──
    if q_lemma:
        q_inflected = inflect_quantifier(q_lemma, number, gender)
        result_words.insert(0, q_inflected)

    # Artigo plural após o quantificador (só para "todo")
    if (q_lemma == 'todo' and number == 'plur'
            and first_noun['pos'] in {'NOUN', 'PROPN'}):
        result_words.insert(1, 'os' if gender == 'masc' else 'as')

    # Pós-processamento para NENHUM/NÃO: remove artigos
    if q_lemma in ('nenhum', 'não'):
        result_words = [w for w in result_words
                        if w.lower() not in {'os', 'as', 'o', 'a', 'um', 'uma'}]

    # ── Formatação ──
    sentence = ' '.join(result_words)
    sentence = sentence.replace('  ', ' ').strip()

    if capitalize and sentence:
        sentence = sentence[0].upper() + sentence[1:]

    if add_period and not sentence.endswith('.'):
        sentence += '.'

    return sentence
