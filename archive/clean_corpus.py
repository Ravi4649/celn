#!/usr/bin/env python3
"""Limpa corpus_pt_expandido.txt removendo contaminação de Wikipedia/markup."""
import re
import unicodedata


def is_portuguese_word(w):
    """Heurística: palavra é reconhecível como português."""
    if len(w) <= 1:
        return False
    # Caracteres isolados ilegítimos (letras soltas sem acento que não são artigos)
    pt_singletons = {'o', 'a', 'e', 'é', 'à', 'ó', 'ú', 'í', 'ê', 'ô', 'ã', 'õ'}
    if len(w) == 1 and w.lower() not in pt_singletons:
        return False
    # Palavras só com consoantes = provavelmente sigla/código
    if len(w) <= 3 and re.match(r'^[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]+$', w):
        return False
    return True


ENGLISH_BLOCKLIST = {
    'wayback', 'handbook', 'encyclopedia', 'encyclopædia',
    'cscr', 'doi', 'http', 'https', 'www', 'publishing',
    'bibcode', 'crossref', 'isbn', 'issn', 'pmid',
    'jstor', 'arxiv', 'hdl', 's2cid', 'pmc',
    'oxford', 'routledge', 'stanford', 'palgrave', 'cambridge',
    'sage', 'wiley', 'springer', 'elsevier', 'columbia',
    'blackwell', 'macmillan', 'kessinger', 'lulu',
    'philosophy', 'political', 'psychology', 'sociology',
    'international', 'encyclopedia', 'handbook',
    'locke', 'hobbes', 'hume', 'kant', 'hegel',
    'review', 'journal', 'press', 'university',
    'accessed', 'retrieved', 'archived', 'original',
    'available', 'online', 'edition', 'volume', 'issue',
    'pp', 'vol', 'ed', 'trans', 'rev',
    'we', 'they', 'their', 'them', 'our', 'your',
    'the', 'and', 'for', 'that', 'with', 'this', 'from',
    'are', 'was', 'were', 'been', 'have', 'has', 'had',
    'not', 'but', 'who', 'which', 'what', 'when', 'where',
    'how', 'all', 'each', 'will', 'would', 'could', 'should',
    'may', 'might', 'shall', 'can',
    'smith', 'johnson', 'williams', 'jones', 'brown',
    'journal', 'quarterly', 'annual', 'monthly',
    'north', 'south', 'east', 'west',
    'pdf', 'html', 'xml', 'svg', 'jpg', 'png',
    'ficheiro', 'arquivo', 'predefinição',
}

ENGLISH_BLOCKLIST_RES = [
    r'^\d{4}$',
    r'^\d+-\d+$',
    r'^\d+\.\d+$',
    r'^pp\.\s*\d',
    r'^vol\.?\s*\d',
    r'^isbn',
    r'^issn',
    r'^doi:\s',
    r'^10\.\d{4}',
    r'^arxiv:',
    r'^\d+$',
]

REF_LINE_PATTERNS = [
    r'^\s*↑',
    r'doi:\s*10\.',
    r'Arquivado em.*Wayback',
    r'Wayback Machine',
    r'[Ee]ncyclopedia [Oo]f',
    r'[Hh]andbook [Oo]f',
    r'Stanford Encyclopedia',
    r'Internet Encyclopedia',
    r'Routledge Encyclopedia',
    r'Oxford Handbook',
    r'Oxford Encyclopedia',
    r'\(em inglês\)',
    r'src="//upload',
    r'resource="\./Ficheiro',
    r'!CS1 maint',
    r'!Páginas com DOI',
    r'acessodata= requer',
    r'Clique aqui para',
    r'Bibcode:',
    r'CrossRef',
    r'pp\.\s*\d',
    r'\|acessodata=',
    r'\|url=',
    r'\|data=',
    r'\|título=',
    r'\|publicação=',
    r'\|volume=',
    r'\|autor=',
    r'\|páginas=',
    r'\|editora=',
    r'\|edição=',
    r'\|local=',
    r'CQ Press\b',
    r'\bMacmillan\b',
    r'\bPalgrave\b',
    r'\bSAGE\b',
    r'ISBN\s*[\d\-]+',
    r'ISSN\s*[\d\-]+',
    r'^\d+\s+\d{4}',
    r'^\s*\d{3,}\s+\d{3,}',
]


def load_corpus(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


def is_reference_line(line):
    for p in REF_LINE_PATTERNS:
        if re.search(p, line, re.IGNORECASE):
            return True
    return False


def tokenize(line):
    tokens = re.findall(r'[a-zA-ZáàãâéêíóôõúüçÁÀÃÂÉÊÍÓÔÕÚÜÇ]+', line)
    return [t.lower() for t in tokens]


def has_excessive_english(tokens, threshold=0.5):
    """Returns True if more than (1-threshold) of tokens are English blocklisted."""
    if not tokens:
        return True
    en_count = 0
    for t in tokens:
        if t in ENGLISH_BLOCKLIST:
            en_count += 1
            continue
        for p in ENGLISH_BLOCKLIST_RES:
            if re.match(p, t, re.IGNORECASE):
                en_count += 1
                break
    ratio = en_count / len(tokens)
    return ratio > (1 - threshold)


def has_illegitimate_singletons(tokens):
    """True if sentence has isolated letters that aren't valid Portuguese words."""
    pt_singletons = {'o', 'a', 'e', 'é', 'à', 'ó', 'ú', 'í', 'ê', 'ô', 'ã', 'õ'}
    non_pt_singletons = [t for t in tokens if len(t) == 1 and t.lower() not in pt_singletons]
    return len(non_pt_singletons) >= 2


def portuguese_ratio(tokens):
    """Fraction of tokens that look like Portuguese words."""
    if not tokens:
        return 0.0
    pt_count = 0
    for t in tokens:
        if len(t) == 1:
            continue
        # Contains accented chars → very likely PT
        if re.search(r'[áàãâéêíóôõúüç]', t, re.IGNORECASE):
            pt_count += 1
            continue
        # Known PT function words
        pt_words = {
            'que', 'não', 'um', 'uma', 'com', 'para', 'dos', 'das',
            'por', 'mais', 'como', 'foi', 'sua', 'seu', 'nos', 'nas',
            'aos', 'pela', 'pelo', 'entre', 'após', 'sobre', 'antes',
            'depois', 'desde', 'até', 'contra', 'sem', 'sob',
            'tem', 'era', 'foram', 'está', 'são', 'ser', 'ter',
            'este', 'esta', 'esse', 'essa', 'aquele', 'aquela',
            'muito', 'pouco', 'todos', 'todas', 'outro', 'outra',
            'onde', 'quando', 'como', 'ainda', 'já', 'só', 'cada',
            'se', 'ou', 'nem', 'logo', 'pois', 'porém', 'mas',
            'sim', 'tal', 'toda', 'todo', 'aquilo', 'isto', 'isso',
            'bem', 'mal', 'aqui', 'ali', 'lá', 'cá',
            'ele', 'ela', 'eles', 'elas', 'nós', 'vocês', 'vós',
            'meu', 'minha', 'teu', 'tua', 'seu', 'sua',
            'nosso', 'nossa', 'seus', 'suas',
            'muito', 'pouco', 'bastante', 'tanto', 'quanto',
            'grande', 'pequeno', 'novo', 'velho', 'bom', 'melhor',
            'primeiro', 'segundo', 'terceiro', 'último',
            'dois', 'três', 'quatro', 'cinco', 'seis', 'sete',
            'oito', 'nove', 'dez', 'cem', 'mil',
        }
        if t.lower() in pt_words:
            pt_count += 1
            continue
        # Ends in PT suffixes
        if re.search(r'(ção|ões|dade|mente|imento|amento|ista|ismo|oso|osa|ivo|iva|al|el|er|ir|ar)$', t, re.IGNORECASE):
            pt_count += 1
            continue
        # Not in English blocklist and reasonable length → give benefit of doubt
        if t.lower() not in ENGLISH_BLOCKLIST and len(t) >= 3:
            pt_count += 1
            continue
    return pt_count / len(tokens)


def main():
    input_path = '/home/ravizin/celn-v3/corpus_pt_expandido.txt'
    output_path = '/home/ravizin/celn-v3/corpus_pt_limpo.txt'

    lines = load_corpus(input_path)
    print(f"Corpus original: {len(lines)} frases")

    kept = []
    removed_reasons = {
        'reference_line': 0,
        'too_short': 0,
        'too_long': 0,
        'excessive_english': 0,
        'illegitimate_singletons': 0,
        'low_pt_ratio': 0,
    }

    key_words = {'conduz', 'cobre', 'eletricidade', 'energia', 'metal',
                 'gato', 'felino', 'brasil', 'brasileiro', 'água',
                 'elétrica', 'corrente', 'condutividade'}

    key_word_found = {w: 0 for w in key_words}

    for line in lines:
        if is_reference_line(line):
            removed_reasons['reference_line'] += 1
            continue

        tokens = tokenize(line)

        if len(tokens) < 3:
            removed_reasons['too_short'] += 1
            continue

        if len(tokens) > 30:
            removed_reasons['too_long'] += 1
            continue

        if has_excessive_english(tokens, threshold=0.5):
            removed_reasons['excessive_english'] += 1
            continue

        if has_illegitimate_singletons(tokens):
            removed_reasons['illegitimate_singletons'] += 1
            continue

        pt_ratio = portuguese_ratio(tokens)
        if pt_ratio < 0.5:
            removed_reasons['low_pt_ratio'] += 1
            continue

        kept.append(line)

        for w in key_words:
            if w in tokens:
                key_word_found[w] += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        for line in kept:
            f.write(line + '\n')

    total_removed = sum(removed_reasons.values())
    print(f"\n{'='*60}")
    print(f"RELATÓRIO DE LIMPEZA")
    print(f"{'='*60}")
    print(f"Frases originais:   {len(lines)}")
    print(f"Frases removidas:   {total_removed}")
    print(f"Frases mantidas:    {len(kept)}")
    print(f"Taxa de remoção:    {total_removed/len(lines)*100:.1f}%")
    print(f"\nMotivos de remoção:")
    for reason, count in sorted(removed_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count} ({count/len(lines)*100:.1f}%)")
    print(f"\nPresença de palavras-chave no corpus limpo:")
    for w, count in sorted(key_word_found.items(), key=lambda x: -x[1]):
        status = "✓" if count > 0 else "✗ AUSENTE"
        print(f"  {w}: {count} ocorrências {status}")
    print(f"\nSalvo em: {output_path}")


if __name__ == '__main__':
    main()
