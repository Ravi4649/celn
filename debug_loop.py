#!/usr/bin/env python3
"""Debug the Wikipedia loop."""
import requests
import re
import time

def split_sentences_from_text(text):
    text = re.sub(r'^\s*=+\s*.*?=+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n+', '\n', text)
    pieces = re.split(r'(?<=[.!?])\s+|\n+', text)
    out = []
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        p = p.strip(' "\'')
        if len(p) < 6:
            continue
        out.append(p)
    return out

def clean_candidate(s):
    if re.search(r'<[^>]+>|\{\{|\}\}|\[\[|\|url=|class=|File:|Ficheiro:|\{\||=\s*\n', s):
        return None
    s = re.sub(r'\[\s*note[^\]]*\]', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\[\s*\d+\s*\]', '', s)
    s = re.sub(r'\(verifica-se\)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'&[a-z]+?;', ' ', s)
    if re.search(r'https?://|www\.', s, flags=re.IGNORECASE):
        return None
    if re.search(r'Categoria:|categoria:|Referências|External links', s, re.IGNORECASE):
        return None
    if not re.search(r'[\.\!\?]\"?$|[\.\!\?]\)$', s):
        return None
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.strip(' \n\r')
    return s

session = requests.Session()
session.headers.update({
    'User-Agent': 'celn-v3-auto/1.0 (+https://github.com/ravizin/celn-v3)',
    'Accept-Language': 'pt'
})

out_sentences = []
collected = 0
batch = 100
target_articles = 2500

print(f'Starting loop: target={target_articles}, batch={batch}')

for iteration in range(5):  # First 5 iterations
    take = min(batch, target_articles - collected)
    print(f'Iteration {iteration}: collected={collected}, take={take}')
    
    try:
        r = session.get('https://pt.wikipedia.org/w/api.php', params={
            'action': 'query', 'format': 'json', 'list': 'random', 'rnnamespace': 0, 'rnlimit': take
        }, timeout=10)
        print(f'  Random API status: {r.status_code}')
        
        data = r.json()
        ids = [str(p['id']) for p in data.get('query', {}).get('random', []) if 'id' in p]
        print(f'  Got {len(ids)} article IDs')
        
        if not ids:
            print('  No IDs, breaking')
            break
        
        pages_r = session.get('https://pt.wikipedia.org/w/api.php', params={
            'action': 'query', 'format': 'json', 'prop': 'extracts', 'explaintext': 1, 
            'pageids': '|'.join(ids), 'exlimit': 'max'
        }, timeout=10)
        print(f'  Extracts API status: {pages_r.status_code}')
        
        pages = pages_r.json().get('query', {}).get('pages', {})
        print(f'  Got {len(pages)} pages')
        
        batch_sents = 0
        for pid, info in pages.items():
            text = info.get('extract', '')
            if not text:
                continue
            sents = split_sentences_from_text(text)
            for sent in sents:
                cs = clean_candidate(sent)
                if cs:
                    out_sentences.append(cs)
                    batch_sents += 1
        
        collected += len(ids)
        print(f'  Batch sentences: {batch_sents}, total so far: {len(out_sentences)}')
        time.sleep(0.5)
        
    except Exception as e:
        print(f'  Exception: {e}')
        import traceback
        traceback.print_exc()
        break

print(f'Final: collected={collected}, sentences={len(out_sentences)}')