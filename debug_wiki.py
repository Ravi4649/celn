#!/usr/bin/env python3
"""Debug Wikipedia extraction with 20 articles."""
import requests
import re

def split_sentences_from_text(text):
    pieces = re.split(r'(?<=[.!?])\s+|\n+', text)
    out = []
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        p = p.strip(' \"\'')
        if len(p) < 6:
            continue
        out.append(p)
    return out

def clean_candidate(s):
    reasons = []
    
    # Reject markup-heavy lines
    if re.search(r'<[^>]+>|\{\{|\}\}|\[\[|\|url=|class=|File:|Ficheiro:|\{\||=\s*\n', s):
        reasons.append("markup")
    
    # Remove reference markers
    s_clean = re.sub(r'\[\s*note[^\]]*\]', '', s, flags=re.IGNORECASE)
    s_clean = re.sub(r'\[\s*\d+\s*\]', '', s_clean)
    s_clean = re.sub(r'\(verifica-se\)', '', s_clean, flags=re.IGNORECASE)
    s_clean = re.sub(r'&[a-z]+?;', ' ', s_clean)
    
    # No URLs
    if re.search(r'https?://|www\.', s, flags=re.IGNORECASE):
        reasons.append("URL")
    
    # No leftover wiki markers
    if re.search(r'==+ |Categoria:|categoria:|Referências|External links', s, re.IGNORECASE):
        reasons.append("wiki_marker")
    
    # Must end with punctuation
    if not re.search(r'[\.\!\?]"?$|[\.\!\?]\)$', s):
        reasons.append("no_final_punct")
    
    if reasons:
        return None, reasons
    
    # Final normalization
    s_final = re.sub(r'\s+', ' ', s_clean).strip()
    s_final = s_final.strip(' \n\r')
    return s_final, []

def test_wikipedia(num_articles=20):
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'celn-v3-auto/1.0 (+https://github.com/ravizin/celn-v3)',
        'Accept-Language': 'pt'
    })
    
    print(f"Fetching {num_articles} random articles...")
    
    # Get random article IDs
    r = session.get('https://pt.wikipedia.org/w/api.php', params={
        'action': 'query', 'format': 'json', 'list': 'random', 'rnnamespace': 0, 'rnlimit': num_articles
    }, timeout=10)
    
    print(f"Status: {r.status_code}")
    
    try:
        data = r.json()
        random_ids = [str(p['id']) for p in data.get('query', {}).get('random', []) if 'id' in p]
        print(f"Got {len(random_ids)} article IDs: {random_ids[:5]}...")
    except Exception as e:
        print(f"JSON parse error: {e}")
        print(f"Response: {r.text[:500]}")
        return
    
    # Fetch extracts
    pages_r = session.get('https://pt.wikipedia.org/w/api.php', params={
        'action': 'query', 'format': 'json', 'prop': 'extracts', 'explaintext': 1, 
        'pageids': '|'.join(random_ids), 'exlimit': 'max'
    }, timeout=10)
    
    print(f"Extracts status: {pages_r.status_code}")
    
    try:
        pages_data = pages_r.json()
        pages = pages_data.get('query', {}).get('pages', {})
        print(f"Got {len(pages)} pages")
    except Exception as e:
        print(f"JSON parse error on extracts: {e}")
        print(f"Response: {pages_r.text[:500]}")
        return
    
    # Analyze extraction
    total_raw_sentences = 0
    passed_split = 0
    passed_clean = 0
    reject_reasons = {}
    
    for pid, info in pages.items():
        text = info.get('extract', '')
        if not text:
            continue
        
        # Show sample of raw text
        print(f"\n--- Page {pid}: {info.get('title', 'Unknown')[:50]} ---")
        print(f"Raw text preview: {text[:200]}...")
        
        raw_sents = split_sentences_from_text(text)
        total_raw_sentences += len(raw_sents)
        
        print(f"Split into {len(raw_sents)} sentences")
        
        # Show first 5 raw sentences
        for i, sent in enumerate(raw_sents[:5]):
            print(f"  Raw [{i}]: {sent[:100]}")
        
        for sent in raw_sents:
            passed_split += 1
            cleaned, reasons = clean_candidate(sent)
            if cleaned:
                passed_clean += 1
                if passed_clean <= 3:
                    print(f"  ✓ CLEAN: {cleaned[:80]}")
            else:
                for reason in reasons:
                    reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
    
    print(f"\n=== SUMMARY ===")
    print(f"Total raw sentences from splitter: {total_raw_sentences}")
    print(f"Passed split stage: {passed_split}")
    print(f"Passed clean stage: {passed_clean}")
    print(f"Rejection reasons: {reject_reasons}")
    
    # Test specific regex patterns
    print(f"\n=== REGEX TESTS ===")
    test_strings = [
        "O Brasil é um país.",
        "== Referências ==",
        "Veja mais em [1].",
        "Isso é importante.",
        "Categoria:Biografias",
    ]
    
    for test in test_strings:
        has_punct = re.search(r'[\.\!\?]"?$|[\.\!\?]\)$', test)
        has_wiki = re.search(r'==+ |Categoria:|categoria:|Referências|External links', test, re.IGNORECASE)
        print(f"  '{test[:40]}': punct={bool(has_punct)}, wiki={bool(has_wiki)}")

if __name__ == '__main__':
    test_wikipedia(20)