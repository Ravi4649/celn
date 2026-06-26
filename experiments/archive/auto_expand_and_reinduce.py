#!/usr/bin/env python3
"""Automate corpus expansion and re-induction (PCFG + PairGraph).

Pipeline (automatic, best-effort):
- Load existing corpus_final.txt (avoid duplicates)
- Fetch Portuguese text from: Wikipedia (random articles), OSCAR (HF) and OpenSubtitles (HF)
- Apply automatic, robust filters (token length, punctuation, markup, URLs, language heuristics)
- Merge, deduplicate and save expanded corpus
- Re-induce PCFG and prune it
- Build PairGraph from the expanded corpus

The script is resilient: if a source or dependency fails, it continues with
other sources. It aims for target_total sentences (default 100000) but will
stop earlier if sources are exhausted.

Run: python3 experiments/auto_expand_and_reinduce.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import math
from pathlib import Path
from typing import Iterable, List, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import random

try:
    import requests
except Exception:
    requests = None

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

from celn.train import tokenize, load_corpus

# Reuse some heuristics from clean_corpus if available
try:
    from clean_corpus import portuguese_ratio
except Exception:
    def portuguese_ratio(tokens: Iterable[str]) -> float:
        toks = list(tokens)
        if not toks:
            return 0.0
        pt_count = 0
        for t in toks:
            if re.search(r'[áàãâéêíóôõúüç]', t, re.IGNORECASE):
                pt_count += 1
                continue
            if t.lower() in {'que','não','um','uma','com','para','dos','das','por','mais','como','foi','sua','seu','nos','nas'}:
                pt_count += 1
                continue
            if re.search(r'(ção|ões|dade|mente|imento|amento|ista|ismo|oso|osa|ivo|iva|al)$', t, re.IGNORECASE):
                pt_count += 1
                continue
            if t.lower() not in {'the','and','for','that','with','this','from'} and len(t) >= 3:
                pt_count += 1
                continue
        return pt_count / len(toks)


def normalize_key(s: str) -> str:
    s2 = s.strip().lower()
    s2 = re.sub(r'\s+', ' ', s2)
    return s2


def split_sentences_from_text(text: str) -> List[str]:
    # First, remove section headers (== Title ==) to avoid polluting sentence splitting
    text = re.sub(r'^\s*=+\s*.*?=+\s*$', '', text, flags=re.MULTILINE)
    # Normalize whitespace but preserve sentence boundaries
    text = re.sub(r'\n+', '\n', text)
    # Split on sentence-ending punctuation followed by space or newline
    pieces = re.split(r'(?<=[.!?])\s+|\n+', text)
    out = []
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        # remove leading/trailing leftover quotes or dashes
        p = p.strip(' \"\'')
        if len(p) < 6:
            continue
        out.append(p)
    return out


def clean_candidate(s: str) -> str | None:
    # Reject markup-heavy lines
    if re.search(r'<[^>]+>|\{\{|\}\}|\[\[|\|url=|class=|File:|Ficheiro:|\{\||=\s*\n', s):
        return None
    # Remove reference markers like [1], [nota 2]
    s = re.sub(r'\[\s*note[^\]]*\]', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\[\s*\d+\s*\]', '', s)
    s = re.sub(r'\(verifica-se\)', '', s, flags=re.IGNORECASE)
    # Remove HTML entities
    s = re.sub(r'&[a-z]+?;', ' ', s)
    # No URLs
    if re.search(r'https?://|www\.', s, flags=re.IGNORECASE):
        return None
    # No leftover wiki markers (section headers already removed in split_sentences_from_text)
    if re.search(r'Categoria:|categoria:|Referências|External links', s, re.IGNORECASE):
        return None
    # Must end with punctuation
    if not re.search(r'[\.\!\?]"?$|[\.\!\?]\)$', s):
        return None

    # Tokenize (use train.tokenize) with min_len=1 to preserve short words
    toks = tokenize(s, min_len=1)
    # Auto-calibrated length later; here we return tokens for further decision
    if len(toks) == 0:
        return None
    # Portuguese ratio heuristic
    if portuguese_ratio(toks) < 0.35:
        return None

    # Strange characters ratio: allow mostly printable letters, some punctuation
    nonprint = sum(1 for ch in s if ord(ch) < 32 and ch not in '\n\r\t')
    if nonprint > 0:
        return None

    # Final normalization: collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    # Remove leading/trailing punctuation anomalies
    s = s.strip(' \n\r')
    return s


def fetch_wikipedia_random(target_articles: int = 2500, session=None, timeout=10):
    if requests is None:
        print('requests not available; skipping Wikipedia')
        return []
    s = session or requests.Session()
    # polite headers to avoid simple bot blocking
    s.headers.update({
        'User-Agent': 'celn-v3-auto/1.0 (+https://github.com/ravizin/celn-v3)',
        'Accept-Language': 'pt'
    })

    out_sentences = []
    collected = 0
    # Wikipedia API limit: max 50 pageids per request
    batch = 50
    print(f"[WIKI] Fetching up to {target_articles} random articles (batch={batch})")

    while collected < target_articles:
        take = min(batch, target_articles - collected)
        # request with a few retries on transient errors / JSON decode failures
        max_retries = 3
        success = False
        for attempt in range(max_retries):
            try:
                r = s.get('https://pt.wikipedia.org/w/api.php', params={
                    'action': 'query', 'format': 'json', 'list': 'random', 'rnnamespace': 0, 'rnlimit': take
                }, timeout=timeout)
                if r.status_code != 200:
                    print(f"[WIKI] unexpected status {r.status_code}; attempt {attempt+1}/{max_retries}")
                    time.sleep(0.5 * (attempt + 1))
                    continue
                try:
                    data = r.json()
                except ValueError:
                    # sometimes Wikipedia returns non-json due to throttling; retry
                    print(f"[WIKI] JSON decode failed on random list (attempt {attempt+1}/{max_retries}); resp[:200]={r.text[:200]!r}")
                    time.sleep(0.5 * (attempt + 1))
                    continue

                ids = [str(p['id']) for p in data.get('query', {}).get('random', []) if 'id' in p]
                if not ids:
                    # nothing returned — stop
                    success = True
                    break

                # fetch extracts by pageids (retry on JSON issues)
                pages = {}
                for p_attempt in range(max_retries):
                    pages_r = s.get('https://pt.wikipedia.org/w/api.php', params={
                        'action': 'query', 'format': 'json', 'prop': 'extracts', 'explaintext': 1, 'pageids': '|'.join(ids), 'exlimit': 'max'
                    }, timeout=timeout)
                    if pages_r.status_code != 200:
                        time.sleep(0.5 * (p_attempt + 1))
                        continue
                    try:
                        pages = pages_r.json().get('query', {}).get('pages', {})
                        break
                    except ValueError:
                        print(f"[WIKI] JSON decode failed on extracts (attempt {p_attempt+1}/{max_retries})")
                        time.sleep(0.5 * (p_attempt + 1))
                        continue

                for pid, info in (pages or {}).items():
                    text = info.get('extract', '')
                    if not text:
                        continue
                    sents = split_sentences_from_text(text)
                    for sent in sents:
                        cs = clean_candidate(sent)
                        if cs:
                            out_sentences.append(cs)

                collected += len(ids)
                success = True
                # be a bit polite
                time.sleep(0.5)
                break
            except requests.RequestException as e:
                print(f"[WIKI] Request error: {e}; attempt {attempt+1}/{max_retries}")
                time.sleep(0.5 * (attempt + 1))
                continue

        if not success:
            print("[WIKI] Failed to fetch random articles after retries; stopping")
            break

    print(f"[WIKI] Extracted ~{len(out_sentences)} candidate sentences from random articles")
    return out_sentences


def fetch_hf_wikipedia(n_sentences: int = 40000) -> List[str]:
    if load_dataset is None:
        print('datasets not available; skipping HF Wikipedia')
        return []
    out = []
    try:
        print('[HF-WIKI] Loading Wikimedia Wikipedia PT (streaming)')
        ds = load_dataset('wikimedia/wikipedia', '20231101.pt', split='train', streaming=True)
        for ex in ds:
            text = ex.get('text', '')
            if not text:
                continue
            for s in split_sentences_from_text(text):
                cs = clean_candidate(s)
                if cs:
                    out.append(cs)
                    if len(out) >= n_sentences:
                        print(f'[HF-WIKI] Collected {len(out)} sentences')
                        return out
            if len(out) % 5000 == 0 and len(out) > 0:
                print(f'[HF-WIKI] {len(out)} sentences so far...')
        print(f'[HF-WIKI] Collected {len(out)} sentences (exhausted)')
    except Exception as e:
        print(f'[HF-WIKI] Warning: {e}; skipping')
    return out


def fetch_hf_oscar(n_sentences: int = 30000) -> List[str]:
    if load_dataset is None:
        print('datasets not available; skipping OSCAR')
        return []
    out = []
    oscar_variants = [
        ('oscar', 'unshuffled_deduplicated_pt'),
        ('oscar-corpus/OSCAR-2301', 'pt'),
    ]
    for ds_name, config in oscar_variants:
        try:
            print(f'[OSCAR] Trying {ds_name} config={config} (streaming)')
            ds = load_dataset(ds_name, config, split='train', streaming=True)
            for ex in ds:
                text = ex.get('text') or ex.get('sentence') or ''
                if not text:
                    continue
                for s in split_sentences_from_text(text):
                    cs = clean_candidate(s)
                    if cs:
                        out.append(cs)
                        if len(out) >= n_sentences:
                            print(f'[OSCAR] Collected {len(out)} sentences')
                            return out
            print(f'[OSCAR] Collected {len(out)} sentences (from {ds_name})')
            if out:
                return out
        except Exception as e:
            print(f'[OSCAR] {ds_name} failed: {e}; trying next')
    return out


def fetch_hf_opensubtitles(n_sentences: int = 20000) -> List[str]:
    if load_dataset is None:
        print('datasets not available; skipping OpenSubtitles')
        return []
    out = []
    candidates = [
        ('Helsinki-NLP/opus-100', 'en-pt'),
    ]
    for ds_name, config in candidates:
        try:
            print(f'[SUB] Trying dataset {ds_name} config={config} (streaming)')
            ds = load_dataset(ds_name, config, split='train', streaming=True)
            for ex in ds:
                text = ''
                trans = ex.get('translation', {})
                if isinstance(trans, dict):
                    text = trans.get('pt', '') or trans.get('target', '')
                if not text:
                    text = ex.get('text') or ex.get('sentence') or ''
                if not text:
                    continue
                for s in split_sentences_from_text(text):
                    cs = clean_candidate(s)
                    if cs:
                        out.append(cs)
                        if len(out) >= n_sentences:
                            print(f'[SUB] Collected {len(out)} sentences')
                            return out
            print(f'[SUB] Collected {len(out)} sentences (from {ds_name})')
            if out:
                return out
        except Exception as e:
            print(f'[SUB] Dataset {ds_name} failed: {e}; trying next')
    return out


def auto_calibrate_len_bounds(candidate_lengths: List[int], hard_min=4, hard_max=50):
    if not candidate_lengths:
        return hard_min, hard_max
    arr = sorted(candidate_lengths)
    n = len(arr)
    lo = arr[int(n * 0.02)] if n > 50 else arr[0]
    hi = arr[int(n * 0.98) - 1] if n > 50 else arr[-1]
    lo = max(hard_min, lo)
    hi = min(hard_max, max(lo + 1, hi))
    return lo, hi


def main():
    random.seed(42)
    existing_path = ROOT / 'corpus_expanded.txt'
    if not existing_path.exists():
        print('ERROR: corpus_expanded.txt not found in repo root')
        return

    existing_lines = [l.strip() for l in open(existing_path, 'r', encoding='utf-8') if l.strip()]
    existing_keys = {normalize_key(l) for l in existing_lines}
    n_existing = len(existing_lines)
    print(f'Existing corpus: {n_existing} sentences')

    target_total = 100000
    need = max(0, target_total - n_existing)
    print(f'Target total: {target_total} sentences -> need ~{need} new sentences')

    # Accumulate candidates
    candidates: List[str] = []

    # 1) HF Wikimedia Wikipedia PT (~40k sentences) — most reliable
    try:
        hf_wiki_sents = fetch_hf_wikipedia(40000)
        candidates.extend(hf_wiki_sents)
    except Exception as e:
        print(f'[HF-WIKI] failed: {e}')

    # 2) Wikipedia random articles via API (~3k sentences)
    try:
        wiki_sents = fetch_wikipedia_random(2500)
        candidates.extend(wiki_sents)
    except Exception as e:
        print(f'[WIKI] failed: {e}')

    # 3) OSCAR (~30k sentences)
    try:
        oscar_sents = fetch_hf_oscar(30000)
        candidates.extend(oscar_sents)
    except Exception as e:
        print(f'[OSCAR] failed: {e}')

    # 4) Opus-100 en-pt (~20k sentences)
    try:
        subs_sents = fetch_hf_opensubtitles(20000)
        candidates.extend(subs_sents)
    except Exception as e:
        print(f'[SUBS] failed: {e}')

    print(f'Collected {len(candidates)} raw candidate sentences from sources')

    # Shuffle and deduplicate candidates internally
    random.shuffle(candidates)

    # Auto-calibrate token length bounds from candidate distribution
    lengths = []
    tokenized_candidates = []
    for s in candidates:
        toks = tokenize(s, min_len=1)
        tokenized_candidates.append((s, toks))
        lengths.append(len(toks))

    min_tok, max_tok = auto_calibrate_len_bounds(lengths, hard_min=4, hard_max=50)
    print(f'Auto-calibrated length bounds: {min_tok} .. {max_tok} tokens')

    added = []
    seen = set(existing_keys)
    for s, toks in tokenized_candidates:
        k = normalize_key(s)
        if k in seen:
            continue
        if len(toks) < min_tok or len(toks) > max_tok:
            continue
        # final punctuation check again
        if not re.search(r'[\.\!\?]"?$|[\.\!\?]\)$', s):
            continue
        seen.add(k)
        added.append(s)
        if len(added) >= need:
            break

    total_final = n_existing + len(added)
    print(f'New sentences added: {len(added)}  -> total {total_final}')

    out_path = ROOT / 'corpus_final_expanded.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        # write existing first, then added
        for l in existing_lines:
            f.write(l.strip() + '\n')
        for s in added:
            f.write(s.strip() + '\n')

    print(f'Saved expanded corpus to {out_path} ({os.path.getsize(out_path)} bytes)')

    # Now run PCFG induction on expanded corpus
    print('\n=== PCFG induction (expanded corpus) ===')
    try:
        from celn.pcfg_inducer import PCFGInducer
        inducer = PCFGInducer(verbose=True)
        sentences_tok = load_corpus(str(out_path), min_len=2)
        print(f'Loaded {len(sentences_tok)} tokenized sentences for PCFG induction')
        pcfg = inducer.induce(sentences_tok)
        pcfg_out = ROOT / 'pcfg_induced_expanded.json'
        inducer.save_pcfg(pcfg, str(pcfg_out))
        print(f'PCFG induced and saved to {pcfg_out}')
        # prune
        from celn.pcfg_pruner import prune_pcfg
        pruned = prune_pcfg(pcfg, str(ROOT / 'pcfg_pruned_expanded.json'), verbose=True)
    except Exception as e:
        print(f'PCFG induction failed: {e}')

    # Build PairGraph using the experiments script (best-effort)
    print('\n=== Build PairGraph (expanded) ===')
    try:
        # call the build_pair_graph script to produce pair_graph_expanded.npz
        import subprocess
        script = str(ROOT / 'experiments' / 'build_pair_graph.py')
        out_npz = str(ROOT / 'pair_graph_expanded.npz')
        cmd = [sys.executable, script, str(out_path), out_npz]
        print('Running:', ' '.join(cmd))
        subprocess.run(cmd, check=True)
        print(f'PairGraph saved to {out_npz}')
    except Exception as e:
        print(f'PairGraph build failed: {e}')

    print('\nDone. Summary:')
    print(f'  Original sentences: {n_existing}')
    print(f'  Added sentences:    {len(added)}')
    print(f'  Expanded corpus:    {total_final} sentences')


if __name__ == '__main__':
    main()
