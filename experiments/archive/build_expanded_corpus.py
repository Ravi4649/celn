#!/usr/bin/env python3
"""Consolidate, deduplicate, and clean the expanded Portuguese corpus."""
import re

SOURCES = [
    '/home/ravizin/sentencas_portugues.txt',
    '/tmp/opencode/sentences_clean.txt',
    '/tmp/pt_sentences.txt',
    '/tmp/opencode/todas_frases.txt',
    '/tmp/opencode/agent6_diverse.txt',
]

ORIGINAL = '/home/ravizin/celn-v3/corpus_final.txt'
OUTPUT = '/home/ravizin/celn-v3/corpus_pt_expandido.txt'

def extract_sentences(text: str) -> list[str]:
    lines = text.split('\n')
    sentences = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = re.split(r'(?<=[.!?])\s+', line)
        for part in parts:
            part = part.strip()
            if len(part) >= 10:
                part = re.sub(r'\s+', ' ', part)
                sentences.append(part)
    return sentences

all_sentences = []

with open(ORIGINAL, 'r', encoding='utf-8') as f:
    original_text = f.read()
orig = extract_sentences(original_text)
all_sentences.extend(orig)
print(f"Original corpus: {len(orig)} sentences")

for path in SOURCES:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        sents = extract_sentences(text)
        all_sentences.extend(sents)
        print(f"{path}: {len(sents)} sentences")
    except FileNotFoundError:
        print(f"WARNING: {path} not found, skipping")

# Deduplicate (case-insensitive)
seen = set()
unique = []
for s in all_sentences:
    key = s.lower().strip()
    if key not in seen:
        seen.add(key)
        unique.append(s)

print(f"\nTotal raw: {len(all_sentences)}")
print(f"Unique:    {len(unique)}")

with open(OUTPUT, 'w', encoding='utf-8') as f:
    for s in unique:
        f.write(s + '\n')

print(f"\nSaved to {OUTPUT}")
print(f"File size: {len(open(OUTPUT).read())} bytes")
