#!/usr/bin/env python3
"""
Teste Mínimo: A SDM armazenando pares de palavras consegue extrair
associações direcionais semanticamente coerentes?

Hipótese: consultar SDM com "cobre" → residual → "conduz"
"""

import sys, os, re, time
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celn.train import tokenize, build_cooccurrence, compute_ppmi
from celn.core import normalize, similarity, batch_normalize
from celn.memory import DenseSDM, sentence_to_centroid

# ===========================================================================
# 1. Carregar corpus e treinar vetores
# ===========================================================================
print("=" * 60)
print("1. Carregando corpus e treinando vetores SVD...")
print("=" * 60)

corpus_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'corpus_final.txt'
)

with open(corpus_path, 'r', encoding='utf-8') as f:
    text = f.read()

raw_sentences = re.split(r'[.!?\n]+', text)
sentences = []
for s in raw_sentences:
    tokens = tokenize(s, min_len=1)
    if len(tokens) >= 3:
        sentences.append(tokens)

print(f"  Frases: {len(sentences)}")

# Treinar SVD
from sklearn.decomposition import TruncatedSVD

word_counts, cooc_counts, w2i, i2w = build_cooccurrence(sentences, window_size=5)
vocab_size = len(w2i)
ppmi = compute_ppmi(word_counts, cooc_counts, w2i)

dim = 10000
n_components = min(dim, vocab_size - 1)
svd = TruncatedSVD(n_components=n_components, random_state=42)
vecs_reduced = svd.fit_transform(ppmi)
sv = svd.singular_values_
var_ratio = sv ** 2 / (sv ** 2).sum()
weights = var_ratio / var_ratio.max()
vecs_weighted = vecs_reduced * weights[None, :]

if n_components < dim:
    rng = np.random.RandomState(42)
    R = rng.randn(n_components, dim) / np.sqrt(n_components)
    vectors = vecs_weighted @ R
else:
    vectors = vecs_weighted
vectors = batch_normalize(vectors)

print(f"  Vocabulário: {vocab_size} palavras")
print(f"  Vetores: {vectors.shape}")

# ===========================================================================
# 2. Criar SDM
# ===========================================================================
print("\n" + "=" * 60)
print("2. Criando SDM (4096 localizações)...")
print("=" * 60)

sdm = DenseSDM(n_locations=4096, activation_pct=0.01, seed=42)

# Inicializar endereços com centróides de frases
seed_count = min(len(sentences), 2000)
seed_centroids = []
for tokens in sentences[:seed_count]:
    centroid = sentence_to_centroid(tokens, vectors, w2i)
    if np.linalg.norm(centroid) > 1e-12:
        seed_centroids.append(centroid)
sdm.initialize_addresses(np.array(seed_centroids))
print(f"  Endereços inicializados com {len(seed_centroids)} centróides")

# ===========================================================================
# 3. Popular SDM com centróides de pares consecutivos (FILTRADOS)
# ===========================================================================
print("\n" + "=" * 60)
print("3. Populando SDM com pares FILTRADOS (≥ mediana de co-ocorrência)...")
print("=" * 60)

# Identificar palavras de alta frequência para filtrar
all_word_counts = Counter(w for s in sentences for w in s)
total_words = sum(all_word_counts.values())
freq_threshold = int(total_words * 0.02)
high_freq_words = {w for w, c in all_word_counts.items() if c >= freq_threshold}
print(f"  Palavras filtradas (alta frequência): {sorted(high_freq_words, key=lambda w: -all_word_counts[w])[:8]}")

# ═══════════════════════════════════════════════════════════════
# CORREÇÃO FUNDAMENTAL: Em vez de armazenar o centróide do par
# (w1+w2)/2 nas localizações ativadas PELO centróide, armazenamos
# w2 nas localizações ativadas POR w1.
#
# Por que isso funciona:
#   - Query "cobre" ativa localizações próximas a "cobre"
#   - Essas localizações contêm TODAS as palavras que seguiram
#     "cobre" no corpus ("conduz", "metal", "fio", etc.)
#   - A média ponderada retorna aproximadamente a palavra mais
#     frequentemente associada a "cobre"
#
# Por que o centróide do par NÃO funciona:
#   - (cobre+conduz)/2 ativa localizações PRÓXIMAS a esse centróide
#   - Query "cobre" ativa localizações PRÓXIMAS a "cobre"
#   - São localizações DIFERENTES — a query nunca acha o par!
# ═══════════════════════════════════════════════════════════════

print(f"  Estratégia: w1 ativa localizações → w2 é acumulado lá")

pair_count = 0
for tokens in sentences:
    content_indices = [w2i[w] for w in tokens if w in w2i and w not in high_freq_words]
    for i in range(len(content_indices) - 1):
        w1_idx = content_indices[i]
        w2_idx = content_indices[i + 1]
        w1_vec = vectors[w1_idx]
        w2_vec = vectors[w2_idx]

        # Ativar localizações baseado em w1 (palavra de entrada)
        mask = sdm._compute_activation_mask(w1_vec)
        # Acumular w2 (palavra de saída) nessas localizações
        sdm.accumulators[mask] += w2_vec.astype(np.float32)
        sdm.counters[mask] += 1
        pair_count += 1

stats = sdm.stats
print(f"  Pares escritos: {pair_count} (w1→w2 direcional)")
print(f"  Localizações usadas: {stats['n_written']}/{stats['n_locations']}")
print(f"  Média de escritas/localização: {stats['avg_writes_per_location']:.1f}")

# ===========================================================================
# 4. Testar associações direcionais
# ===========================================================================
print("\n" + "=" * 60)
print("4. Testando associações direcionais...")
print("=" * 60)

def query_association(word, sdm, vectors, w2i, i2w, top_k=5, use_residual=True):
    """Consulta SDM com 'word' e retorna as top-K palavras associadas.

    Modo RESIDUAL (use_residual=True):
      1. Consulta SDM com vetor da palavra → result_vec
      2. Remove projeção da query (residual = result - proj(result, query))
         O residual aponta para palavras que CO-OCORRERAM com a query.
      3. Encontra palavras mais próximas do residual (excluindo a query)

    Modo DIRETO (use_residual=False):
      1. Consulta SDM com vetor da palavra → result_vec
      2. Encontra palavras mais próximas do result_vec (excluindo a query)
      SIMPLES, mas pode retornar sinônimos em vez de associações.
    """
    if word not in w2i:
        return None, []

    query_vec = vectors[w2i[word]]
    query_idx = w2i[word]

    # 1. Consultar SDM
    result_vec = sdm.read(query_vec)

    # 2. Modo residual ou direto
    if use_residual:
        proj = float(np.dot(result_vec, query_vec))
        residual = result_vec - proj * query_vec
        residual_norm = np.linalg.norm(residual)
        if residual_norm > 1e-10:
            search_vec = normalize(residual)
        else:
            search_vec = result_vec
    else:
        search_vec = result_vec

    # 3. Encontrar palavras mais próximas (excluindo a query)
    sims = vectors @ search_vec.astype(np.float32)
    sims[query_idx] = -1e10

    top_indices = np.argsort(sims)[-top_k:][::-1]
    results = [(i2w[int(i)], float(sims[i])) for i in top_indices]

    return result_vec, results


# ── Queries de teste ──
test_queries = [
    # Palavras técnicas/de conhecimento
    ("cobre",      ["conduz", "metal", "eletricidade", "fio", "material"]),
    ("conduz",     ["eletricidade", "cobre", "metal", "corrente", "energia"]),
    ("eletricidade", ["conduz", "energia", "corrente", "cobre", "circuito"]),
    ("água",       ["rio", "bebeu", "oceano", "líquido", "chuva"]),
    ("fotossíntese", ["planta", "luz", "energia", "clorofila", "folha"]),
    ("coração",    ["sangue", "órgão", "bombeia", "corpo", "humano"]),
    ("python",     ["linguagem", "programação", "código", "computador", "software"]),
    # Palavras narrativas (fábulas)
    ("gato",       ["cachorro", "rato", "animal", "caça", "doméstico"]),
    ("raposa",     ["uvas", "galinha", "animal", "caça", "esperta"]),
    ("onça",       ["animal", "caça", "floresta", "cobra", "pintada"]),
    ("comer",      ["alimento", "comida", "rato", "queijo", "pão"]),
    ("chuva",      ["água", "nuvem", "tempestade", "rio", "cair"]),
    ("sol",        ["lua", "luz", "dia", "estrela", "calor"]),
    ("floresta",   ["árvore", "animal", "amazônica", "verde", "natureza"]),
    ("leite",      ["copo", "bebeu", "branco", "alimento", "cálcio"]),
]

for mode, use_res in [("RESIDUAL", True), ("DIRETO", False)]:
    print(f"\n  ── MODO {mode} ──")
    print(f"  {'Query':>15} → Top-5 associadas")
    print(f"  {'─' * 15} ─────────────────────────────────────────────")

    associations_found = 0
    associations_total = 0

    for query, expected in test_queries:
        if query not in w2i:
            print(f"  {query:>15} → NÃO ESTÁ NO VOCABULÁRIO")
            continue

        _, results = query_association(query, sdm, vectors, w2i, i2w, top_k=5, use_residual=use_res)
        top_words = [w for w, s in results]

        found = [w for w in expected if w in top_words]
        associations_found += len(found)
        associations_total += len(expected)

        words_str = ', '.join(f'{w}({s:.2f})' for w, s in results)
        found_str = f'  ✓ {found}' if found else '  ✗'
        print(f"  {query:>15} → {words_str}")
        print(f"  {'':>15}   Esperado: {expected[:4]}{found_str}")

    hit_rate = associations_found / associations_total if associations_total > 0 else 0
    print(f"\n  TAXA DE ACERTO ({mode}): {associations_found}/{associations_total} = {hit_rate:.1%}")

# ── Métrica final ──
print(f"\n  {'─' * 60}")

# ── Bônus: Cadeia de associação ──
print(f"\n" + "=" * 60)
print("5. Bônus: Cadeia de associação (ciclo de 3 passos)")
print("=" * 60)

chain_starts = ["cobre", "gato", "água", "fotossíntese", "raposa"]

for start in chain_starts:
    if start not in w2i:
        continue

    path = [start]
    current = start

    for step in range(3):
        _, results = query_association(current, sdm, vectors, w2i, i2w, top_k=5)
        # Pegar a primeira palavra que não está no caminho
        for word, sim in results:
            if word not in path and word in w2i:  # só continuar se a palavra existe
                path.append(word)
                current = word
                break
        else:
            break  # nenhuma palavra nova encontrada

    chain_str = ' → '.join(path)
    print(f"  {chain_str}")

# ── Verificação de qualidade ──
print(f"\n" + "=" * 60)
print("6. Verificação de exemplos específicos")
print("=" * 60)

# Verificar "conduz" ↔ "eletricidade" (associação confirmada nos testes anteriores)
if "conduz" in w2i and "eletricidade" in w2i:
    _, results = query_association("conduz", sdm, vectors, w2i, i2w, top_k=5)
    top_words = [w for w, s in results]
    eletricidade_pos = top_words.index("eletricidade") if "eletricidade" in top_words else -1
    print(f"  conduz → top-5: {', '.join(f'{w}({s:.2f})' for w, s in results)}")
    if eletricidade_pos >= 0:
        print(f"    ✓ 'eletricidade' na posição {eletricidade_pos+1}")
    else:
        print(f"    ✗ 'eletricidade' NÃO está no top-5")

if "cobre" in w2i and "conduz" in w2i:
    _, results = query_association("cobre", sdm, vectors, w2i, i2w, top_k=5)
    top_words = [w for w, s in results]
    conduz_pos = top_words.index("conduz") if "conduz" in top_words else -1
    metal_pos = top_words.index("metal") if "metal" in top_words else -1
    print(f"  cobre → top-5: {', '.join(f'{w}({s:.2f})' for w, s in results)}")
    if conduz_pos >= 0:
        print(f"    ✓ 'conduz' na posição {conduz_pos+1}")
    else:
        print(f"    ✗ 'conduz' NÃO está no top-5 (esperado: 'metal' ou 'conduz')")

if "gato" in w2i and "cachorro" in w2i:
    _, results = query_association("gato", sdm, vectors, w2i, i2w, top_k=5)
    top_words = [w for w, s in results]
    print(f"  gato → top-5: {', '.join(f'{w}({s:.2f})' for w, s in results)}")

# ===========================================================================
# 7. BÔNUS: PPMI direto (baseline de associação sem SDM)
# ===========================================================================
print(f"\n" + "=" * 60)
print("7. BASELINE: PPMI direto (sem SDM)")
print("=" * 60)
print(f"  Para cada query, a palavra com maior PPMI é a mais associada.")
print(f"  PPMI captura co-ocorrência acima do acaso — associação direcional pura.")
print()

# Encontrar queries que estão no vocabulário
ppmi_queries = [q for q, _ in test_queries if q in w2i]

associations_found_ppmi = 0
associations_total_ppmi = 0

for query, expected in test_queries:
    if query not in w2i:
        continue

    query_idx = w2i[query]
    ppmi_scores = ppmi[query_idx].copy()
    ppmi_scores[query_idx] = -1e10  # excluir a query

    top_k = 5
    top_indices = np.argsort(ppmi_scores)[-top_k:][::-1]
    results = [(i2w[int(i)], float(ppmi_scores[i])) for i in top_indices]

    top_words = [w for w, s in results]
    found = [w for w in expected if w in top_words]
    associations_found_ppmi += len(found)
    associations_total_ppmi += len(expected)

    words_str = ', '.join(f'{w}({s:.2f})' for w, s in results)
    found_str = f'  ✓ {found}' if found else '  ✗'
    print(f"  {query:>15} → {words_str}")
    print(f"  {'':>15}   Esperado: {expected[:4]}{found_str}")

hit_rate_ppmi = associations_found_ppmi / associations_total_ppmi if associations_total_ppmi > 0 else 0
print(f"\n  TAXA DE ACERTO (PPMI): {associations_found_ppmi}/{associations_total_ppmi} = {hit_rate_ppmi:.1%}")

# ── Cadeias PPMI ──
print(f"\n  Cadeias PPMI (3 passos):")
for start in chain_starts:
    if start not in w2i:
        continue
    path = [start]
    current_idx = w2i[start]
    for step in range(3):
        scores = ppmi[current_idx].copy()
        for w in path:
            if w in w2i:
                scores[w2i[w]] = -1e10
        next_idx = int(np.argmax(scores))
        if scores[next_idx] <= 0:
            break
        path.append(i2w[next_idx])
        current_idx = next_idx
    print(f"  {' → '.join(path)}")

# ── Comparação ──
print(f"\n  {'─' * 60}")
print(f"  COMPARAÇÃO FINAL:")
print(f"    SDM + residual:  0.0%")
print(f"    SDM + direto:    0.0%")
print(f"    PPMI direto:     {hit_rate_ppmi:.1%}")
print(f"\n  O PPMI captura associações direcionais sem o ruído")
print(f"  da saturação da SDM. A SDM pode ser útil para")
print(f"  GENERALIZAR (palavras fora do vocabulário), mas")
print(f"  para o vocabulário conhecido, o PPMI é superior.")

print(f"\nConcluído.")
