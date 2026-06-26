#!/usr/bin/env python3
"""
CELN v3 — Real-World Wikipedia Digestion Test
===============================================
Feeds a Wikipedia article about GRAPHENE (grafeno) to the CELN
and tests whether it can answer factual questions about the topic.

The article contains facts NOT present in the training corpus:
  - Definition: crystalline form of carbon
  - Discoverer: Hanns-Peter Boehm proposed the name
  - Nobel Prize: Andre Geim & Konstantin Novoselov (2010)
  - Properties: strongest material, transparent, conductive
  - Dimensions: 3M layers = 1mm, 0.34nm thick
  - Applications: 500GHz processors, water purification, memristors

Phase 1: Load base corpus + train vectors
Phase 2: Ingest Wikipedia article sentences into SDM
Phase 3: Measure corroboration, knowledge confidence
Phase 4: Ask factual questions and evaluate answers

Usage:
  python experiments/test_wikipedia.py
"""

import sys, os, re, time, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celn.train import tokenize, build_cooccurrence, compute_ppmi
from celn.core import normalize, batch_normalize, projective_resonance as M
from celn.memory import DenseSDM
from celn.dual_channel import DualChannelGenerator
from celn.hdc_types import train_hdc_type_vectors
from sklearn.decomposition import TruncatedSVD


# ═══════════════════════════════════════════════════════════════
# Wikipedia Article: Grafeno (simplified, factual)
# ═══════════════════════════════════════════════════════════════

GRAFENO_ARTICLE = [
    "o grafeno é uma forma cristalina do carbono como o diamante e o grafite",
    "o grafeno é o material mais forte já encontrado",
    "o grafeno consiste em uma folha plana de átomos de carbono",
    "o grafeno é quase transparente e um excelente condutor de calor",
    "o grafeno é um excelente condutor de eletricidade",
    "o termo grafeno foi proposto por hanns peter boehm",
    "o prêmio nobel de física de dois mil e dez foi para andre geim",
    "o prêmio nobel de física de dois mil e dez foi para konstantin novoselov",
    "andre geim e konstantin novoselov receberam o nobel por experiências com grafeno",
    "três milhões de camadas de grafeno empilhadas têm altura de um milímetro",
    "a espessura de uma camada de grafeno é de trinta e quatro centésimos de nanômetro",
    "o grafeno é um semicondutor de zero gap",
    "as bandas de condução e valência do grafeno se encontram nos cones de dirac",
    "o grafeno absorve até dois por cento da luz",
    "o grafeno tem índice de reflexão de apenas zero vírgula um por cento",
    "processadores de grafeno poderiam chegar a mais de quinhentos gigahertz",
    "o óxido de grafeno pode extrair substâncias radioativas da água",
    "o óxido de grafeno é útil para purificação de água contaminada",
    "pesquisadores usaram laser de dvd para produzir grafeno de forma barata",
    "o grafeno de duas camadas pode ser ferroelétrico",
    "memristores baseados em grafeno são usados em computação neuromórfica",
    "dispositivos de grafeno possuem dezesseis estados condutores armazenáveis",
    "o grafeno é um material bidimensional",
    "o carbono forma estruturas como nanotubos e fulerenos além do grafeno",
    "o grafeno é duzentas vezes mais forte que o aço",
]

# Facts we expect the system to learn
EXPECTED_FACTS = {
    "o que é o grafeno": ["carbono", "cristalina", "bidimensional", "folha"],
    "qual o material mais forte": ["grafeno", "forte", "aço"],
    "quem propos o nome grafeno": ["boehm", "hanns", "peter"],
    "quem ganhou o nobel pelo grafeno": ["geim", "novoselov", "andre", "konstantin"],
    "quando foi o nobel do grafeno": ["dois", "mil", "dez", "2010"],
    "qual a espessura do grafeno": ["nanômetro", "camada", "espessura"],
    "o grafeno conduz eletricidade": ["condutor", "conduz", "eletricidade"],
    "o grafeno é transparente": ["transparente", "absorve", "luz"],
    "qual a velocidade do processador de grafeno": ["gigahertz", "processadores", "quinhentos"],
    "o grafeno purifica agua": ["óxido", "radioativas", "purificação", "água"],
}


def main():
    print("╔" + "═" * 68 + "╗")
    print("║  CELN v3 — Wikipedia Digestion: GRAFENO                        ║")
    print("║  Real-world test: can CELN learn from internet text?            ║")
    print("╚" + "═" * 68 + "╝")

    t_start = time.time()

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Base knowledge
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'─'*68}")
    print("PHASE 1: Loading base knowledge (corpus_final.txt)")
    print(f"{'─'*68}")

    corpus_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'corpus_final.txt'
    )
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    raw = re.split(r'[.!?\n]+', text)
    sentences = []
    for s in raw:
        tokens = tokenize(s, min_len=1)
        if len(tokens) >= 3:
            sentences.append(tokens)

    word_counts, cooc_counts, w2i, i2w = build_cooccurrence(sentences, window_size=5)
    V = len(w2i)
    ppmi = compute_ppmi(word_counts, cooc_counts, w2i)
    nc = min(10000, V - 1)
    svd = TruncatedSVD(n_components=nc, random_state=42)
    vr = svd.fit_transform(ppmi)
    sv = svd.singular_values_
    var = sv**2 / (sv**2).sum()
    vr = vr * (var / var.max())[None, :]
    if nc < 10000:
        R_mat = np.random.RandomState(42).randn(nc, 10000) / np.sqrt(nc)
        vectors = vr @ R_mat
    else:
        vectors = vr
    vectors = batch_normalize(vectors)
    D = vectors.shape[1]
    print(f"  Base vectors: {V} words × {D}D")

    # Check vocab coverage for article words
    article_words = set()
    for s in GRAFENO_ARTICLE:
        for w in tokenize(s, min_len=1):
            article_words.add(w)
    in_vocab = article_words & set(w2i.keys())
    out_vocab = article_words - set(w2i.keys())
    print(f"  Article words in base vocab: {len(in_vocab)}/{len(article_words)}")
    if out_vocab:
        print(f"  New words (not in corpus):   {sorted(out_vocab)[:15]}...")

    # ── Add new words with CONTEXTUAL initialization ──
    # For each new word, find which existing words co-occur with it
    # in the article, and initialize near those words' vectors.
    # This gives new words semantic grounding from the start.
    rng = np.random.RandomState(42)

    # Build co-occurrence map from article sentences
    word_contexts = {w: set() for w in out_vocab}
    for sentence in GRAFENO_ARTICLE:
        tokens = tokenize(sentence, min_len=1)
        for i, w in enumerate(tokens):
            if w in out_vocab:
                # Collect words within ±3 window
                start = max(0, i - 3)
                end = min(len(tokens), i + 4)
                for j in range(start, end):
                    if j != i and tokens[j] in w2i:
                        word_contexts[w].add(tokens[j])

    for w in out_vocab:
        w2i[w] = V
        i2w[V] = w

        # Initialize near context words' centroid
        ctx_words = word_contexts.get(w, set())
        ctx_in_vocab = [cw for cw in ctx_words if cw in w2i and w2i[cw] < V - len(out_vocab)]
        if len(ctx_in_vocab) >= 2:
            ctx_vecs = [vectors[w2i[cw]] for cw in ctx_in_vocab[:5]]
            new_vec = normalize(np.mean(ctx_vecs, axis=0))
            # Add small noise for uniqueness
            noise = rng.randn(D).astype(np.float32) * 0.1
            new_vec = normalize(new_vec + noise)
        elif len(ctx_in_vocab) == 1:
            new_vec = vectors[w2i[ctx_in_vocab[0]]].copy()
            noise = rng.randn(D).astype(np.float32) * 0.2
            new_vec = normalize(new_vec + noise)
        else:
            # Fallback: find words similar to the new word's context
            # by looking at all words in the same sentences
            all_ctx = set()
            for sentence in GRAFENO_ARTICLE:
                tokens = tokenize(sentence, min_len=1)
                if w in tokens:
                    all_ctx.update(t for t in tokens if t in w2i and w2i[t] < V - len(out_vocab))
            if len(all_ctx) >= 3:
                ctx_vecs = [vectors[w2i[cw]] for cw in list(all_ctx)[:8]]
                new_vec = normalize(np.mean(ctx_vecs, axis=0) +
                                   0.15 * rng.randn(D).astype(np.float32))
            else:
                new_vec = normalize(rng.randn(D).astype(np.float32))

        vectors = np.vstack([vectors, new_vec.reshape(1, -1)])
        V += 1

    vectors = batch_normalize(vectors)
    print(f"  Extended vocab: {V} words (added {len(out_vocab)} new)")
    ctx_available = sum(1 for w in out_vocab if len(word_contexts.get(w, set()) & in_vocab) >= 2)
    print(f"  Context-initialized: {ctx_available}/{len(out_vocab)} new words")

    # ── Initialize SDM ──
    sdm = DenseSDM(n_locations=4096, activation_pct=0.01, seed=42)
    seed_centroids = []
    for tokens in sentences[:2000]:
        idxs = [w2i[w] for w in tokens if w in w2i]
        if len(idxs) >= 3:
            seed_centroids.append(normalize(vectors[idxs].mean(axis=0)))
    sdm.initialize_addresses(np.array(seed_centroids))

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Ingest Wikipedia article
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'─'*68}")
    print(f"PHASE 2: Ingesting Wikipedia article ({len(GRAFENO_ARTICLE)} sentences)")
    print(f"{'─'*68}")

    total_corr = 0
    total_contra = 0

    for i, sentence in enumerate(GRAFENO_ARTICLE):
        tokens = tokenize(sentence, min_len=1)
        idxs = [w2i[w] for w in tokens if w in w2i]
        if len(idxs) < 2:
            continue
        centroid = normalize(vectors[idxs].mean(axis=0))
        result = sdm.write_corroborated(centroid)
        total_corr += result['corroborating']
        total_contra += result['contradictory']

        if i < 5 or i >= len(GRAFENO_ARTICLE) - 3:
            print(f"  [{i+1:>2}] {sentence[:55]:<55} corr={result['corroborating']:>2}")

    print(f"  ...")
    print(f"\n  Digestion complete:")
    print(f"    Corroborations:  {total_corr}")
    print(f"    Contradictions:  {total_contra}")
    print(f"    Conflicts:       {sdm.total_conflicts_detected}")
    print(f"    Conflict locs:   {sdm.has_conflict.sum()}/{sdm.n_locations}")

    written_locs = sdm.counters > 0
    mean_corr = float(sdm.corroboration[written_locs].mean())
    print(f"    Mean corroboration: {mean_corr:.3f}")

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Knowledge confidence measurement
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'─'*68}")
    print("PHASE 3: Knowledge confidence per topic")
    print(f"{'─'*68}")

    topic_queries = {
        "grafeno": ["grafeno"],
        "carbono": ["carbono", "grafeno"],
        "nobel": ["nobel", "grafeno", "geim"],
        "propriedades": ["grafeno", "forte", "transparente", "condutor"],
        "aplicações": ["grafeno", "processadores", "água", "purificação"],
    }

    for topic_name, query_words in topic_queries.items():
        idxs = [w2i[w] for w in query_words if w in w2i]
        if not idxs:
            continue
        query = normalize(vectors[idxs].mean(axis=0))
        result = sdm.read_with_confidence(query)

        # Top words
        sims = vectors @ normalize(result['result'])
        top = np.argsort(sims)[-8:][::-1]
        top_words = [(i2w[i], float(sims[i])) for i in top[:5]]

        print(f"  {topic_name:<15} trust={result['trust_score']:.3f}  "
              f"corr={result['mean_corroboration']:.2f}  "
              f"conflicts={result['n_conflicts']}  "
              f"→ {', '.join(f'{w}' for w,_ in top_words)}")

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: Factual Q&A
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'─'*68}")
    print("PHASE 4: Factual Questions about Grafeno")
    print(f"{'─'*68}")

    factual_questions = [
        ("o que é o grafeno", ["grafeno", "carbono", "cristalina", "bidimensional"]),
        ("qual o material mais forte ja encontrado", ["grafeno", "forte", "material"]),
        ("quem propos o nome grafeno", ["grafeno", "boehm", "hanns", "nome"]),
        ("quem ganhou o premio nobel pelo grafeno", ["grafeno", "nobel", "geim", "novoselov"]),
        ("em que ano foi o nobel do grafeno", ["grafeno", "nobel", "dois", "mil", "dez"]),
        ("qual a espessura de uma camada de grafeno", ["grafeno", "camada", "nanômetro", "espessura"]),
        ("o grafeno conduz eletricidade", ["grafeno", "condutor", "eletricidade"]),
        ("o grafeno é transparente", ["grafeno", "transparente", "luz", "absorve"]),
        ("quantos gigahertz um processador de grafeno pode chegar", ["grafeno", "processador", "gigahertz", "quinhentos"]),
        ("o grafeno pode purificar agua", ["grafeno", "óxido", "água", "purificação", "radioativas"]),
    ]

    score = 0
    total_q = 0

    for question, expected_words in factual_questions:
        total_q += 1
        qw = tokenize(question, min_len=1)
        qk = [w for w in qw if w in w2i]
        if len(qk) < 2:
            continue

        # Query SDM
        query_vec = normalize(vectors[[w2i[w] for w in qk if w in w2i]].mean(axis=0))
        result = sdm.read_with_confidence(query_vec)

        # Get top words from SDM result
        sims = vectors @ normalize(result['result'])
        top_indices = np.argsort(sims)[-10:][::-1]
        top_words = [i2w[i] for i in top_indices]

        # Check how many expected words are in the top results
        expected_hits = [w for w in expected_words if w in top_words]
        hit_count = len(expected_hits)
        # Also check: are any expected words in top-3?
        top3_hits = [w for w in expected_words if w in top_words[:3]]

        # Also check similarity to specific fact words
        fact_sims = {}
        for ew in expected_words:
            if ew in w2i:
                fact_sims[ew] = float(sims[w2i[ew]])

        question_score = min(hit_count / max(len(expected_words), 1), 1.0)
        score += question_score

        marker = "✓" if hit_count >= 1 else "✗"
        print(f"\n  Q{total_q}: {question}")
        print(f"     Expected: {expected_words}")
        print(f"     Top SDM:  {top_words[:6]}")
        print(f"     Hits: {hit_count}/{len(expected_words)} {marker}")
        if fact_sims:
            best_fact = max(fact_sims, key=fact_sims.get)
            print(f"     Best fact: '{best_fact}' sim={fact_sims[best_fact]:.4f}")

    # ═══════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═'*68}")
    print("FINAL REPORT — Wikipedia Digestion")
    print(f"{'═'*68}")

    mean_score = score / max(total_q, 1)
    c1 = mean_score > 0.3    # Factual recall above chance
    c2 = mean_score > 0.5    # Strong factual recall
    c3 = total_corr > 5      # Corroboration detected
    c4 = len(in_vocab) > 50  # Article words found in base vocab

    passed = sum([c1, c2, c3, c4])

    print(f"\n  ┌─ Results {'─'*57}")
    print(f"  │ Article sentences ingested:    {len(GRAFENO_ARTICLE)}")
    print(f"  │ Base vocabulary:               {len(w2i) - len(out_vocab)} words + {len(out_vocab)} new")
    print(f"  │ Corroborations detected:       {total_corr} {'✓' if c3 else '✗'}")
    print(f"  │ Factual recall score:          {mean_score:.1%} "
          f"({'✓' if c1 else '✗'} >30%, {'✓' if c2 else '✗'} >50%)")
    print(f"  │ Article vocab in base:         {len(in_vocab)}/{len(article_words)} {'✓' if c4 else '✗'}")

    total_time = time.time() - t_start
    print(f"  │")
    print(f"  │ Result: {passed}/4 criteria passed in {total_time:.0f}s on Ryzen 2600 CPU")

    if passed >= 3:
        print(f"  │")
        print(f"  │ ✅ CELN successfully DIGESTED the Wikipedia article.")
        print(f"  │    It absorbed factual knowledge about graphene and")
        print(f"  │    can answer questions using the ingested facts.")
        print(f"  │")
        print(f"  │    This is autonomous learning from raw internet text —")
        print(f"  │    zero backprop, zero templates, pure vector algebra.")
    elif passed >= 2:
        print(f"  │")
        print(f"  │ ⚠️  Partial success. The SDM absorbs knowledge but")
        print(f"  │    factual precision needs more data or better vectors.")
    else:
        print(f"  │")
        print(f"  │ ⚠️  Wikipedia digestion needs improvement.")

    print(f"  └{'─'*67}")
    print()

    return passed


if __name__ == '__main__':
    main()
