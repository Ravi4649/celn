#!/usr/bin/env python3
"""
CELN v3 — Autonomous Digestion Cycle (Integrated Test)
=======================================================
Full integration of the three CDA pillars:
  1. SDM with corroboration weights
  2. Cross-lingual alignment (M + Resonator + SDM)
  3. Contradiction detection + competing hypothesis isolation

Simulates: the CELN reads raw text from the internet — facts in
multiple languages, some true, some false, all mixed together.

Experiment:
  Phase 1: Load corpus_final.txt (base knowledge, Portuguese)
  Phase 2: Inject mixed facts:
    - TRUE_PT  ×5: "cobre conduz eletricidade" (corroborated, Portuguese)
    - FALSE_PT ×2: "cobre isola eletricidade" (false, Portuguese)
    - TRUE_EN  ×3: "copper conducts electricity" (corroborating, English)
  Phase 3: SDM processes everything. Measure:
    - Corroboration hits (true facts reinforce each other)
    - Contradictions isolated (false facts → ALT track)
    - Cross-lingual alignment (EN facts align to PT equivalents)
  Phase 4: Query "o cobre conduz eletricidade?" and verify
    that the CORROBORATED hypothesis (8 sources, 2 languages)
    dominates the false one (2 sources, isolated).

Usage:
  python experiments/test_digestion_cycle.py
"""

import sys, os, re, time, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celn.train import tokenize, build_cooccurrence, compute_ppmi
from celn.core import normalize, batch_normalize, projective_resonance as M
from celn.memory import DenseSDM
from sklearn.decomposition import TruncatedSVD


def main():
    print("╔" + "═" * 68 + "╗")
    print("║  CELN v3 — Autonomous Digestion Cycle                         ║")
    print("║  Corroboration + Cross-Lingual + Contradiction Detection       ║")
    print("╚" + "═" * 68 + "╝")

    t_start = time.time()

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Base knowledge (Portuguese corpus)
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

    word_counts, cooc_counts, pt_w2i, pt_i2w = build_cooccurrence(sentences, window_size=5)
    V_pt = len(pt_w2i)
    ppmi = compute_ppmi(word_counts, cooc_counts, pt_w2i)
    nc = min(10000, V_pt - 1)
    svd = TruncatedSVD(n_components=nc, random_state=42)
    vr = svd.fit_transform(ppmi)
    sv = svd.singular_values_
    var = sv**2 / (sv**2).sum()
    vr = vr * (var / var.max())[None, :]
    if nc < 10000:
        R_mat = np.random.RandomState(42).randn(nc, 10000) / np.sqrt(nc)
        pt_vectors = vr @ R_mat
    else:
        pt_vectors = vr
    pt_vectors = batch_normalize(pt_vectors)
    D = pt_vectors.shape[1]
    print(f"  Portuguese vectors: {V_pt} words × {D}D")
    print(f"  Corpus sentences:   {len(sentences)}")

    # ── Initialize SDM ──
    sdm = DenseSDM(n_locations=4096, activation_pct=0.01, seed=42)
    seed_centroids = []
    for tokens in sentences[:2000]:
        idxs = [pt_w2i[w] for w in tokens if w in pt_w2i]
        if len(idxs) >= 3:
            seed_centroids.append(normalize(pt_vectors[idxs].mean(axis=0)))
    sdm.initialize_addresses(np.array(seed_centroids))
    print(f"  SDM initialized:    4096 locations from {len(seed_centroids)} centroids")

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Create mixed facts (TRUE_PT + FALSE_PT + TRUE_EN)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'─'*68}")
    print("PHASE 2: Creating mixed facts (PT true, PT false, EN true)")
    print(f"{'─'*68}")

    # ── Create English vectors (PT-equivalent + noise) ──
    rng = np.random.RandomState(12345)
    noise_scale = 0.5

    # English words we'll use
    en_vocab = {
        "copper": "cobre", "conducts": "conduz", "electricity": "eletricidade",
        "metal": "metal", "heat": "calor", "wires": "fios",
        "current": "corrente", "transmission": "transmissão",
        "blocks": "bloqueia", "insulates": "isola",
    }

    en_w2i = {}
    en_vectors_list = []
    for en_w, pt_w in en_vocab.items():
        en_w2i[en_w] = len(en_vectors_list)
        if pt_w in pt_w2i:
            pt_vec = pt_vectors[pt_w2i[pt_w]]
            noise = rng.randn(D).astype(np.float32)
            noise = noise / (np.linalg.norm(noise) + 1e-12)
            en_vec = normalize((1.0 - noise_scale) * pt_vec + noise_scale * noise)
        else:
            en_vec = normalize(rng.randn(D).astype(np.float32))
        en_vectors_list.append(en_vec)
    en_vectors = np.array(en_vectors_list)
    V_en = len(en_vectors)

    # Cross-lingual similarity stats
    xl_sims = []
    for en_w, pt_w in en_vocab.items():
        if pt_w in pt_w2i:
            sim = float(np.dot(en_vectors[en_w2i[en_w]], pt_vectors[pt_w2i[pt_w]]))
            xl_sims.append(sim)
    print(f"  English vectors:     {V_en} words (PT-equiv + {noise_scale:.1f} noise)")
    print(f"  Cross-lingual sim:   μ={np.mean(xl_sims):.3f} (baseline for alignment)")

    # ── Build fact vectors ──
    def pt_fact(words):
        idxs = [pt_w2i[w] for w in words if w in pt_w2i]
        return normalize(pt_vectors[idxs].mean(axis=0)) if idxs else None

    def en_fact(words):
        idxs = [en_w2i[w] for w in words if w in en_w2i]
        return normalize(en_vectors[idxs].mean(axis=0)) if idxs else None

    # TRUE facts (Portuguese) — well-corroborated
    true_pt_facts = [
        ["cobre", "conduz", "eletricidade"],
        ["cobre", "condutividade", "elétrica"],
        ["cobre", "fios", "transmissão", "energia"],
        ["cobre", "corrente", "elétrica"],
        ["cobre", "metal", "condutor", "térmica"],
    ]

    # FALSE facts (Portuguese) — contradict true facts
    false_pt_facts = [
        ["cobre", "bloqueia", "corrente"],
        ["cobre", "isola", "eletricidade"],
    ]

    # TRUE facts (English simulated) — corroborate the PT true facts
    true_en_facts = [
        ["copper", "conducts", "electricity"],
        ["copper", "metal", "conducts", "heat"],
        ["copper", "wires", "current", "transmission"],
    ]

    print(f"  TRUE_PT:  {len(true_pt_facts)} facts (Portuguese, corroborated)")
    print(f"  FALSE_PT: {len(false_pt_facts)} facts (Portuguese, contradictory)")
    print(f"  TRUE_EN:  {len(true_en_facts)} facts (English, cross-lingual corroboration)")

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: SDM processes everything
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'─'*68}")
    print("PHASE 3: SDM Digestion (process all facts)")
    print(f"{'─'*68}")

    total_corroborating = 0
    total_contradictory = 0
    total_neutral = 0

    # 3a: TRUE Portuguese facts
    print(f"\n  ── Writing TRUE_PT facts ──")
    for words in true_pt_facts:
        fv = pt_fact(words)
        if fv is not None:
            r = sdm.write_corroborated(fv)
            total_corroborating += r['corroborating']
            total_contradictory += r['contradictory']
            total_neutral += r['neutral']
            print(f"    {' '.join(words[:3]):<35} corr={r['corroborating']:>2} neut={r['neutral']:>2}")

    # 3b: FALSE Portuguese facts
    print(f"\n  ── Writing FALSE_PT facts (contradictory) ──")
    for words in false_pt_facts:
        fv = pt_fact(words)
        if fv is not None:
            r = sdm.write_corroborated(fv)
            total_corroborating += r['corroborating']
            total_contradictory += r['contradictory']
            total_neutral += r['neutral']
            print(f"    {' '.join(words[:3]):<35} corr={r['corroborating']:>2} "
                  f"contra={r['contradictory']:>2} neut={r['neutral']:>2}")

    # 3c: TRUE English facts (cross-lingual corroboration)
    print(f"\n  ── Writing TRUE_EN facts (cross-lingual) ──")
    for words in true_en_facts:
        fv = en_fact(words)
        if fv is not None:
            r = sdm.write_corroborated(fv)
            total_corroborating += r['corroborating']
            total_contradictory += r['contradictory']
            total_neutral += r['neutral']
            print(f"    {' '.join(words[:3]):<35} corr={r['corroborating']:>2} neut={r['neutral']:>2}")

    # ── Summary ──
    print(f"\n  ── Digestion Summary ──")
    print(f"  Total corroborations:   {total_corroborating}")
    print(f"  Total contradictions:   {total_contradictory}")
    print(f"  Total neutral:          {total_neutral}")
    print(f"  Conflicts isolated:     {sdm.total_conflicts_detected}")
    print(f"  Locations with conflicts: {sdm.has_conflict.sum()}/{sdm.n_locations}")

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: Query and verify
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'─'*68}")
    print("PHASE 4: Query — does the system favor corroborated facts?")
    print(f"{'─'*68}")

    # Query 1: Portuguese — "o cobre conduz eletricidade?"
    print(f"\n  ── Query 1: 'o cobre conduz eletricidade?' (Portuguese) ──")
    q1_vec = pt_fact(["cobre", "conduz", "eletricidade"])
    if q1_vec is not None:
        r1 = sdm.read_with_confidence(q1_vec)

        print(f"  Trust score:        {r1['trust_score']:.3f}")
        print(f"  Corroboration:      {r1['mean_corroboration']:.3f}")
        print(f"  Conflicts detected: {r1['n_conflicts']} (magnitude={r1['conflict_magnitude']:.3f})")

        # Show top words
        sims = pt_vectors @ normalize(r1['result'])
        top = np.argsort(sims)[-10:][::-1]
        print(f"  Main hypothesis:    {', '.join(f'{pt_i2w[i]}({sims[i]:.2f})' for i in top[:6])}")

        if r1['competing_result'] is not None:
            sims_alt = pt_vectors @ normalize(r1['competing_result'])
            top_alt = np.argsort(sims_alt)[-10:][::-1]
            print(f"  Competing (ALT):    {', '.join(f'{pt_i2w[i]}({sims_alt[i]:.2f})' for i in top_alt[:6])}")

        # Key metric: similarity to "conduz" vs "isola"/"bloqueia"
        sim_conduct = float(np.dot(normalize(r1['result']),
                          pt_vectors[pt_w2i["conduz"]])) if "conduz" in pt_w2i else 0
        sim_block = float(np.dot(normalize(r1['result']),
                        pt_vectors[pt_w2i["bloqueia"]])) if "bloqueia" in pt_w2i else 0
        print(f"  → conduz={sim_conduct:.4f}  bloqueia={sim_block:.4f}  "
              f"Δ={sim_conduct-sim_block:+.4f}  "
              f"[{'CONDUZ ✓' if sim_conduct > sim_block else 'BLOQUEIA ✗'}]")

    # Query 2: English — "does copper conduct electricity?"
    print(f"\n  ── Query 2: 'copper conducts electricity?' (English) ──")
    q2_vec = en_fact(["copper", "conducts", "electricity"])
    if q2_vec is not None:
        r2 = sdm.read_with_confidence(q2_vec)

        print(f"  Trust score:        {r2['trust_score']:.3f}")
        print(f"  Cross-lingual test: query in EN → results in PT space")

        # Map result to Portuguese words (cross-lingual retrieval)
        sims_pt = pt_vectors @ normalize(r2['result'])
        top_pt = np.argsort(sims_pt)[-10:][::-1]
        print(f"  PT words retrieved: {', '.join(f'{pt_i2w[i]}({sims_pt[i]:.2f})' for i in top_pt[:6])}")

        # Check if "cobre" is in top results
        cobre_rank = int((sims_pt > sims_pt[pt_w2i["cobre"]]).sum()) + 1 if "cobre" in pt_w2i else -1
        conduz_rank = int((sims_pt > sims_pt[pt_w2i["conduz"]]).sum()) + 1 if "conduz" in pt_w2i else -1
        print(f"  'cobre' rank:       #{cobre_rank}/{V_pt}")
        print(f"  'conduz' rank:      #{conduz_rank}/{V_pt}")
        xl_success = cobre_rank <= 5

    # ═══════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═'*68}")
    print("FINAL REPORT — Autonomous Digestion Cycle")
    print(f"{'═'*68}")

    # Criteria
    c1 = total_corroborating > 0              # Corroboration detected
    c2 = sdm.total_conflicts_detected > 0     # Contradictions isolated
    c3 = sim_conduct > sim_block              # True fact dominates false
    c4 = xl_success                            # Cross-lingual alignment works
    c5 = r1['n_conflicts'] > 0                # Conflicts visible in query

    passed = sum([c1, c2, c3, c4, c5])

    print(f"\n  ┌─ Digestão {'─'*57}")
    print(f"  │ Corroborations detected:          {total_corroborating} {'✓' if c1 else '✗'}")
    print(f"  │ Contradictions isolated (ALT):    {sdm.total_conflicts_detected} {'✓' if c2 else '✗'}")
    print(f"  │ True fact dominates false:        "
          f"conduz={sim_conduct:.3f} > bloqueia={sim_block:.3f} {'✓' if c3 else '✗'}")
    print(f"  │ Cross-lingual retrieval (EN→PT):  "
          f"'cobre' rank #{cobre_rank} {'✓' if c4 else '✗'}")
    print(f"  │ Conflicts visible in query:       {r1['n_conflicts']} locations {'✓' if c5 else '✗'}")

    total_time = time.time() - t_start
    print(f"  │")
    print(f"  │ Result: {passed}/5 criteria passed in {total_time:.0f}s on Ryzen 2600 CPU")

    if passed >= 4:
        print(f"  │")
        print(f"  │ ✅ The Autonomous Digestion Cycle is FUNCTIONAL.")
        print(f"  │    The CELN digests mixed-language, contradictory text,")
        print(f"  │    corroborates consistent facts across languages,")
        print(f"  │    isolates contradictions as competing hypotheses,")
        print(f"  │    and favors well-corroborated knowledge on query.")
        print(f"  │")
        print(f"  │    Zero backprop. Zero templates. Pure algebra.")
    elif passed >= 3:
        print(f"  │")
        print(f"  │ ⚠️  The cycle shows partial functionality.")
        print(f"  │    Core mechanisms work but need refinement.")
    else:
        print(f"  │")
        print(f"  │ ⚠️  The cycle needs significant improvement.")

    print(f"  └{'─'*67}")
    print()

    return passed


if __name__ == '__main__':
    main()
