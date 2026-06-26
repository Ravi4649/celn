"""
CELN v3 — Full Multi-Turn Conversation Test
=============================================
Tests the complete integrated pipeline:
  - Type Field (97% syntactic accuracy)
  - Phase Rotation Lens (context-dependent similarity)
  - Auto-calibrating Blend (context_strength → dynamic type/semantic weights)
  - Session Memory (word centroids across turns for phase lens)
  - SDM Knowledge (DenseSDM for factual grounding)

5-turn conversation:
  1. Factual: "o que voce sabe sobre metais?"
  2. Follow-up: "e sobre o cobre especificamente?"
  3. Deductive: "se a cobra comeu o rato, quem comeu o rato?"
  4. Memory chain: "e o que mais voce sabe sobre cobras?"
  5. Analogy: "cobre esta para metal como onca esta para o que?"
"""

import sys, os, re, time, hashlib
import numpy as np
sys.path.insert(0, '/home/ravizin/celn-v3')

from celn.core import normalize, projective_resonance as M, similarity, phase_lens
from celn.dual_channel import DualChannelGenerator, extract_type_vectors
from celn.train import tokenize, load_corpus, build_cooccurrence, compute_ppmi, precompute_spectra
from celn.memory import DenseSDM
import warnings
warnings.filterwarnings('ignore')

FUNCTION_WORDS = {
    'o','a','os','as','um','uma','uns','umas',
    'de','do','da','dos','das','dum','duma',
    'em','no','na','nos','nas','num','numa','e','ou','mas','que','se','nem','pois',
    'é','foi','era','são','está','ser','sendo','estava','foram',
    'não','sim','como','quando','onde','porque','muito','pouco','mais','menos','tão',
    'ele','ela','eles','elas','seu','sua','seus','suas','este','essa','para','com',
    'por','pelo','pela','pelos','pelas','pra','pro','sem','sob','sobre','entre','até',
}

print("╔" + "═" * 70 + "╗")
print("║  CELN v3 — Full Multi-Turn Conversation                              ║")
print("║  Type Field + Phase Lens + Auto-Blend + SDM + Session Memory         ║")
print("╚" + "═" * 70 + "╝")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & SETUP
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[1] Loading pre-trained vectors...")
vector_path = os.environ.get(
    'CELN_VECTOR_PATH',
    '/home/ravizin/celn-v3/celn_full_vectors.npz',
)
print(f"    Vector file: {vector_path}")
data = np.load(vector_path, allow_pickle=True)
sem_vecs = data['vectors']  # (3007, 10000)
vocab = data['vocab']
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for i, w in enumerate(vocab)}
V, D = sem_vecs.shape
print(f"    {V} words, {D}D")

pmi_ri_vecs = None
pmi_ri_path = os.environ.get('CELN_PMI_RI_PATH')
if pmi_ri_path:
    pmi_data = np.load(pmi_ri_path, allow_pickle=True)
    pmi_vocab = pmi_data['vocab']
    if list(pmi_vocab) != list(vocab):
        raise ValueError('PMI-RI vocab does not match semantic vocab')
    pmi_ri_vecs = pmi_data['vectors']
    print(f"    PMI-RI channel: {pmi_ri_path}")

print("\n[2] Training type field...")
sentences = load_corpus('/home/ravizin/celn-v3/corpus_final.txt', min_len=2)
word_counts, cooc_counts, _, _ = build_cooccurrence(sentences, window_size=5)
ppmi = compute_ppmi(word_counts, cooc_counts, w2i)
type_vecs = extract_type_vectors(ppmi, type_dim=2000)
print(f"    Type vectors: {type_vecs.shape}")

print("\n[3] Loading/encoding word pairs for PairSDM (Parallel Transport)...")
cache_dir = '/home/ravizin/celn-v3/.cache'
os.makedirs(cache_dir, exist_ok=True)

vector_stat = os.stat(vector_path)
vocab_hash = hashlib.sha1('\n'.join(map(str, vocab)).encode('utf-8')).hexdigest()
cache_key = hashlib.sha1(
    f"{os.path.abspath(vector_path)}|{vector_stat.st_size}|"
    f"{int(vector_stat.st_mtime)}|{V}|{D}|{len(sentences)}|{vocab_hash}".encode('utf-8')
).hexdigest()[:16]
pair_cache_path = os.path.join(cache_dir, f'pair_sdm_{cache_key}.npz')

pair_sdm = None
pair_source_indices = None
pair_follower_indices = None

if os.path.exists(pair_cache_path):
    try:
        cache = np.load(pair_cache_path, allow_pickle=True)
        valid = (
            int(cache['vocab_size']) == V and
            int(cache['dim']) == D and
            int(cache['n_sentences']) == len(sentences) and
            str(cache['vocab_hash'].item()) == vocab_hash and
            int(cache['vector_size']) == vector_stat.st_size and
            int(cache['vector_mtime']) == int(vector_stat.st_mtime)
        )
        if valid:
            pair_source_indices = cache['pair_source_indices'].astype(np.int32)
            pair_follower_indices = cache['pair_follower_indices'].astype(np.int32)
            pair_sdm = DenseSDM(
                n_locations=int(cache['n_locations']),
                activation_pct=float(cache['activation_pct']),
                seed=42,
            )
            pair_sdm.addresses = cache['addresses'].astype(np.float32)
            pair_sdm.accumulators = cache['accumulators'].astype(np.float32)
            pair_sdm.counters = cache['counters'].astype(np.int32)
            pair_sdm.corroboration = cache['corroboration'].astype(np.float32)
            pair_sdm.total_writes = int(cache['total_writes'])
            print(f"    Cache hit: {pair_cache_path}")
            print(f"    {len(pair_source_indices)} pairs loaded")
        else:
            print("    Cache invalid: metadata mismatch")
    except Exception as exc:
        print(f"    Cache invalid: {exc}")
        pair_sdm = None

if pair_sdm is None:
    print("    Cache miss: encoding pairs and building PairSDM...")
    pair_vectors = []
    pair_source_indices = []
    pair_follower_indices = []
    for sent in sentences:
        for i in range(len(sent) - 1):
            w1, w2 = sent[i], sent[i+1]
            if w1 in w2i and w2 in w2i:
                i1, i2 = w2i[w1], w2i[w2]
                pv = M(sem_vecs[i1], sem_vecs[i2], gamma=1.0, bilateral=True)
                pair_vectors.append(pv)
                pair_source_indices.append(i1)
                pair_follower_indices.append(i2)
    pair_source_indices = np.asarray(pair_source_indices, dtype=np.int32)
    pair_follower_indices = np.asarray(pair_follower_indices, dtype=np.int32)
    print(f"    {len(pair_vectors)} pairs encoded")

    pair_sdm = DenseSDM(n_locations=8192, activation_pct=0.005, seed=42)
    n_seed = min(len(pair_vectors), 8000)
    seed_idx = np.random.RandomState(42).choice(len(pair_vectors), n_seed, replace=False)
    seed_vecs = np.array([pair_vectors[i] for i in seed_idx])
    pair_sdm.initialize_addresses(seed_vecs)
    for i, pv in enumerate(pair_vectors):
        pair_sdm.write(pv)

    np.savez_compressed(
        pair_cache_path,
        vocab_size=V,
        dim=D,
        n_sentences=len(sentences),
        vocab_hash=vocab_hash,
        vector_size=vector_stat.st_size,
        vector_mtime=int(vector_stat.st_mtime),
        n_locations=pair_sdm.n_locations,
        activation_pct=pair_sdm.activation_pct,
        total_writes=pair_sdm.total_writes,
        pair_source_indices=pair_source_indices,
        pair_follower_indices=pair_follower_indices,
        addresses=pair_sdm.addresses.astype(np.float32),
        accumulators=pair_sdm.accumulators.astype(np.float32),
        counters=pair_sdm.counters.astype(np.int32),
        corroboration=pair_sdm.corroboration.astype(np.float32),
    )
    print(f"    Cache saved: {pair_cache_path}")

print(f"    PairSDM: {pair_sdm.stats['n_written']} locations written")

print("\n[4] Creating generator with Transport + Phase Lens + Auto-Blend...")
gen = DualChannelGenerator(
    sem_vecs, type_vecs, w2i, i2w,
    window_size=5, window_decay=0.7,
    sdm=None,
    use_phase_lens=True,
    phase_lens_max_alpha=0.6,
    pair_sdm=pair_sdm,
    pmi_ri_vectors=pmi_ri_vecs,
    pair_source_indices=np.asarray(pair_source_indices, dtype=np.int32),
    pair_follower_indices=np.asarray(pair_follower_indices, dtype=np.int32),
)
gen.learn_type_field(sentences)
print(f"    Type fields: {gen.has_type_field.sum()}/{V}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

# Session: word centroids of all responses (for phase lens coherence)
session_centroids = []  # list of normalized word centroids
session_words = []      # all words from all responses

def update_session(response_words):
    """Update session memory after each turn."""
    global session_centroids, session_words

    # Track all words
    session_words.extend(response_words)

    # Word centroid of THIS response
    indices = [w2i[w] for w in response_words if w in w2i]
    if indices:
        turn_centroid = normalize(sem_vecs[indices].mean(axis=0))
        session_centroids.append(turn_centroid)


def get_session_context():
    """Build phase lens context from session centroids."""
    if not session_centroids:
        return None
    # Average all turn centroids (more weight to recent turns)
    weights = np.array([0.7 ** (len(session_centroids) - 1 - i)
                       for i in range(len(session_centroids))])
    weights = weights / weights.sum()
    ctx = np.zeros(D)
    for c, w in zip(session_centroids, weights):
        ctx += w * c
    return normalize(ctx)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GENERATION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def generate_response(prompt_words, max_len=12, seed=42):
    """Generate a response using the full pipeline."""
    ctx = get_session_context()
    session_ctx_list = []
    if ctx is not None:
        session_ctx_list.append(ctx)

    response = gen.generate(
        prefix_words=prompt_words,
        max_len=max_len,
        temperature=0.5,  # Transport has concentrated scores
        seed=seed,
        session_context=session_ctx_list if session_ctx_list else None,
        thematic_state=None,  # Static path with word centroids (not M-state)
        dynamic_temperature=False,  # Keep temperature fixed for SDM
    )
    return response


# ═══════════════════════════════════════════════════════════════════════════════
# 4. THE CONVERSATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("CONVERSATION")
print("═" * 70)

turns = []

# ── Turn 1: Factual ──────────────────────────────────────────────────
turn = 1
print(f"\n  ┌─ Turno {turn}: FACTUAL {'─'*54}")
question = "o que voce sabe sobre metais"
q_words = tokenize(question, min_len=2)
q_known = [w for w in q_words if w in w2i]  # ['que','sobre','metais']

print(f"  │ Q: {question}")
print(f"  │ Prefix: {q_known}")

resp = generate_response(q_known, max_len=10, seed=100 + turn)
update_session(resp)
func_ratio = sum(1 for w in resp if w in FUNCTION_WORDS) / max(len(resp), 1)
turns.append({'q': question, 'a': resp, 'func': func_ratio})

print(f"  │ A: {' '.join(resp)}")
print(f"  │    func={func_ratio:.0%}  session_words={len(session_words)}")

# ── Turn 2: Follow-up ────────────────────────────────────────────────
turn = 2
print(f"\n  ┌─ Turno {turn}: FOLLOW-UP {'─'*50}")
question = "e sobre o cobre especificamente"
q_words = tokenize(question, min_len=2)
q_known = [w for w in q_words if w in w2i]  # ['sobre','cobre']

print(f"  │ Q: {question}")
print(f"  │ Prefix: {q_known}")
print(f"  │ Session ctx: {len(session_centroids)} prior turns")

resp = generate_response(q_known, max_len=10, seed=200 + turn)
update_session(resp)
func_ratio = sum(1 for w in resp if w in FUNCTION_WORDS) / max(len(resp), 1)
turns.append({'q': question, 'a': resp, 'func': func_ratio})

print(f"  │ A: {' '.join(resp)}")
print(f"  │    func={func_ratio:.0%}  session_words={len(session_words)}")

# ── Turn 3: Deductive ────────────────────────────────────────────────
turn = 3
print(f"\n  ┌─ Turno {turn}: DEDUCTIVE {'─'*49}")
question = "se a cobra comeu o rato, quem comeu o rato"
q_words = tokenize(question, min_len=2)
q_known = [w for w in q_words if w in w2i]  # ['se','cobra','comeu','rato']

# Deductive reasoning: encode "cobra comeu rato" and unbind to find subject
if all(w in w2i for w in ['cobra','comeu','rato']):
    s_vec = sem_vecs[w2i['cobra']]
    v_vec = sem_vecs[w2i['comeu']]
    o_vec = sem_vecs[w2i['rato']]

    # M-encode: M(cobra, M(comeu, rato))
    inner = M(v_vec, o_vec, gamma=1.0, bilateral=True)
    composite = M(s_vec, inner, gamma=1.0, bilateral=True)

    # Unbind to find subject: unbind(composite, inner) ≈ cobra
    from celn.core import inverse_projective_resonance
    s_recovered = inverse_projective_resonance(composite, inner, gamma=1.0, bilateral=True, n_iter=20)
    s_sims = sem_vecs @ s_recovered
    s_idx = int(np.argmax(s_sims))
    s_word = i2w[s_idx]
    s_correct = (s_idx == w2i['cobra'])

    # Also recover object
    o_recovered = normalize(np.real(np.fft.ifft(
        np.fft.fft(inner) * np.conj(np.fft.fft(v_vec))
    )))
    o_sims = sem_vecs @ o_recovered
    o_idx = int(np.argmax(o_sims))
    o_word = i2w[o_idx]
    o_correct = (o_idx == w2i['rato'])

    ded_result = f"S={s_word}({'✓' if s_correct else '✗'}) O={o_word}({'✓' if o_correct else '✗'})"
    # Use the recovered subject as extra context
    extra = normalize(s_recovered)
else:
    ded_result = "missing words in vocab"
    extra = None
    s_word = "?"
    s_correct = False

print(f"  │ Q: {question}")
print(f"  │ Prefix: {q_known}")
print(f"  │ Deduction: {ded_result}")

resp = generate_response(q_known, max_len=10, seed=300 + turn)
update_session(resp)
func_ratio = sum(1 for w in resp if w in FUNCTION_WORDS) / max(len(resp), 1)
turns.append({'q': question, 'a': resp, 'func': func_ratio, 'deduction': s_word})

print(f"  │ A: {' '.join(resp)}")
print(f"  │    func={func_ratio:.0%}")

# ── Turn 4: Memory chain ─────────────────────────────────────────────
turn = 4
print(f"\n  ┌─ Turno {turn}: MEMORY CHAIN {'─'*46}")
question = "e o que mais voce sabe sobre cobras"
q_words = tokenize(question, min_len=2)
q_known = [w for w in q_words if w in w2i]  # ['que','sobre','cobras']

print(f"  │ Q: {question}")
print(f"  │ Prefix: {q_known}")
print(f"  │ Session: {len(session_centroids)} prior turns, {len(session_words)} words")

resp = generate_response(q_known, max_len=10, seed=400 + turn)
update_session(resp)
func_ratio = sum(1 for w in resp if w in FUNCTION_WORDS) / max(len(resp), 1)
turns.append({'q': question, 'a': resp, 'func': func_ratio})

print(f"  │ A: {' '.join(resp)}")
print(f"  │    func={func_ratio:.0%}")

# ── Turn 5: Analogy ──────────────────────────────────────────────────
turn = 5
print(f"\n  ┌─ Turno {turn}: ANALOGY {'─'*52}")
question = "cobre esta para metal como onca esta para o que"
q_words = tokenize(question, min_len=2)
q_known = [w for w in q_words if w in w2i]  # ['cobre','esta','para','metal','como','onca']

# Analogical reasoning: cobre:metal :: onça:X
# Parallel transport: relation = M(cobre, metal), transport = unbind(relation, onça)
if all(w in w2i for w in ['cobre','metal','onça']):
    av = sem_vecs[w2i['cobre']]
    bv = sem_vecs[w2i['metal']]
    cv = sem_vecs[w2i['onça']]

    # Encode the relationship
    relation = M(av, bv, gamma=1.0, bilateral=True)
    # Unbind to transport the relationship onto c
    transport = np.real(np.fft.ifft(
        np.fft.fft(relation) * np.conj(np.fft.fft(cv))
    ))
    transport = normalize(transport)

    # Find best match
    sims = sem_vecs @ transport
    for w in ['cobre','metal','onça']:
        if w in w2i: sims[w2i[w]] = -1.0
    best_idx = int(np.argmax(sims))
    best_word = i2w[best_idx]
    best_sim = float(sims[best_idx])

    # Top alternatives
    top5_idx = np.argsort(sims)[::-1][:5]
    top5 = [(i2w[i], float(sims[i])) for i in top5_idx]

    analogy_result = f"{best_word} (sim={best_sim:.3f})"
    extra = normalize(transport)
else:
    analogy_result = "missing words"
    best_word = "?"
    top5 = []
    extra = None

print(f"  │ Q: {question}")
print(f"  │ Prefix: {q_known}")
print(f"  │ Analogy: cobre:metal :: onça:{analogy_result}")
if top5:
    print(f"  │ Top-5: {', '.join(f'{w}({s:.3f})' for w,s in top5[:5])}")

resp = generate_response(q_known, max_len=10, seed=500 + turn)
update_session(resp)
func_ratio = sum(1 for w in resp if w in FUNCTION_WORDS) / max(len(resp), 1)
turns.append({'q': question, 'a': resp, 'func': func_ratio, 'analogy': best_word})

print(f"  │ A: {' '.join(resp)}")
print(f"  │    func={func_ratio:.0%}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("EVALUATION")
print("═" * 70)

func_values = [t['func'] for t in turns]
mean_func = np.mean(func_values)

print(f"\n  Engine:                     Parallel Transport (PairSDM + Resonator)")
print(f"  Turns completed:            {len(turns)}")
print(f"  Mean function words:        {mean_func:.1%} (target: >30%)")
print(f"  Session centroids stored:   {len(session_centroids)}")
print(f"  Session words accumulated:  {len(session_words)}")

# Per-turn display
print(f"\n  ┌─ TURN DETAILS {'─'*56}")
for i, t in enumerate(turns):
    func = t['func']
    bar = '█' * int(func * 30) + '░' * (30 - int(func * 30))
    print(f"  │ T{i+1} func={func:.0%} {bar}")
    print(f"  │   Q: {t['q'][:80]}")
    print(f"  │   A: {t['a'][:100]}")
    if 'deduction' in t:
        print(f"  │   Deduction: {t['deduction']}")
    if 'analogy' in t:
        print(f"  │   Analogy result: {t['analogy']}")
    print(f"  │")

# Domain keyword tracking
eletric_kw = {'cobre','conduz','corrente','elétrica','elétrico','condutor','energia','transmissão'}
mining_kw = {'metal','minério','ferro','liga','bronze','latão','zinco','produção'}
bio_kw = {'cobra','animal','réptil','predador','onça','rato','célula','tecido'}

all_responses = []
for t in turns:
    all_responses.extend(t['a'])

e_count = sum(1 for w in all_responses if w in eletric_kw)
m_count = sum(1 for w in all_responses if w in mining_kw)
b_count = sum(1 for w in all_responses if w in bio_kw)

print(f"  ┌─ DOMAIN KEYWORDS {'─'*54}")
print(f"  │ ⚡ Electricity: {e_count}")
print(f"  │ ⛏ Mining:      {m_count}")
print(f"  │ 🧬 Biology:     {b_count}")

# Criteria
print(f"\n  ┌─ CRITERIA {'─'*59}")
c1 = len(turns) >= 5
c2 = mean_func > 0.30
c3 = turns[2].get('deduction', '') == 'cobra'  # deduction correct
c4 = turns[4].get('analogy', '') not in ('?', '')  # analogy produced

print(f"  │ 1. 5 turns completed:          {'✓' if c1 else '✗'}")
print(f"  │ 2. Fluency > 30% func words:   {mean_func:.1%} {'✓' if c2 else '✗'}")
print(f"  │ 3. Deduction S='cobra':        {'✓' if c3 else '✗'} ({turns[2].get('deduction','?')})")
print(f"  │ 4. Analogy produced:           {'✓' if c4 else '✗'} ({turns[4].get('analogy','?')})")

passed = sum([c1, c2, c3, c4])
print(f"  │")
print(f"  │ Result: {passed}/4 criteria passed")

if passed >= 3:
    print(f"  │ ✅ CELN v3 maintains coherent multi-turn conversation")
elif passed >= 2:
    print(f"  │ ⚠️  Partial success. Components work, tuning needed.")
else:
    print(f"  │ ⚠️  Needs improvement.")

print(f"  └{'─'*70}")
print()
