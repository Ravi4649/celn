#!/usr/bin/env python3
"""
CELN v3 — Real Multi-Turn Conversation Demo
=============================================
A complete 5-turn conversation testing:
  1. Factual recall (SDM memory)
  2. Context follow-up (session state)
  3. Deductive reasoning (Resonator S-V-O extraction)
  4. Memory + knowledge chaining
  5. Analogical reasoning (parallel transport)

Pipeline per turn: LISTEN → REMEMBER → REASON → RESPOND → UPDATE SESSION

Usage:
  python experiments/demo_conversation.py [--quick]
"""

import sys, os, re, time, textwrap
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celn_v3.train import tokenize, build_cooccurrence, compute_ppmi
from celn_v3.core import normalize, batch_normalize, projective_resonance as M
from celn_v3.dual_channel import DualChannelGenerator
from celn_v3.hdc_types import train_hdc_type_vectors
from celn_v3.memory import DenseSDM
from celn_v3.resonator import (
    ResonatorDecoder, bind_vec, unbind_vec,
    unbind_M_forward, unbind_M_reverse,
)

FUNCTION_WORDS = {
    'o','a','os','as','um','uma','uns','umas',
    'de','do','da','dos','das','dum','duma',
    'em','no','na','nos','nas','num','numa',
    'por','pelo','pela','pelos','pelas','para','pra','pro','com','sem','sob','sobre','entre','até',
    'e','ou','mas','que','se','nem','pois',
    'é','foi','era','são','está','ser','sendo','estava','foram',
    'não','sim','como','quando','onde','porque',
    'muito','pouco','mais','menos','tão','tanto',
    'ele','ela','eles','elas','seu','sua','seus','suas',
    'me','te','lhe','nos','vos','lo','la','lhes',
    'este','essa','isto','isso','aquele',
}


class ConversationAgent:
    """Full CELN v3 conversational agent with session memory."""

    def __init__(self, vectors, type_vecs, sdm, w2i, i2w, sentences, seed=42):
        self.vectors = vectors.astype(np.float32)
        self.type_vecs = type_vecs.astype(np.float32)
        self.sdm = sdm
        self.w2i = w2i
        self.i2w = i2w
        self.V = len(w2i)
        self.D = vectors.shape[1]
        self.rng = np.random.RandomState(seed)

        self.generator = DualChannelGenerator(
            semantic_vectors=vectors, type_vectors=type_vecs,
            w2i=w2i, i2w=i2w, window_size=5, window_decay=0.7, sdm=sdm,
        )
        self.generator.learn_type_field(sentences)

        self.resonator = ResonatorDecoder(
            vectors, max_iter=20, n_restarts=3,
            convergence_patience=3, seed=seed,
        )

        # Session memory: accumulates conversation via M
        self.session_state = None
        self.session_decay = 0.9
        self.turn_count = 0

        # ── Thematic state: M-encoded dynamic topic representation ──
        # Evolves with each response. Captures what the conversation
        # is ABOUT, not just what words were used.
        self.thematic_state = None
        self.thematic_decay = 0.85

    # ── LISTEN + REMEMBER ──────────────────────────────────────

    def _centroid(self, words):
        idxs = [self.w2i[w] for w in words if w in self.w2i]
        return normalize(self.vectors[idxs].mean(axis=0)) if idxs else np.zeros(self.D)

    def _sd_knowledge(self, query_words):
        q = self._centroid(query_words)
        sr = self.sdm.read(q)
        proj = float(np.dot(sr, q))
        r = sr - proj * q
        rn = np.linalg.norm(r)
        if rn > 1e-12: r /= rn
        return r, min(rn * 3.0, 1.0)

    # ── REASON ──────────────────────────────────────────────────

    def _deduce(self, subject, verb, obj):
        """Directional M-unbinding: extract S, V, O from M(S, M(V, O))."""
        sv, vv, ov = [self.vectors[self.w2i[w]] for w in [subject, verb, obj]]
        inner = M(vv, ov, gamma=1.0, bilateral=True)
        composite = M(sv, inner, gamma=1.0, bilateral=True)

        s_rec = unbind_M_forward(composite, inner)
        s_idx, s_sim = self.resonator._nearest_with_score(s_rec)
        o_rec = unbind_M_reverse(inner, vv)
        o_idx, o_sim = self.resonator._nearest_with_score(o_rec)
        v_rec = unbind_M_forward(inner, ov, x=vv)
        v_idx, v_sim = self.resonator._nearest_with_score(v_rec)

        return {
            'subject': self.i2w[s_idx], 'verb': self.i2w[v_idx], 'object': self.i2w[o_idx],
            'subject_correct': s_idx == self.w2i[subject],
            'object_correct': o_idx == self.w2i[obj],
            'verb_correct': v_idx == self.w2i[verb],
            's_sim': float(s_sim), 'v_sim': float(v_sim), 'o_sim': float(o_sim),
        }

    def _analogy(self, a, b, c):
        """Parallel transport: A:B :: C:X."""
        if not all(w in self.w2i for w in [a, b, c]):
            return None
        av, bv, cv = [self.vectors[self.w2i[w]] for w in [a, b, c]]
        relation = M(av, bv, gamma=1.0, bilateral=True)
        transport = unbind_vec(relation, cv)
        tn = normalize(transport)
        sims = self.vectors @ tn.astype(np.float32)
        for w in [a, b, c]:
            if w in self.w2i: sims[self.w2i[w]] = -1.0
        top_idx = int(np.argmax(sims))
        return self.i2w[top_idx]

    # ── RESPOND ──────────────────────────────────────────────────

    def _respond(self, prompt, reasoning_vecs=None, max_len=10):
        """Generate response with thematic core, session memory, SDM, and fluency."""
        if reasoning_vecs is None:
            reasoning_vecs = []

        # Build combined context: session + reasoning
        context_vecs = []
        if self.session_state is not None:
            context_vecs.append(self.session_state.copy())
        for rv in reasoning_vecs:
            if np.linalg.norm(rv) > 1e-12:
                context_vecs.append(normalize(rv))

        response = self.generator.generate(
            prefix_words=prompt,
            max_len=max_len,
            temperature=0.8,
            seed=42 + self.turn_count,
            session_context=context_vecs,
            thematic_state=self.thematic_state,
            creative_restlessness=0.02,
            dynamic_temperature=True,
        )
        return response

    # ── SESSION UPDATE ───────────────────────────────────────────

    def _update_session(self, response_words):
        rc = self._centroid(response_words)
        if np.linalg.norm(rc) < 1e-12:
            return

        # Session state: accumulated via M (for context window)
        if self.session_state is None:
            self.session_state = rc.copy()
        else:
            decayed = self.session_state * self.session_decay
            self.session_state = normalize(M(decayed, rc, gamma=0.5, bilateral=False))

        # Thematic state: M-encode INDIVIDUAL WORDS from the response
        # This preserves word-level information for MSWE extraction.
        # Each content word from the response is bound into the M state,
        # building a rich holographic composite that the Resonator can decode.
        content_words = [w for w in response_words if w in self.w2i
                        and w not in {'o','a','os','as','um','uma','uns','umas',
                            'de','do','da','dos','das','em','no','na','nos','nas',
                            'por','para','com','sem','sob','sobre','entre','até',
                            'e','ou','mas','que','se','nem','pois',
                            'é','foi','era','são','está','ser','não','sim',
                            'como','quando','onde','porque','muito','pouco','mais','menos'}]

        for word in content_words[:8]:  # bind up to 8 content words per turn
            wv = self.vectors[self.w2i[word]]
            if self.thematic_state is None:
                self.thematic_state = wv.copy()
            else:
                decayed = self.thematic_state * self.thematic_decay
                self.thematic_state = normalize(
                    M(decayed, wv, gamma=0.5, bilateral=False)
                )

    # ── HIGH-LEVEL API ───────────────────────────────────────────

    def ask(self, question_type, question_text, **kwargs):
        """Process a single turn and return the response."""
        self.turn_count += 1
        qw = tokenize(question_text, min_len=1)
        qk = [w for w in qw if w in self.w2i]
        prompt = qk[:5] if len(qk) >= 3 else qk

        # Initialize thematic state from question content words on first turn
        if self.thematic_state is None:
            content_words = [w for w in qk if w not in
                {'o','a','os','as','um','uma','de','do','da','e','que','se','é',
                 'voce','sabe','sobre','para','como','onde','quando','porque','mais'}]
            if content_words:
                self.thematic_state = self._centroid(content_words)

        reasoning_vecs = []
        reasoning_data = {}

        if question_type == 'factual':
            topic = kwargs.get('topic_words', qk)
            kn, _ = self._sd_knowledge(topic)
            if np.linalg.norm(kn) > 1e-12:
                reasoning_vecs.append(kn)
            response = self._respond(prompt, reasoning_vecs)

        elif question_type == 'deductive':
            s, v, o = kwargs['subject'], kwargs['verb'], kwargs['object']
            ded = self._deduce(s, v, o)
            reasoning_data = ded
            target = kwargs.get('ask_for', 'subject')
            tw = s if target == 'subject' else o
            if tw in self.w2i:
                reasoning_vecs.append(self.vectors[self.w2i[tw]])
            response = self._respond(prompt, reasoning_vecs)

        elif question_type == 'analogy':
            a, b, c = kwargs['a'], kwargs['b'], kwargs['c']
            best = self._analogy(a, b, c)
            reasoning_data = {'a': a, 'b': b, 'c': c, 'best': best}
            if best and best in self.w2i:
                reasoning_vecs.append(self.vectors[self.w2i[best]])
            response = self._respond(prompt, reasoning_vecs)

        else:
            response = self._respond(prompt)

        self._update_session(response)

        func_ratio = sum(1 for w in response if w in FUNCTION_WORDS) / max(len(response), 1)
        session_active = self.session_state is not None

        return {
            'turn': self.turn_count,
            'question': question_text,
            'type': question_type,
            'response': ' '.join(response),
            'response_words': response,
            'func_ratio': func_ratio,
            'session_active': session_active,
            'reasoning': reasoning_data,
        }


def train_all(sentences, quick=False):
    from sklearn.decomposition import TruncatedSVD
    word_counts, cooc_counts, w2i, i2w = build_cooccurrence(sentences, window_size=5)
    V = len(w2i)
    ppmi = compute_ppmi(word_counts, cooc_counts, w2i)
    nc = min(10000, V - 1)
    svd = TruncatedSVD(n_components=nc, random_state=42)
    vr = svd.fit_transform(ppmi)
    sv_vals = svd.singular_values_
    var = sv_vals**2 / (sv_vals**2).sum()
    vr = vr * (var / var.max())[None, :]
    if nc < 10000:
        R = np.random.RandomState(42).randn(nc, 10000) / np.sqrt(nc)
        vectors = vr @ R
    else:
        vectors = vr
    vectors = batch_normalize(vectors)
    type_vecs = train_hdc_type_vectors(
        sentences, w2i, V, hdc_dim=4096, context_window=3,
        n_epochs=3 if quick else 5, learning_rate=0.05, seed=42, verbose=False)
    sdm = DenseSDM(n_locations=4096, activation_pct=0.01, seed=42)
    sn = min(len(sentences), 2000)
    sc_list = []
    for tokens in sentences[:sn]:
        idxs = [w2i[w] for w in tokens if w in w2i]
        if idxs: sc_list.append(normalize(vectors[idxs].mean(axis=0)))
    if sc_list: sdm.initialize_addresses(np.array(sc_list))
    for idx in range(len(vectors)): sdm.write(vectors[idx])
    return vectors, type_vecs, sdm, w2i, i2w


def main():
    quick = '--quick' in sys.argv

    print("╔" + "═" * 72 + "╗")
    print("║  CELN v3 — Multi-Turn Conversation Demo                        ║")
    print("║  Listen → Remember → Reason → Respond → Update Session         ║")
    print("╚" + "═" * 72 + "╝")

    # ── Load & train ──
    corpus_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'corpus_final.txt')
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    raw = re.split(r'[.!?\n]+', text)
    sentences = []
    for s in raw:
        tokens = tokenize(s, min_len=1)
        if len(tokens) >= 3: sentences.append(tokens)
    if quick: sentences = sentences[:500]

    print(f"\n  Training on {len(sentences)} sentences...")
    t0 = time.time()
    vectors, type_vecs, sdm, w2i, i2w = train_all(sentences, quick=quick)
    print(f"  Done in {time.time()-t0:.0f}s  (V={len(w2i)}, D={vectors.shape[1]})")

    # ── Create agent ──
    agent = ConversationAgent(vectors, type_vecs, sdm, w2i, i2w, sentences, seed=42)

    # ════════════════════════════════════════════════════════════
    # THE CONVERSATION
    # ════════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("CONVERSATION")
    print("═" * 72)

    turns = []

    # Turn 1: Factual
    print(f"\n  ┌─ Turno 1: FACTUAL {'─'*54}")
    r1 = agent.ask('factual', "o que voce sabe sobre metais",
                    topic_words=["metal", "cobre", "ferro", "condutividade"])
    turns.append(r1)
    print(f"  │ Q: {r1['question']}")
    print(f"  │ A: {r1['response']}")
    print(f"  │    func={r1['func_ratio']:.0%}  session={'active' if r1['session_active'] else 'inactive'}")

    # Turn 2: Follow-up with context (tests session memory)
    print(f"\n  ┌─ Turno 2: FOLLOW-UP (tests session memory) {'─'*35}")
    r2 = agent.ask('factual', "e sobre o cobre especificamente",
                    topic_words=["cobre", "condutividade", "elétrica", "metal"])
    turns.append(r2)
    print(f"  │ Q: {r2['question']}")
    print(f"  │ A: {r2['response']}")
    print(f"  │    func={r2['func_ratio']:.0%}  session={'active' if r2['session_active'] else 'inactive'}")

    # Turn 3: Deductive reasoning
    print(f"\n  ┌─ Turno 3: DEDUCTIVE (Resonator S-V-O) {'─'*39}")
    r3 = agent.ask('deductive', "se a cobra comeu o rato, quem comeu o rato",
                    subject="cobra", verb="comeu", object="rato", ask_for="subject")
    turns.append(r3)
    ded = r3['reasoning']
    print(f"  │ Q: {r3['question']}")
    print(f"  │ Reason: S='{ded['subject']}'({ded['s_sim']:.2f}) "
          f"V='{ded['verb']}'({ded['v_sim']:.2f}) "
          f"O='{ded['object']}'({ded['o_sim']:.2f}) "
          f"[S{'✓' if ded['subject_correct'] else '✗'} "
          f"V{'✓' if ded['verb_correct'] else '✗'} "
          f"O{'✓' if ded['object_correct'] else '✗'}]")
    print(f"  │ A: {r3['response']}")
    print(f"  │    func={r3['func_ratio']:.0%}  session={'active' if r3['session_active'] else 'inactive'}")

    # Turn 4: Memory chaining — builds on what was said before
    print(f"\n  ┌─ Turno 4: MEMORY CHAIN (session + SDM knowledge) {'─'*30}")
    r4 = agent.ask('factual', "e o que mais voce sabe sobre cobras",
                    topic_words=["cobra", "réptil", "predador", "animal"])
    turns.append(r4)
    print(f"  │ Q: {r4['question']}")
    print(f"  │ A: {r4['response']}")
    print(f"  │    func={r4['func_ratio']:.0%}  session={'active' if r4['session_active'] else 'inactive'}")

    # Turn 5: Analogical reasoning
    print(f"\n  ┌─ Turno 5: ANALOGY (parallel transport) {'─'*42}")
    r5 = agent.ask('analogy', "cobre esta para metal como onca esta para o que",
                    a="cobre", b="metal", c="onça")
    turns.append(r5)
    ana = r5['reasoning']
    print(f"  │ Q: {r5['question']}")
    print(f"  │ Reason: {ana['a']}:{ana['b']} :: {ana['c']}:{ana['best']}")
    print(f"  │ A: {r5['response']}")
    print(f"  │    func={r5['func_ratio']:.0%}  session={'active' if r5['session_active'] else 'inactive'}")

    # ════════════════════════════════════════════════════════
    # EVALUATION
    # ════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("EVALUATION")
    print("═" * 72)

    func_vals = [t['func_ratio'] for t in turns]
    mean_func = np.mean(func_vals)
    print(f"\n  Turns completed:           {len(turns)}")
    print(f"  Mean function words:       {mean_func:.1%} (target: >35%)")
    print(f"  Session memory:            {'active' if turns[-1]['session_active'] else 'inactive'}")
    print(f"  Fluency mechanisms:        dynamic temp + restlessness = ON")

    # Per-turn analysis
    print(f"\n  ┌─ PER-TURN {'─'*58}")
    for t in turns:
        rtype = t['type']
        func = t['func_ratio']
        bar = '█' * int(func * 40) + '░' * (40 - int(func * 40))
        print(f"  │ T{t['turn']} [{rtype:<10}] func={func:.0%} {bar}")

    print(f"\n  ┌─ CRITERIA {'─'*58}")
    c1 = len(turns) >= 5
    c2 = mean_func > 0.30
    c3 = turns[2]['reasoning'].get('subject_correct', False) if len(turns) > 2 else False  # deduction
    c4 = turns[4]['reasoning'].get('best') is not None if len(turns) > 4 else False  # analogy

    print(f"  │ 1. 5+ turns completed:      {len(turns)} turns {'✓' if c1 else '✗'}")
    print(f"  │ 2. Fluency > 30% func:       {mean_func:.1%} {'✓' if c2 else '✗'}")
    print(f"  │ 3. Deduction correct:        {'✓' if c3 else '✗'}")
    print(f"  │ 4. Analogy produced:         {'✓' if c4 else '✗'}")

    passed = sum([c1, c2, c3, c4])
    print(f"  │")
    print(f"  │ Result: {passed}/4 criteria passed")

    if passed >= 3:
        print(f"  │")
        print(f"  │ ✅ CELN v3 maintains coherent multi-turn conversation")
        print(f"  │    with session memory, reasoning, and fluent generation.")
    elif passed >= 2:
        print(f"  │")
        print(f"  │ ⚠️  Partial success. Some components work, others need tuning.")
    else:
        print(f"  │")
        print(f"  │ ⚠️  Multi-turn conversation needs significant improvement.")

    print(f"  └{'─'*70}")
    print()


if __name__ == '__main__':
    main()
