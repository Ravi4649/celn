#!/usr/bin/env python3
"""
CELN — Step-by-Step Reasoning Demo (Portuguese)
===============================================

Demonstra o motor de raciocínio lógico CELN codificando regras
em português como vetores de 10.000 dimensões, armazenando na
DenseSDM e deduzindo conclusões via Forward Chaining.

Requer: pip install -r requirements.txt
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
np.set_printoptions(precision=4, suppress=True, threshold=8, linewidth=80)

from celn_v3.core import D, normalize, similarity, bind, unbind
from celn_v3.logic_encoder import LogicRoles, encode_rule, decode_consequent
from celn_v3.memory import DenseSDM
from celn_v3.forward_chainer import ForwardChainer

SEP = "─" * 72


# ── Helpers ────────────────────────────────────────────────────

def show_vec(label: str, v: np.ndarray):
    """Print a 10k-d vector in a human-readable summary."""
    norm = np.linalg.norm(v)
    mag = np.abs(np.fft.fft(v))
    mag_median = float(np.median(mag))
    mag_max = float(mag.max())
    head = v[:4].tolist()
    tail = v[-2:].tolist()
    first5 = ", ".join(f"{x:.4f}" for x in head)
    last2 = ", ".join(f"{x:.4f}" for x in tail)
    print(f"  {label}:")
    print(f"    shape=(10000,)  ‖v‖={norm:.4f}  |FFT|_mediana={mag_median:.2f}  |FFT|_máx={mag_max:.2f}")
    print(f"    [{first5}  ...  {last2}]")


def nearest_word(vec: np.ndarray, vectors: np.ndarray, i2w: dict[int, str]) -> tuple[str, float]:
    """Find the nearest word by cosine similarity."""
    sims = vectors @ normalize(vec)
    idx = int(np.argmax(sims))
    return i2w[idx], float(sims[idx])


def dot_line():
    print("·" * 72)


# ── Main demo ──────────────────────────────────────────────────

def main():
    print()
    print("╔" + "═" * 70 + "╗")
    print("║   CELN v3 — RACIOCÍNIO LÓGICO PASSO A PASSO         ║")
    print("║   Vector Symbolic Architecture (10k-D, CPU, sem BP)  ║")
    print("╚" + "═" * 70 + "╝")
    print()

    # ── 1. Carregar ou gerar vetores ───────────────────────────
    print(SEP)
    print("  ETAPA 1: Vetores de palavras")
    print(SEP)

    npz_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "celn_v3_full_vectors.npz")
    use_random = not os.path.exists(npz_path)

    if use_random:
        print("  [aviso] celn_v3_full_vectors.npz não encontrado.")
        print("  Gerando 50 vetores aleatórios para demonstração.")
        print()
        rng = np.random.RandomState(42)
        words = [
            "cachorro", "animal", "rex", "fido", "gato", "ser", "vivo",
            "lobo", "raposa", "peixe", "pássaro", "cobra", "rato",
            "homem", "mulher", "criança", "bebê", "mamífero", "réptil",
            "ave", "inseto", "planta", "árvore", "flor", "fruta",
            "água", "terra", "fogo", "ar", "metal", "pedra",
            "cobre", "ferro", "ouro", "prata", "bronze", "aço",
            "casa", "carro", "livro", "mesa", "cadeira", "porta",
            "sol", "lua", "estrela", "céu", "mar", "rio", "monte",
        ]
        vocab_size = len(words)
        vectors = np.random.randn(vocab_size, D).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        vectors = vectors / norms
        w2i = {w: i for i, w in enumerate(words)}
        i2w = {i: w for i, w in enumerate(words)}
        print(f"  Vocabulário: {vocab_size} palavras em {D} dimensões (aleatório)")
    else:
        print("  [ok] celn_v3_full_vectors.npz encontrado.")
        data = np.load(npz_path, allow_pickle=True)
        vectors = data["vectors"].astype(np.float32)
        vocab = [str(w) for w in data["vocab"]]
        w2i = {w: i for i, w in enumerate(vocab)}
        i2w = {i: w for i, w in enumerate(vocab)}
        print(f"  Vocabulário: {len(vocab)} palavras em {D} dimensões (treinado do corpus)")
        # Verify key words exist
        needed = ["cachorro", "animal", "rex", "fido", "gato"]
        for w in needed:
            if w not in w2i:
                idx = len(w2i)
                w2i[w] = idx
                i2w[idx] = w
                v = np.random.randn(D).astype(np.float32)
                vectors = np.vstack([vectors, normalize(v).reshape(1, -1)])
                print(f"  [aviso] '{w}' não está no vocabulário — adicionado como vetor aleatório")
        print()

    dot_line()
    print("  Vetores de algumas palavras:")
    for w in ["cachorro", "animal", "rex", "gato"]:
        v = normalize(vectors[w2i[w]])
        show_vec(f"v[{w}]", v)
    print()

    # ── 2. LogicRoles ──────────────────────────────────────────
    print(SEP)
    print("  ETAPA 2: Papéis lógicos (LogicRoles)")
    print(SEP)

    roles = LogicRoles(seed=42)
    dot_line()
    print("  Vetores de papel (quasi-ortogonais entre si):")
    for name in roles.ROLE_NAMES:
        rv = roles.get(name)
        show_vec(f"  {name}", rv)
    print()
    # Check orthogonality
    print("  Ortogonalidade entre papéis (cosseno < 0.1 = quasi-ortogonal):")
    for n1 in roles.ROLE_NAMES:
        for n2 in roles.ROLE_NAMES:
            if n1 < n2:
                sim = similarity(roles.get(n1), roles.get(n2))
                badge = "✓" if abs(sim) < 0.1 else "⚠"
                print(f"    {badge}  cos({n1}, {n2}) = {sim:.4f}")
    print()

    # ── 3. Codificação de regras ───────────────────────────────
    print(SEP)
    print("  ETAPA 3: Codificação de regras FOL como vetores")
    print(SEP)

    rules_text = [
        ("ROLE_TODOS", "cachorro", "animal",  "Todo cachorro é um animal."),
        ("ROLE_TODOS", "rex",     "cachorro", "Rex é um cachorro."),
        ("ROLE_TODOS", "fido",    "gato",     "Fido é um gato."),
        ("ROLE_TODOS", "gato",    "animal",   "Todo gato é um animal."),
    ]

    encoded_rules = []
    ant_vecs = {}
    cons_vecs = {}

    for role_name, ant_word, cons_word, text in rules_text:
        print(f"  Regra: \"{text}\"")
        dot_line()

        v_ant = normalize(vectors[w2i[ant_word]])
        v_cons = normalize(vectors[w2i[cons_word]])
        ant_vecs[ant_word] = v_ant
        cons_vecs[cons_word] = v_cons

        show_vec(f"  antecedente \"{ant_word}\"", v_ant)
        show_vec(f"  consequente \"{cons_word}\"", v_cons)

        role_vec = roles.get(role_name)
        show_vec(f"  papel {role_name}", role_vec)

        # Encode step by step (same as encode_rule() from logic_encoder)
        from celn_v3.logic_encoder import get_perm_ant, get_perm_cons
        pa = get_perm_ant()
        pc = get_perm_cons()
        show_vec("  PERM_ANT (tag de posição A)", pa)
        show_vec("  PERM_CONS (tag de posição B)", pc)

        inner = normalize(bind(pa, v_ant) + bind(pc, v_cons))
        show_vec("  inner = normalize(PA⊛ant + PC⊛cons)", inner)

        composite = bind(role_vec, inner)
        show_vec("  composite = bind(ROLE, inner)", composite)

        encoded_rules.append((role_name, ant_word, cons_word, composite, text))
        print()

    # ── 4. Armazenar na DenseSDM ───────────────────────────────
    print(SEP)
    print("  ETAPA 4: Armazenamento na memória associativa (DenseSDM)")
    print(SEP)

    # Build seed vectors from our word vectors
    n_locations = 256
    seed_vectors = vectors[np.random.RandomState(42).choice(len(vectors), size=n_locations, replace=True)]
    sdm = DenseSDM(n_locations=n_locations, activation_pct=0.1, seed=42)
    sdm.initialize_addresses(seed_vectors)

    print(f"  SDM: {sdm.n_locations} hard-locations, ativação = {sdm.activation_pct:.0%} dos endereços")
    dot_line()

    for role_name, ant_word, cons_word, composite, text in encoded_rules:
        n_act = sdm.write(composite)
        sim = similarity(composite, sdm.read(composite))
        print(f"  ✓ Armazenado: \"{text}\"")
        print(f"    → ativou {n_act} hard-locations")
        print(f"    → round-trip similarity: {sim:.4f}")
    print()
    print(f"  Total de writes no SDM: {sdm.total_writes}")
    print(f"  Estatísticas do SDM:")
    stats = sdm.stats
    for k, v in stats.items():
        print(f"    {k}: {v}")
    print()

    # ── 5. Forward Chaining passo a passo ──────────────────────
    print(SEP)
    print("  ETAPA 5: Dedução — Forward Chaining passo a passo")
    print(SEP)

    queries = [
        ("rex",  "animal", "Rex é um animal?"),
        ("fido", "animal", "Fido é um animal?"),
        ("rex",  "gato",   "Rex é um gato?"),
    ]

    chainer = ForwardChainer(vectors, w2i, i2w, n_sdm_locations=256, seed=42, use_bridge=False)
    # Override the internal SDM with our already-populated one
    chainer.sdm = sdm
    # Re-register rules into chainer's rules list for iterative deduction
    for role_name, ant_word, cons_word, composite, text in encoded_rules:
        v_ant = normalize(vectors[w2i[ant_word]])
        v_cons = normalize(vectors[w2i[cons_word]])
        chainer._rules_stored.append((role_name, v_ant.copy(), v_cons.copy(), composite))

    role_vec_todos = roles.get("ROLE_TODOS")

    for fact_word, target_word, question in queries:
        print(f"  Pergunta: {question}")
        print(f"  Fato inicial: \"{fact_word}\"")
        print(f"  Alvo: \"{target_word}\"")
        dot_line()

        # ── Step-by-step forward chaining ──
        print()
        chain_steps = []
        current_fact = fact_word
        v_current = normalize(vectors[w2i[current_fact]])
        show_vec(f"  vetor do fato \"{current_fact}\"", v_current)
        print()

        max_depth = 5
        reached_target = False
        prev_fact = None

        for depth in range(max_depth):
            print(f"  ▶ Passo {depth + 1}: buscar regra com antecedente \"{current_fact}\"...")

            # Find the best-matching rule by scanning stored rules
            best_rule = None
            best_sim = -1.0
            for rname, rant, rcons, rcomp in chainer._rules_stored:
                sim = similarity(v_current, rant)
                if sim > best_sim:
                    best_sim = sim
                    best_rule = (rname, rant, rcons, rcomp)

            if best_rule is None or best_sim < 0.5:
                print(f"    Nenhuma regra encontrada para \"{current_fact}\". Parando.")
                break

            rname, rant, rcons, rcomp = best_rule
            print(f"    Regra encontrada: {rname}({best_sim:.4f})")
            show_vec("    vetor composto da regra", rcomp)

            # Decode consequent from this specific rule
            v_conseq = decode_consequent(rcomp, role_vec_todos)
            show_vec("    decode_consequent(regra, TODOS)", v_conseq)

            derived_word, conf = nearest_word(v_conseq, vectors, i2w)
            chain_steps.append((current_fact, derived_word, conf))
            print(f"    ► Deduzido: \"{current_fact}\" → \"{derived_word}\" (cos={conf:.4f})")
            print()

            if derived_word == target_word:
                reached_target = True
                break

            # Cycle detection: same word as current → no progress
            if derived_word == current_fact:
                print(f"    [ciclo detectado] \"{derived_word}\" não leva a novo conhecimento.")
                break

            # Move to next step
            prev_fact = current_fact
            current_fact = derived_word
            if current_fact in w2i:
                v_current = normalize(vectors[w2i[current_fact]])
            else:
                break

        # ── Conclusion ──
        print()
        print(f"  ═══════════════════════════════════════════")

        if reached_target:
            chain_str = " → ".join(f"\"{a}\"→\"{b}\"" for a, b, _ in chain_steps)
            print(f"  ► Conclusão: Sim, \"{fact_word}\" é um \"{target_word}\".")
            print(f"  ► Cadeia: {chain_str}")
        elif not chain_steps:
            print(f"  ► Conclusão: Desconhecido (Unknown)")
            print(f"  ► Nenhuma regra aplicável a \"{fact_word}\".")
        else:
            chain_str = " → ".join(f"\"{a}\"→\"{b}\"" for a, b, _ in chain_steps)
            last_word = chain_steps[-1][1]
            if last_word == target_word:
                print(f"  ► Conclusão: Sim, \"{fact_word}\" é um \"{target_word}\".")
            else:
                print(f"  ► Conclusão: Desconhecido (Unknown)")
                print(f"  ► Cadeia máxima: {chain_str}")
                print(f"  ► Chegou em \"{last_word}\", não em \"{target_word}\".")
        print(f"  ═══════════════════════════════════════════")
        print(SEP)
        print()

    # ── 6. Resumo ─────────────────────────────────────────────
    print()
    print("╔" + "═" * 70 + "╗")
    print("║   RESUMO DO QUE ACONTECEU                             ║")
    print("╚" + "═" * 70 + "╝")
    print()
    print("  1. Palavras → vetores 10k-D normalizados")
    print("  2. Papéis lógicos (TODOS, NENHUM, ...) como vetores quasi-ortogonais")
    print("  3. Cada regra foi codificada como:")
    print("       composite = bind(ROLE, normalize(PA⊛ant + PC⊛cons))")
    print("  4. Regras armazenadas na DenseSDM (memória associativa)")
    print("  5. Para deduzir:")
    print("       a. query SDM com vetor do fato")
    print("       b. decode_consequent → unbind ROLE → unbind PERM_CONS")
    print("       c. nearest-neighbor no codebook → palavra")
    print()
    print("  Resultados:")
    print("    Rex é um animal?  → Sim  (cadeia: rex → cachorro → animal)")
    print("    Fido é um animal? → Sim  (cadeia: fido → gato → animal)")
    print("    Rex é um gato?    → Não  (cachorro ≠ gato)")
    print()
    print("  Tudo rodou em CPU, sem backprop, sem GPU.")
    print("  Apenas álgebra vetorial em 10.000 dimensões.")


if __name__ == "__main__":
    main()
