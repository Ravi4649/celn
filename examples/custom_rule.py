#!/usr/bin/env python3
"""
CELN — Custom Rule Example
===========================

Define your own logical rules, encode them as 10k-d vectors,
store them in associative memory, and run forward-chaining deduction.

    pip install -r requirements.txt
    python examples/custom_rule.py
"""

import sys, os, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from celn.core import D, normalize, similarity, bind, unbind
from celn.logic_encoder import LogicRoles, encode_rule, decode_consequent

np.set_printoptions(precision=4, suppress=True, threshold=6, linewidth=80)

_SEP = "\n" + "─" * 72 + "\n"


# ── Hash-based word vectors (no .npz needed) ─────────────────────

def word_vec(word: str) -> np.ndarray:
    seed = int(hashlib.sha256(word.lower().encode()).hexdigest()[:8], 16)
    return normalize(np.random.RandomState(seed).randn(D).astype(np.float32))


def nearest(vec: np.ndarray, lookup: dict[str, np.ndarray]) -> tuple[str, float]:
    best_w, best_s = "", -1.0
    for w, v in lookup.items():
        s = similarity(vec, v)
        if s > best_s:
            best_s, best_w = s, w
    return best_w, best_s


def show(label: str, v: np.ndarray):
    mag = np.abs(np.fft.fft(v))
    head = ", ".join(f"{x:.4f}" for x in v[:4])
    tail = ", ".join(f"{x:.4f}" for x in v[-2:])
    print(f"  {label}")
    print(f"    ‖v‖={np.linalg.norm(v):.4f}  |FFT|_med={np.median(mag):.2f}  "
          f"|FFT|_max={mag.max():.2f}")
    print(f"    [{head}  ...  {tail}]")


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("\n╔" + "═" * 70 + "╗")
    print("║   CELN — CUSTOM RULE EXAMPLE                          ║")
    print("║   Define, encode, store, and query your own rules.    ║")
    print("╚" + "═" * 70 + "╝")

    # ── 1. Define your vocabulary ─────────────────────────────────
    print(_SEP + "  STEP 1: Define vocabulary (your domain)\n" + _SEP)

    words = ["sparrow", "eagle", "robin", "bat", "worm",
             "bird", "mammal", "flies", "swims", "eats_insects"]
    vectors = {w: word_vec(w) for w in words}

    print("  Every word is a deterministic 10k-d vector:\n")
    for w in words:
        show(f"  {w:>15}", vectors[w])
    print()

    # ── 2. Create logic roles ─────────────────────────────────────
    print(_SEP + "  STEP 2: Create logic roles (∀ = ROLE_TODOS)\n" + _SEP)

    roles = LogicRoles(seed=42)
    todos = roles.get("ROLE_TODOS")

    show("  ROLE_TODOS (∀)", todos)
    print()

    # ── 3. Encode YOUR rules as vectors ──────────────────────────
    print(_SEP + "  STEP 3: Encode your custom rules\n" + _SEP)

    rules = [
        ("ROLE_TODOS", "sparrow",       "bird",          "Every sparrow is a bird."),
        ("ROLE_TODOS", "eagle",         "bird",          "Every eagle is a bird."),
        ("ROLE_TODOS", "robin",         "bird",          "Every robin is a bird."),
        ("ROLE_TODOS", "bat",           "mammal",        "Every bat is a mammal."),
        ("ROLE_TODOS", "worm",          "eats_insects",  "Every worm eats insects."),
        ("ROLE_TODOS", "bird",          "flies",         "Every bird flies."),
        ("ROLE_TODOS", "mammal",        "flies",         "Every mammal flies."),
    ]

    encoded = []
    for role_name, ant_word, cons_word, text in rules:
        print(f'  Rule: "{text}"')
        v_ant = vectors[ant_word]
        v_cons = vectors[cons_word]
        composite = encode_rule(todos, v_ant, v_cons)
        sim = similarity(v_ant, decode_consequent(composite, todos))
        print(f"    antecedent→consequent round-trip sim: {sim:.4f}")
        encoded.append((ant_word, cons_word, composite, text))
    print()

    # ── 4. Store rules in associative memory ─────────────────────
    print(_SEP + "  STEP 4: Store rules in DenseSDM\n" + _SEP)

    from celn.memory import DenseSDM

    seed_vecs = np.stack(list(vectors.values()))
    sdm = DenseSDM(n_locations=128, activation_pct=0.1, seed=42)
    sdm.initialize_addresses(seed_vecs)

    for ant_word, cons_word, composite, text in encoded:
        n = sdm.write(composite)
        rt = similarity(composite, sdm.read(composite))
        print(f'  ✓ Stored: "{text}"  ({n} locations, round-trip sim={rt:.4f})')
    print()

    # ── 5. Query ─────────────────────────────────────────────────
    print(_SEP + "  STEP 5: Deduction — forward chaining\n" + _SEP)

    queries = [
        ("robin",  "flies",  "Does a robin fly?"),
        ("bat",     "flies",  "Does a bat fly?"),
        ("worm",    "bird",   "Is a worm a bird?"),
    ]

    for fact, target, question in queries:
        print(f"  Query: {question}")
        print(f"  Goal:  {fact} → ... → {target}")
        print("  " + "·" * 60)

        chain = []
        current = fact
        reached = False

        for _ in range(10):
            v_current = vectors[current]

            best_sim = -1.0
            best_rule = None
            for ant, cons, comp, _ in encoded:
                s = similarity(v_current, vectors[ant])
                if s > best_sim:
                    best_sim = s
                    best_rule = (ant, cons, comp)

            if best_rule is None or best_sim < 0.5:
                break

            ant, cons, comp = best_rule
            v_derived = decode_consequent(comp, todos)
            derived, conf = nearest(v_derived, vectors)

            chain.append(f"    [{len(chain)+1}] {current} → {derived}  (cos={conf:.3f})")
            if derived == target:
                reached = True
                break
            if derived == current:
                break
            current = derived

        print()
        if reached:
            print("  ═══════════════════════════════════════════════")
            print(f'  ► YES: "{fact}" IS a "{target}".')
            print("  ═══════════════════════════════════════════════")
        elif not chain:
            print("  ═══════════════════════════════════════════════")
            print(f'  ► UNKNOWN: no matching rule for "{fact}".')
            print("  ═══════════════════════════════════════════════")
        else:
            print("  ═══════════════════════════════════════════════")
            last = chain[-1].split("→")[-1].strip().split()[0]
            print(f'  ► UNKNOWN: chain stopped at "{last}" (not "{target}").')
            print("  ═══════════════════════════════════════════════")
        print()
        for s in chain:
            print(s)
        print()

    # ── Summary ─────────────────────────────────────────────────
    print(_SEP)
    print("  How to create your own rules:")
    print(_SEP)
    print("  1. Define your words → hash-based vectors")
    print("  2. Create LogicRoles, pick ROLE_TODOS (∀)")
    print("  3. Call encode_rule(TODOS, antecedent_vec, consequent_vec)")
    print("  4. Store in DenseSDM (or keep in a list)")
    print("  5. Query: decode_consequent(rule, TODOS) → nearest word")
    print()
    print("  Edit the 'words' and 'rules' lists above to")
    print("  adapt this example to your own domain.\n")


if __name__ == "__main__":
    main()
