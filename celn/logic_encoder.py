"""
CELN v3 — Logic Encoder v3
===========================
Codificação de regras lógicas de primeira ordem (FOL) como vetores de 10k dims.

Estrutura:
  composite = bind(ROLE, PERM_ANT ⊛ ant + PERM_CONS ⊛ conseq)

Vetores de permutação PERM_ANT e PERM_CONS são unitários quasi-ortogonais.
O inner é uma SUPERPOSIÇÃO de position-tagged bindings:

  inner = PERM_ANT ⊛ ant + PERM_CONS ⊛ conseq

Cada tag define um "slot" espectralmente separado do outro.
unbind(inner, PERM_ANT) ≈ ant  (com ruído ~1/√D do crosstalk)
unbind(inner, PERM_CONS) ≈ conseq

A ORDEM é determinística pela PERM, não por frágil diferença de similaridade.
PERM_ANT sempre marca o antecedente, PERM_CONS sempre marca o consequente.
Nunca há ambiguidade de ordem.

Comparado com v1 (M(ant,conseq)):
  - v1: M(A,B) vs M(B,A) difere ~1% — decode troca ordem com frequência
  - v2: M(PA⊛A, PC⊛B) — PERM dentro de M, ainda ~1% diff, não resolve
  - v3: PA⊛A + PC⊛B — ORDEM é estrutural, zero ambiguidade

Princípios:
  ZERO backprop/transformers. ZERO templates. ZERO thresholds mágicos.
  Tudo auto-calibrável. 100% álgebra vetorial.
"""

import numpy as np
from typing import Tuple, Optional, Dict

from .core import D, bind, unbind, normalize
from .port_adapter import make_unitary_vector


PERM_SEED_ANT = 1701
PERM_SEED_CONS = 1702

_PERM_ANT: Optional[np.ndarray] = None
_PERM_CONS: Optional[np.ndarray] = None


def get_perm_ant() -> np.ndarray:
    global _PERM_ANT
    if _PERM_ANT is None:
        _PERM_ANT = make_unitary_vector(D, np.random.RandomState(PERM_SEED_ANT))
    return _PERM_ANT


def get_perm_cons() -> np.ndarray:
    global _PERM_CONS
    if _PERM_CONS is None:
        _PERM_CONS = make_unitary_vector(D, np.random.RandomState(PERM_SEED_CONS))
    return _PERM_CONS


class LogicRoles:
    """
    Vetores de papel (ROLE) para operadores lógicos FOL.
    
    Cada ROLE é um vetor unitário quasi-ortogonal (FFT de magnitude 1).
    Quasi-ortogonalidade garante que bindings com roles diferentes
    não interferem no unbinding.
    """

    ROLE_NAMES = [
        'ROLE_TODOS',       # ∀ (universal affirmative)
        'ROLE_NENHUM',      # ∀x (P(x) → ¬Q(x))
        'ROLE_ALGUM',       # ∃ (existential)
        'ROLE_SE_ENTAO',    # → (material conditional)
        'ROLE_NEGACAO',     # ¬ (negation)
    ]

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self._roles: Dict[str, np.ndarray] = {}
        self._generate_roles()

    def _generate_roles(self):
        for name in self.ROLE_NAMES:
            self._roles[name] = make_unitary_vector(D, self.rng)

    def get(self, name: str) -> np.ndarray:
        if name not in self._roles:
            raise ValueError(f"ROLE desconhecido: {name}. Válidos: {list(self._roles.keys())}")
        return self._roles[name].copy()

    @property
    def TODOS(self) -> np.ndarray:
        return self.get('ROLE_TODOS')

    @property
    def NENHUM(self) -> np.ndarray:
        return self.get('ROLE_NENHUM')

    @property
    def ALGUM(self) -> np.ndarray:
        return self.get('ROLE_ALGUM')

    @property
    def SE_ENTAO(self) -> np.ndarray:
        return self.get('ROLE_SE_ENTAO')

    @property
    def NEGACAO(self) -> np.ndarray:
        return self.get('ROLE_NEGACAO')

    def verify_orthogonality(self) -> Dict[str, float]:
        similarities = {}
        names = self.ROLE_NAMES
        for i, n1 in enumerate(names):
            for n2 in names[i+1:]:
                sim = float(self._roles[n1] @ self._roles[n2])
                similarities[f"{n1}·{n2}"] = sim
        return similarities


def encode_rule(
    role: np.ndarray,
    antecedent: np.ndarray,
    consequent: np.ndarray,
) -> np.ndarray:
    """
    Codifica uma regra lógica como vetor composto.

    Estrutura:
      composite = bind(ROLE, normalize(PERM_ANT ⊛ ant + PERM_CONS ⊛ conseq))

    Inner = superposição de position-tagged bindings.
    PERM_ANT marca a POSIÇÃO de antecedent.
    PERM_CONS marca a POSIÇÃO de consequent.
    Cada slot é espectralmente distinto por construção (permutação unitária).

    Args:
        role: Vetor ROLE (TODOS, NENHUM, ALGUM, SE_ENTAO, NEGACAO)
        antecedent: Vetor do antecedente
        consequent: Vetor do consequente
    """
    pa = get_perm_ant()
    pc = get_perm_cons()
    inner = normalize(bind(pa, antecedent) + bind(pc, consequent))
    return bind(role, inner)


def decode_antecedent(composite: np.ndarray, role_vec: np.ndarray) -> np.ndarray:
    """Extrai o vetor do antecedente de uma regra codificada.
    
    unbind(composite, ROLE) → inner
    unbind(inner, PERM_ANT) → antecedent (com crosstalk ~1/√D)
    """
    inner = unbind(composite, role_vec)
    pa = get_perm_ant()
    return normalize(unbind(inner, pa))


def decode_consequent(composite: np.ndarray, role_vec: np.ndarray) -> np.ndarray:
    """Extrai o vetor do consequente de uma regra codificada.
    
    unbind(composite, ROLE) → inner
    unbind(inner, PERM_CONS) → consequent (com crosstalk ~1/√D)
    """
    inner = unbind(composite, role_vec)
    pc = get_perm_cons()
    return normalize(unbind(inner, pc))


def decode_rule(
    composite: np.ndarray,
    roles: LogicRoles,
    codebook: np.ndarray,
    w2i: Dict[str, int],
    i2w: Dict[int, str],
    top_k: int = 30,
) -> Tuple[Optional[str], Optional[str], Optional[str], Dict]:
    """
    Decodifica uma regra lógica em (role, antecedent, consequent).

    Algoritmo:
      1. Para cada role R, unbind(composite, R) → inner
      2. unbind(inner, PERM_ANT) → candidate_ant; nearest neighbor → palavra
      3. unbind(inner, PERM_CONS) → candidate_conseq; nearest neighbor → palavra
      4. O role com melhor similaridade global (média ant+conseq) vence

    A ordem é DETERMINÍSTICA pela PERM — nunca há ambiguidade.

    Args:
        composite: Vetor composto da regra
        roles: Instância LogicRoles
        codebook: Matriz (V, D)
        w2i, i2w: Mapeamentos palavra↔índice
        top_k: Número de candidatos (não usado na v3 — decode é direto)

    Returns:
        (role_name, antecedent_word, consequent_word, metadata_dict)
    """
    pa = get_perm_ant()
    pc = get_perm_cons()

    best_role = None
    best_ant = None
    best_conseq = None
    best_global_score = -np.inf

    for role_name in roles.ROLE_NAMES:
        role_vec = roles.get(role_name)
        inner = unbind(composite, role_vec)

        v_ant = normalize(unbind(inner, pa))
        v_conseq = normalize(unbind(inner, pc))

        sims_ant = codebook @ v_ant.astype(np.float32)
        sims_conseq = codebook @ v_conseq.astype(np.float32)

        idx_ant = int(np.argmax(sims_ant))
        idx_conseq = int(np.argmax(sims_conseq))
        sim_ant = float(sims_ant[idx_ant])
        sim_conseq = float(sims_conseq[idx_conseq])

        avg_sim = (sim_ant + sim_conseq) / 2.0

        if avg_sim > best_global_score:
            best_global_score = avg_sim
            best_role = role_name
            best_ant = i2w.get(idx_ant)
            best_conseq = i2w.get(idx_conseq)

    if best_role is None:
        return None, None, None, {
            'success': False,
            'error': 'Falha na decodificação — nenhum role produziu par válido',
        }

    return best_role, best_ant, best_conseq, {
        'success': True,
        'role_name': best_role,
        'antecedent_word': best_ant,
        'consequent_word': best_conseq,
        'ant_sim': float(sims_ant[idx_ant]) if 'sims_ant' in dir() else 0,
        'conseq_sim': float(sims_conseq[idx_conseq]) if 'sims_conseq' in dir() else 0,
        'reconstruction_sim': best_global_score,
    }


def negate(concept: np.ndarray) -> np.ndarray:
    """¬P = −P (reflexão antipódica). cosine_sim(P, ¬P) = −1.0."""
    return normalize(-concept)


def test_logic_encoder():
    print("=" * 60)
    print("TESTE: Logic Encoder v3 (Superposição + Permutação)")
    print("=" * 60)

    roles = LogicRoles(seed=42)
    print("\n1. ROLE vectors:")
    print(f"   {roles.ROLE_NAMES}")

    print("\n2. Quasi-ortogonalidade:")
    sims = roles.verify_orthogonality()
    max_sim = max(abs(v) for v in sims.values())
    for pair, sim in sims.items():
        status = "✓" if abs(sim) < 0.3 else "⚠"
        print(f"   {status} {pair}: {sim:.4f}")
    print(f"   Max |sim|: {max_sim:.4f} {'✓' if max_sim < 0.05 else '⚠'}")

    pa = get_perm_ant()
    pc = get_perm_cons()
    perm_sim = float(pa @ pc)
    print(f"\n3. Ortogonalidade PERM_ANT vs PERM_CONS: {perm_sim:.4f}")

    try:
        data = np.load('celn_full_vectors.npz', allow_pickle=True)
        vocab = [str(w) for w in data['vocab']]
        vectors = data['vectors']
        w2i = {w: i for i, w in enumerate(vocab)}
        i2w = {i: w for i, w in enumerate(vocab)}
    except FileNotFoundError:
        print("   ✗ celn_full_vectors.npz não encontrado")
        return False

    print("\n4. Teste de encode/decode (superposição + permutação):")

    test_cases = [
        ('gato', 'animal'),
        ('cachorro', 'animal'),
        ('aluno', 'escola'),
        ('professor', 'escola'),
        ('peixe', 'animal'),
    ]

    all_passed = True
    for ant, conseq in test_cases:
        if ant not in w2i or conseq not in w2i:
            print(f"   ⊘ {ant}/{conseq} — fora do vocab")
            continue

        v_ant = normalize(vectors[w2i[ant]].astype(np.float32))
        v_conseq = normalize(vectors[w2i[conseq]].astype(np.float32))

        rule = encode_rule(roles.TODOS, v_ant, v_conseq)
        decoded_role, decoded_ant, decoded_conseq, meta = decode_rule(
            rule, roles, vectors, w2i, i2w
        )

        ok = (decoded_role == 'ROLE_TODOS' and
              decoded_ant == ant and
              decoded_conseq == conseq)
        all_passed &= ok

        print(f"   {'✓' if ok else '✗'} TODOS({ant} → {conseq}): "
              f"decoded=({decoded_role}, {decoded_ant} → {decoded_conseq}) "
              f"avg_sim={meta['reconstruction_sim']:.4f}")

    print("\n5. Ordem é determinística (teste de swap):")
    for ant, conseq in test_cases[:3]:
        if ant not in w2i or conseq not in w2i:
            continue
        v_a = normalize(vectors[w2i[ant]].astype(np.float32))
        v_c = normalize(vectors[w2i[conseq]].astype(np.float32))

        rule_ab = encode_rule(roles.TODOS, v_a, v_c)
        rule_ba = encode_rule(roles.TODOS, v_c, v_a)

        _, ant_ab, conseq_ab, _ = decode_rule(rule_ab, roles, vectors, w2i, i2w)
        _, ant_ba, conseq_ba, _ = decode_rule(rule_ba, roles, vectors, w2i, i2w)

        ab_ok = (ant_ab == ant and conseq_ab == conseq)
        ba_ok = (ant_ba == conseq and conseq_ba == ant)
        print(f"   {ant}→{conseq}: AB=({ant_ab}→{conseq_ab}) {'✓' if ab_ok else '✗'}, "
              f"BA=({ant_ba}→{conseq_ba}) {'✓' if ba_ok else '✗'}")

    print("\n6. Teste de negação:")
    v_gato = normalize(vectors[w2i['gato']].astype(np.float32))
    v_not_gato = negate(v_gato)
    sim = float(v_gato @ v_not_gato)
    print(f"   cos(gato, neg(gato)) = {sim:.4f} {'✓' if sim < -0.99 else '⚠'}")

    print("\n" + "=" * 60)
    print(f"RESULTADO: {'✓ TODOS PASSARAM' if all_passed else '⚠ ALGUNS FALHARAM'}")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    test_logic_encoder()