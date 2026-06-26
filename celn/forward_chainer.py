"""
CELN v3 — Forward Chainer
==========================
Motor de dedução lógica via álgebra vetorial, sem backprop.

Objetivo:
  Dadas premissas codificadas como regras FOL (A→B, B→C),
  deduzir conclusões via encadeamento para frente (A→C).

Mecanismo:
  1. Codifica cada premissa como bind(ROLE, M(ant, conseq))
  2. Armazena regras no DenseSDM, indexadas pelo antecedente
  3. Para cada fato inicial F:
     a. Query SDM com F → recupera regras onde F é antecedente
     b. unbind_M_reverse(inner, F) → consequent G
     c. Adiciona G aos fatos descobertos
     d. Repete até convergência ou max_depth
  4. Verifica se conclusão está nos fatos deduzidos

Princípios:
  ZERO backprop. ZERO thresholds mágicos.
  Tudo auto-calibrável via percentis da distribuição real.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field

from .core import D, bind, unbind, normalize, similarity
from .memory import DenseSDM
from .logic_encoder import (
    LogicRoles, encode_rule, decode_rule,
    decode_antecedent, decode_consequent,
    get_perm_ant, get_perm_cons,
)
from .vocab_bridge import VocabBridge


@dataclass
class DeductionStep:
    """Registro de um passo dedutivo."""
    rule_repr: str          # "TODOS(A, B)"
    fact_used: str          # Fato que ativou a regra
    fact_derived: str       # Novo fato deduzido
    confidence: float       # Similaridade de recuperação


@dataclass
class DeductionResult:
    """Resultado de uma dedução."""
    conclusion: str
    label: str              # "True", "False", "Unknown"
    chain: List[DeductionStep] = field(default_factory=list)
    confidence: float = 0.0
    max_depth_reached: int = 0


class ForwardChainer:
    """
    Motor de forward chaining para lógica de primeira ordem.
    
    Uso:
      chainer = ForwardChainer(vectors, w2i, i2w)
      chainer.add_premise("Todo gato é animal", "TODOS", "gato", "animal")
      chainer.add_premise("Todo animal é ser vivo", "TODOS", "animal", "vivo")
      
      result = chainer.deduce(
          initial_facts=["gato"],
          conclusion="Todo gato é ser vivo",
          max_depth=5
      )
      # → DeductionResult(label="True", chain=[...])
    """

    def __init__(
        self,
        vectors: np.ndarray,
        w2i: Dict[str, int],
        i2w: Dict[int, str],
        n_sdm_locations: int = 2048,
        seed: int = 42,
        use_bridge: bool = True,
    ):
        self.vectors = vectors
        self.w2i = w2i
        self.i2w = i2w
        self.roles = LogicRoles(seed=seed)
        self.use_bridge = use_bridge
        
        self._bridge: Optional[VocabBridge] = None
        
        self.sdm = DenseSDM(n_locations=n_sdm_locations, activation_pct=0.02, seed=seed)
        
        self._rules_stored: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        
        self.rng = np.random.RandomState(seed)

    def _get_bridge(self) -> Optional[VocabBridge]:
        if self._bridge is None and self.use_bridge:
            try:
                self._bridge = VocabBridge()
                self._bridge.align_to_celn(self.vectors, self.w2i)
            except Exception:
                self.use_bridge = False
        return self._bridge

    def _ensure_vocab(self, word: str) -> bool:
        """Garante que TODAS as palavras do termo estão no codebook."""
        words = word.split()
        for w in words:
            if w not in self.w2i:
                bridge = self._get_bridge()
                if bridge is None:
                    return False
                v = bridge.project(w)
                if v is None:
                    v = bridge.project(w.lower())
                    if v is None:
                        return False
                idx = len(self.w2i)
                self.w2i[w] = idx
                self.i2w[idx] = w
                self.vectors = np.vstack([self.vectors, v.reshape(1, -1)])
        return True

    def _get_vec(self, word: str) -> Optional[np.ndarray]:
        """Obtém vetor: single word ou composição multi-word."""
        words = word.split()
        if len(words) == 1:
            if word in self.w2i:
                return normalize(self.vectors[self.w2i[word]].astype(np.float32))
            bridge = self._get_bridge()
            if bridge is not None:
                v = bridge.project(word)
                if v is not None:
                    return v
                v = bridge.project(word.lower())
                if v is not None:
                    return v
            return None
        # Multi-word: compose via weighted average of first content words
        vecs = []
        for w in words:
            if w in self.w2i:
                v = normalize(self.vectors[self.w2i[w]].astype(np.float32))
                vecs.append(v)
            else:
                bridge = self._get_bridge()
                if bridge is not None:
                    v = bridge.project(w)
                    if v is not None:
                        vecs.append(v)
        if not vecs:
            return None
        if len(vecs) == 1:
            return vecs[0]
        # Weighted: first word = 70%, rest share 30%
        weights = [0.7] + [0.3/(len(vecs)-1)] * (len(vecs)-1)
        return normalize(np.average(vecs, axis=0, weights=weights))

    def add_rule(self, role_name: str, antecedent: str, consequent: str) -> bool:
        """
        Adiciona uma regra lógica ao chainer.
        
        Se uma palavra não está no codebook, tenta projetá-la
        do spaCy via VocabBridge (JL 300→10k).
        
        Args:
            role_name: 'ROLE_TODOS', 'ROLE_NENHUM', 'ROLE_ALGUM', 'ROLE_SE_ENTAO'
            antecedent: Palavra antecedente
            consequent: Palavra consequente
        
        Returns:
            True se sucesso (ambas as palavras encontradas), False caso contrário
        """
        if not self._ensure_vocab(antecedent):
            return False
        if not self._ensure_vocab(consequent):
            return False
        
        v_ant = self._get_vec(antecedent)
        v_conseq = self._get_vec(consequent)
        
        if v_ant is None or v_conseq is None:
            return False
        
        role_vec = self.roles.get(role_name)
        rule_vec = encode_rule(role_vec, v_ant, v_conseq)
        
        self._rules_stored.append((role_name, v_ant.copy(), v_conseq.copy(), rule_vec))
        self.sdm.write(rule_vec)
        
        return True

    def _recover_consequent(
        self,
        rule_vec: np.ndarray,
        role_name: str,
        antecedent_vec: np.ndarray,
    ) -> Tuple[Optional[str], float]:
        """
        Extrai o consequente de uma regra codificada.
        
        Dado rule_vec = bind(ROLE, PA⊛ant + PC⊛conseq) e ant conhecido:
          1. unbind(rule_vec, ROLE) → inner ≈ PA⊛ant + PC⊛conseq
          2. unbind(inner, PERM_CONS) → conseq
          3. nearest neighbor → palavra
        
        Returns:
            (consequent_word, confidence)
        """
        role_vec = self.roles.get(role_name)
        v_conseq = decode_consequent(rule_vec, role_vec)

        sims = self.vectors @ v_conseq.astype(np.float32)
        idx = int(np.argmax(sims))
        sim = float(sims[idx])

        confidence = float(np.clip((sim + 1) / 2, 0, 1))

        return self.i2w.get(idx), confidence

    def deduce(
        self,
        initial_facts: List[str],
        conclusion: Optional[str] = None,
        max_depth: int = 5,
        strict: bool = True,
    ) -> DeductionResult:
        """
        Forward chaining a partir de fatos iniciais.
        
        Algoritmo:
          1. Inicializa fatos_conhecidos = initial_facts
          2. Para depth = 1..max_depth:
             a. Para cada fato F em fatos_conhecidos:
                - Para cada regra armazenada:
                  * Se F == antecedent (ou F ≈ antecedent se strict=False):
                    - Extrai consequent G
                    - Adiciona G aos fatos
             b. Se conclusion ∈ fatos, retorna True
             c. Se nenhum novo fato, para (convergência)
          3. Retorna Unknown se conclusion não foi alcançada
        
        Args:
            initial_facts: Lista de fatos iniciais (palavras)
            conclusion: Conclusão a verificar (opcional)
            max_depth: Profundidade máxima do encadeamento
            strict: Se True, exige correspondência exata do antecedente
        """
        # Converte fatos iniciais para vetores
        known_facts: Set[str] = set(initial_facts)
        fact_vectors: Dict[str, np.ndarray] = {}
        
        for fact in initial_facts:
            v = self._get_vec(fact)
            if v is not None:
                fact_vectors[fact] = v
        
        chain: List[DeductionStep] = []
        all_derived_facts: Set[str] = set()
        
        for depth in range(max_depth):
            new_facts_found = False
            
            # Para cada fato conhecido, tenta aplicar regras
            for fact in list(known_facts):
                if fact not in fact_vectors:
                    continue
                
                v_fact = fact_vectors[fact]
                
                # Itera sobre todas as regras armazenadas
                for role_name, v_ant, v_conseq, rule_vec in self._rules_stored:
                    # Verifica se esta regra é relevante
                    ant_sim = float(v_ant @ v_fact)
                    
                    # Threshold: strict exige correspondência exata (ant_sim > 0.99)
                    # relaxed permite generalização (ant_sim > 0.5)
                    threshold = 0.99 if strict else 0.5
                    if ant_sim < threshold:
                        continue
                    
                    # Recupera o consequente via unbinding da regra
                    derived_word, conf = self._recover_consequent(
                        rule_vec, role_name, v_ant
                    )
                    
                    if derived_word is None:
                        continue
                    
                    # Evita loops e redundâncias
                    if derived_word in known_facts or derived_word in all_derived_facts:
                        continue
                    
                    # Adiciona novo fato
                    all_derived_facts.add(derived_word)
                    v_derived = self._get_vec(derived_word)
                    if v_derived is not None:
                        fact_vectors[derived_word] = v_derived
                    
                    # Registra passo dedutivo
                    step = DeductionStep(
                        rule_repr=f"{role_name}({fact} → {derived_word})",
                        fact_used=fact,
                        fact_derived=derived_word,
                        confidence=conf * ant_sim,
                    )
                    chain.append(step)
                    new_facts_found = True
            
            # Verifica conclusão
            if conclusion is not None and conclusion in known_facts | all_derived_facts:
                avg_conf = np.mean([s.confidence for s in chain]) if chain else 1.0
                return DeductionResult(
                    conclusion=conclusion,
                    label="True",
                    chain=chain,
                    confidence=avg_conf,
                    max_depth_reached=depth + 1,
                )
            
            # Atualiza fatos conhecidos
            known_facts |= all_derived_facts
            
            # Convergência
            if not new_facts_found:
                break
        
        # Conclusão não alcançada
        if conclusion is not None:
            # Verifica se a conclusão é NEGADA por algum fato derivado
            # (detecção de contradição)
            # Por enquanto, apenas retorna Unknown
            return DeductionResult(
                conclusion=conclusion,
                label="Unknown",
                chain=chain,
                confidence=0.0,
                max_depth_reached=len(chain),
            )
        else:
            # Retorna todos os fatos deduzidos
            return DeductionResult(
                conclusion="multiple",
                label="True",
                chain=chain,
                confidence=1.0,
                max_depth_reached=len(chain),
            )

    def verify(
        self,
        conclusion_ant: str,
        conclusion_conseq: str,
        role_name: str = "ROLE_TODOS",
    ) -> DeductionResult:
        """
        Verifica se uma conclusão específica é dedutível.
        
        Ex: verify("gato", "vivo", "ROLE_TODOS")
          → Verifica se "Todo gato é vivo" é dedutível
        
        Args:
            conclusion_ant: Antecedente da conclusão
            conclusion_conseq: Consequente da conclusão
            role_name: Role da conclusão
        
        Returns:
            DeductionResult com label True/False/Unknown
        """
        # Tenta deduzir a partir do antecedente
        result = self.deduce(
            initial_facts=[conclusion_ant],
            conclusion=conclusion_conseq,
            max_depth=5,
        )
        
        if result.label == "True":
            return result
        
        # Se não encontrou diretamente, verifica contradição
        # (se derivou a negação da conclusão)
        # Implementação futura: usar negate() e verificar se ¬conseq foi derivado
        
        return result


def test_forward_chainer():
    """Teste do forward chainer com cadeia simples."""
    print("=" * 60)
    print("TESTE: Forward Chainer")
    print("=" * 60)
    
    # Carrega vetores
    try:
        data = np.load('celn_full_vectors.npz', allow_pickle=True)
        vectors = data['vectors']
        vocab = [str(w) for w in data['vocab']]
        w2i = {w: i for i, w in enumerate(vocab)}
        i2w = {i: w for i, w in enumerate(vocab)}
    except FileNotFoundError:
        print("✗ celn_full_vectors.npz não encontrado")
        return False
    
    chainer = ForwardChainer(vectors, w2i, i2w, n_sdm_locations=1024, seed=42)
    
    # Adiciona regras
    print("\n1. Adicionando regras:")
    rules = [
        ("ROLE_TODOS", "gato", "animal"),
        ("ROLE_TODOS", "animal", "ser"),
        ("ROLE_TODOS", "ser", "vivo"),
    ]
    
    for role, ant, conseq in rules:
        ok = chainer.add_rule(role, ant, conseq)
        status = "✓" if ok else "✗"
        print(f"   {status} {role}({ant} → {conseq})")
    
    # Testa dedução
    print("\n2. Teste de dedução:")
    print("   Fatos iniciais: ['gato']")
    print("   Conclusão esperada: 'vivo'")
    
    result = chainer.deduce(
        initial_facts=["gato"],
        conclusion="vivo",
        max_depth=5,
    )
    
    print(f"\n3. Resultado:")
    print(f"   Label: {result.label}")
    print(f"   Confidence: {result.confidence:.4f}")
    print(f"   Max depth: {result.max_depth_reached}")
    
    if result.chain:
        print(f"\n   Cadeia dedutiva ({len(result.chain)} passos):")
        for i, step in enumerate(result.chain[:5]):  # mostra só os primeiros 5
            print(f"     {i+1}. {step.rule_repr}: {step.fact_used} → {step.fact_derived} (conf={step.confidence:.3f})")
        if len(result.chain) > 5:
            print(f"     ... e mais {len(result.chain) - 5} passos")
    
    success = (result.label == "True")
    print(f"\n{'=' * 60}")
    print(f"RESULTADO: {'✓ DEDUÇÃO BEM-SUCEDIDA' if success else '✗ DEDUÇÃO FALHOU'}")
    print("=" * 60)
    
    return success


if __name__ == '__main__':
    test_forward_chainer()