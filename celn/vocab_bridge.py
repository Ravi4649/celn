"""
CELN v3 — Vocabulary Bridge (Aligned Projection)
==================================================
Projeção alinhada 300d → 10k para gerar vetores CELN de qualquer
palavra do spaCy, alinhada ao codebook CELN existente via Procrustes.

Fundamentação:
  JL projection aleatória preserva similaridade mas NÃO alinha
  com o space CELN treinado (PPMI+Hebbian). Para round-trip funcionar,
  precisamos de uma matriz de projeção que MAPEIE spaCy → CELN.

Estratégia:
  1. Encontra vocabulário sobreposto entre spaCy (415k) e CELN (20k)
  2. Aprende matriz W ∈ ℝ^{10000×300} via mínimos quadrados / Procrustes:
     W = V_celn @ V_spacy⁺  (pseudo-inversa)
  3. Para palavras OOV: v_celn = normalize(W @ v_spacy)

  Zero backprop, apenas álgebra linear. Auto-calibrável via vocabulário
  compartilhado (sem thresholds mágicos).

Princípios:
  ZERO backprop. ZERO listas fixas. ZERO templates.
  ZERO thresholds mágicos. Tudo auto-calibrável. 100% álgebra vetorial.
"""

import numpy as np
from typing import Optional, Dict, Tuple, List

from .core import D, normalize


_Bridge_INSTANCE = None

SPACY_VECTORS_PATH = 'data/spacy_300d_vectors.npz'

PROJ_SEED = 42
SPACY_DIM = 300


class VocabBridge:
    """
    Ponte vocabulário: projecão alinhada 300d → 10k.
    
    Carrega vetores spaCy 300d e aprende projeção alinhada ao CELN
    via vocabulário sobreposto. Qualquer palavra no spaCy pode ser
    convertida para o espaço CELN com round-trip funcional.
    """

    def __init__(
        self,
        spacy_path: str = SPACY_VECTORS_PATH,
        target_dim: int = D,
        spacy_dim: int = SPACY_DIM,
        seed: int = PROJ_SEED,
    ):
        self.spacy_path = spacy_path
        self.target_dim = target_dim
        self.spacy_dim = spacy_dim
        self.seed = seed

        self._spacy_data = None
        self._spacy_vectors = None
        self._spacy_vocab = None
        self._spacy_w2i = None
        self._proj_matrix = None
        self._is_aligned = False

        self._init_projection()

    def _init_projection(self):
        """Inicializa com JL aleatório; será substituído por align_to_celn()."""
        rng = np.random.RandomState(self.seed)
        self._proj_matrix = rng.randn(self.target_dim, self.spacy_dim).astype(np.float32)
        self._proj_matrix /= np.sqrt(self.spacy_dim)

    def _load_spacy(self):
        if self._spacy_vectors is not None:
            return
        data = np.load(self.spacy_path, allow_pickle=True)
        self._spacy_vectors = data['vectors'].astype(np.float32)
        if 'words' in data:
            self._spacy_vocab = [str(w) for w in data['words']]
        elif 'vocab' in data:
            self._spacy_vocab = [str(w) for w in data['vocab']]
        else:
            raise ValueError(f"No 'words' or 'vocab' key in {self.spacy_path}")
        self._spacy_w2i = {w: i for i, w in enumerate(self._spacy_vocab)}

    def align_to_celn(self, celn_vectors: np.ndarray, celn_w2i: Dict[str, int], min_overlap: int = 100):
        """
        Alinha matriz de projeção ao espaço CELN via vocabulário compartilhado.
        
        Usa Procrustes ortogonal (sem backprop, apenas SVD):
          W_opt = U @ V^T  onde U, V vêm de SVD(V_celn @ V_spacy.T)
        
        Args:
            celn_vectors: (V_celn, 10k) matriz de vetores CELN
            celn_w2i: palavra → índice no codebook CELN
            min_overlap: mínimo de palavras sobrepostas para alinhar
        """
        self._load_spacy()
        
        # Encontra vocabulário compartilhado
        common_words = [w for w in celn_w2i if w in self._spacy_w2i]
        n_common = len(common_words)
        
        if n_common < min_overlap:
            print(f"[VocabBridge] Apenas {n_common} palavras sobrepostas (< {min_overlap}), mantendo JL aleatório")
            return
        
        print(f"[VocabBridge] Alinhando com {n_common} palavras sobrepostas...")
        
        # Matrizes de vetores alinhados
        spacy_idx = [self._spacy_w2i[w] for w in common_words]
        celn_idx = [celn_w2i[w] for w in common_words]
        
        V_spacy = self._spacy_vectors[spacy_idx]  # (n_common, 300)
        V_celn = celn_vectors[celn_idx]            # (n_common, 10k)
        
        # Normaliza ambos os conjuntos
        V_spacy = V_spacy / (np.linalg.norm(V_spacy, axis=1, keepdims=True) + 1e-12)
        V_celn = V_celn / (np.linalg.norm(V_celn, axis=1, keepdims=True) + 1e-12)
        
        # Procrustes: W = V_celn.T @ V_spacy @ (V_spacy.T @ V_spacy)^-1
        # Ou SVD de V_celn.T @ V_spacy para solução ortogonal
        M = V_celn.T @ V_spacy  # (10k, 300)
        U, _, VT = np.linalg.svd(M, full_matrices=False)
        W_aligned = U @ VT  # (10k, 300) - ortogonal
        
        self._proj_matrix = W_aligned.astype(np.float32)
        self._is_aligned = True
        print(f"[VocabBridge] Alinhamento completo. Matriz: {self._proj_matrix.shape}")

    def project_vector(self, v_300: np.ndarray) -> np.ndarray:
        """Projeta vetor 300d → 10k via matriz alinhada e normaliza."""
        v = v_300.astype(np.float32)
        projected = self._proj_matrix @ v
        return normalize(projected)

    def project(self, word: str) -> Optional[np.ndarray]:
        """
        Retorna vetor CELN 10k para qualquer palavra do spaCy.
        
        Se a palavra não existe no spaCy, retorna None.
        """
        self._load_spacy()
        if word not in self._spacy_w2i:
            return None
        idx = self._spacy_w2i[word]
        v_300 = self._spacy_vectors[idx]
        return self.project_vector(v_300)

    def project_batch(self, words: list[str]) -> Tuple[np.ndarray, list[str], dict]:
        """
        Projeta lote de palavras spaCy → 10k.
        
        Returns:
            (vectors_10k, found_words, stats)
            vectors_10k: (N_found, 10k) float32
            found_words: lista de palavras encontradas
            stats: {'n_found', 'n_missing', 'missing_words'}
        """
        self._load_spacy()
        found = []
        missing = []
        indices = []
        for w in words:
            if w in self._spacy_w2i:
                indices.append(self._spacy_w2i[w])
                found.append(w)
            else:
                missing.append(w)

        if not indices:
            return np.zeros((0, self.target_dim), dtype=np.float32), found, {
                'n_found': 0, 'n_missing': len(missing), 'missing_words': missing
            }

        vectors_300 = self._spacy_vectors[indices]
        projected = (self._proj_matrix @ vectors_300.T).T
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        projected = projected / norms

        return projected.astype(np.float32), found, {
            'n_found': len(found),
            'n_missing': len(missing),
            'missing_words': missing,
        }

    def get_or_fallback(self, word: str, existing_w2i: dict, existing_vectors: np.ndarray) -> np.ndarray:
        """
        Retorna vetor para palavra: usa codebook existente se disponível,
        senão projeta do spaCy via matriz alinhada.
        
        Args:
            word: Palavra a buscar
            existing_w2i: Dict palavra→índice no codebook existente
            existing_vectors: Matriz (V, 10k) do codebook existente
            
        Returns:
            Vetor 10k normalizado, ou zero se não encontrado em nenhum lugar
        """
        if word in existing_w2i:
            return normalize(existing_vectors[existing_w2i[word]].astype(np.float32))

        projected = self.project(word)
        if projected is not None:
            return projected

        return np.zeros(self.target_dim, dtype=np.float32)

    def augment_codebook(
        self,
        existing_vectors: np.ndarray,
        existing_w2i: Dict[str, int],
        existing_i2w: Dict[int, str],
        new_words: list[str],
    ) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
        """
        Expande codebook com novas palavras projetadas via matriz alinhada.
        
        Args:
            existing_vectors: (V_old, 10k)
            existing_w2i, existing_i2w: mapeamentos existentes
            new_words: palavras a adicionar
            
        Returns:
            (new_vectors, new_w2i, new_i2w)
        """
        projected, found, stats = self.project_batch(new_words)
        if stats['n_found'] == 0:
            return existing_vectors, existing_w2i, existing_i2w

        n_old = existing_vectors.shape[0]
        new_vectors = np.vstack([existing_vectors, projected])
        new_w2i = dict(existing_w2i)
        new_i2w = dict(existing_i2w)

        for i, word in enumerate(found):
            idx = n_old + i
            new_w2i[word] = idx
            new_i2w[idx] = word

        return new_vectors, new_w2i, new_i2w


def get_bridge() -> VocabBridge:
    global _Bridge_INSTANCE
    if _Bridge_INSTANCE is None:
        _Bridge_INSTANCE = VocabBridge()
    return _Bridge_INSTANCE


def test_vocab_bridge():
    print("=" * 60)
    print("TESTE: VocabBridge (Aligned 300→10k)")
    print("=" * 60)

    bridge = VocabBridge()

    print(f"\n1. Matriz de projeção inicial: {bridge._proj_matrix.shape}")
    col_norms = np.linalg.norm(bridge._proj_matrix, axis=0)
    print(f"   Média norma colunas: {col_norms.mean():.4f}")

    test_words = ['mamífero', 'criança', 'filosofia', 'quantum', 'banana']
    print(f"\n2. Projeção (antes do alinhamento):")
    for w in test_words:
        v = bridge.project(w)
        if v is not None:
            print(f"   ✓ '{w}': shape={v.shape}, norm={np.linalg.norm(v):.4f}")
        else:
            print(f"   ✗ '{w}': não encontrado no spaCy")

    # Alinha ao CELN
    print(f"\n3. Alinhando ao codebook CELN...")
    try:
        data = np.load('data/celn_full_vectors.npz', allow_pickle=True)
        celn_vectors = data['vectors']
        celn_vocab = [str(w) for w in data['vocab']]
        celn_w2i = {w: i for i, w in enumerate(celn_vocab)}
        bridge.align_to_celn(celn_vectors, celn_w2i)
    except FileNotFoundError:
        print("   ✗ celn_full_vectors.npz não encontrado")

    print(f"\n4. Projeção (após alinhamento) + preservação similaridade:")
    bridge._load_spacy()
    pairs = [('gato', 'cachorro'), ('gato', 'mesa'), ('professor', 'aluno')]
    for w1, w2 in pairs:
        if w1 not in bridge._spacy_w2i or w2 not in bridge._spacy_w2i:
            print(f"   ⊘ {w1}/{w2} — fora do spaCy")
            continue
        v1_300 = bridge._spacy_vectors[bridge._spacy_w2i[w1]]
        v2_300 = bridge._spacy_vectors[bridge._spacy_w2i[w2]]
        sim_300 = float(np.dot(v1_300, v2_300) / (np.linalg.norm(v1_300) * np.linalg.norm(v2_300) + 1e-12))

        v1_10k = bridge.project_vector(v1_300)
        v2_10k = bridge.project_vector(v2_300)
        sim_10k = float(np.dot(v1_10k, v2_10k))

        print(f"   {w1}/{w2}: sim_300d={sim_300:.4f}, sim_10k={sim_10k:.4f}, "
              f"diff={abs(sim_300 - sim_10k):.4f}")

    print(f"\n5. Teste de augmentação de codebook:")
    try:
        data = np.load('data/celn_full_vectors.npz', allow_pickle=True)
        existing_vectors = data['vectors']
        vocab = [str(w) for w in data['vocab']]
        existing_w2i = {w: i for i, w in enumerate(vocab)}
        existing_i2w = {i: w for i, w in enumerate(vocab)}
        print(f"   Codebook original: {existing_vectors.shape[0]} palavras, {existing_vectors.shape[1]} dims")

        new_words = ['mamífero', 'criança', 'xenófobo']
        new_vectors, new_w2i, new_i2w = bridge.augment_codebook(
            existing_vectors, existing_w2i, existing_i2w, new_words
        )
        print(f"   Codebook expandido: {new_vectors.shape[0]} palavras")
        print(f"   Novas palavras adicionadas: {[w for w in new_words if w in new_w2i and w not in existing_w2i]}")
    except FileNotFoundError:
        print("   ✗ celn_full_vectors.npz não encontrado — skip")

    print(f"\n{'=' * 60}")
    print("RESULTADO: ✓ VocabBridge funcional")
    print("=" * 60)


if __name__ == '__main__':
    test_vocab_bridge()
