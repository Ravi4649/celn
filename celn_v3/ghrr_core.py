"""
CELN v3 — GHRR Core Operations
===============================
Baseado em: Generalized Holographic Reduced Representations (Yeung et al., 2024)
arXiv: 2405.09689

Binding com matrizes m×m (m>1) implementa atenção nativamente via Q·K†.
FHRR (m=1) é o caso especial onde binding = multiplicação escalar comutativa.

Espaço vetorial: ℝ^{D×M×M} — D fatias (attention heads), cada uma com M×M matriz.
Binding = multiplicação de matrizes bloco-a-bloco (não-comutativa).
Atenção emerge naturalmente: sigma(Q * K^T) * V.
Similaridade: trace-based com normalização cosseno.

Princípios:
  ZERO backprop/transformers. ZERO listas fixas. ZERO templates.
  ZERO thresholds mágicos. Tudo auto-calibrável.
  100% álgebra vetorial.
"""

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
D = 400          # Número de fatias (attention heads independentes)
M = 5            # Tamanho do bloco (m>1 para não-comutatividade e atenção)
DTYPE = np.float32
EFFECTIVE_DIM = D * M * M  # 400 * 25 = 10000

# ---------------------------------------------------------------------------
# Conversão ℝ^10000 ↔ ℝ^{D×M×M}
# ---------------------------------------------------------------------------

def vec_10k_to_ghrr(v: np.ndarray) -> np.ndarray:
    """Converte vetor ℝ^10000 → GHRR ℝ^{D×M×M} com normalização por fatia."""
    h = v.reshape(D, M, M).astype(DTYPE)
    return normalize_slices(h)


def ghrr_to_10k(h: np.ndarray) -> np.ndarray:
    """Achata GHRR ℝ^{D×M×M} → ℝ^10000."""
    return h.reshape(-1).astype(DTYPE)


def bulk_10k_to_ghrr(vectors_10k: np.ndarray) -> np.ndarray:
    """Converte matriz (V, 10000) → (V, D, M, M). VETORIZADO."""
    V = vectors_10k.shape[0]
    h = vectors_10k.reshape(V, D, M, M).astype(DTYPE)
    return normalize_slices_batch(h)


# ---------------------------------------------------------------------------
# Normalização
# ---------------------------------------------------------------------------

def normalize_slices(h: np.ndarray) -> np.ndarray:
    """Normaliza cada fatia j para Frobenius norm = sqrt(M).
    
    Isso garante que auto-similaridade = 1 com a fórmula da similaridade GHRR:
      δ(H, H) = (1/(M*D)) * Σ_j tr(H_j @ H_j^T)
    Com ||H_j||_F = sqrt(M): tr(H_j @ H_j^T) = M, logo δ = (D*M)/(D*M) = 1.
    """
    result = h.copy()
    for j in range(D):
        fn = np.linalg.norm(h[j], 'fro')
        if fn > 1e-12:
            result[j] = h[j] * (np.sqrt(M) / fn)
    return result


def normalize_slices_batch(h_batch: np.ndarray) -> np.ndarray:
    """Normaliza (V, D, M, M) por fatia — VETORIZADO."""
    fnorms = np.linalg.norm(h_batch, axis=(-2, -1), keepdims=True)
    fnorms[fnorms < 1e-12] = np.sqrt(M)
    return h_batch * (np.sqrt(M) / fnorms)


def normalize_vector_batch(h_batch: np.ndarray) -> np.ndarray:
    """Alias: normaliza (V, D, M, M) por fatia em batch. VETORIZADO."""
    return normalize_slices_batch(h_batch)


# ---------------------------------------------------------------------------
# Binding GHRR (multiplicação de matrizes bloco-a-bloco)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _ghrr_bind_jit(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    D = h1.shape[0]
    result = np.zeros_like(h1)
    for j in range(D):
        result[j] = h1[j] @ h2[j]
    return result


def ghrr_bind(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    return _ghrr_bind_jit(h1.astype(DTYPE), h2.astype(DTYPE))


def ghrr_bind_normalized(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Binding GHRR com normalização por fatia do resultado."""
    return normalize_slices(ghrr_bind(h1, h2))


# ---------------------------------------------------------------------------
# Unbinding (aproximação via inversa da matriz)
# ---------------------------------------------------------------------------

def ghrr_unbind(bound: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Unbinding GHRR: bound * key^{-1}.
    
    Para key com fatias bem-condicionadas, key^{-1} ≈ key^T (se ortogonal)
    ou usa np.linalg.inv para precisão.
    
    Args:
        bound: (D, M, M) float32
        key: (D, M, M) float32
    
    Returns:
        (D, M, M) float32
    """
    result = np.zeros((D, M, M), dtype=DTYPE)
    for j in range(D):
        try:
            k_inv = np.linalg.inv(key[j])
        except np.linalg.LinAlgError:
            k_inv = key[j].T  # fallback: transpose (pseudo-ortogonal)
        result[j] = bound[j] @ k_inv
    return result


# ---------------------------------------------------------------------------
# Similaridade GHRR (trace-based)
# ---------------------------------------------------------------------------

def ghrr_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """Similaridade GHRR: (1/(M*D)) Σ_j tr(H1_j @ H2_j^T).
    
    Range: [-1, 1] para vetores normalizados por fatia.
    
    O(D * M^2) — rápido.
    """
    h2_T = np.swapaxes(h2, -1, -2)
    product = np.matmul(h1, h2_T)
    traces = np.trace(product, axis1=-2, axis2=-1)
    return float(traces.sum()) / (M * D)


def ghrr_similarity_batch(h_state: np.ndarray, h_candidates: np.ndarray) -> np.ndarray:
    """Similaridade GHRR em batch: estado × N candidatos.
    
    Args:
        h_state: (D, M, M) float32
        h_candidates: (N, D, M, M) float32
    
    Returns:
        (N,) float32 — similaridades
    """
    state_expanded = h_state[None, ...]  # (1, D, M, M)
    h2_T = np.swapaxes(h_candidates, -2, -1)  # (N, D, M, M)
    product = np.matmul(state_expanded, h2_T)  # (N, D, M, M)
    traces = np.trace(product, axis1=-2, axis2=-1)  # (N, D)
    return (traces.sum(axis=-1) / (M * D)).astype(DTYPE)


# ---------------------------------------------------------------------------
# Atenção GHRR — a operação-chave do paper
# ---------------------------------------------------------------------------

def ghrr_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray,
                   temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Atenção GHRR: sigma(Q * K^T) * V.
    
    sigma = softmax sobre a parte real de Q·K^T (matriz M×M por fatia).
    Esta é a correspondência EXATA com a atenção de Transformers,
    demonstrada no paper (Eq. 22).
    
    Estrutura:
      Q·K^T → softmax → pesos de atenção → aplicados a V
    
    Args:
        query: (D, M, M) float32 — estado atual / query
        key: (D, M, M) float32 — candidato / key
        value: (D, M, M) float32 — candidato / value (geralmente = key)
        temperature: suavização do softmax
    
    Returns:
        (attended_value, attention_weights) ambos (D, M, M) float32
    """
    key_T = np.swapaxes(key, -1, -2)
    attn_raw = np.matmul(query, key_T)  # (D, M, M)

    # Softmax por fatia sobre os M*M elementos
    attn_flat = attn_raw.reshape(D, -1)  # (D, M*M)
    attn_flat = attn_flat - attn_flat.max(axis=-1, keepdims=True)
    exp_attn = np.exp(attn_flat / (temperature + 1e-12))
    attn_prob_flat = exp_attn / exp_attn.sum(axis=-1, keepdims=True)
    attn_weights = attn_prob_flat.reshape(D, M, M).astype(DTYPE)

    attended = np.matmul(attn_weights, value)
    return attended, attn_weights


def ghrr_attention_score(state: np.ndarray, candidate: np.ndarray,
                         temperature: float = 0.3) -> float:
    """Score de atenção GHRR: quão concentrada é Q·K^T estado→candidato.
    
    Mede a CONCENTRAÇÃO ESPECTRAL da matriz de atenção Q·K^T:
    - Alta concentração = estado e candidato se alinham fortemente
    - Baixa concentração = alinhamento difuso
    
    Usa a ENTROPIA da distribuição softmax:
      H = -Σ p_i log(p_i)  (25 elementos por fatia, média sobre D fatias)
      score = 1 - H/H_max  onde H_max = log(M*M)
    
    Range: [0, 1] — 0 = uniforme (ruim), 1 = totalmente focado (bom)
    """
    key_T = np.swapaxes(candidate, -1, -2)
    attn_raw = np.matmul(state, key_T)  # (D, M, M)
    
    attn_flat = attn_raw.reshape(D, -1)  # (D, M*M)
    attn_flat = attn_flat - attn_flat.max(axis=-1, keepdims=True)
    exp_attn = np.exp(attn_flat / (temperature + 1e-12))
    probs = exp_attn / exp_attn.sum(axis=-1, keepdims=True)
    
    # Entropia por fatia
    probs_safe = probs + 1e-12
    entropy_per_slice = -np.sum(probs_safe * np.log(probs_safe), axis=-1)  # (D,)
    H_max = np.log(M * M)
    
    concentration = 1.0 - np.mean(entropy_per_slice) / H_max
    return float(np.clip(concentration, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Encoding de sequências (scan left-to-right)
# ---------------------------------------------------------------------------

def ghrr_encode_sequence(words_ghrr: list[np.ndarray]) -> np.ndarray:
    """Codifica uma sequência de palavras GHRR via binding left-to-right.
    
    Com GHRR (M>1), o binding retém informação bloco-diagonal,
    permitindo recuperação seletiva do histórico (atenção implícita).
    
    state_0 = w[0]
    state_i = ghrr_bind(N(state_{i-1}), w_i)
    
    Args:
        words_ghrr: lista de (D, M, M) float32
    
    Returns:
        (D, M, M) float32 — estado da sequência
    """
    if not words_ghrr:
        return np.zeros((D, M, M), dtype=DTYPE)

    state = normalize_slices(words_ghrr[0].copy())
    for w in words_ghrr[1:]:
        state = normalize_slices(ghrr_bind(state, w))
    return state


# ---------------------------------------------------------------------------
# Auto-calibração (mesmo princípio do core original)
# ---------------------------------------------------------------------------

def auto_threshold(values: np.ndarray, percentile: float = 90.0) -> float:
    """Threshold auto-calibrável baseado no percentil da distribuição."""
    return float(np.percentile(values, percentile))


def competitive_filter(scores: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    """Zera scores abaixo do threshold competitivo auto-calibrável."""
    threshold = auto_threshold(scores, percentile)
    filtered = scores.copy()
    filtered[filtered < threshold] = 0.0
    return filtered


# ---------------------------------------------------------------------------
# Inicialização de vetores GHRR aleatórios (para novos vocabulários)
# ---------------------------------------------------------------------------

def make_random_ghrr_vector(seed: int = None) -> np.ndarray:
    """Gera vetor GHRR aleatório quasi-ortogonal.
    
    Segue Corolário 2 do paper:
      H_j = Q_j @ Λ_j
    onde Q_j é aleatória (QR de matriz normal) e Λ_j = I (identidade).
    
    As fatias são quasi-ortogonais: δ(H_a, H_b) → 0 para D grande.
    """
    rng = np.random.RandomState(seed)
    h = np.zeros((D, M, M), dtype=DTYPE)
    for j in range(D):
        A = rng.randn(M, M).astype(DTYPE)
        q, _ = np.linalg.qr(A.astype(np.float64))
        h[j] = q.astype(DTYPE)
    return normalize_slices(h)


def make_random_ghrr_vectors(n: int, seed: int = 42) -> np.ndarray:
    """Gera N vetores GHRR aleatórios quasi-ortogonais."""
    rng = np.random.RandomState(seed)
    h = np.zeros((n, D, M, M), dtype=DTYPE)
    for i in range(n):
        for j in range(D):
            A = rng.randn(M, M).astype(DTYPE)
            q, _ = np.linalg.qr(A.astype(np.float64))
            h[i, j] = q.astype(DTYPE)
    return normalize_vector_batch(h)
