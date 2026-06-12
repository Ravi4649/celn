# A Operação Unificadora — Exploração Matemática para o CELN v3

> **Pergunta central:** Qual operação matemática única pode fazer pelo CELN o que a atenção fez pelos Transformers — unificar fluência e raciocínio em um sistema vetorial, sem backprop, rodando em CPU?

---

## 0. O Que a Atenção Fez e Por Que Deu Certo

Antes de propor alternativas, precisamos entender exatamente o que a atenção resolveu:

```
Attention(Q,K,V) = softmax(QKᵀ / √dₖ) V
```

Essa equação fez 3 coisas simultaneamente:

1. **Roteamento dinâmico**: Cada token decide, via similaridade, de quais outros tokens extrair informação. Não há conexões fixas — o grafo de dependência é recalculado a cada forward pass.

2. **Contextualização**: A representação de saída de cada token é uma mistura ponderada de todos os outros. "Banco" recebe informação de "rio" ou "dinheiro" dependendo do contexto.

3. **Composicionalidade implícita**: As somas ponderadas permitem que significados compostos emerjam sem que ninguém programe explicitamente a regra de composição.

**Por que isso funcionou TÃO bem**: A atenção resolveu o problema de _crédito dinâmico_ — num grafo de dependências que muda a cada frase, qual palavra influencia qual? A resposta é: aquela com maior similaridade Q·K, aprendida via backprop.

**O que precisamos reproduzir SEM backprop**: A capacidade de rotear informação dinamicamente com base em similaridade, mas onde a "similaridade" é aprendida continuamente (Hebbian), não otimizada por gradiente.

---

## 1. Candidato A — FHRR com Gating Espectral Aprendido

### A operação

```
bind_G(x, y) = F⁻¹( G ⊙ F(x) ⊙ F(y) )
```

onde:
- `F` = FFT (Fast Fourier Transform)
- `⊙` = multiplicação elemento-a-elemento de números complexos
- `G ∈ ℂᵈ` = vetor de gates espectrais aprendidos (um peso complexo por frequência)
- `F⁻¹` = IFFT

Na prática, isto é convolução circular modulada por frequência.

### Por que FHRR?

FHRR (Plate, 1995) usa vetores complexos onde cada componente é um número complexo unitário: `vₖ = e^{iθₖ}`. O binding via convolução circular é：

```
(a ∗ b)ₖ = Σⱼ aⱼ · b₍ₖ₋ⱼ₎ mod d
```

No domínio de Fourier, isso colapsa para multiplicação elemento-a-elemento:
```
F(a ∗ b)ₖ = F(a)ₖ · F(b)ₖ
```

Isso é computacionalmente LINDO: O(d log d) via FFT, comparado a O(d²) da convolução direta.

### A inovação: Gating espectral aprendido

Em vez de multiplicação cega, o gate `G` aprende quais frequências são relevantes para cada par de conceitos:

```
ΔGₖ = η · (targetₖ - outputₖ) · conj(F(x)ₖ) · conj(F(y)ₖ)
```

Esta é uma regra Hebbiana no domínio espectral: se o output difere do target, ajusta o gate na direção que reduz o erro, ponderado pela atividade em cada frequência.

### Forças

| Dimensão | Avaliação |
|----------|-----------|
| **Fluência** | ✅ O gate G aprende as correlações espectrais que produzem transições naturais entre palavras |
| **Raciocínio** | ✅ Analogia é unbind seguido de bind; com G aprendido, a "transformação analógica" é mais precisa |
| **CPU** | ✅ 10k-dim FFT roda em ~100μs; binding = 2 FFTs + 1 IFFT = ~300μs |
| **Auto-calibração** | ✅ O gate é atualizado comparando com a distribuição atual, sem thresholds fixos |
| **Anti-colisão** | ✅ A seletividade espectral impede que Halos colapsem (frequências diferentes = interferência destrutiva) |

### Fraquezas

| Dimensão | Avaliação |
|----------|-----------|
| **Não-linearidade** | ❌ Convolução é linear; fluência requer não-linearidade. Atenção resolve isso com softmax. Precisaríamos de uma não-linearidade pós-binding. |
| **Gargalo do gate** | ⚠️ G tem d elementos complexos = 20k parâmetros reais. É pouco para capturar toda a estrutura da língua. |
| **Aprendizado** | ⚠️ A regra Hebbiana espectral é instável se o target não for bem definido. Quem define o target na geração de texto? |

### Veredito parcial
Forte para raciocínio, frágil para fluência. A linearidade da convolução limita a diversidade de saída — tende a produzir médias, não extremos. Precisaria de um mecanismo competitivo adicional (tipo um "softmax sem backprop").

---

## 2. Candidato B — Sparse Distributed Memory (SDM) com Endereçamento Contínuo

### A operação

```
read(x) = Σ_{h ∈ activate(x)} w_h · content_h
```

onde:
- `activate(x)` = os K hard-locations cujo endereço é mais próximo de x (distância de Hamming ou cosseno)
- `w_h` = peso de ativação (exponencial decrescente com a distância)
- `content_h` = vetor de conteúdo armazenado na localização h

A versão "contínua" substitui endereços binários por vetores reais e Hamming por similaridade de cosseno.

### Por que SDM?

Kanerva (1988) propôs o SDM como modelo de memória de longo prazo no cérebro. A ideia central é genial na sua simplicidade:

1. **Endereçamento por conteúdo**: Você acessa a memória não por índice, mas por similaridade ao que quer encontrar
2. **Escrever é somar**: Cada escrita adiciona o vetor de dados às localizações ativadas (não sobrescreve)
3. **Ler é média ponderada**: A leitura retorna a soma dos conteúdos das localizações ativadas, ponderada por similaridade

### A inovação: SDM contínuo com ativação competitiva

O SDM clássico usa endereços binários e ativação por threshold fixo (raio de Hamming). A versão contínua:

1. Endereços são vetores reais unitários (mesmo espaço dos dados)
2. Ativação usa percentil da distribuição de similaridades (auto-calibrável):
   ```
   activate(x) = {h : sim(x, addr_h) > percentile(sim_distribution, 95)}
   ```
3. O número de localizações ativas varia naturalmente com a densidade local do espaço

### Forças

| Dimensão | Avaliação |
|----------|-----------|
| **Fluência** | ✅ Memória auto-associativa: dado um contexto parcial, completa o padrão (≈ gerar próxima palavra) |
| **Raciocínio** | ✅ Encadeamento: read(read(x, rel1), rel2) implementa dedução em 2 passos |
| **CPU** | ✅ Com 100k localizações, uma leitura é O(d · K) onde K ≈ 1000 localizações ativas. 10k × 1000 = 10M ops — factível |
| **Auto-calibração** | ✅ Percentil dinâmico da distribuição real de similaridades |
| **Aprendizado contínuo** | ✅ Escrever é somar — cada nova frase adiciona informação sem degradar a anterior (propriedade de superposição) |

### Fraquezas

| Dimensão | Avaliação |
|----------|-----------|
| **Memória** | ❌ 100k localizações × 10k dimensões × 4 bytes = 4GB. No limite do Ryzen (16GB). |
| **Vocabulário diverso** | ❌ O problema dos Halos: com muitas palavras, as médias ponderadas borram as distinções |
| **Geração de texto** | ❌ SDM foi projetado para recuperação, não geração. A saída é uma média de vetores armazenados — tende ao "centroide" do corpus. |
| **Sequenciamento** | ⚠️ SDM não tem noção nativa de ordem. Precisaria de binding com vetores de posição (como HRR), adicionando complexidade. |

### Veredito parcial
Excelente como memória de longo prazo, mas a operação de "média ponderada" embaça distinções sutis. Para fluência, precisaríamos de algo mais seletivo — que retorne UM item, não uma média. Mas como subsistema de memória, é fortíssimo.

---

## 3. Candidato C — Álgebra Geométrica / Rotores de Clifford

### A operação

```
transform(x, R) = R x R⁻¹
```

onde:
- `R = e^{-θB/2}` = rotor (generalização de quatérnios para d dimensões)
- `B` = bivetor unitário que define o plano de rotação
- `θ` = ângulo de rotação
- `R x R⁻¹` = conjugação: rotaciona o vetor x no plano B pelo ângulo θ

### Por que Álgebra Geométrica?

A Álgebra Geométrica (Clifford) unifica:
- **Produto interno** (a·b): similaridade, projeção
- **Produto externo** (a∧b): área orientada, independência, novidade
- **Produto geométrico** (ab = a·b + a∧b): a operação fundamental da álgebra

O rotor é a ferramenta perfeita para analogias: "rei está para homem assim como rainha está para mulher" é literalmente uma rotação no espaço semântico.

Seja:
- `H` = vetor de "homem"
- `K` = vetor de "rei"
- `M` = vetor de "mulher"

O rotor que mapeia H → K é: `R = sqrt(K · H⁻¹)` (assumindo representação conforme)

Então: `rainha ≈ R M R⁻¹`

Isso é elegantíssimo. A analogia É uma rotação.

### Forças

| Dimensão | Avaliação |
|----------|-----------|
| **Raciocínio analógico** | ✅✅ Perfeito. Esta é a matemática natural da analogia. |
| **Composicionalidade** | ✅ Rotores compõem: R₁(R₂ x R₂⁻¹)R₁⁻¹ = (R₁R₂) x (R₁R₂)⁻¹ |
| **Invertibilidade** | ✅ Toda operação tem uma inversa exata |
| **Interpretabilidade** | ✅ O plano de rotação B mostra QUAIS dimensões semânticas estão sendo transformadas |

### Fraquezas

| Dimensão | Avaliação |
|----------|-----------|
| **Dimensionalidade** | ❌❌ O espaço completo de multivétores tem 2ᵈ dimensões. Para d=10k, é 2^10000 — impossível. Precisamos restringir a k-vetores de baixo grau (k=1,2,3), mas será que isso captura toda a estrutura da língua? |
| **Fluência** | ❌ Como usar rotores para gerar texto fluente? Rotacionar "contexto atual" produz o "próximo estado", mas que rotação corresponde à transição entre palavras? Não é óbvio. |
| **Aprendizado** | ❌ Como aprender o rotor "correto" sem backprop? Comparar ângulos e planos é não-linear e instável. |
| **CPU** | ⚠️ Produto geométrico de dois multivétores de grau k é combinatório. Mesmo restrito a k≤2, um bivetor em 10k dims tem 10k×10k/2 = 50M componentes — 200MB. |

### Veredito parcial
A ferramenta matemática mais elegante para raciocínio analógico. Mas a ponte entre "rotações em espaços geométricos" e "geração de texto fluente" não é clara. A álgebra geométrica brilha em espaços de baixa dimensão (3D, 4D). Em 10k dimensões, a complexidade combinatória dos k-vetores explode.

---

## 4. Candidato D (Síntese Original) — Ressonância Vetorial com Competição Local (RVCL)

### A intuição física

Pense num sistema de partículas em um espaço 10k-dimensional. Cada palavra é uma partícula com posição `v ∈ ℝᵈ`. O "significado" de uma frase não é a posição das partículas, mas o **padrão de ressonância** entre elas — quais partículas oscilam em fase, quais em oposição, e como essas oscilações se propagam.

A pergunta "qual a próxima palavra?" torna-se: **se eu perturbar o sistema no ponto `context`, qual partícula vizinha entra em ressonância?**

### A operação

```
resonate(x, C) = x ⊙ normalize( Σ_{c ∈ C} sim(x, c) · c )
```

onde:
- `x` = vetor de consulta (query)
- `C` = conjunto de vetores de contexto (palavras da frase atual, tópico, etc.)
- `sim(x, c)` = similaridade normalizada (cosseno)
- `⊙` = binding via convolução circular (FHRR)
- `normalize` = projeção na hiperesfera unitária

**Versão simplificada (uma única equação):**

```
y = F⁻¹( F(x) ⊙ F( Σ_{c ∈ C} w_c · c ) )
```

com pesos competitivos auto-calibráveis:

```
w_c = exp(β · sim(x, c)) / percentile( exp(β · sim(x, ·)), 90 )
```

O denominador é o percentil 90 das similaridades — isso garante que apenas os top ~10% de contexto realmente influenciam, sem threshold fixo.

### Por que isto é diferente?

**Diferença crucial da atenção**: A atenção dos Transformers compara TODOS os tokens entre si (O(n²)). A ressonância compara o query apenas com um conjunto pequeno de contexto e o resto é feito via binding (O(n · d log d)).

**Diferença crucial do VSA 2.0**: O VSA 2.0 usava binding fixo (XOR). Aqui, o binding é modulado pelos pesos competitivos `w_c`, que são auto-calibráveis a cada operação.

**Diferença crucial de FHRR puro**: O gate espectral NÃO é um vetor fixo G, mas emerge dinamicamente da interação competitiva entre os vetores de contexto. É um "gate implícito" gerado pela distribuição de similaridades do momento.

### Como unifica fluência e raciocínio

#### Fluência (geração de texto)

```
# Estado inicial: vetor de contexto da frase
state = encode("O cobre é um")

# Loop de geração:
for step in range(max_len):
    # Ressonância: quais palavras ressoam com o estado atual?
    candidates = resonante(state, recent_words)
    
    # Seleção: a palavra com maior ressonância (com sampling por temperatura)
    next_word = select(candidates, temperature=auto_calibrated)
    
    # Atualização: binding do estado com a palavra escolhida
    state = bind(state, next_word)
```

A **ressonância** produz fluência porque:
1. Palavras que frequentemente co-ocorrem desenvolvem similaridade alta (via Hebbian)
2. O binding sequencial `bind(bind(bind(w1, w2), w3), w4)` codifica a estrutura da frase
3. A competição (top 10%) impede que o sistema produza a palavra "média" — força escolhas definidas

#### Raciocínio (dedução, analogia)

```
# Analogia: A está para B assim como C está para ?
# Passo 1: extrair a transformação
T = unbind(B, A)   # "o que transforma A em B"
# Passo 2: aplicar a transformação
D = bind(C, T)     # "aplique a mesma transformação a C"
# Passo 3: verificar consistência
if similarity(bind(A, T), B) > auto_threshold:
    return D
```

O **binding** produz raciocínio porque:
1. `unbind` extrai a relação entre dois conceitos como um vetor
2. `bind` aplica essa relação a um novo conceito
3. A operação é a mesma usada para fluência — unificação real

### Aprendizado sem backprop

A matriz de similaridade entre palavras é atualizada via Hebbian:

```
Δsim(v_i, v_j) = η · (co-occurrence_count(i,j) / expected_by_chance(i,j) - 1)
```

Isso é PMI (Pointwise Mutual Information) implementado como atualização vetorial. Sem backprop. Sem gradientes. Apenas contagem e normalização.

Adicionalmente, os próprios vetores são ajustados:

```
v_i += η · Σ_{j ∈ context(i)} v_j · PMI(i,j)
```

Isso aproxima vetores de palavras que co-ocorrem mais que o esperado ao acaso.

### Forças

| Dimensão | Avaliação |
|----------|-----------|
| **Unificação** | ✅✅ A MESMA operação (`resonate` + `bind`) é usada para gerar texto e deduzir |
| **Fluência** | ✅ Competição (percentil) gera diversidade; binding sequencial gera coerência |
| **Raciocínio** | ✅ Binding/unbinding são invertíveis e transitivos |
| **CPU** | ✅ Cada passo: 2 FFTs + similaridades com contexto (O(|C|·d)) |
| **Auto-calibração** | ✅ Percentil dinâmico, sem thresholds mágicos |
| **Anti-colisão** | ✅ Competição esparsa (top 10%) impede que Halos colapsem |

### Fraquezas

| Dimensão | Avaliação |
|----------|-----------|
| **Qualidade da competição** | ⚠️ Se o percentil for mal calibrado, o sistema pode ficar "estreito" demais (sempre as mesmas palavras) ou "largo" demais (volta ao problema da média) |
| **Binding sequencial** | ⚠️ bind(bind(bind(...))) acumula ruído. Após ~20 palavras, o sinal degrada. Precisamos de um mecanismo de "reset" ou compressão. |
| **Inicialização** | ⚠️ Vetores aleatórios não têm estrutura. O sistema precisa de um "bootstrapping" a partir do corpus — a qualidade da inicialização determinará tudo. |
| **Corpus pequeno** | ⚠️ 3000 frases é pouco para estimar PMI confiável. Precisamos de aumento de dados ou regularização. |

---

## 5. Análise Comparativa

| Critério | FHRR+Gating (A) | SDM Contínuo (B) | Geom. Algebra (C) | RVCL (D) |
|----------|-----------------|-------------------|--------------------|----------|
| Unifica fluência+raciocínio | ⚠️ Parcial | ❌ Não | ❌ Não | ✅ Sim |
| Uma única operação | ✅ bind_G | ✅ read/write | ✅ rotor | ✅ resonate+bind* |
| Roda em CPU (10k dims) | ✅ ~300μs | ⚠️ ~10ms | ❌ Combinatória | ✅ ~500μs |
| Aprendizado sem backprop | ✅ Hebbian espectral | ✅ Hebbian (write) | ❌ Instável | ✅ PMI + Hebbian |
| Auto-calibrável | ✅ | ✅ | ❌ | ✅ |
| Anti-colisão de Halos | ✅ Gate espectral | ❌ Médias borram | ✅ Planos separados | ✅ Competição top-10% |
| Gera texto diverso | ❌ (linear) | ❌ (média) | ❌ (rotações?) | ✅ (competição+temp) |
| Raciocínio analógico | ✅ unbind+bind | ⚠️ (2-pass read) | ✅✅ rotor natural | ✅ unbind+bind |
| Memória RAM (16GB) | ✅ ~10MB | ❌ ~4GB | ❌ ~200MB+ | ✅ ~50MB |
| Complexidade de implementação | Média | Alta | Muito Alta | Média |

*Nota: `resonate` e `bind` são duas "operações", mas `resonate` é inteiramente definida em termos de `bind` + `similaridade` + `normalização` — assim como a atenção dos Transformers é definida em termos de `matmul` + `softmax`. É uma composição de primitivas que funciona como UMA unidade conceitual.

---

## 6. Recomendações

### Para um sistema Puramente Unificado (uma única equação):

**Aposta na RVCL (Candidato D)**, com a operação unificadora sendo:

```
y = F⁻¹( F(x) ⊙ Σ_{c ∈ top_k(C, x)} w_c(x) · F(c) )
```

onde `w_c(x)` são pesos competitivos auto-calibráveis e `top_k` seleciona os k vizinhos mais similares (k = percentil 90 da distribuição de similaridades).

Esta equação:
- **É uma**: definida como composição de primitivas algébricas (FFT, ⊙, soma ponderada, seleção por percentil)
- **Gera texto**: quando `x = estado atual` e `C = palavras candidatas` → seleciona a próxima palavra por ressonância
- **Deduz**: quando `x = premissa` e `C = relações conhecidas` → infere a conclusão por binding
- **Faz analogia**: quando `x = C` e usamos `unbind(B,A)` como query → encontra D

### Para um sistema Híbrido (múltiplos subsistemas com a MESMA operação):

**SDM para memória de longo prazo + RVCL para processamento ativo**, ambos usando **binding espectral (FHRR)** como operação fundamental comum.

- SDM armazena o corpus como memória associativa (com primeiros resultados de experimentos em ~1 hora)
- RVCL processa o contexto ativo e gera saída (com primeiros resultados em ~1 dia)
- Ambos compartilham a mesma álgebra de binding (FHRR via FFT)

### Próximo Passo Imediato (Recomendado)

Implementar um **teste mínimo de fluência** com ~100 frases do corpus:

1. Inicializar vetores de palavras (10k dims, aleatórios + normalização)
2. Treinar similaridades via PMI no corpus (1 iteração, sem backprop)
3. Gerar continuação de 5 palavras dado um prefixo de 3 palavras
4. Avaliar: coerência semântica, diversidade, repetição

Este teste responde em ~2 horas se a RVCL tem potencial para fluência.

---

## 7. Apêndice: Ideias Descartadas (e Por Quê)

### A. Atenção com softmax sem backprop
O softmax precisa de gradientes para aprender Q, K, V. Sem backprop, Q, K, V seriam aleatórios → a atenção seria essencialmente aleatória.

### B. Memória de Hopfield contínua (Modern Hopfield)
A atualização `x^{t+1} = softmax(X x^t) X` é elegante, mas:
- Requer armazenar toda a matriz de padrões X (grande)
- O softmax é sensível a thresholds
- Não tem binding/unbinding → sem composicionalidade

### C. K-Means dinâmico no espaço vetorial
Agrupar palavras em "significados" viola o princípio de zero classificação. Os clusters seriam fixos, não adaptativos.

### D. Autômatos celulares em espaços vetoriais
Divertido, mas a ponte entre "regras locais de atualização" e "fluência linguística" é longa demais para um primeiro teste.

### E. Operador de Koopman (teoria de sistemas dinâmicos)
Avançar o estado via `x_{t+1} = K x_t` onde K é aprendido. Mas K é uma matriz d×d (100M parâmetros para d=10k). Inviável em CPU.

---

## 8. Referências Conceituais

- **Plate, T. (1995)** — Holographic Reduced Representations. IEEE Trans. Neural Networks.
- **Kanerva, P. (1988)** — Sparse Distributed Memory. MIT Press.
- **Gayler, R. (2004)** — Vector Symbolic Architectures. AAAI Spring Symposium.
- **Kleyko, D. et al. (2022)** — A Survey on HDC/VSA. ACM Computing Surveys.
- **Furlong & Eliasmith (2022)** — VSAs as Generative Models. AAAI Spring Symposium.
- **Schlegel et al. (2025)** — Attention as Binding: A VSA Perspective on Transformer Reasoning. arXiv 2512.14709.
- **VaCoAl (2026)** — Beyond LLMs, SDM and Neuromorphics. arXiv 2604.11665.
- **GHRR (2024)** — Generalized Holographic Reduced Representations. arXiv 2405.09689.
- **Geometric Algebra for NLP (2026)** — Toward a Functional Geometric Algebra. arXiv 2604.25902.
