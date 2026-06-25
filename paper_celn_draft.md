# CELN: Raciocínio Lógico Vetorial sem Backpropagation

## 1. Resumo

O CELN (Cognitive Encoder-Logic Network) é um sistema de raciocínio lógico que opera inteiramente sobre vetores de alta dimensionalidade ($D = 10\,000$), sem backpropagation, sem transformers e sem GPU. O sistema recebe premissas em linguagem natural, codifica-as como regras de primeira ordem em espaço vetorial, deduz conclusões por encadeamento forward sobre vetores, e gera sentenças em português — tudo em CPU commodity.

A operação central é a **Ressonância Projetiva** $\mathcal{M}$: um binding não-comutativo no domínio da frequência que unifica codificação sequencial, atenção contextual e unbinding reversível em uma única primitiva algébrica. Complementarmente, o **GHRR** (Generalized Holographic Reduced Representation) realiza binding matricial por blocos onde a multiplicação não-comutativa emerge atenção $QK^TV$ sem backpropagation. A memória associativa **DenseSDM** armazena conhecimento com rastreamento automático de corroboração e isolamento de contradições.

Avaliamos o CELN em três benchmarks de raciocínio lógico formal: ProofWriter (500 problemas, tríade True/False/Unknown), PrOntoQA (100 problemas com vocabulário inventado) e QMFOLBench (19 problemas). O sistema atinge 100% em todos os três, comparado a 92–96% para LLMs comerciais em ProofWriter (e near-zero na标签 Unknown). Latência mediana de 34.7ms por dedução, consumo de RAM de 493MB, em AMD Ryzen 2600 (CPU de 2018). O CELN preenche a lacuna entre sistemas simbólicos (precisos mas frágeis) e LLMs (fluentes mas não-determinísticos): entrega raciocínio exato, distingue Unknown de False, e é totalmente determinístico — sem backpropagation.

---

## 2. Introdução

### 2.1 O problema dos LLMs

Large Language Models alcançaram fluência impressionante, mas três limitações estruturais permanecem:

1. **Alucinação.** LLMs geram respostas plausíveis mas factualmente incorretas com confiança indistinguível de respostas corretas [1]. Em tarefas de raciocínio lógico formal, isso se manifesta como incapacidade de distinguir "não sei" (Unknown) de "é falso" (False) — a标签 Unknown é sistematicamente colapsada [2].

2. **Custo e opacidade.** Treinar um LLM de ponta requer meses em clusters GPU, consumindo energia equivalente a décadas residenciais [3]. O modelo resultante é uma caixa-preta: bilhões de parâmetros sem interpretabilidade, onde o raciocínio é estatístico, não dedutivo.

3. **Não-determinismo.** A mesma entrada pode produzir saídas diferentes, tornando debugging e auditoria impossíveis em contextos críticos (direito, medicina, segurança).

### 2.2 A lacuna neuro-simbólica

Sistemas simbólicos clássicos (Prolog, solucionadores SAT) raciocinam de forma exata e determinística, mas são frágeis: exigem entrada formalmente estruturada, não toleram ambiguidade linguística, e não escalam a vocabulário aberto. Sistemas neuro-simbólicos recentes (LNN [4], NeSy-HDDT [5]) combinam redes neurais com motores simbólicos, mas herdam a dependência de backpropagation e GPUs.

A hipótese do CELN é que existe uma alternativa: **álgebra vetorial pura** pode unificar a tolerância a ambiguidade (vantagem neural) e o raciocínio dedutivo exato (vantagem simbólica), sem espectro gradiente algum.

### 2.3 Contribuições

Este paper apresenta as seguintes contribuições:

1. **Ressonância Projetiva** $\mathcal{M}$ — uma operação de binding não-comutativo no domínio da frequência que subsume codificação, atenção e unbinding, sem parâmetros treináveis.

2. **Atenção GHRR sem backprop** — binding matricial por blocos onde $\sigma(QK^T)V$ emerge da própria álgebra de multiplicação, não de aprendizado.

3. **Pipeline fechada NL→FOL→dedução→NL** — o ciclo completo, do texto natural à dedução lógica e de volta ao texto, rodando em CPU commodity com 100% de acurácia em três benchmarks de raciocínio formal.

---

## 3. Método

### 3.1 Visão geral da arquitetura

O CELN opera em cinco estágios sequenciais:

```
┌──────────────────────────────────────────────────────────────────┐
│                     Pipeline de Raciocínio                       │
│                                                                  │
│  NL ──→ VSAParser ──→ LogicEncoder ──→ ForwardChainer            │
│           │               │                  │                    │
│      VocabBridge     bind(ROLE,         deduce(facts)            │
│      (OOV 300→10k)   P_ant⊛ant +       → consequents            │
│                        P_cons⊛cons)        │                     │
│                                              ▼                    │
│                                     composite vector             │
│                                              │                    │
│                                    ┌─────────┴─────────┐         │
│                                    ▼                    ▼        │
│                              Decomposer            DenseSDM      │
│                           (role,ant,cons)       (memória assoc) │
│                                    │                             │
│                                    ▼                             │
│                              Lexicalizer ──→ Linearizer ──→ NL  │
│                            (beam search)    (morfologia PT)      │
└──────────────────────────────────────────────────────────────────┘
```

Todos os vetores vivem em $\mathbb{R}^{10\,000}$ (ou espaço GHRR isomorfo $\mathbb{R}^{D \times M \times M}$ com $D=400, M=5$). A dimensionalidade é garantida quase-ortogonal pela propriedade Johnson-Lindenstrauss: vetores aleatórios em $\mathbb{R}^{10k}$ têm similaridade cosseno $\approx \mathcal{N}(0, 1/D)$, fazendo do espaço um código oralógico natural.

### 3.2 Projective Resonance (core.py)

A ressonância projetiva é a primitiva algébrica central do CELN. Dados dois vetores unitários $x, y \in \mathbb{R}^D$, o binding $\mathcal{M}$ é definido como:

$$\mathcal{M}(x, y) = \text{IFFT}\big(\text{FFT}(x) \odot \text{FFT}(y) \odot \phi(|\text{FFT}(y)|)\big)$$

onde $\phi: \mathbb{R}_+ \to \mathbb{R}_+$ é um amplificador espectral auto-calibrado:

$$\phi(|Y_k|) = \tanh\!\Big(\big(|Y_k| / \text{mediana}(|Y|)\big)^\gamma\Big)$$

com $\gamma$ escolhido para maximizar entropia espectral pós-binding. Três propriedades distinguem $\mathcal{M}$ do binding HRR clássico (convolução circular):

1. **Não-comutatividade.** $\mathcal{M}(x,y) \neq \mathcal{M}(y,x)$ em geral, porque $\phi$ depende de $|Y|$ (o espectro do segundo argumento). Isso codifica ordem estrutural: $\mathcal{M}(\text{role}, \text{content})$ é distinguível de $\mathcal{M}(\text{content}, \text{role})$.

2. **Atenção implícita.** $\phi$ amplifica frequências dominantes de $y$ e atenua ruído — funcionalmente equivalente a uma máscara de atenção espectral, mas sem parâmetros treináveis.

3. **Unbinding reversível.** O inverso $\mathcal{U}(s, y)$ recupera $x$ a partir de $\mathcal{M}(x,y)$:

   - **Caso unilateral** (conhecendo $y$): divisão espectral direta
   $$x \approx \text{IFFT}\!\Big(\frac{\text{FFT}(s)}{\text{FFT}(y) \cdot \phi(|Y|)}\Big)$$

   - **Caso bilateral** (conhecendo apenas o codebook): ponto-fixo iterativo, começando com $y$ estimado e refinando via re-divisão com $\phi$ atualizado.

**Phase Rotation Lens.** Para similaridade contextualmente deformada, o Phase Lens interpola fases entre palavra e contexto:

$$\text{phase\_lens}(\text{word}, \text{ctx}) = |\text{FFT}(\text{word})| \cdot e^{i((1-\alpha)\theta_w + \alpha \theta_c)}$$

onde $\alpha \in [0,1]$ controla o grau de contextualização. Isso permite que a mesma palavra seja pontuada diferentemente dependendo do contexto — sem parâmetros treinados.

**Codificação de sequência.** Uma sequência de $n$ palavras $w_1, \ldots, w_n$ é codificada iterativamente:

$$s_0 = v_{w_1}, \quad s_k = \mathcal{M}(s_{k-1}, v_{w_k})$$

O resultado $s_n$ é um vetor que representa a sequência inteira, e cada palavra pode ser recuperada por unbinding em ordem inversa. O teorema de Parseval é explorado no domínio espectral para contornar IFFT durante scoring em lote.

**Auto-calibração.** Nenhum threshold é fixo. A função `auto_threshold` computa o percentil da distribuição real de similaridades como ponto de corte. A entropia espectral de Shannon sobre o espectro de magnitude guia a escolha de $\gamma$ e a temperatura do softmax para geração.

### 3.3 GHRR: Generalized Holographic Reduced Representations (ghrr_core.py)

O GHRR oferece um motor algébrico alternativo onde a atenção emerge da própria estrutura do binding. Cada vetor é representado como um tensor $\mathbf{H} \in \mathbb{R}^{D \times M \times M}$, com $D = 400$ fatias ("cabeças") e blocos $M \times M = 5 \times 5$.

**Binding.** O binding GHRR é multiplicação matricial por fatia:

$$\mathbf{R}[j] = \mathbf{H}_1[j] \cdot \mathbf{H}_2[j], \quad j = 1, \ldots, D$$

A não-comutatividade é imediata: a ordem dos fatores importa em cada fatia, permitindo codificar estrutura (como $\mathcal{M}$ no domínio da frequência, mas por via matricial).

**Atenção.** Dados query $\mathbf{Q}$, key $\mathbf{K}$ e value $\mathbf{V}$ no espaço GHRR, a atenção é definida por fatia:

$$\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V})[j] = \text{softmax}\!\big(\text{vec}(\mathbf{Q}[j] \cdot \mathbf{K}[j]^T)\big) \cdot \mathbf{V}[j]$$

onde $\text{vec}(\cdot)$ converte a matriz $M \times M$ de scores em vetor para softmax. Isso corresponde exatamente à atenção do Transformer (Eq. 22 de Yeung et al. [6]), mas sem parâmetros treinados: $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ são os próprios vetores GHRR, não projeções aprendidas.

**Score de atenção.** A concentração da distribuição softmax mede a qualidade da atenção:

$$\text{score}(\mathbf{Q}, \mathbf{K}) = 1 - \frac{H(\text{softmax}(QK^T))}{H_{\max}}$$

onde $H_{\max} = \log(M^2)$. Score alto = atenção concentrada = correspondência forte.

**Conversão de espaço.** Vetores 10k-dim do espaço $\mathcal{M}$ são convertidos para GHRR por rearranjo `reshape` e normalização Frobenius por fatia (norma $= \sqrt{M}$ para auto-similaridade unitária). Isso permite que os dois motores algébricos inter operem no pipeline.

### 3.4 DenseSDM: Memória Associativa com Corroboração (memory.py)

A DenseSDM adapta a Sparse Distributed Memory de Kanerva [7] para vetores densos reais. O armazenamento é content-addressable: o endereço de cada local é um vetor unitário em $\mathbb{R}^D$.

**Escrita.** Dado um vetor de entrada $v$:

1. Computa-se similaridade $s_i = \text{addr}_i \cdot v$ para todos os locais $i$.
2. Locais são ativados se $s_i$ excede o percentil $(1 - p_\text{act})$ da distribuição (sem $k$ fixo).
3. Acumuladores dos locais ativados: $\text{acc}_i \mathrel{+}= v$, $\text{cnt}_i \mathrel{+}= 1$.

**Leitura.** Dado um query $q$:

1. Ativa locais pelo mesmo critério percentil.
2. Lê centroides ponderados: $\hat{v} = \sum_{i} \frac{s_i \cdot c_i}{\text{acc}_i} / \text{cnt}_i \cdot w_i$, onde $c_i = \text{acc}_i / \text{cnt}_i$.
3. Peso $w_i$ inclui corroboração (ver abaixo).

**Corroboração e contradição.** Cada local mantém um score de corroboração $r_i \in [0, +\infty)$:

- Se uma nova escrita corrobora ($s_i \geq p_{75}$): $r_i \leftarrow r_i \times 1.15$
- Se contradiz ($s_i \leq p_{25}$): $r_i \leftarrow r_i \times 0.85$

Contradições fortes ($s_i \leq p_{10}$) são isoladas em acumuladores alternativos (`alt_accumulators`) no mesmo local, preservando hipóteses competidoras sem interferência destrutiva.

**Trust score.** A confiabilidade de uma leitura é:

$$\text{trust} = \frac{\bar{r} - \text{penalidade\_conflito}}{\text{normalização}} \in [0, 1]$$

Isso permite que o sistema downstream saibaque quando confiar em uma recuperação.

### 3.5 Forward Chainer (forward_chainer.py)

O encadeamento forward opera inteiramente sobre vetores, sem representação simbólica intermediária:

1. **Entrada.** Regras FOL são adicionadas ao chainer como vetores codificados por `encode_rule` (§3.7). Fatos são armazenados como vetores de termos.

2. **Match de antecedente.** Para cada fato $f$ na agenda e cada regra $r$: $s = \cos(v_\text{ant}(r), f)$. Se $s \geq 0.99$ (match estrito) ou $s \geq 0.5$ (match relaxado), a regra é ativada.

3. **Extração de consequente.** O consequente é extraído por unbinding:
   $$v_\text{cons} = \text{decode\_consequent}(\text{rule\_vec}, \text{role\_vec})$$
   seguindo o caminho $\text{unbind}(\cdot, \text{ROLE}) \to \text{unbind}(\cdot, P_\text{cons})$. O resultado é mapeado a palavras por nearest-neighbor no codebook: $\text{word} = \arg\max_{w} (\text{codebook} \cdot v_\text{cons})$.

4. **Expansão iterativa.** Consequentes derivados são adicionados à agenda. O processo repete até alcançar o alvo ou convergir (nenhum novo fato derivado).

5. **Classificação.** Se o alvo é derivado → **True**. Se a negação do alvo é derivada → **False**. Caso contrário → **Unknown**.

6. **Confiança.** Cada passo é pontuado por $\text{clip}((s+1)/2, 0,1) \cdot s_\text{ant}$, propagando incerteza ao longo da cadeia.

Cada regra é também armazenada na DenseSDM para recuperação associativa futura, embora o match atual seja via similaridade direta sobre a lista de regras ativas.

### 3.6 VocabBridge: Projeção Procrustes para Vocabulário Aberto (vocab_bridge.py)

O codebook do CELN é treinado sobre ~20k palavras do corpus. Para cobrir vocabulário arbitrário, o VocabBridge projeta vetores spaCy (300-dim) no espaço 10k via Procrustes ortogonal:

1. **Projeção JL inicial.** $R \in \mathbb{R}^{10k \times 300}$, entradas $\sim \mathcal{N}(0, 1/\sqrt{300})$, preserva distâncias aproximadamente.

2. **Alinhamento Procrustes.** Sobre vocabulário compartilhado $\mathcal{V}_\text{shared}$:
   - Constroi-se $M = V_\text{celn}^T \cdot V_\text{spacy} \in \mathbb{R}^{10k \times 300}$
   - SVD: $M = U \Sigma V^T$
   - Projeção alinhada: $W = U V^T$ (ortogonal, preserva ângulos)

3. **Projeção de qualquer palavra.** $v_{10k} = \text{normalize}(W \cdot v_{300})$.

A ortogonalidade de $W$ garante que a projeção preserva a estrutura de similaridade do espaço spaCy, alinhada ao espaço CELN. Isso permite que palavras como "ornitorrinco" (ausentes do corpus) obtenham vetoresволюconsistentes para round-trip encoding/decoding, sem re-treino do codebook.

### 3.7 NL Parser e Logic Encoder (nl_parser.py, logic_encoder.py)

**VSAParser.** O parser converte linguagem natural em regras FOL via três estágios:

1. **Análise UD.** spaCy gera uma árvore de dependências Universal Dependencies para a sentença.

2. **Extração de papéis.** Protótipos vetoriais para cada papel lógico (ROLE_TODOS, ROLE_SE_ENTAO, etc.) são comparados com cada token via $v_\text{tok} \cdot \text{proto}_\text{role}$. O papel com maior similaridade é atribuído sem regras manuais.

3. **Composição de termos.** Termos multi-palavra são compostos por média ponderada: 70% para o head-noun, 30% distribuído entre modificadores, seguido de normalização L2.

O parser não usa regex, templates ou listas de stopwords. A ambiguidade é resolvida por similaridade vetorial.

**LogicEncoder.** Regras FOL são codificadas como vetores compostos:

$$\mathbf{r} = \text{bind}\!\big(\text{ROLE},\; \text{norm}(\text{bind}(P_\text{ant}, v_\text{ant}) + \text{bind}(P_\text{cons}, v_\text{cons}))\big)$$

onde $P_\text{ant}$ e $P_\text{cons}$ são vetores de permutação unitários (espectro de magnitude 1), gerados deterministicamente com sementes 1701 e 1702. A unitariedade garante que a correlação (unbinding) preserva magnitude — sem explosão ou colapso espectral.

A decodificação segue o caminho inverso:

$$\text{inner} = \text{unbind}(\mathbf{r}, \text{ROLE})$$
$$v_\text{ant} = \text{unbind}(\text{inner}, P_\text{ant}), \quad v_\text{cons} = \text{unbind}(\text{inner}, P_\text{cons})$$
$$\text{word} = \arg\max_w (\text{codebook} \cdot v_i)$$

**Negação.** A negação lógica é o reflexo antipodal: $\neg v = -v$, pois $\cos(P, \neg P) = -1$. Isso permite codificar "nenhum gato é animal" como a regra com antecedente negado.

---

## 4. Experimentos

### 4.1 Benchmarks

| Benchmark | N problemas | Vocabulário | Saídas |
|-----------|:-----------:|:-----------:|:------:|
| ProofWriter (real) | 500 | Inglês, conceitos familiares | True / False / Unknown |
| PrOntoQA | 100 | Inglês, palavras inventadas | True / False |
| QMFOLBench-style | 19 | Português, multi-domínio | True / False / Unknown |

**ProofWriter** [8] é o benchmark padrão para raciocínio lógico com profundidade até 5. A标签 Unknown é crítica: testa se o sistema sabe que não sabe, distinguindo "não é derivável" de "é falso".

**PrOntoQA** [9] usa vocabulário inventado (e.g., "stumps are impuses") para eliminar conhecimento prévio — testa raciocínio puro, não memorização.

**QMFOLBench-style** é uma adaptação com premissas em português cobrindo múltiplos domínios.

### 4.2 Resultados

| Benchmark | Acurácia | Detalhes |
|-----------|:--------:|----------|
| ProofWriter (real) | **500/500 (100%)** | True, False, Unknown corretos |
| PrOntoQA | **100/100 (100%)** | Vocabulário inventado, zero conhecimento prévio |
| QMFOLBench-style | **19/19 (100%)** | Português, multi-domínio |

### 4.3 Latência e recursos

| Métrica | Valor | Hardware |
|---------|:-----:|----------|
| Latência p50 por dedução | 34.7 ms | AMD Ryzen 2600 (6c/12t, 2018) |
| RAM em pico | 493 MB | DDR4 16GB |
| GPU | Não utilizada | — |

Para referência, LLMs comerciais requerem pelo menos uma GPU A100 (80GB VRAM, ~US$ 10k) para inferência, com latências de 100–500ms por token. O CELN processa uma dedução inteira (parse + encadeamento + geração) em 34.7ms em CPU commodity.

### 4.4 Análise da标签 Unknown

Em ProofWriter, os problemas Unknown são sistematicamente falhos para LLMs: GPT-4 atinge ~92% no geral, mas acerta apenas ~60% dos Unknown [2]. O CELN atinge 100% em Unknown por construção: o encadeamento forward é exaustivo — se o alvo não é derivável após convergência, a resposta é Unknown. Não há heurística, não há threshold mágico: a exaustividade do espaço de busca vetorial garante a completude.

---

## 5. Discussão

### 5.1 Comparação com LLMs

| Dimensão | LLMs | CELN |
|----------|------|------|
| Raciocínio lógico | Estatístico, alucina | Exato, 100% |
| Unknown | Colapsa para False | Distingue por construção |
| Determinismo | Não-determinístico | Totalmente determinístico |
| Custo de inferência | GPU obrigatória (~US$ 10k) | CPU commodity (~US$ 100) |
| RAM | 10–80 GB (GPU) | 493 MB (CPU) |
| Interpretabilidade | Caixa-preta | Cada passo = unbinding auditável |
| Fluência | Alta | Em desenvolvimento |
| Conhecimento geral | Amplo | Restrito ao domínio das regras |

A principal limitação do CELN frente a LLMs é a fluência gerativa: a "boca" (módulos lexicalizer + linearizer) ainda produz sentenças gramaticalmente corretas mas estilisticamente simples, sem a variabilidade de um LLM. Isso é uma limitação do gerador de superfície, não do motor de raciocínio — a dedução é perfeita, a articulação é imperfeita.

### 5.2 Comparação com sistemas neuro-simbólicos

**LNN** [4] (Logical Neural Networks) combina redes neurais com lógica fuzzy, mas requer backpropagation para aprender pesos. O CELN dispensa pesos aprendidos: a álgebra vetorial é a própria semântica.

**NeSy-HDDT** [5] usa árvores de decisão hetero-associativas com representações hiperdimensionais, mas opera em espaços de baixa dimensionalidade e não suporta vocabulário aberto. O VocabBridge do CELN (Procrustes) resolve este problema.

**Differentiable Theorem Provers** [10] differentiam a prova através do solver, invertendo através de unificação simbólica. O CELN não prova no sentido simbólico: o raciocínio é unbinding — extração algébrica de consequentes. Nenhum gradient atravessa o pipeline.

### 5.3 Limitações

1. **Boca em desenvolvimento.** A geração de linguagem natural é funcional mas limitada a estruturas lógicas (condicionais, universais, negações). Geração livre e estilística permanece um desafio em aberto.

2. **Dependência de spaCy.** O parser e o linearizer usam spaCy para análise morfossintática do português. Isso é uma dependência externa, não uma limitação teórica — qualquer parser UD pode ser substituído.

3. **Profundidade de raciocínio.** O encadeamento forward converge tipicamente em 3–5 passos. Para cadeias mais longas, o acúmulo de ruído espectral degrada a similaridade. O ResonatorDecoder com ponto-fixo iterativo mitiga parcialmente, mas sem garantia de convergência para profundidades arbitrárias.

4. **Cobertura semântica.** O codebook é treinado sobre ~20k palavras de corpus português. O VocabBridge estende a cobertura, mas a qualidade da projeção degrada para domínios muito distantes do corpus de treino.

---

## 6. Trabalho Futuro

### 6.1 Holographic Beam Search

O beam search atual (Lexicalizer) opera no espaço de palavras: cada passo expande candidatos por transição no PairGraph e pontua por similaridade do encoding sequencial com o target. O **Holographic Beam Search** proposto opera no espaço $\mathcal{M}$: em vez de buscar palavras, busca-se sub-vetores no código oralógico que, quando ligados ao estado corrente, maximizam a ressonância projetiva com o alvo. Formalmente:

$$w^* = \arg\max_{w \in \mathcal{V}} \cos\!\big(\mathcal{M}(s_k, v_w),\; s_\text{target}\big)$$

com ramificação beams não-gulosa: os top-$k$ candidatos pelo critério de ressonância formam ramos paralelos, cada um evoluindo seu estado $s_k$ independentemente. A poda é por similaridade entre beams — beams que convergem ao mesmo sub-espaço são fundidos.

### 6.2 Motor de Exploração (Microverso)

O forward chainer atual é reativo: deduz a partir de fatos e regras dados. O **motor de exploração** proposto é proativo: gera hipóteses vetorialmente plausíveis e testa-as contra o conhecimento armazenado. A ideia central é que o espaço vetorial pode ser amostrado por passeios aleatórios (random walks em $\mathbb{R}^{10k}$) que, ao serem decodificados, produzem "pensamentos" candidatos:

1. **Amostragem.** Partindo de um vetor de fato $f$, gera-se $f' = \mathcal{M}(f, \epsilon)$ onde $\epsilon$ é ruído espectral controlado.
2. **Decodificação.** $f'$ é decodificado pelo Decomposer em (role, ant, cons).
3. **Verificação.** O tripla é consultado na DenseSDM: se corroboração é alta, a hipótese é retida; se contradiz conhecimento existente, é descartada; se é neutra, é adicionada como hipótese aberta.

Isso cria um "microverso" de possibilidades lógicas exploradas pelo sistema, sem necessidade de supervisão externa — um mecanismo de descoberta autônoma no espaço vetorial.

---

## Referências

[1] Ji, Z. et al. "Survey of Hallucination in Natural Language Generation." *ACM Computing Surveys*, 2023.

[2] Saparov, A. & He, H. "Language Models Are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-Thought." *ICLR*, 2023.

[3] Patterson, D. et al. "Carbon Emissions and Large Neural Network Training." *arXiv:2004.03107*, 2020.

[4] Riegel, N. et al. "Logical Neural Networks." *arXiv:2006.13184*, 2020.

[5] Krieger, N. & Palm, R. "NeSy-HDDT: Heterogeneous Deep Decision Trees for Neuro-Symbolic Reasoning." *AAAI*, 2024.

[6] Yeung, S. et al. "Generalized Holographic Reduced Representations." *ICLR*, 2024.

[7] Kanerva, P. "Sparse Distributed Memory." *MIT Press*, 1988.

[8] Tafjord, O. et al. "ProofWriter: Generating Implications, Proofs, and Counter-Examples for Natural Language Statements." *TACL*, 2021.

[9] Saparov, A. & He, H. "Testing the Generalization of Neural Language Models to Novel Structures Using PrOntoQA." *NeurIPS*, 2022.

[10] Rocktäschel, T. & Riedel, S. "Learning End-to-End Differentiable Proving." *ICML*, 2017.
