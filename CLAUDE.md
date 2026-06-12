# CELN-v3 — A Nova Arquitetura

## Objetivo
Criar uma IA que supere todos os LLMs em fluência, raciocínio, eficiência e descoberta.
Sem backprop. Sem GPU. Rodando em CPU (Ryzen 2600, 16GB RAM).

## Princípios Imutáveis
- ZERO backpropagation, transformers, LLMs
- ZERO listas fixas de palavras (stopwords, artigos, preposições)
- ZERO templates de resposta
- ZERO classificação gramatical de palavras (sujeito, verbo, etc.)
- ZERO thresholds mágicos (tudo percentil da distribuição real)
- Tudo auto-calibrável
- 100% álgebra vetorial

## Herança do VSA 2.0
- Vetores de 10k dimensões são expressivos e eficientes
- Operações algébricas (binding, unbinding, similaridade) rodam em nano segundos
- Aprendizado contínuo sem backprop é possível
- A estrutura da frase pode ser codificada sem classificar palavras
- Corpus disponível: corpus_final.txt (~3000 frases em português, múltiplos domínios)

## Lições Aprendidas (NÃO REPETIR)
- XOR é limitado para fluência — é binário, sem unbinding real
- Halos colapsam com vocabulário diverso (extração de intenções falhou com 200 frases)
- Atenção geométrica pura (cosseno + softmax) não gera fluência — produz drift semântico
- Codebook pré-atribuído limita a expressividade
- Separar "codebook", "Halos", "boca", "dedução" em módulos estanques cria gargalos
- Estatística acumulada (Halos) não substitui projeção aprendida

## Pergunta Central
Qual operação matemática única pode fazer pelo CELN o que a atenção fez pelos Transformers —
unificar fluência e raciocínio em um sistema vetorial, sem backprop, rodando em CPU?
