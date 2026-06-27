# Análise de Sentimentos em Comunidades de Saúde Mental do Reddit Brasileiro

Projeto prático da disciplina de Processamento de Linguagem Natural (PLN) — 2ª Etapa.

O objetivo é classificar o **tom emocional de comentários** coletados dos subreddits brasileiros `r/AnsiedadeDepressao` e `r/desabafos`, distinguindo respostas de suporte (**B**om), **N**eutras ou de não-suporte (**M**au). O projeto cobre todo o pipeline: coleta e pré-processamento dos dados, anotação humana, fine-tuning do BERTimbau e avaliação comparativa com modelos do estado da arte.

---

## Estrutura do Repositório

```
├── PLN_Trabalho_02.ipynb   # Notebook principal com todo o pipeline
└── Relatorio.pdf           # Relatório completo do projeto
```

---

## Pipeline

### Parte 1 — Dados

**1.1 Importação** — Os dados são baixados do Google Drive via `gdown`: posts e comentários de `r/AnsiedadeDepressao` e `r/desabafos`.

**1.2 Normalização e Filtragem** — Comentários são filtrados (removendo respostas do autor do post, comentários removidos/redact/bot e textos muito curtos) e normalizados (emojis convertidos para texto, marcações Markdown removidas, espaços padronizados).

**1.3 Amostragem** — 3.000 comentários são amostrados de forma estratificada por subreddit a partir do corpus filtrado.

**1.4 Anotação Humana** — O corpus amostrado é dividido em 6 partes iguais e distribuído entre os 6 membros do grupo em planilhas `.xlsx` com menu suspenso restrito às opções B/N/M. Os critérios de anotação são:
- **B (Bom / Suporte):** acolhe, encoraja, demonstra empatia
- **N (Neutro):** informa, relata ou pergunta sem carga emocional
- **M (Mau / Não-suporte):** minimiza sofrimento, é hostil ou ofensivo

**1.5 Divisão Treino/Val/Teste** — Dois métodos implementados: Holdout (80/10/10) e K-Fold estratificado com k=5.

---

### Parte 2 — Modelo

O modelo base é o **BERTimbau** (`neuralmind/bert-base-portuguese-cased`), um BERT pré-treinado em português.

**2.1 TAPT (Task-Adaptive Pre-Training)** — Antes do fine-tuning, o BERTimbau passa por pré-treino MLM (Masked Language Modeling) adicional nos ~10 mil comentários não rotulados do corpus, adaptando o encoder ao vocabulário do domínio (saúde mental no Reddit).

**2.2 Fine-tuning com Focal Loss** — O modelo é ajustado para classificação em 3 classes (B/N/M) usando:
- **Focal Loss** com `γ=2` e pesos de classe balanceados como `α`, substituindo a entropia cruzada ponderada que tendia a super-prever a classe minoritária (M)
- **2 épocas** de treinamento (a partir da 3ª o modelo overfittava)
- Learning rate `2e-5`, batch 16, warmup de 10%
- Seleção do melhor modelo por F1-macro na validação

**2.3 Validação Cruzada (K-Fold)** — 5 modelos independentes são treinados, cada um com um fold diferente de validação, permitindo estimar a variância do desempenho.

**2.4 Ensemble** — Os logits dos 5 modelos do K-Fold são mediados para produzir predições mais estáveis.

**2.5 Temperature Scaling** — A temperatura `T` é calibrada nas predições out-of-fold (sem vazamento de dados) para corrigir a sub-confiança do modelo (ECE reduzido em ~57–75%).

**Modelo Final:** ensemble dos 5 folds + temperature scaling, encapsulado na função `prever_ensemble(textos)`.

---

### Parte 3 — Avaliação

Como os rótulos humanos não passaram por validação de concordância entre anotadores (Kappa), eles foram usados apenas para treino. A avaliação de qualidade é feita comparando o BERTimbau com dois modelos do **estado da arte** em análise de sentimentos para português:

- **[Pysentimiento](https://huggingface.co/pysentimiento/bertweet-pt-sentiment)** — BERTabaporu ajustado em 15.000 tweets PT-BR anotados manualmente
- **[XLM-RoBERTa](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment)** — Modelo multilingual ajustado em ~10 milhões de tweets para classificação de sentimentos
- **Simbólico** — Modelo baseado em léxico (baseline determinístico)

A avaliação inclui matrizes de concordância par a par, classificação de discordâncias em leves (POS↔NEU, NEU↔NEG) e severas (POS↔NEG), métricas formais usando cada modelo de referência como ground truth, e um ensemble Pysentimiento+XLM-RoBERTa (mantendo apenas os casos em que ambos concordam) como ground truth consensual com Cohen's Kappa como justificativa.

---

## Requisitos

```bash
pip install gdown emoji scikit-learn pandas matplotlib seaborn openpyxl
pip install torch transformers datasets accelerate
pip install pysentimiento scipy
```

> O notebook foi desenvolvido para rodar no **Google Colab** com GPU. Em CPU o treinamento é consideravelmente mais lento.

---

## Como Reproduzir

1. Abra `PLN_Trabalho_02.ipynb` no Google Colab
2. Execute a **Parte 1** completa (coleta, filtragem, amostragem e divisão dos dados)
   - A anotação humana (seção 1.4) já está consolidada e disponível via Google Drive
3. Execute a **Parte 2** para treinar o BERTimbau
   - Se os checkpoints já existirem em disco, o notebook os recarrega sem retreinar
4. Execute a **Parte 3** para a avaliação comparativa
   - O corpus classificado pelos modelos de referência também está disponível via Google Drive para pular a inferência

---

## Referências

- Gururangan et al. (2020). *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks*. ACL.
- Lin et al. (2017). *Focal Loss for Dense Object Detection*. ICCV.
- Guo et al. (2017). *On Calibration of Modern Neural Networks*. ICML.
- Lakshminarayanan et al. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. NeurIPS.
- Kohavi (1995). *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection*. IJCAI.
- Brum & das Graças Volpe Nunes (2018). *Building a Sentiment Corpus of Tweets in Brazilian Portuguese*. arXiv.
- Pérez et al. (2021). *pysentimiento: A Python Toolkit for Opinion Mining and Social NLP tasks*. arXiv.
- Barbieri et al. (2021). *XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis and Beyond*. arXiv.
