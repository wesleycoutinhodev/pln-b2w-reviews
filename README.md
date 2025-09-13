# Atividade de Participação - Análise Dataset B2W-Reviews01

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![UFPI](https://img.shields.io/badge/UFPI-Graduação%2FPós--Graduação-red.svg)](https://ufpi.br)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wesleycoutinhodev/pln-representacao-textual)

## 📋 Sobre o Projeto

Este repositório contém a implementação e documentação da **Atividade de Participação em Grupo** da disciplina de **Processamento de Linguagem Natural (PLN)** oferecida conjuntamente para alunos de graduação e pós-graduação do Departamento de Computação (DC) da UFPI.

O trabalho foca na análise do **Dataset B2W-Reviews01**, um corpus aberto de avaliações de produtos contendo mais de 130 mil reviews de clientes de e-commerce coletadas no site americanas.com entre janeiro e maio de 2018, com implementações práticas desenvolvidas no **Google Colab**.

## 📊 Dataset B2W-Reviews01

- **Fonte**: [GitHub - americanas-tech/b2w-reviews01](https://github.com/americanas-tech/b2w-reviews01)
- **Tamanho**: +130.000 avaliações de clientes
- **Período**: Janeiro a Maio de 2018
- **Plataforma**: americanas.com
- **Artigo**: [b2w-reviews01_stil2019.pdf](https://github.com/americanas-tech/b2w-reviews01/blob/main/b2wreviews01_stil2019.pdf)

## 👥 Equipe

| Nome | GitHub | Nível | Responsabilidade |
|------|--------|-------|------------------|
| Wesley Coutinho | [@wesleycoutinhodev](https://github.com/wesleycoutinhodev) | Graduação | Operações 1, 2, 3 |
| Averinaldo Oscar | [@AveiaPodre](https://github.com/AveiaPodre) | Graduação | Operações 4, 5, 6 |
| Carlos Daniel | [@carlos-dani-dev](https://github.com/carlos-dani-dev) | Graduação | Operações 7, 8 |
| Paulo Henrique | [@PauloHenriqueRod](https://github.com/PauloHenriqueRod) | Graduação | Operações 9, 10 e Documentação |

## 🔧 Operações sobre Dataset

### 🧹 **Operação 1** - Tratamento de Dados
- **Objetivo**: Analisar informações do dataset para identificar valores ausentes (NaN) e/ou inválidos
- **Validação**: Verificar se coluna "reviewer_gender" possui apenas valores 'M' e 'F'
- **Ferramentas**: Pandas, análise exploratória de dados

### 🎯 **Operação 2** - Seleção de Colunas Relevantes
- **Objetivo**: Selecionar apenas colunas: "review_text", "overall_rating", "recommend_to_a_friend"
- **Filtro**: Apenas produtos da marca "Samsung"
- **Output**: Dataset reduzido com colunas específicas

### ⚖️ **Operação 3** - Análise de Inconsistências
- **Objetivo**: Verificar coerência entre avaliações e recomendações
- **Regras**: 
  - Notas 1-2 → "recommend_to_a_friend" = 'No'
  - Notas 4-5 → "recommend_to_a_friend" = 'Yes'
- **Ação**: Remover inconsistências identificadas

### 🔄 **Operação 4** - Conversão de Dados
- **Objetivo**: Converter coluna "recommend_to_a_friend" de str para int
- **Mapeamento**: "Yes" → 1, "No" → 0
- **Tipo**: Transformação de dados categóricos

### 📊 **Operação 5** - Divisão Train/Test
- **Objetivo**: Separar dataset em conjuntos de treino e teste
- **Método**: `train_test_split()` do Scikit-Learn
- **Proporção**: 90/10 ou 80/20
- **Estratificação**: Manter distribuição das classes

### 🔤 **Operação 6** - Tokenização com TextVectorization
- **Objetivo**: Tokenizar reviews usando TextVectorization (Keras/TensorFlow)
- **Processo**: 
  - Usar `adapt()` para vocabulário
  - Padding/truncamento para mesmo comprimento
  - Plotar histograma de comprimentos
- **Análise**: Tamanho do vocabulário, ajuste do parâmetro `maxlen`

### 🚫 **Operação 7** - Remoção de Stopwords
- **Objetivo**: Remover stopwords e refazer vetorização
- **Comparação**: Tamanho do vocabulário antes vs. depois
- **Análise**: Impacto na dimensionalidade

### 📈 **Operação 8** - Representação com CountVectorizer e TFIDFVectorizer
- **Objetivo**: Representar palavras usando Scikit-Learn
- **Métodos**: CountVectorizer e TFIDFVectorizer
- **Análises**:
  - Aplicação da Lei de Zipf
  - Cortes de Luhn
  - Redução de dimensionalidade
- **Filtros**: Remoção de stopwords e palavras por frequência

### 🔤➡️🔢 **Operação 9** - Word Embeddings
- **Objetivo**: Representar palavras usando embeddings 100D
- **Embeddings Pré-treinadas NILC**:
  - Word2Vec (CBOW e Skip-gram)
  - GloVe
- **Embeddings Próprias**: Treinar com dados dos reviews
- **Conjunto**: Usar treino para gerar embeddings próprias

### 📄📊 **Operação 10** - Representação de Reviews
- **Objetivo**: Representar reviews completos usando embeddings
- **Algoritmos**:
  - Doc2Vec
  - SentenceBERT
- **Comparação**: Análise de performance dos métodos

## 🚀 Executar no Google Colab

**Recomendado**: Use o Google Colab para execução com GPU gratuita e bibliotecas pré-instaladas!

| Operação | Notebook | Colab Link | Status |
|----------|----------|------------|--------|
| 1-3 | Análise e Tratamento de Dados | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wesleycoutinhodev/pln-b2w-reviews/blob/main/notebooks/Operacoes_1_2_3_Tratamento_Dados.ipynb) | ✅ |
| 4-5 | Conversão e Divisão dos Dados | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wesleycoutinhodev/pln-b2w-reviews/blob/main/notebooks/Operacoes_4_5_Conversao_Split.ipynb) | ✅ |
| 6 | Tokenização com TextVectorization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wesleycoutinhodev/pln-b2w-reviews/blob/main/notebooks/Operacao_6_TextVectorization.ipynb) | ✅ |
| 7 | Remoção de Stopwords | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wesleycoutinhodev/pln-b2w-reviews/blob/main/notebooks/Operacao_7_Stopwords.ipynb) | ✅ |
| 8 | CountVectorizer e TFIDF | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wesleycoutinhodev/pln-b2w-reviews/blob/main/notebooks/Operacao_8_Vectorizers.ipynb) | ✅ |
| 9 | Word Embeddings | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wesleycoutinhodev/pln-b2w-reviews/blob/main/notebooks/Operacao_9_Word_Embeddings.ipynb) | ✅ |
| 10 | Doc2Vec e SentenceBERT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wesleycoutinhodev/pln-b2w-reviews/blob/main/notebooks/Operacao_10_Doc_Embeddings.ipynb) | ✅ |

### 💡 Vantagens do Colab
- ✅ **GPU gratuita**: Acelera treinamento de embeddings e processamento
- ✅ **RAM abundante**: Processa dataset grande (130k+ reviews)
- ✅ **Bibliotecas pré-instaladas**: TensorFlow, Scikit-learn, Pandas
- ✅ **Fácil compartilhamento**: Colaboração em tempo real
- ✅ **Persistência**: Salva automaticamente no Google Drive

### ⚠️ Dicas Importantes para o Dataset
- **Download**: Use `!wget` ou `!gdown` para baixar o dataset
- **Memória**: Dataset grande, monitore uso de RAM
- **Backup**: Salve resultados intermediários no Drive
- **Checkpoints**: Salve modelos treinados periodicamente

## 📁 Estrutura do Projeto

```
📦 pln-b2w-reviews/
├── 📄 README.md                    # Este arquivo
├── 📂 notebooks/                   # Notebooks Google Colab
│   ├── Operacoes_1_2_3_Tratamento_Dados.ipynb
│   ├── Operacoes_4_5_Conversao_Split.ipynb
│   ├── Operacao_6_TextVectorization.ipynb
│   ├── Operacao_7_Stopwords.ipynb
│   ├── Operacao_8_Vectorizers.ipynb
│   ├── Operacao_9_Word_Embeddings.ipynb
│   └── Operacao_10_Doc_Embeddings.ipynb
├── 📂 data/                        # Datasets e arquivos
│   ├── B2W-Reviews01.csv          # Dataset original
│   ├── samsung_reviews.csv        # Reviews Samsung filtrados
│   └── processed/                 # Dados pré-processados
├── 📂 models/                      # Modelos treinados
│   ├── word2vec_custom.model
│   ├── doc2vec.model
│   └── embeddings/
├── 📂 results/                     # Resultados e análises
│   ├── exploratory_analysis/
│   ├── vocabulary_analysis/
│   ├── embeddings_comparison/
│   └── visualizations/
├── 📂 docs/                        # Documentação
│   ├── dataset_analysis.md
│   ├── metodologia.md
│   └── resultados_finais.md
└── 📄 requirements_colab.txt       # Dependências específicas
```

## 🛠️ Tecnologias e Bibliotecas

### Google Colab Environment
- **Google Colab**: Ambiente principal com GPU
- **Google Drive**: Persistência de dados e modelos

### Processamento de Dados
```python
# Bibliotecas principais
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Pré-processamento
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
```

### Deep Learning e NLP
```python
# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.utils import TextVectorization

# Word Embeddings
import gensim
from gensim.models import Word2Vec, Doc2Vec
from sentence_transformers import SentenceBERT

# NLTK
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
```

### Instalações Adicionais no Colab
```python
# Executar no início dos notebooks
!pip install gensim sentence-transformers
!pip install wordcloud plotly

# Para downloads do dataset
!pip install gdown
```

## 📈 Análises e Métricas

### Análise Exploratória
- **Distribuição de ratings**: Histogramas e estatísticas
- **Comprimento de reviews**: Análise para definir `maxlen`
- **Inconsistências**: Detecção e remoção de dados contraditórios

### Análise de Vocabulário
- **Tamanho do vocabulário**: Antes e depois das transformações
- **Lei de Zipf**: Análise da distribuição de frequência
- **Stopwords**: Impacto na dimensionalidade

### Comparação de Embeddings
- **Word2Vec vs GloVe**: Qualidade das representações
- **Embeddings próprias vs pré-treinadas**: Performance
- **Doc2Vec vs SentenceBERT**: Representação de documentos

## 💡 Como Usar os Notebooks

### 1. Setup Inicial
```python
# Monte o Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Download do dataset
!gdown --id [ID_DO_DATASET] -O /content/drive/MyDrive/pln_data/
```

### 2. Processamento Sequencial
- Execute os notebooks na ordem das operações (1→10)
- Cada notebook salva resultados para o próximo
- Verifique dependências entre operações

### 3. Monitoramento de Recursos
```python
# Verificar uso de RAM
!cat /proc/meminfo | head -n 3

# Verificar GPU disponível
!nvidia-smi
```

## 📊 Resultados Esperados

### Operações 1-5: Pré-processamento
- Dataset Samsung limpo e consistente
- Divisão treino/teste balanceada
- Transformações de dados aplicadas

### Operações 6-8: Vetorização
- Vocabulários de diferentes tamanhos
- Comparação de métodos de vetorização
- Aplicação de filtros de frequência

### Operações 9-10: Embeddings
- Modelos Word2Vec e GloVe treinados
- Comparação de embeddings pré-treinadas vs próprias
- Representações Doc2Vec e SentenceBERT

## 📖 Documentação Adicional

- 📋 [Enunciado Original](docs/enunciado_atividade.pdf)
- 📊 [Artigo B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01/blob/main/b2wreviews01_stil2019.pdf)
- 🔗 [Dataset B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01)
- 🔤 [Embeddings NILC](http://nilc.icmc.usp.br/nilc/index.php/repositorio-de-word-embeddings-do-nilc)
- 📖 [Tutorial Google Colab](https://colab.research.google.com/notebooks/intro.ipynb)

## 🎓 Contexto Acadêmico

**Disciplina**: Processamento de Linguagem Natural  
**Modalidade**: Disciplina conjunta Graduação/Pós-Graduação  
**Departamento**: DC - Departamento de Computação / CCN - Centro de Ciências da Natureza  
**Universidade**: UFPI - Universidade Federal do Piauí  
**Semestre**: 2024.2  
**Tipo**: Atividade de Participação em Grupo

## 📧 Contato

Para dúvidas ou sugestões sobre este projeto:
- 📧 Email: [wesleysousa@ufpi.edu.br](mailto:wesleysousa@ufpi.edu.br)
- 🐙 GitHub: [@wesleycoutinhodev](https://github.com/wesleycoutinhodev)
- 💼 LinkedIn: [Wesley Coutinho](https://www.linkedin.com/in/wesleycoutinhodev/)

---

⭐ **Se este projeto foi útil para você, considere dar uma estrela!**

🚀 **Desenvolvido com Google Colab e ❤️ pela equipe de PLN da UFPI**
