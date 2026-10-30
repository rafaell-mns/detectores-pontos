# Visão Computacional - Atividade Prática 2

Este repositório contém a implementação do script de avaliação, descrição e correspondência de pontos de interesse (POIs) utilizando OpenCV em Python.

## 📌 Informações Acadêmicas
* **Instituição:** Universidade Federal do Piauí (UFPI)
* **Curso:** Ciência da Computação (CCN)
* **Disciplina:** Visão Computacional (2026.2)
* **Docente:** Prof. Kelson Rômulo Teixeira Aires
* **Discente:** Francisco Rafael Meneses Gonçalves

---

## 🛠️ Descrição do Código (`main.py`)

O código principal realiza:
1. **Carregamento e Pré-processamento:** Leitura dos pares de imagens, redimensionamento (largura máxima de 1600px) e conversão para escala de cinza.
2. **Detecção de POIs:** Execução dos detectores **Harris**, **SIFT** e **ORB** com parâmetros configuráveis.
3. **Cálculo de Sobreposição:** Medição do percentual de pontos coincidentes entre diferentes detectores com limiar de distância de $3$ pixels.
4. **Descrição e Feature Matching:**
   * Extração de descritores SIFT (L2, 128D) e ORB (Hamming, Binário).
   * Matching por Força Bruta (BFMatcher) com o **teste da razão de Lowe** (ratio threshold = 0.75).
5. **Validação da Ground Truth:**
   * Estimação da homografia entre os pares usando SIFT + RANSAC.
   * Cálculo da taxa de acerto considerando como *match correto* reprojeções a até $3$ pixels de distância.
6. **Geração de Saídas:** Salvamento de métricas no terminal/logs e exportação das imagens com os matches visualizados (linhas verdes para corretos e vermelhas para incorretos).

---

## 🚀 Requisitos e Instalação

### Pré-requisitos
Certifique-se de ter o Python 3.8+ instalado.

### Dependências
Instale as bibliotecas necessárias com o comando:

```bash
pip install opencv-python numpy matplotlib
```

---

## 📂 Estrutura de Diretórios

```text
.
├── main.py                # Script principal de execução
├── README.md              # Documentação do código
├── imgs/                  # Conjunto dos 3 pares de imagens de entrada
└── resultados/            # Resultados gerados (visualizações dos matches)
```

---

## 💻 Como Executar

Para rodar todo o pipeline de detecção, descrição e cálculo das métricas:

```bash
python main.py
```

---

## ⚙️ Configuração dos Algoritmos

As configurações padrão utilizadas na pipeline principal do script são:

* **Harris:** `blockSize=2`, `kSize=3`, `k=0.04`, `thr=0.01` (com NMS em vizinhança $5 \times 5$, limite max. 20.000 pts).
* **SIFT:** `contrastThreshold=0.04`, `edgeThreshold=10`, `sigma=1.6`.
* **ORB:** `nfeatures=10000`, `scaleFactor=1.2`, `nlevels=8`, `fastThreshold=20`.
