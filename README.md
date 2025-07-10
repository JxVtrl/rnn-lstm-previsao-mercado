# Previsão de Preço do Bitcoin com RNN e LSTM

Projeto prático da disciplina de Redes Neurais (UERJ/IPRJ) com foco em previsão de curto prazo no mercado financeiro. O sistema utiliza redes neurais recorrentes (RNN e LSTM) para prever o preço do Bitcoin (BTC-USD) com 5 minutos de antecedência, a partir de séries temporais com resolução de 1 minuto.

## 📈 Contexto

O mercado de criptomoedas é altamente volátil e não-linear. Ferramentas tradicionais de regressão não conseguem capturar bem essas dinâmicas. Por isso, empregamos redes neurais recorrentes (RNN) e suas variantes, como LSTM, que são eficazes para lidar com dependências temporais de curto e longo prazo.

## 🎯 Objetivos

- Prever o preço do **Bitcoin** 5 minutos à frente
- Utilizar dados históricos com resolução de 1 minuto
- Simular uma operação contínua com visualização gráfica em tempo real
- Avaliar continuamente o desempenho com métricas padronizadas

## 🛠 Tecnologias Utilizadas

- Python 3 (Google Colab)
- TensorFlow / Keras
- [yfinance](https://pypi.org/project/yfinance/) (para obtenção dos dados do BTC)
- matplotlib / plotly (visualização em tempo real)
- scikit-learn (cálculo de métricas de avaliação)

## 📊 Métricas de Avaliação

O desempenho do modelo será acompanhado ao longo do tempo com:

- MAE – Erro Absoluto Médio
- RMSE – Raiz do Erro Quadrático Médio
- MAPE – Erro Percentual Absoluto Médio
- R² – Coeficiente de Determinação

## 🔄 Simulação Gráfica em Tempo Real

Durante a execução, o sistema atualiza continuamente:

- Série histórica do preço real
- Previsões geradas pelo modelo
- Erros acumulados
- Gráficos comparativos entre real e predito

## 📁 Estrutura do Projeto

```bash
├── data/               # Scripts de coleta e preparação de dados
├── models/             # Modelos RNN e LSTM salvos
├── notebook.ipynb      # Notebook principal (executável no Google Colab)
├── plots/              # Imagens geradas pela simulação
└── README.md
```

##  ▶️ Como Executar
- Clone o repositório:

```bash
git clone https://github.com/SEU_USUARIO/rnn-lstm-btc-predict.git
```

- Acesse o notebook principal no Google Colab (ou abra diretamente via link do repositório).

- Execute as células sequencialmente para carregar dados, treinar modelos e iniciar a simulação.

## 💡 Recomendado: utilizar o Google Colab para melhor performance e integração em grupo.

## 👨‍💻 Equipe
Desenvolvido por alunos da UERJ/IPRJ:

João Vinicius Vitral

[Adicionar nomes dos colegas de grupo aqui]

## 📌 Licença
Este projeto é de uso educacional e não deve ser utilizado para decisões financeiras reais.
