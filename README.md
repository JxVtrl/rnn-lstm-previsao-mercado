# Previsão de Preço do Bitcoin com RNN e LSTM

Projeto prático da disciplina de Redes Neurais (UERJ/IPRJ) com foco em previsão de curto prazo no mercado financeiro. O sistema utiliza redes neurais recorrentes (RNN e LSTM) para prever o preço do Bitcoin (BTC-USD) com 5 minutos de antecedência, a partir de séries temporais com resolução de 1 minuto.

## 📈 Contexto e Motivação

O mercado de criptomoedas é altamente volátil e não-linear. Ferramentas tradicionais de regressão não conseguem capturar bem essas dinâmicas. Por isso, empregamos redes neurais recorrentes (RNN) e suas variantes, como LSTM, que são eficazes para lidar com dependências temporais de curto e longo prazo.

## 🎯 Objetivo Geral

Projetar e avaliar um sistema baseado em redes neurais recorrentes que, a partir de séries históricas de preços com resolução de 1 minuto, seja capaz de prever o preço de fechamento do Bitcoin nos próximos 5 minutos. O sistema apresenta, de forma gráfica e dinâmica, a evolução da série real e das previsões ao longo do tempo.

## �� Como Executar

### Opção 1: Executável Simples
```bash
python run.py
```

### Opção 2: Execução Direta
```bash
python main.py
```

### Opção 3: Execução com Configurações Personalizadas
```bash
python main.py --duration 60 --window-size 60 --predict-ahead 5 --update-interval 60
```

## 📋 Funcionalidades Implementadas

### 1. **Previsão de 5 Minutos à Frente**
- Sistema prevê o preço do BTC para os próximos 5 minutos
- Utiliza dados de 1 minuto de resolução via yfinance
- Compara dois modelos: RNN simples e LSTM

### 2. **Simulação Gráfica em Tempo Real**
- **Série histórica real** acumulada ao longo do tempo
- **Previsões do modelo** a cada novo instante
- **Erros acumulados** da previsão
- **4 gráficos simultâneos**:
  - Preços reais vs previsões dos modelos
  - Erro acumulado ao longo do tempo
  - Métricas LSTM em janela móvel
  - Métricas RNN em janela móvel

### 3. **Métricas Quantitativas em Janela Móvel**
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coeficiente de Determinação)

### 4. **Simulação Contínua**
- Atualização automática a cada minuto
- Reavaliação contínua dos modelos
- Salvamento automático de métricas

## 🛠 Tecnologias Utilizadas

- **Python 3.8+**
- **TensorFlow/Keras** - Redes neurais
- **yfinance** - Dados do BTC em tempo real
- **matplotlib** - Visualização dinâmica
- **scikit-learn** - Métricas de avaliação
- **pandas/numpy** - Manipulação de dados

## 📁 Estrutura do Projeto

```
├── data/
│   └── fetch_btc.py          # Coleta de dados BTC/USD
├── models/
│   ├── lstm_model.py         # Modelo LSTM
│   └── rnn_model.py          # Modelo RNN
├── plots/                    # Gráficos e métricas salvos
├── main.py                   # Sistema principal
├── run.py                    # Executável simples
├── requirements.txt          # Dependências
└── README.md                # Documentação
```

## 📊 Saídas do Sistema

### Arquivos Gerados na Pasta `plots/`:
- `metrics_history.csv`: Histórico completo das métricas
- `final_prediction_plot.png`: Gráfico final da simulação

### Informações Exibidas em Tempo Real:
```
⏰ 14:30:15 | Iteração 25
💰 Preço Atual: $45,250.50
🔮 LSTM (5min): $45,280.30
🔮 RNN (5min): $45,265.80
📊 MAE - LSTM: 125.45 | RNN: 142.30
```

## 🔧 Configurações Disponíveis

### Parâmetros Principais:
- `WINDOW_SIZE`: Tamanho da janela de dados (padrão: 60)
- `PREDICT_AHEAD`: Minutos à frente para prever (padrão: 5)
- `ROLLING_WINDOW`: Janela móvel para métricas (padrão: 100)
- `UPDATE_INTERVAL`: Intervalo de atualização em segundos (padrão: 60)

### Configurações dos Modelos:
- **LSTM**: 2 camadas (128, 64) com dropout
- **RNN**: 2 camadas (128, 64) com dropout
- **Dropout**: 0.2
- **Épocas**: 10
- **Batch Size**: 32

## 📈 Interpretação dos Resultados

### Métricas de Erro:
- **MAE**: Erro absoluto médio (menor = melhor)
- **RMSE**: Erro quadrático médio (menor = melhor)
- **MAPE**: Erro percentual médio (menor = melhor)
- **R²**: Qualidade do ajuste (mais próximo de 1 = melhor)

### Comparação de Modelos:
- **LSTM**: Geralmente melhor para padrões complexos
- **RNN**: Mais simples, pode ser mais rápido

## ⚠️ Requisitos do Sistema

### Dependências:
```bash
pip install -r requirements.txt
```

### Conexão com Internet:
- Necessária para obter dados do BTC em tempo real
- Sistema usa yfinance para dados do Yahoo Finance

### Recursos Computacionais:
- Mínimo: 4GB RAM
- Recomendado: 8GB+ RAM para simulações longas
- GPU opcional para aceleração do TensorFlow

## 🐛 Solução de Problemas

### Erro de Conexão:
```
❌ Erro ao obter dados: [Errno 11001]
```
**Solução**: Verificar conexão com internet

### Erro de Memória:
```
RuntimeError: CUDA out of memory
```
**Solução**: Reduzir batch_size ou usar CPU

### Dados Insuficientes:
```
⚠️ Aviso: Apenas 45 pontos de dados obtidos
```
**Solução**: Sistema automaticamente busca mais dados históricos

## 📝 Análise Crítica e Discussão

### Estabilidade do Modelo:
- Os modelos LSTM e RNN demonstram diferentes níveis de estabilidade
- LSTM geralmente apresenta menor variância nas previsões
- RNN pode ser mais sensível a ruídos de mercado

### Sensibilidade a Ruídos:
- Mercado de criptomoedas é altamente volátil
- Modelos podem ser afetados por eventos externos
- Janela móvel ajuda a suavizar impactos temporários

### Limitações Práticas:
- Previsões são probabilísticas, não determinísticas
- Não deve ser usado como única base para decisões financeiras
- Requer monitoramento contínuo e ajustes

### Aplicabilidade Real:
- Útil para análise de tendências de curto prazo
- Pode auxiliar em estratégias de trading automatizado
- Necessita validação em diferentes condições de mercado

## 🎯 Requisitos Obrigatórios Atendidos

- ✅ **Prever 5 minutos à frente** com base em dados anteriores
- ✅ **Simulação contínua** com atualização de previsões e exibição gráfica em tempo real
- ✅ **Comparação contínua** entre previsões e valores reais
- ✅ **Dois modelos distintos** de RNN (RNN simples e LSTM)
- ✅ **Métricas de avaliação** reportadas em gráfico ao longo da simulação

## 👨‍💻 Equipe
Desenvolvido por alunos da UERJ/IPRJ:

João Vinicius Vitral

[Adicionar nomes dos colegas de grupo aqui]

## 📌 Licença
Este projeto é de uso educacional e não deve ser utilizado para decisões financeiras reais.

## 📚 Documentação Adicional

Para informações detalhadas sobre uso e configuração, consulte:
- `notebook.ipynb` - Versão exploratória (Google Colab)
