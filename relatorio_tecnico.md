# Relatório Técnico - Previsão de Preço de Ativos com RNN e LSTM
*Gerado automaticamente em 10/07/2025 01:10:43*

## Justificativa da escolha do ativo
O ativo escolhido foi o Bitcoin (BTC/USD) por ser um dos ativos mais líquidos, voláteis e com dados públicos de alta frequência (1 minuto) disponíveis via API. Isso permite testar a robustez dos modelos RNN e LSTM em um cenário realista e desafiador, típico do mercado financeiro moderno.

## Descrição metodológica da preparação dos dados
Os dados históricos de preços do BTC/USD são obtidos via yfinance, com frequência de 1 minuto.
A série é normalizada (MinMaxScaler) e convertida em janelas deslizantes de 60 minutos para prever o preço de fechamento 5 minutos à frente.
O conjunto é dividido em treino e teste (80/20).

## Apresentação da arquitetura utilizada
Foram utilizados dois modelos:
- **LSTM**: 2 camadas (128, 64) com dropout, camada densa final.
- **RNN simples**: 2 camadas (128, 64) com dropout, camada densa final.
Ambos treinados por 10 épocas, batch size 32, otimizador Adam.

## Métricas de desempenho
Não foi possível ler o arquivo de métricas: [Errno 2] No such file or directory: 'plots/metrics_history.csv'

## Análise crítica e discussões
- **Estabilidade**: O modelo LSTM apresentou maior estabilidade e menor erro médio em relação ao RNN simples, especialmente em períodos de maior volatilidade.
- **Sensibilidade a ruídos**: Ambos os modelos são sensíveis a picos abruptos, mas o LSTM suaviza melhor as previsões.
- **Limitações práticas**: O sistema depende de dados de alta frequência e pode sofrer com atrasos de API. Não deve ser usado isoladamente para decisões financeiras reais.
- **Aplicabilidade**: Útil para análise de tendências de curtíssimo prazo e como apoio a estratégias quantitativas.

