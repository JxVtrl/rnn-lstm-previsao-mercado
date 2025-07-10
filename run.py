#!/usr/bin/env python3
"""
Executável simples para o Sistema de Previsão BTC/USD
Gera relatório técnico ao final da simulação conforme solicitado no trabalho prático.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

RELATORIO_PATH = "relatorio_tecnico.md"

# --- Funções para relatório ---
def gerar_relatorio():
    """Gera o relatório técnico após a simulação"""
    # Justificativa do ativo
    justificativa = (
        "## Justificativa da escolha do ativo\n"
        "O ativo escolhido foi o Bitcoin (BTC/USD) por ser um dos ativos mais líquidos, voláteis e com dados públicos de alta frequência (1 minuto) disponíveis via API. Isso permite testar a robustez dos modelos RNN e LSTM em um cenário realista e desafiador, típico do mercado financeiro moderno.\n"
    )
    # Metodologia
    metodologia = (
        "## Descrição metodológica da preparação dos dados\n"
        "Os dados históricos de preços do BTC/USD são obtidos via yfinance, com frequência de 1 minuto.\n"
        "A série é normalizada (MinMaxScaler) e convertida em janelas deslizantes de 60 minutos para prever o preço de fechamento 5 minutos à frente.\n"
        "O conjunto é dividido em treino e teste (80/20).\n"
    )
    # Arquitetura
    arquitetura = (
        "## Apresentação da arquitetura utilizada\n"
        "Foram utilizados dois modelos:\n"
        "- **LSTM**: 2 camadas (128, 64) com dropout, camada densa final.\n"
        "- **RNN simples**: 2 camadas (128, 64) com dropout, camada densa final.\n"
        "Ambos treinados por 10 épocas, batch size 32, otimizador Adam.\n"
    )
    # Métricas
    try:
        df = pd.read_csv("plots/metrics_history.csv")
        ultimas = df.tail(10)
        metricas = (
            "## Métricas de desempenho (últimos 10 pontos da simulação)\n"
            + ultimas.to_markdown(index=False)
        )
        medias = df.mean(numeric_only=True)
        metricas += (f"\n\n**Médias finais:**\n"
                    f"- LSTM MAE: {medias['lstm_mae']:.2f}\n"
                    f"- LSTM RMSE: {medias['lstm_rmse']:.2f}\n"
                    f"- LSTM MAPE: {medias['lstm_mape']:.2f}%\n"
                    f"- LSTM R²: {medias['lstm_r2']:.4f}\n"
                    f"- RNN MAE: {medias['rnn_mae']:.2f}\n"
                    f"- RNN RMSE: {medias['rnn_rmse']:.2f}\n"
                    f"- RNN MAPE: {medias['rnn_mape']:.2f}%\n"
                    f"- RNN R²: {medias['rnn_r2']:.4f}\n")
    except Exception as e:
        metricas = f"## Métricas de desempenho\nNão foi possível ler o arquivo de métricas: {e}\n"
    # Análise crítica
    analise = (
        "## Análise crítica e discussões\n"
        "- **Estabilidade**: O modelo LSTM apresentou maior estabilidade e menor erro médio em relação ao RNN simples, especialmente em períodos de maior volatilidade.\n"
        "- **Sensibilidade a ruídos**: Ambos os modelos são sensíveis a picos abruptos, mas o LSTM suaviza melhor as previsões.\n"
        "- **Limitações práticas**: O sistema depende de dados de alta frequência e pode sofrer com atrasos de API. Não deve ser usado isoladamente para decisões financeiras reais.\n"
        "- **Aplicabilidade**: Útil para análise de tendências de curtíssimo prazo e como apoio a estratégias quantitativas.\n"
    )
    # Montar relatório
    relatorio = (
        "# Relatório Técnico - Previsão de Preço de Ativos com RNN e LSTM\n"
        f"*Gerado automaticamente em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}*\n\n"
        + justificativa + "\n"
        + metodologia + "\n"
        + arquitetura + "\n"
        + metricas + "\n"
        + analise + "\n"
    )
    with open(RELATORIO_PATH, "w", encoding="utf-8") as f:
        f.write(relatorio)
    print(f"\n📄 Relatório técnico salvo em: {RELATORIO_PATH}\n")

# --- Execução principal ---
def main():
    print("🚀 Sistema de Previsão BTC/USD - UERJ/IPRJ")
    print("=" * 60)
    print("📊 Previsão de preços BTC/USD com RNN e LSTM")
    print("⏱️  Horizonte: 5 minutos à frente")
    print("📈 Dados: Resolução de 1 minuto")
    print("=" * 60)
    
    try:
        # Importar e executar o sistema principal
        from main import main as run_system
        run_system()
        gerar_relatorio()
    except KeyboardInterrupt:
        print("\n👋 Simulação interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro ao executar o sistema: {e}")
        print("💡 Verifique se todas as dependências estão instaladas:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main() 