#!/usr/bin/env python3
"""
Executável simples para o Sistema de Previsão BTC/USD
"""

import sys
import os

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Executável principal"""
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
        
    except KeyboardInterrupt:
        print("\n👋 Simulação interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro ao executar o sistema: {e}")
        print("💡 Verifique se todas as dependências estão instaladas:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main() 