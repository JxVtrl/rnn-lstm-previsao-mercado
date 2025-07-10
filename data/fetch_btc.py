import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_btc_data():
    """
    Obtém dados do BTC/USD com intervalo de 1 minuto
    Retorna dados das últimas 24 horas para garantir dados suficientes
    """
    try:
        # Obter dados das últimas 24 horas com intervalo de 1 minuto
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        
        df = yf.download(
            "BTC-USD", 
            start=start_time,
            end=end_time,
            interval="1m",
            auto_adjust=True
        )
        
        # Limpar dados
        df = df[['Close']].dropna()
        df.columns = ['price']
        
        # Garantir que temos pelo menos 60 pontos de dados
        if len(df) < 60:
            print(f"⚠️  Aviso: Apenas {len(df)} pontos de dados obtidos")
            # Se não temos dados suficientes, usar dados de 7 dias
            start_time = end_time - timedelta(days=7)
            df = yf.download(
                "BTC-USD", 
                start=start_time,
                end=end_time,
                interval="1m",
                auto_adjust=True
            )
            df = df[['Close']].dropna()
            df.columns = ['price']
        
        print(f"📊 Dados obtidos: {len(df)} pontos de {df.index[0]} até {df.index[-1]}")
        return df
        
    except Exception as e:
        print(f"❌ Erro ao obter dados: {e}")
        # Retornar dados de exemplo em caso de erro
        return pd.DataFrame({
            'price': [50000 + i * 10 for i in range(100)]
        }, index=pd.date_range(start='2024-01-01', periods=100, freq='1min'))
