# main.py

import numpy as np
import pandas as pd
import time
import argparse
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import os

from data.fetch_btc import fetch_btc_data
from models.lstm_model import build_lstm
from models.rnn_model import build_rnn

# Configurações
WINDOW_SIZE = 20  # Janela de 20 minutos (teste rápido)
PREDICT_AHEAD = 5  # Prever 5 minutos à frente
ROLLING_WINDOW = 10  # Janela móvel para métricas (teste rápido)
UPDATE_INTERVAL = 2  # Atualizar a cada 2 segundos (teste rápido)

class BTCPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.lstm_model = None
        self.rnn_model = None
        self.history = []
        self.predictions_lstm = []
        self.predictions_rnn = []
        self.actual_values = []
        self.timestamps = []
        self.metrics_history = {
            'lstm': {'mae': [], 'rmse': [], 'mape': [], 'r2': []},
            'rnn': {'mae': [], 'rmse': [], 'mape': [], 'r2': []}
        }
        
    def create_sequences(self, data, window, ahead):
        """Cria sequências para treinamento"""
        X, y = [], []
        for i in range(len(data) - window - ahead):
            X.append(data[i:i+window])
            y.append(data[i+window+ahead])
        return np.array(X), np.array(y)
    
    def calculate_mape(self, y_true, y_pred):
        """Calcula MAPE (Mean Absolute Percentage Error)"""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def calculate_rolling_metrics(self, actual, predicted, window_size=ROLLING_WINDOW):
        """Calcula métricas em janela móvel"""
        if len(actual) < window_size:
            return None, None, None, None
        
        recent_actual = actual[-window_size:]
        recent_predicted = predicted[-window_size:]
        
        mae = mean_absolute_error(recent_actual, recent_predicted)
        rmse = np.sqrt(mean_squared_error(recent_actual, recent_predicted))
        mape = self.calculate_mape(recent_actual, recent_predicted)
        r2 = r2_score(recent_actual, recent_predicted)
        
        return mae, rmse, mape, r2
    
    def train_models(self, data):
        """Treina os modelos LSTM e RNN"""
        print("🔄 Treinando modelos...")
        
        # Preparar dados
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = self.create_sequences(scaled_data, WINDOW_SIZE, PREDICT_AHEAD)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Dividir dados
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Treinar LSTM
        self.lstm_model = build_lstm((X.shape[1], 1))
        self.lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, 
                           validation_split=0.1, verbose=0)
        
        # Treinar RNN
        self.rnn_model = build_rnn((X.shape[1], 1))
        self.rnn_model.fit(X_train, y_train, epochs=10, batch_size=32, 
                          validation_split=0.1, verbose=0)
        
        print("✅ Modelos treinados com sucesso!")
    
    def predict_next(self, recent_data):
        """Faz previsão para os próximos 5 minutos"""
        if len(recent_data) < WINDOW_SIZE:
            return None, None
        
        # Preparar dados para previsão
        scaled_data = self.scaler.transform(recent_data.reshape(-1, 1))
        sequence = scaled_data[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
        
        # Fazer previsões
        pred_lstm = self.lstm_model.predict(sequence, verbose=0)
        pred_rnn = self.rnn_model.predict(sequence, verbose=0)
        
        # Converter de volta para preços reais
        pred_lstm_price = self.scaler.inverse_transform(pred_lstm)[0][0]
        pred_rnn_price = self.scaler.inverse_transform(pred_rnn)[0][0]
        
        return pred_lstm_price, pred_rnn_price
    
    def update_metrics(self):
        """Atualiza métricas em janela móvel"""
        if len(self.actual_values) < ROLLING_WINDOW:
            return
        
        # Métricas para LSTM
        mae_lstm, rmse_lstm, mape_lstm, r2_lstm = self.calculate_rolling_metrics(
            self.actual_values, self.predictions_lstm
        )
        
        # Métricas para RNN
        mae_rnn, rmse_rnn, mape_rnn, r2_rnn = self.calculate_rolling_metrics(
            self.actual_values, self.predictions_rnn
        )
        
        # Armazenar métricas
        if mae_lstm is not None:
            self.metrics_history['lstm']['mae'].append(mae_lstm)
            self.metrics_history['lstm']['rmse'].append(rmse_lstm)
            self.metrics_history['lstm']['mape'].append(mape_lstm)
            self.metrics_history['lstm']['r2'].append(r2_lstm)
            
            self.metrics_history['rnn']['mae'].append(mae_rnn)
            self.metrics_history['rnn']['rmse'].append(rmse_rnn)
            self.metrics_history['rnn']['mape'].append(mape_rnn)
            self.metrics_history['rnn']['r2'].append(r2_rnn)
    
    def save_metrics_to_csv(self):
        """Salva métricas em arquivo CSV"""
        if not self.metrics_history['lstm']['mae']:
            return
        
        df_metrics = pd.DataFrame({
            'timestamp': self.timestamps[-len(self.metrics_history['lstm']['mae']):],
            'lstm_mae': self.metrics_history['lstm']['mae'],
            'lstm_rmse': self.metrics_history['lstm']['rmse'],
            'lstm_mape': self.metrics_history['lstm']['mape'],
            'lstm_r2': self.metrics_history['lstm']['r2'],
            'rnn_mae': self.metrics_history['rnn']['mae'],
            'rnn_rmse': self.metrics_history['rnn']['rmse'],
            'rnn_mape': self.metrics_history['rnn']['mape'],
            'rnn_r2': self.metrics_history['rnn']['r2']
        })
        
        os.makedirs('plots', exist_ok=True)
        df_metrics.to_csv('plots/metrics_history.csv', index=False)
        print("📊 Métricas salvas em plots/metrics_history.csv")
    
    def create_live_plot(self):
        """Cria gráfico dinâmico em tempo real"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Sistema de Previsão BTC/USD - Tempo Real', fontsize=16)
        
        def animate(frame):
            if len(self.actual_values) < 2:
                # Mensagem em todos os subplots
                ax1.clear(); ax1.set_title('Preços BTC/USD e Previsões'); ax1.set_ylabel('Preço (USD)')
                ax1.text(0.5, 0.5, 'Aguardando dados suficientes...', ha='center', va='center', fontsize=14, color='gray', transform=ax1.transAxes)
                ax2.clear(); ax2.set_title('Erro Acumulado'); ax2.set_ylabel('Erro Acumulado')
                ax2.text(0.5, 0.5, 'Aguardando dados suficientes...', ha='center', va='center', fontsize=14, color='gray', transform=ax2.transAxes)
                ax3.clear(); ax3.set_title('Métricas LSTM (Janela Móvel)'); ax3.set_ylabel('Erro')
                ax3.text(0.5, 0.5, 'Aguardando dados suficientes...', ha='center', va='center', fontsize=14, color='gray', transform=ax3.transAxes)
                ax4.clear(); ax4.set_title('Métricas RNN (Janela Móvel)'); ax4.set_ylabel('Erro')
                ax4.text(0.5, 0.5, 'Aguardando dados suficientes...', ha='center', va='center', fontsize=14, color='gray', transform=ax4.transAxes)
                return
            
            # Limpar gráficos
            for ax, title, ylabel in zip(
                [ax1, ax2, ax3, ax4],
                ['Preços BTC/USD e Previsões', 'Erro Acumulado', 'Métricas LSTM (Janela Móvel)', 'Métricas RNN (Janela Móvel)'],
                ['Preço (USD)', 'Erro Acumulado', 'Erro', 'Erro']):
                ax.clear()
                ax.set_title(title)
                ax.set_ylabel(ylabel)
            
            # Gráfico 1: Preços e Previsões
            ax1.plot(self.timestamps, self.actual_values, 'b-', label='Real', linewidth=2)
            if self.predictions_lstm:
                ax1.plot(self.timestamps, self.predictions_lstm, 'r--', label='LSTM', linewidth=1.5)
            if self.predictions_rnn:
                ax1.plot(self.timestamps, self.predictions_rnn, 'g--', label='RNN', linewidth=1.5)
            ax1.legend()
            ax1.grid(True)
            ax1.tick_params(axis='x', rotation=45)
            if len(self.timestamps) > 20:
                step = max(1, len(self.timestamps) // 20)
                ax1.set_xticks(self.timestamps[::step])
            
            # Gráfico 2: Erro Acumulado
            if len(self.actual_values) > 1 and self.predictions_lstm and self.predictions_rnn:
                error_lstm = np.array(self.actual_values) - np.array(self.predictions_lstm)
                error_rnn = np.array(self.actual_values) - np.array(self.predictions_rnn)
                ax2.plot(self.timestamps, np.cumsum(error_lstm), 'r-', label='Erro LSTM', linewidth=2)
                ax2.plot(self.timestamps, np.cumsum(error_rnn), 'g-', label='Erro RNN', linewidth=2)
                ax2.legend()
                ax2.grid(True)
                ax2.tick_params(axis='x', rotation=45)
                if len(self.timestamps) > 20:
                    step = max(1, len(self.timestamps) // 20)
                    ax2.set_xticks(self.timestamps[::step])
            else:
                ax2.text(0.5, 0.5, 'Aguardando dados suficientes...', ha='center', va='center', fontsize=14, color='gray', transform=ax2.transAxes)
            
            # Gráfico 3: Métricas LSTM
            if self.metrics_history['lstm']['mae']:
                ax3.plot(self.metrics_history['lstm']['mae'], 'r-', label='MAE', linewidth=2)
                ax3.plot(self.metrics_history['lstm']['rmse'], 'g-', label='RMSE', linewidth=2)
                ax3.legend()
                ax3.grid(True)
            else:
                ax3.text(0.5, 0.5, 'Aguardando dados suficientes...', ha='center', va='center', fontsize=14, color='gray', transform=ax3.transAxes)
            
            # Gráfico 4: Métricas RNN
            if self.metrics_history['rnn']['mae']:
                ax4.plot(self.metrics_history['rnn']['mae'], 'r-', label='MAE', linewidth=2)
                ax4.plot(self.metrics_history['rnn']['rmse'], 'g-', label='RMSE', linewidth=2)
                ax4.legend()
                ax4.grid(True)
            else:
                ax4.text(0.5, 0.5, 'Aguardando dados suficientes...', ha='center', va='center', fontsize=14, color='gray', transform=ax4.transAxes)
            
            plt.tight_layout()
        
        return fig, animate
    
    def save_individual_plots(self):
        """Salva cada gráfico em arquivo separado na pasta plots/"""
        import matplotlib.pyplot as plt
        os.makedirs('plots', exist_ok=True)
        # 1. Preços e Previsões
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.set_title('Preços BTC/USD e Previsões')
        ax1.set_ylabel('Preço (USD)')
        ax1.plot(self.timestamps, self.actual_values, 'b-', label='Real', linewidth=2)
        if self.predictions_lstm:
            ax1.plot(self.timestamps, self.predictions_lstm, 'r--', label='LSTM', linewidth=1.5)
        if self.predictions_rnn:
            ax1.plot(self.timestamps, self.predictions_rnn, 'g--', label='RNN', linewidth=1.5)
        ax1.legend()
        ax1.grid(True)
        ax1.tick_params(axis='x', rotation=45)
        if len(self.timestamps) > 20:
            step = max(1, len(self.timestamps) // 20)
            ax1.set_xticks(self.timestamps[::step])
        plt.tight_layout()
        plt.savefig('plots/plot_precos_previsoes.png', dpi=300)
        plt.close(fig1)
        # 2. Erro Acumulado
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.set_title('Erro Acumulado')
        ax2.set_ylabel('Erro Acumulado')
        if len(self.actual_values) > 1 and self.predictions_lstm and self.predictions_rnn:
            error_lstm = np.array(self.actual_values) - np.array(self.predictions_lstm)
            error_rnn = np.array(self.actual_values) - np.array(self.predictions_rnn)
            ax2.plot(self.timestamps, np.cumsum(error_lstm), 'r-', label='Erro LSTM', linewidth=2)
            ax2.plot(self.timestamps, np.cumsum(error_rnn), 'g-', label='Erro RNN', linewidth=2)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Aguardando dados suficientes...', ha='center', va='center', fontsize=14, color='gray', transform=ax2.transAxes)
        ax2.grid(True)
        ax2.tick_params(axis='x', rotation=45)
        if len(self.timestamps) > 20:
            step = max(1, len(self.timestamps) // 20)
            ax2.set_xticks(self.timestamps[::step])
        plt.tight_layout()
        plt.savefig('plots/plot_erro_acumulado.png', dpi=300)
        plt.close(fig2)
        # 3. Métricas LSTM
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.set_title('Métricas LSTM (Janela Móvel)')
        ax3.set_ylabel('Erro')
        if self.metrics_history['lstm']['mae']:
            ax3.plot(self.metrics_history['lstm']['mae'], 'r-', label='MAE', linewidth=2)
            ax3.plot(self.metrics_history['lstm']['rmse'], 'g-', label='RMSE', linewidth=2)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Aguardando dados suficientes...', ha='center', va='center', fontsize=14, color='gray', transform=ax3.transAxes)
        ax3.grid(True)
        plt.tight_layout()
        plt.savefig('plots/plot_metricas_lstm.png', dpi=300)
        plt.close(fig3)
        # 4. Métricas RNN
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        ax4.set_title('Métricas RNN (Janela Móvel)')
        ax4.set_ylabel('Erro')
        if self.metrics_history['rnn']['mae']:
            ax4.plot(self.metrics_history['rnn']['mae'], 'r-', label='MAE', linewidth=2)
            ax4.plot(self.metrics_history['rnn']['rmse'], 'g-', label='RMSE', linewidth=2)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Aguardando dados suficientes...', ha='center', va='center', fontsize=14, color='gray', transform=ax4.transAxes)
        ax4.grid(True)
        plt.tight_layout()
        plt.savefig('plots/plot_metricas_rnn.png', dpi=300)
        plt.close(fig4)

    def run_simulation(self, duration_minutes=30):
        """Executa simulação contínua"""
        print(f"🚀 Iniciando simulação por {duration_minutes} minutos...")
        print("📊 Previsão: 5 minutos à frente")
        print("⏱️  Atualização: a cada minuto")
        print("=" * 50)
        
        # Obter dados iniciais
        df = fetch_btc_data()
        initial_data = df['price'].values
        
        # Treinar modelos
        self.train_models(initial_data)
        
        # Configurar gráfico dinâmico
        fig, animate = self.create_live_plot()
        ani = animation.FuncAnimation(fig, animate, interval=1000, blit=False, cache_frame_data=False)
        
        # Simulação contínua
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < duration_minutes * 60:
            try:
                # Obter novos dados
                df = fetch_btc_data()
                current_price = df['price'].iloc[-1]
                current_time = datetime.now().strftime("%H:%M:%S")
                
                # Adicionar à história
                self.history.append(current_price)
                self.actual_values.append(current_price)
                self.timestamps.append(current_time)
                
                # Fazer previsões se tivermos dados suficientes
                if len(self.history) >= WINDOW_SIZE:
                    pred_lstm, pred_rnn = self.predict_next(np.array(self.history))
                    
                    if pred_lstm is not None:
                        self.predictions_lstm.append(pred_lstm)
                        self.predictions_rnn.append(pred_rnn)
                        
                        # Atualizar métricas
                        self.update_metrics()
                        
                        # Exibir informações
                        print(f"\n⏰ {current_time} | Iteração {iteration}")
                        print(f"💰 Preço Atual: ${current_price:.2f}")
                        print(f"🔮 LSTM (5min): ${pred_lstm:.2f}")
                        print(f"🔮 RNN (5min): ${pred_rnn:.2f}")
                        
                        if self.metrics_history['lstm']['mae']:
                            latest_lstm_mae = self.metrics_history['lstm']['mae'][-1]
                            latest_rnn_mae = self.metrics_history['rnn']['mae'][-1]
                            print(f"📊 MAE - LSTM: {latest_lstm_mae:.2f} | RNN: {latest_rnn_mae:.2f}")
                
                iteration += 1
                time.sleep(UPDATE_INTERVAL)
                
            except Exception as e:
                print(f"❌ Erro: {e}")
                time.sleep(5)
        
        # Salvar métricas
        self.save_metrics_to_csv()
        
        # Salvar gráfico final
        plt.savefig('plots/final_prediction_plot.png', dpi=300, bbox_inches='tight')
        print("\n✅ Simulação concluída!")
        print("📁 Gráficos salvos em plots/")
        
        # Salvar gráficos individuais
        self.save_individual_plots()

        plt.show()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Sistema de Previsão BTC/USD com RNN e LSTM'
    )
    
    parser.add_argument(
        '--duration', 
        type=int, 
        default=30,
        help='Duração da simulação em minutos (padrão: 30)'
    )
    
    parser.add_argument(
        '--window-size', 
        type=int, 
        default=60,
        help='Tamanho da janela de dados (padrão: 60)'
    )
    
    parser.add_argument(
        '--predict-ahead', 
        type=int, 
        default=5,
        help='Minutos à frente para prever (padrão: 5)'
    )
    
    parser.add_argument(
        '--update-interval', 
        type=int, 
        default=60,
        help='Intervalo de atualização em segundos (padrão: 60)'
    )
    
    return parser.parse_args()

def main():
    """Função principal"""
    args = parse_arguments()
    
    # Configurar parâmetros globais
    global WINDOW_SIZE, PREDICT_AHEAD, UPDATE_INTERVAL
    WINDOW_SIZE = args.window_size
    PREDICT_AHEAD = args.predict_ahead
    UPDATE_INTERVAL = args.update_interval
    
    print("🚀 Sistema de Previsão BTC/USD")
    print("=" * 50)
    print(f"⏱️  Duração: {args.duration} minutos")
    print(f"📊 Janela: {args.window_size} minutos")
    print(f"🔮 Previsão: {args.predict_ahead} minutos à frente")
    print(f"🔄 Atualização: a cada {args.update_interval} segundos")
    print("=" * 50)
    
    # Executar simulação
    predictor = BTCPredictor()
    predictor.run_simulation(duration_minutes=args.duration)

if __name__ == "__main__":
    main()
