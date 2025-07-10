# main.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

from data.fetch_btc import fetch_btc_data
from models.lstm_model import build_lstm
# from models.rnn_model import build_rnn  # opcional para alternar

WINDOW_SIZE = 60
PREDICT_AHEAD = 5

def create_sequences(data, window, ahead):
    X, y = [], []
    for i in range(len(data) - window - ahead):
        X.append(data[i:i+window])
        y.append(data[i+window+ahead])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    df = fetch_btc_data()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = create_sequences(scaled_data, WINDOW_SIZE, PREDICT_AHEAD)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm((X.shape[1], 1))  # ou build_rnn((X.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    y_pred = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)

    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    r2 = r2_score(y_test_inv, y_pred_inv)

    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label="Real")
    plt.plot(y_pred_inv, label="Previsto")
    plt.title("Previsão BTC/USD - LSTM")
    plt.legend()
    plt.grid()
    plt.show()
