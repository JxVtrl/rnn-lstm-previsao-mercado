from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Input

def build_rnn(input_shape):
    """
    Constrói modelo RNN simples para previsão de séries temporais
    """
    model = Sequential([
        Input(shape=input_shape),
        SimpleRNN(128, return_sequences=True),
        Dropout(0.2),
        SimpleRNN(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model
