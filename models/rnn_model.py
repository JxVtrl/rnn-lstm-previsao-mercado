from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

def build_rnn(input_shape):
    """
    Constrói modelo RNN simples para previsão de séries temporais
    """
    model = Sequential([
        SimpleRNN(128, return_sequences=True, input_shape=input_shape),
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
