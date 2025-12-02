import joblib
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Embedding, Dense, Bidirectional, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from dataset import load,balance
from utils import plot_history
from evaluate import evaluate

def run_training():

    data = load("R:\nlp\Sentiment Analysis\data\training.1600000.processed.noemoticon.csv")
    data=balance(data)
    tokenizer = Tokenizer(num_words=50000, oov_token="<OOV>")
    tokenizer.fit_on_texts(data['clean_text'])

    sequences = tokenizer.texts_to_sequences(data['clean_text'])
    max_len = max(len(s) for s in sequences)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    X = padded
    y = data['target'].values

    joblib.dump(tokenizer, "models/tokenizer.pkl")
    open("models/max_len.txt","w").write(str(max_len))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )

    def build_lstm():
        model_lstm = Sequential()
        model_lstm.add(Embedding(input_dim=50000, output_dim=64, input_length=max_len))
        model_lstm.add(Bidirectional(LSTM(64)))
        model_lstm.add(Dropout(0.3)) 
        model_lstm.add(Dense(16, activation='relu'))
        model_lstm.add(Dropout(0.5)) 
        model_lstm.add(Dense(1, activation='sigmoid'))
        optimizer = Adam(learning_rate=3e-5)
        model_lstm.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model_lstm

    # Train LSTM
    model_lstm = build_lstm()
    history_lstm = model_lstm.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=15, batch_size=1024, callbacks=callbacks,
        class_weight=dict(enumerate(class_weights))
    )
    model_lstm.save("models/lstm_model.h5")
    plot_history(history_lstm, "LSTM")

