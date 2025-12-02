import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from processing import clean_text

# Load trained model and tokenizer
model = load_model("models/lstm_model.h5")
tokenizer = joblib.load("models/tokenizer.pkl")
max_len = int(open("models/max_len.txt").read())

def predict_sentiment(text: str):
    """
    Predict the sentiment of a given text.
    Returns 0 for Negative, 1 for Positive.
    """
    cleaned_text = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded_seq = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    
    pred = model.predict(padded_seq, verbose=0)[0][0]
    sentiment = 1 if pred >= 0.5 else 0
    return sentiment, float(pred)