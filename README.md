📘 Sentiment Analysis Using LSTM — End-to-End NLP Project

data link "https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset"

A complete Sentiment Analysis System built using an LSTM deep learning model, deployed with Streamlit, and trained on the 1.6M Tweets Dataset.
This project demonstrates a full machine-learning workflow: data preparation, preprocessing, training, evaluation, model saving, and app deployment.

🚀 Project Overview

This project predicts whether a given text expresses Positive or Negative sentiment using a trained LSTM neural network.

It includes:

A clean and scalable project structure

Separate modules for training, evaluation, preprocessing, and prediction

A Streamlit web app for real-time sentiment classification

Ready-to-deploy setup for GitHub + Streamlit Cloud

Saved TensorFlow LSTM model & tokenizer

📁 Project Architecture
Sentiment-Analysis/
```│
├── data/
│   ├── training.1600000.processed.noemoticon.csv
│   └── testdata.manual.2009.06.14.csv
│
├── models/
│   ├── lstm_model.h5
│   ├── tokenizer.pkl
│   └── max_len.txt
│
├── notebooks/
│   ├── sentiment-analysis.ipynb
│   └── sentiment-analysis (1).ipynb
│
├── src/
│   ├── app.py            # Streamlit app
│   ├── train.py          # Training the LSTM model
│   ├── evaluate.py       # Model evaluation
│   ├── predict.py        # Real-time prediction logic
│   ├── processing.py     # Text cleaning & preprocessing
│   ├── dataset.py        # Dataset utilities
│   └── test.py
│
├── utils/
│   └── plot_history.py   # Training curve visualization
│
├── .env (optional)
├── .gitignore
├── requirements.txt
└── README.md
```
🔍 Model Architecture (LSTM)

The final trained model includes:

Tokenizer → Sequence Conversion

Embedding Layer

LSTM Layer (128 units)

Dense Output Layer + Sigmoid Activation

Why LSTM?

LSTMs capture long-term context in text and perform well for sentiment classification compared to classical ML models.

🧹 Text Preprocessing Pipeline

Defined in src/processing.py:

✔ Convert text to lowercase
✔ Remove URLs
✔ Remove mentions & hashtags
✔ Remove punctuation & digits
✔ Remove extra spaces
✔ Tokenization
✔ Padding/truncation

This ensures the same preprocessing is applied during training & real-time predictions.

🏋️ Training the Model

Run:

```python src/train.py```


This script:

Loads and processes the dataset

Tokenizes and pads text

Trains the LSTM model

Saves:

```
models/lstm_model.h5
models/tokenizer.pkl
models/max_len.txt
```

📊 Model Evaluation

Run:

```python src/evaluate.py```


You will get:

Accuracy

Precision

Recall

F1-score

Confusion matrix

Training curves (via utils/plot_history.py)

⚡ Real-Time Sentiment Prediction

Example code from predict.py:
```
model = load_model("models/lstm_model.h5")
tokenizer = joblib.load("models/tokenizer.pkl")
max_len = int(open("models/max_len.txt").read())
```

To test manually:

from predict import predict_sentiment
predict_sentiment("I love this project!")

🌐 Streamlit Web App

Run locally:
```
streamlit run src/app.py
```

The app:

Accepts input text

Preprocesses it

Predicts sentiment using the LSTM

Displays:

Sentiment label

Model confidence score

☁ Deploy on Streamlit Cloud
1️⃣ Push your project to GitHub

Make sure these files exist:

✔ models/
✔ src/app.py
✔ requirements.txt

2️⃣ Go to Streamlit Cloud → “New app”

Select your GitHub repo:
```
Branch: main

Startup file:

src/app.py
```
3️⃣ Streamlit automatically installs:
tensorflow
numpy
pandas
nltk
joblib
sklearn

4️⃣ App goes live with a public URL 🎉
📦 requirements.txt

Make sure you include:
```
tensorflow
streamlit
joblib
numpy
pandas
scikit-learn
nltk
h5py
```

🧪 Example Predictions
Text	Prediction	Confidence
"I love this!"	Positive	0.97
"This is terrible!"	Negative	0.89
"Nothing special but okay"	Positive	0.61
🙌 Author

Your Name
Machine Learning & NLP Engineer

GitHub: (your link)

🎯 Final Notes

✔ No absolute paths → portable & deployable
✔ models/ paths must remain exactly:

models/lstm_model.h5
models/tokenizer.pkl
models/max_len.txt


✔ The project is fully compatible with GitHub & Streamlit Cloud
✔ Perfect for your portfolio or production demo
