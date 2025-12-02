import pandas as pd
from dataset import load
from evaluate import evaluate
from predict import predict_sentiment

# Path to your test CSV
test_file_path = r"data/testdata.manual.2009.06.14.csv"
data=load(test_file_path)
y_test=data['target']
y_test_pred=[]

for text in data['text']:
     sentiment, confidence = predict_sentiment(text)
     label = "Positive" if sentiment == 1 else "Negative"
     y_test_pred.append(sentiment)
  
acc, cm , report=evaluate(y_test,y_test_pred)

print(f"Accuracy: {acc:.4f}")
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(report)
