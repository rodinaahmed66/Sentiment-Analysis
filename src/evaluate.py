import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

def evaluate( y_val, y_pred):
    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred, digits=4)
    return acc, cm , report
