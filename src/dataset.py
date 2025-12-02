import pandas as pd
from sklearn.utils import resample
from tqdm import tqdm
from processing import clean_text

def load(path):
    data = pd.read_csv(path, encoding='ISO-8859-1')
    data.columns = ['target','id','data','quary','user','text']

    data = data[['target','text']]
    data['target'] = data['target'].replace(4, 1)
    return data

def balance(data):
    majority = data[data['target'] == 0]
    minority = data[data['target'] == 1]

    minority_upsampled = resample(
        minority, replace=True,
        n_samples=len(majority),
        random_state=42
    )

    balanced = pd.concat([majority, minority_upsampled])
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    tqdm.pandas()
    balanced["clean_text"] = balanced["text"].progress_apply(clean_text)

    return balanced
