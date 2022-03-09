import numpy as np
import librosa
import librosa.feature
import librosa.display
import glob
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from fastdtw import fastdtw
import random
from PIL import Image
from sklearn.metrics import accuracy_score
import operator
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def mean_mfccs(p):
    x, sr = librosa.load(p)
    return [np.mean(feature) for feature in librosa.feature.mfcc(x)]

def feature_extraction(data):
    fitur, label = [], []
    for key, values in data.items():
        for value in values:
            fitur.append(mean_mfccs(value))
            label.append(key)
    return fitur, label

directory = ["ayu","indri", "dede"]
konsonan = ["h", "n", "c", "r", "k", "d", "t", "s", "w", "l", "m", "g", "b", "ng", "p", "j", "y", "ny"]
vokal_a = {k+"a": [] for k in konsonan}

for d in directory:
    for v, l in vokal_a.items():
        path = glob.glob("data/split/"+d+"/a/"+v+"/*.wav")
        vokal_a[v].extend(path)

fitur_data_a, label_data_a = feature_extraction(vokal_a)
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(fitur_data_a, label_data_a, test_size=0.15, random_state=42)

model_a = KNeighborsClassifier(n_neighbors = 1)
model_a.fit(X_train_a, y_train_a)
y_pred_a = model_a.predict(X_test_a)
result_a = accuracy_score(y_test_a, y_pred_a, normalize = True)*100
print("akurasi model-a :", round(result_a,2), end= "%\n\n")

pickle.dump(model_a, open('flask/model.pkl','wb'))