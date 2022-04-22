import numpy as np
import librosa.feature
import librosa.display
import pandas as pd
from sklearn.metrics import accuracy_score
import pickle
from sklearn.model_selection import train_test_split
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from collections import Counter
import operator
import noisereduce as nr
from knn import KNN
from extraction import Extraction
from preprocessing import Preprocessing

class MyModule(object):
    def __init__(self, k = 11):
        self.model_a = KNN(k=k)
        self.model_i = KNN(k=k)
        self.model_u = KNN(k=k)
        self.model_ē = KNN(k=k)
        self.model_o = KNN(k=k)
        self.model_e = KNN(k=k)
        self.model_kata = KNN(k=k)
        self.train_model()
        self.prep = Preprocessing()
        self.extraction = Extraction()
    def train_model(self):
        df_fitur_a = pd.read_csv("train/data_a.csv")
        df_label_a = df_fitur_a.pop("label")
        fitur_a = df_fitur_a.stack().groupby(level=0).apply(list).tolist()
        label_a = df_label_a.values.tolist()
        self.model_a.train(fitur_a, label_a)
        
        df_fitur_kata = pd.read_csv("train/data_kata.csv")
        df_label_kata = df_fitur_kata.pop("label")
        fitur_kata = df_fitur_kata.stack().groupby(level=0).apply(list).tolist()
        label_kata = df_label_kata.values.tolist()
        self.model_kata.train(fitur_kata, label_kata)
        
    def predict(self, x, sr):
        prep_data = self.prep.get(x=x)
        extract_data = self.extraction.get(prep_data)
        clasification = self.model_kata.predict(np.reshape(extract_data, (1,-1)))[0]
        return clasification