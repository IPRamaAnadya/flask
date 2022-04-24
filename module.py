import numpy as np
import pandas as pd
from preprocessing import Preprocessing
from knn import KNN
from extraction import Extraction

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

        df_fitur_i = pd.read_csv("train/data_i.csv")
        df_label_i = df_fitur_i.pop("label")
        fitur_i = df_fitur_i.stack().groupby(level=0).apply(list).tolist()
        label_i = df_label_i.values.tolist()
        self.model_i.train(fitur_i, label_i)
        
        df_fitur_u = pd.read_csv("train/data_u.csv")
        df_label_u = df_fitur_u.pop("label")
        fitur_u = df_fitur_u.stack().groupby(level=0).apply(list).tolist()
        label_u = df_label_u.values.tolist()
        self.model_u.train(fitur_u, label_u)

        df_fitur_ē = pd.read_csv("train/data_ē.csv")
        df_label_ē = df_fitur_ē.pop("label")
        fitur_ē = df_fitur_ē.stack().groupby(level=0).apply(list).tolist()
        label_ē = df_label_ē.values.tolist()
        self.model_ē.train(fitur_ē, label_ē)

        df_fitur_o = pd.read_csv("train/data_o.csv")
        df_label_o = df_fitur_o.pop("label")
        fitur_o = df_fitur_o.stack().groupby(level=0).apply(list).tolist()
        label_o = df_label_o.values.tolist()
        self.model_o.train(fitur_o, label_o)

        df_fitur_e = pd.read_csv("train/data_e.csv")
        df_label_e = df_fitur_e.pop("label")
        fitur_e = df_fitur_e.stack().groupby(level=0).apply(list).tolist()
        label_e = df_label_e.values.tolist()
        self.model_e.train(fitur_e, label_e)

    def predict(self, x, sr, model = ""):
        prep_data = self.prep.get(x=x)
        dist = []
        extract_data = self.extraction.get(prep_data)
        if(model == "kata"): clasification, dist = self.model_kata.predict(np.reshape(extract_data, (1,-1)))
        elif(model == "a"): clasification, dist = self.model_a.predict(np.reshape(extract_data, (1,-1)))
        elif(model == "i"): clasification, dist = self.model_i.predict(np.reshape(extract_data, (1,-1)))
        elif(model == "u"): clasification, dist = self.model_u.predict(np.reshape(extract_data, (1,-1)))
        elif(model == "ē"): clasification, dist = self.model_ē.predict(np.reshape(extract_data, (1,-1)))
        elif(model == "o"): clasification, dist = self.model_o.predict(np.reshape(extract_data, (1,-1)))
        elif(model == "e"): clasification, dist = self.model_e.predict(np.reshape(extract_data, (1,-1)))
        else: clasification, dist = "Error#ModelNotFound", "None"
        return clasification, dist