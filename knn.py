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

class KNN(object):
    def __init__(self, k=5):
        self.k = k
    def train(self, x, y):
        self.x = x
        self.y = y
    def distance(self,a,b):
        return np.sqrt(np.sum(np.square(a-b)))
    def distance2(self,a,b):
        d,p = fastdtw(a,b)
        return d
    def get_label(self, arr):
        label = [x[-1] for x in arr]
        label = Counter(label).most_common(1)[0][0]
        return label
    def predict(self, X):
        result = []
        dist_arr = []
        for x in X:
            dist = []
            for j in range(len(self.x)):
                distance = self.distance2(x, self.x[j])
                dist.append([distance, self.y[j]])
            dist.sort(key=lambda row: (row[0]), reverse=False)
            result.append(self.get_label(dist[:self.k]))
            dist_arr.append(dist[:self.k])
        return result, dist_arr