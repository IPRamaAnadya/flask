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

class Extraction(object):
    def __init__(self,sr = 22050):
        self.sr = sr
    def get(self,x):
        mfcc = librosa.feature.mfcc(x, n_mfcc = 13)
        mfcc = np.delete(mfcc,0,0)
        return np.ndarray.flatten(mfcc)