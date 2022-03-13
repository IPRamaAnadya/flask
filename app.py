# import library
from crypt import methods
from pyexpat import model
from flask import Flask, render_template, request
from flask_restful import Resource, Api
from flask_cors import CORS
import os
import librosa
import numpy as np
import librosa.feature
import pickle
# init object flask
app = Flask(__name__)

# init object flask restfull
api = Api(app)

# init cors
CORS(app)

res = {}
model = pickle.load(open("model.pkl", 'rb'))


@app.route("/")
def landing():
    res["value"] = "entahlah"
    return res

@app.route("/data", methods=["GET", "POST"])
def coba():
    if request.method == "GET":
        return res
    if request.method == 'POST':
        save_path = os.path.join("audio/", "temp.wav")
        request.files['audio_data'].save(save_path)
        data = prediction()
        res["result"] = data
        return res

@app.route("/aksara", methods=["GET"])
def aksara():
    data = prediction()
    res["result"] = data
    return res

def prediction():
    global model
    mfcc = np.array(mean_mfccs("audio/temp.wav"))
    X = np.reshape(mfcc,(1, mfcc.size))
    result = model.predict(X)
    return result[0]

def mean_mfccs(p):
    x, sr = librosa.load(p)
    return [np.mean(feature) for feature in librosa.feature.mfcc(x)]

if __name__ == "__main__":
    app.run(debug=True, port = int(os.environ.get('PORT', 5000)))