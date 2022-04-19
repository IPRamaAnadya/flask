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
import noisereduce as nr

# init object flask
app = Flask(__name__)

# init object flask restfull
api = Api(app)

# init cors
CORS(app)

res = {}
aksara = ["h","n","c","r","k","d","t",'s',"w","l","m","g","b","ng","p","j","y","ny"]

model_a = pickle.load(open("models/model_a.pkl", 'rb'))
label_a = [x+"a" for x in aksara]

model_i = pickle.load(open("models/model_i.pkl", 'rb'))
label_i = [x+"i" for x in aksara]

model_u = pickle.load(open("models/model_u.pkl", 'rb'))
label_u = [x+"u" for x in aksara]

model_ē = pickle.load(open("models/model_ē.pkl", 'rb'))
label_ē = [x+"ē" for x in aksara]

model_o = pickle.load(open("models/model_o.pkl", 'rb'))
label_o = [x+"o" for x in aksara]

model_e = pickle.load(open("models/model_e.pkl", 'rb'))
label_e = [x+"e" for x in aksara]

model_kata = pickle.load(open("models/model_kata.pkl", 'rb'))
label_kata = ["adi", "satē", "mēmē", "bape", "melali"]

def prediction_a():
    global model_a
    mfcc = get_mfcc("audio/temp.wav")
    X = np.reshape(mfcc,(1, mfcc.size))
    result = model_a.predict(X)
    result_string = label_a[result[0]]
    return result_string

def prediction_i():
    global model_i
    mfcc = get_mfcc("audio/temp.wav")
    X = np.reshape(mfcc,(1, mfcc.size))
    result = model_i.predict(X)
    result_string = label_i[result[0]]
    return result_string

def prediction_u():
    global model_u
    mfcc = get_mfcc("audio/temp.wav")
    X = np.reshape(mfcc,(1, mfcc.size))
    result = model_u.predict(X)
    result_string = label_u[result[0]]
    return result_string

def prediction_ē():
    global model_ē
    mfcc = get_mfcc("audio/temp.wav")
    X = np.reshape(mfcc,(1, mfcc.size))
    result = model_ē.predict(X)
    result_string = label_ē[result[0]]
    return result_string

def prediction_o():
    global model_o
    mfcc = get_mfcc("audio/temp.wav")
    X = np.reshape(mfcc,(1, mfcc.size))
    result = model_o.predict(X)
    result_string = label_o[result[0]]
    return result_string

def prediction_e():
    global model_e
    mfcc = get_mfcc("audio/temp.wav")
    X = np.reshape(mfcc,(1, mfcc.size))
    result = model_e.predict(X)
    result_string = label_e[result[0]]
    return result_string

def prediction_kata():
    global model_kata
    mfcc = get_mfcc("audio/temp.wav")
    X = np.reshape(mfcc,(1, mfcc.size))
    result = model_kata.predict(X)
    result_string = label_kata[result[0]]
    return result_string

def normalize_sample(x):
    x -= np.mean(x)
    return x

def noise_reduce(x, sr):
    reduced_noise = nr.reduce_noise(y=x, sr=sr)
    return reduced_noise

def get_mfcc(p):
    x, sr = librosa.load(p)
    x = noise_reduce(x, sr)
    x = normalize_sample(x)
    mfcc = librosa.feature.mfcc(x, n_mfcc = 13)
#     mfcc /= np.amax(np.absolute(mfcc))
    mfcc = np.delete(mfcc,0,0)
    return np.ndarray.flatten(np.array([x[0:10] for x in mfcc]))

@app.route("/")
def landing():
    return res

@app.route("/data", methods=["GET", "POST"])
def get_prediction():
    if request.method == "GET":
        data = prediction_a()
        res["result"] = data
        return res
    if request.method == 'POST':
        save_path = os.path.join("audio/", "temp.wav")
        request.files['audio_data'].save(save_path)
        request_model = request.form["request_model"]
        try:
            data = prediction_a() if (request_model == "a") else prediction_i() if (request_model == "i") else prediction_u() if (request_model == "u") else prediction_ē() if (request_model == "ē") else prediction_o() if (request_model == "o") else prediction_e() if (request_model == "e") else prediction_kata() if (request_model == "kata") else "error#ModelNotFound"
        except:
            data = "error#AudioIsToShort"
        res["result"] = data
        return res

if __name__ == "__main__":
    app.run(debug=True, port = int(os.environ.get('PORT', 5000)))
