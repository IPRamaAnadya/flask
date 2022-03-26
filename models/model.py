import numpy as np
import librosa.feature
import librosa.display
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

df_fitur_a = pd.read_csv("data_a.csv", sep=',')
df_label_a = df_fitur_a.pop("label")
fitur_a = df_fitur_a.stack().groupby(level=0).apply(list).tolist()
label_a = df_label_a.values.tolist()
label_a = np.array(label_a)
fitur_a = np.array(fitur_a, dtype = object)

df_fitur_i = pd.read_csv("data_i.csv", sep=',')
df_label_i = df_fitur_i.pop("label")
fitur_i = df_fitur_i.stack().groupby(level=0).apply(list).tolist()
label_i = df_label_i.values.tolist()
label_i = np.array(label_i)
fitur_i = np.array(fitur_i, dtype = object)

df_fitur_u = pd.read_csv("data_u.csv", sep=',')
df_label_u = df_fitur_u.pop("label")
fitur_u = df_fitur_u.stack().groupby(level=0).apply(list).tolist()
label_u = df_label_u.values.tolist()
label_u = np.array(label_u)
fitur_u = np.array(fitur_u, dtype = object)

df_fitur_ē = pd.read_csv("data_ē.csv", sep=',')
df_label_ē = df_fitur_ē.pop("label")
fitur_ē = df_fitur_ē.stack().groupby(level=0).apply(list).tolist()
label_ē = df_label_ē.values.tolist()
label_ē = np.array(label_ē)
fitur_ē = np.array(fitur_ē, dtype = object)

df_fitur_o = pd.read_csv("data_o.csv", sep=',')
df_label_o = df_fitur_o.pop("label")
fitur_o = df_fitur_o.stack().groupby(level=0).apply(list).tolist()
label_o = df_label_o.values.tolist()
label_o = np.array(label_o)
fitur_o = np.array(fitur_o, dtype = object)

df_fitur_e = pd.read_csv("data_e.csv", sep=',')
df_label_e = df_fitur_e.pop("label")
fitur_e = df_fitur_e.stack().groupby(level=0).apply(list).tolist()
label_e = df_label_e.values.tolist()
label_e = np.array(label_e)
fitur_e = np.array(fitur_e, dtype = object)

df_fitur_kata = pd.read_csv("data_kata.csv", sep=',')
df_label_kata = df_fitur_kata.pop("label")
fitur_kata = df_fitur_kata.stack().groupby(level=0).apply(list).tolist()
label_kata = df_label_kata.values.tolist()
label_kata = np.array(label_kata)
fitur_kata = np.array(fitur_kata, dtype = object)

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(fitur_a, label_a, test_size=0.15, random_state=42)

X_train_kata, X_test_kata, y_train_kata, y_test_kata = train_test_split(fitur_kata, label_kata, test_size=0.15, random_state=42)

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(fitur_i, label_i, test_size=0.15, random_state=42)

X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(fitur_u, label_u, test_size=0.15, random_state=42)

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(fitur_e, label_e, test_size=0.15, random_state=42)

X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(fitur_o, label_o, test_size=0.15, random_state=42)

X_train_ē, X_test_ē, y_train_ē, y_test_ē = train_test_split(fitur_ē, label_ē, test_size=0.15, random_state=42)

model_a = KNeighborsClassifier(n_neighbors = 3)
model_a.fit(X_train_a, y_train_a)
y_pred_a = model_a.predict(X_test_a)
result_a = accuracy_score(y_test_a, y_pred_a, normalize = True)*100
print("akurasi model-a :", round(result_a,2), end= "%\n\n")

model_i = KNeighborsClassifier(n_neighbors=3)
model_i.fit(X_train_i, y_train_i)
y_pred_i = model_i.predict(X_test_i)
# new_pred_i = [y_pred_i[x][1] for x in range(len(y_pred_i))]
result_i = accuracy_score(y_test_i, y_pred_i, normalize = True)*100
print("akurasi model-i :", round(result_i,2), end= "%\n\n")

model_u = KNeighborsClassifier(n_neighbors=3)
model_u.fit(X_train_u, y_train_u)
y_pred_u = model_u.predict(X_test_u)
# new_pred_u = [y_pred_u[x][1] for x in range(len(y_pred_u))]
result_u = accuracy_score(y_test_u, y_pred_u, normalize = True)*100
print("akurasi model-u :", round(result_u,2), end= "%\n\n")

model_e = KNeighborsClassifier(n_neighbors=3)
model_e.fit(X_train_e, y_train_e)
y_pred_e = model_e.predict(X_test_e)
# new_pred_e = [y_pred_e[x][1] for x in range(len(y_pred_e))]
result_e = accuracy_score(y_test_e, y_pred_e, normalize = True)*100
print("akurasi model-e :", round(result_e,2), end= "%\n\n")

model_o = KNeighborsClassifier(n_neighbors=3)
model_o.fit(X_train_o, y_train_o)
y_pred_o = model_o.predict(X_test_o)
# new_pred_o = [y_pred_o[x][1] for x in range(len(y_pred_o))]
result_o = accuracy_score(y_test_o, y_pred_o, normalize = True)*100
print("akurasi model-o :", round(result_o,2), end= "%\n\n")

model_ē = KNeighborsClassifier(n_neighbors=3)
model_ē.fit(X_train_ē, y_train_ē)
y_pred_ē = model_ē.predict(X_test_ē)
# new_pred_ē = [y_pred_ē[x][1] for x in range(len(y_pred_ē))]
result_ē = accuracy_score(y_test_ē, y_pred_ē, normalize = True)*100
print("akurasi model-ē :", round(result_ē,2), end= "%\n\n")

model_kata = KNeighborsClassifier(n_neighbors=3)
model_kata.fit(X_train_kata, y_train_kata)
y_pred_kata = model_kata.predict(X_test_kata)
# new_pred_kata = [y_pred_kata[x][1] for x in range(len(y_pred_kata))]
result_kata = accuracy_score(y_test_kata, y_pred_kata, normalize = True)*100
print("akurasi model-kata :", round(result_kata,2), end= "%\n\n")

avg_acc = round((result_ē + result_e + result_o + result_u + result_i + result_a + result_kata)/7,2)
print("rata-rata akurasi: ",avg_acc, end="%")

pickle.dump(model_a, open('model_a.pkl','wb'))
pickle.dump(model_i, open('model_i.pkl','wb'))
pickle.dump(model_u, open('model_u.pkl','wb'))
pickle.dump(model_ē, open('model_ē.pkl','wb'))
pickle.dump(model_o, open('model_o.pkl','wb'))
pickle.dump(model_e, open('model_e.pkl','wb'))
pickle.dump(model_kata, open('model_kata.pkl','wb'))