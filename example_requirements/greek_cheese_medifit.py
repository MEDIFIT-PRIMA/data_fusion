# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 18:49:44 2023

@author: kdola
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import  train_test_split
import tensorflow as tf
from tensorflow.keras import regularizers
import pickle

data = pd.read_excel(r"C:\Users\anton\OneDrive\Desktop\data-greek-goat-cheese.xlsx")
X = data.values[0:,1:].astype('float32')
y=data.values[0:,0].astype('float32')


from scipy.signal import savgol_filter
window_length = 19
polyorder = 2
filtered_data = savgol_filter(X, window_length, polyorder)
with open('savgol_filter.pkl', 'wb') as file:
    pickle.dump(filtered_data, file)

mean = np.mean(filtered_data, axis=0)
mean_centered_data = filtered_data - mean

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
pca = PCA(n_components=3)
X = pca.fit_transform(mean_centered_data)
explained_variance = pca.explained_variance_ratio_
with open('pca.pkl', 'wb') as file:
    pickle.dump((mean, X, pca), file)

X, X_test, y, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
X_target = X[y == 0]
X_non_target = X[y == 1]
y_target = y[y == 0]
y_non_target = y[y ==1]
X_non_target_train, X_non_target_test, _, _ = train_test_split(X_non_target, y_non_target, test_size=0.9, random_state=42)
X_train_som = np.concatenate((X_target, X_non_target_train))
y_train_som = np.concatenate((np.zeros(len(X_target)), np.ones(len(X_non_target_train))))


from minisom import MiniSom
som = MiniSom(15, 15, 3, sigma=1, learning_rate=0.1)
som.random_weights_init(X_train_som)

# train the SOM
som.train_random(X_train_som, 100)
with open('classification_som.pkl', 'wb') as file:
    pickle.dump(som, file)

plt.pcolor(som.distance_map().T, cmap = 'bone_r')
plt.colorbar()
plt.show()

som_classes = np.zeros((som.get_weights().shape[0], som.get_weights().shape[1]))
for i, x in enumerate(X_train_som):
    w = som.winner(x)
    som_classes[w[0], w[1]] = y_train_som[i]


y_pred = []
for x in X_test:
    w = som.winner(x)
    class_label = som_classes[w[0], w[1]]
    y_pred.append(class_label)

# convert lists to arrays
y_pred = np.array(y_pred)


# evaluate the performance of the model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import precision_score, recall_score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Precision: ", precision)
print("Recall: ", recall)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(confusion_matrix, annot=True, fmt='d')
plt.show()

#load the filter from the file and reuse it
#with open('savgol_filter.pkl', 'rb') as file:
    #filtered_data = pickle.load(file)
    
#load the mean, PCA components, and PCA object
#with open('pca.pkl', 'rb') as file:
    #mean, X, pca = pickle.load(file)
    
# load the SOM object
#with open('classification_som.pkl', 'rb') as file:
    #som = pickle.load(file)


#bmu_new = np.array([som.winner(sample) for sample in X_test])
#print("Predictions:", bmu_new)
#bmu_classes = np.array([som_classes[coord[0], coord[1]] for coord in bmu_new])
#print("Predictions:", bmu_classes)
#acc = accuracy_score(y_test, bmu_classes)
#print("Accuracy:", acc)
#from sklearn.metrics import precision_score, recall_score
#precision = precision_score(y_test, bmu_classes)
#recall = recall_score(y_test, bmu_classes)
#print("Precision: ", precision)
#print("Recall: ", recall)

#from sklearn.metrics import classification_report
#print(classification_report(y_test, bmu_classes))

#from sklearn.metrics import confusion_matrix
#confusion_matrix = confusion_matrix(y_test, bmu_classes)
#import matplotlib.pyplot as plt
#import seaborn as sns
#sns.heatmap(confusion_matrix, annot=True, fmt='d')
#plt.show()