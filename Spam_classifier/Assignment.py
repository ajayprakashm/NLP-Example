# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 11:08:11 2019

@author: ajay
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 10:47:33 2019

@author: ajay
"""
#%% Importing librries and datasets
% clear
% reset -f
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn import preprocessing 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import seaborn as sns
import os as os
import scipy.io as spio
sns.set(style="whitegrid")
np.random.seed(203)
os.chdir('C:/Users/ajay/Desktop/Assignment')
mat1 = spio.loadmat('IGBOutput7.mat') ## Normal
mat2 = spio.loadmat('IGBOutput21.mat') ## Slight faulty
mat3 = spio.loadmat('IGBOutput34.mat') ## Severely faulty

d1 = mat1['Output'][0]
d2 = mat2['Output'][0]
d3 = mat3['Output'][0]
X=pd.DataFrame(d1)

# =============================================================================
# X=np.float64(pd.concat([D1[0],D2[0],D3[0]]))
# Y=np.float64(pd.concat([D1['Output'],D2['Output'],D3['Output']]))
# d=[d1,d2,d3]
# =============================================================================

#%% For plotting 
def tsne_plot(x1, y1, name="graph.png"):
    tsne = TSNE(n_components=2, random_state=0)
    X_t = tsne.fit_transform(x1)

    plt.figure(figsize=(12, 8))
    plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', color='g', linewidth='1', alpha=0.8, label='Normal')
    plt.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker='o', color='r', linewidth='1', alpha=0.8, label='Faulty')

    plt.legend(loc='best');
    plt.savefig(name);
    plt.show();
    
#%% Keras autoencoder model is used along with one input layer and one output layer having identical dimentions 
#   ie. the shape of faulty data
## input layer 
input_layer = Input(shape=(X.shape[1],))

## encoding part
encoded = Dense(100, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoded = Dense(5, activation='relu')(encoded)

## decoding part
decoded = Dense(5, activation='tanh')(encoded)
decoded = Dense(100, activation='tanh')(decoded)

## output layer
output_layer = Dense(X.shape[1], activation='relu')(decoded)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer="adadelta", loss="mse")

x_norm = preprocessing.MinMaxScaler().fit_transform(X.values)
x_faulty = preprocessing.MinMaxScaler().fit_transform(pd.DataFrame(d3).values) # Severe faultyh data is used for prediction

# First 10000 samples is used from normal data because of time constrain
autoencoder.fit(x_norm[0:10000], x_norm[0:10000],batch_size = 256, epochs = 100, 
                shuffle = True, validation_split = 0.20)

hidden_representation = Sequential()
hidden_representation.add(autoencoder.layers[0])
hidden_representation.add(autoencoder.layers[1])
hidden_representation.add(autoencoder.layers[2])

norm_hid_rep = hidden_representation.predict(x_norm[:1000])
faulty_hid_rep = hidden_representation.predict(x_faulty[:500])

#%% Ploting to visualiuze results
rep_x = np.append(norm_hid_rep, faulty_hid_rep, axis = 0)
y_g = np.zeros(norm_hid_rep.shape[0])
y_f = np.ones(faulty_hid_rep.shape[0])
rep_y = np.append(y_g, y_f)
tsne_plot(rep_x, rep_y, "comparision plot.png")

#%%
plt.plot(x_norm[:500])
plt.plot(norm_hid_rep[:500])