#!/usr/bin/env python
# coding: utf-8

import math
import datetime
import numpy as np
import pandas as pd
import keras.backend as K
import matplotlib.pyplot as plt
from itertools import repeat
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

fps = 60
interval = int(60/fps)

num_classes = 2  # 3 for LSTM-3Class, 2 for others(Baseline, LSTM-2classLR)

hidden_units = 128             # one from [60, 128, 256, 512]
batch_size = 5                 # one from [5, 10]
epochs = 250                   # one from [128, 256, 500, 1000]
dropout = 0.25                 # one from [0, 0.25, 0.5, 1]
optimizer = "Adam"             # one from ["Adam", "RMSProp", "AdaDelta"]
activation = "tanh"            # one from ["tanh", "ReLU", "sigmoid"]
last_activation = "sigmoid"    # one from ["tanh", "ReLU", "sigmoid"]

# fix random seed for reproducibility
np.random.seed(7)

X_train = pd.read_csv("x_train.csv", header=None)

for i in range(0, len(X_train)):
    if fps == 45:
        X_train[i] = X_train[i][7:52]
    else:
        X_train[i] = X_train[i][::interval]

X_train = np.array(X_train, dtype='float32')


X_val = pd.read_csv("x_val.csv", header=None)

for i in range(0, len(X_val)):
    if fps == 45:
        X_val[i] = X_val[i][7:52]
    else:
        X_val[i] = X_val[i][::interval]


X_val = np.array(X_val, dtype='float32')


X_test = pd.read_csv("x_test.csv", header=None)

for i in range(0, len(X_test)):
    if fps == 45:
        X_test[i] = X_test[i][7:52]
    else:
        X_test[i] = X_test[i][::interval]


X_test = np.array(X_test, dtype='float32')


y_train = pd.read_csv("y_train.csv", header=None)
y_val = pd.read_csv("y_val.csv", header=None)
y_test = pd.read_csv("y_test.csv", header=None)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes, dtype='float32')
y_val = tf.keras.utils.to_categorical(y_val, num_classes=num_classes, dtype='float32')
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes, dtype='float32')


loss_func= "wbce_sigmoid"

def wbce(y_true, y_pred, weight1=1, weight0=1):
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    logloss = -(y_true * K.log(y_pred) * weight1 + (1 - y_true) * K.log(1 - y_pred) * weight0 )
    return K.mean(logloss, axis=-1)


params = {
     'hidden_units': hidden_units,
     'batch_size': batch_size,
     'epochs': epochs,
     'dropout': dropout,
     'optimizer': optimizer,
     'activation': activation,
     'last_activation': last_activation
}


steps = fps
features = 4 #because the maximum number of coordinates=2 x,y for each
model = Sequential()
model.add(LSTM(params['hidden_units'], input_shape=(steps, features), activation=params['activation'], return_sequences=False, name='single_layer_LSTM'))
model.add(Dropout(params['dropout']))
model.add(Dense(2, activation=params['last_activation'], name='dense_output'))
model.compile(loss=wbce, optimizer=params["optimizer"], metrics=[tf.keras.metrics.categorical_accuracy])

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001, verbose=1)
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=params["batch_size"],
                    epochs=params["epochs"],
                    callbacks=[reduce_lr])


model.save("lstm_model_{}fps_{}units_{}epochs_{}loss_{}dropout_{}bsize_reducelr_yolov4.h5".format(steps, params['hidden_units'], params['epochs'], loss_func, params['dropout'], params['batch_size']))

import time
start_time = time.time()
results = model.evaluate(X_test, y_test, batch_size=20)
print('test loss, test acc:', results)
print("--- %s seconds ---" % (time.time() - start_time))
print(len(X_test))

train_loss = model.history.history['loss'][-1] 
train_acc = model.history.history['categorical_accuracy'][-1]
val_loss = model.history.history['val_loss'][-1]
val_acc = model.history.history['val_categorical_accuracy'][-1]

print(results)
test_loss = results[0]
test_acc = results[1]

predictions = model.predict(X_test)
y_pred = (predictions > 0.5)

import sklearn
c_matrix = sklearn.metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

print(c_matrix)
