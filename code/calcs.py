# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:01:42 2020

@author: Mario-User
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM
import datetime as tm
from matplotlib import pyplot

root = './AIDA_patients/'

posicion_glucosa = 6

pacientes = ['p1']
x_glucose = []
y_glucose = []

for paciente in pacientes:

    data = pd.read_csv(root+paciente+'.csv', delimiter=';', names=['Time', 'Glucose', 'PlasmaInsuline', 'SlowInsuline', 'FastInsuline', 'Carbohydrates'])
    print('paciente: ', paciente)
    print('Tamaño del csv: ',data.shape)
    datosProcesados = data.fillna(value=0)
    print(datosProcesados.head())

    for x in range(datosProcesados.shape[0]-24-posicion_glucosa):
        x_muestras = []
        for y in range(24):
            # x_muestras.append([datosProcesados['Glucose'][x+y], datosProcesados['SlowInsuline'][x+y], datosProcesados['FastInsuline'][x+y], datosProcesados['Carbohydrates'][x+y]])
            x_muestras.append([datosProcesados['Glucose'][x+y]])
        x_glucose.append(x_muestras)     
        y_glucose.append(datosProcesados['Glucose'][x+24+posicion_glucosa])

x = np.array(x_glucose)
print('Tamaño de x: ', x.shape)     

y = np.array(y_glucose)
print('Tamaño de y: ', y.shape)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.1, random_state = 0)

model = Sequential()
model.add(LSTM(units=10, input_shape=(xTrain.shape[1], xTrain.shape[2])))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.001))

history = model.fit(xTrain, yTrain,
    epochs=500,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
    shuffle=False)

# plot train and validation loss
pyplot.plot(history.history['loss'][:])
pyplot.plot(history.history['val_loss'][:])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()

y_pred = model.predict(xTest)
score = model.evaluate(xTest, yTest, batch_size=32)
print('El modelo de predicción tiene un score: ',score)

eje_x = np.arange(yTest.shape[0])
print(eje_x.shape)

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.plot(eje_x[0:100], yTest[0:100], color='blue', label='yTest')
ax.plot(eje_x[0:100], y_pred[0:100], color='red', label='y_pred')
plt.title('Resultados obtenidos')
plt.xlabel('Número de muestra')
plt.ylabel('Glucosa')
leg = ax.legend();

plt.savefig(root+'ResultadosObtenidos.png')

