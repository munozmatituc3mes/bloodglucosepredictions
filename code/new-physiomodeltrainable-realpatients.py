import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import SimpleRNNCell
from tensorflow.keras.models import Model
import datetime as tm
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ClarkeErrorGrid import clarke_error_grid 


root = './D1NAMO_patients/'

posicion_glucosa = 9
time_span = 36
bs=16

pacientes = ['004/glucose','001/glucose']
x_glucose = []
y_glucose = []
glucose = []
datavalues = []

for paciente in pacientes:

    data = pd.read_csv(root+paciente+'.csv', delimiter=';', names=['Date','Time', 'Glucose', 'SlowInsuline', 'FastInsuline', 'Carbohydrates'], skiprows=1)
    print('paciente: ', paciente)
    print('Tamaño del csv: ',data.shape)
    datosProcesados = data.fillna(value=0)
    print(datosProcesados.head())
    datavalues=datosProcesados.values

    for x in range(1, datosProcesados.shape[0]-time_span-posicion_glucosa, 1):
        x_muestras = []
        for y in range(time_span):
            # x_muestras.append([datosProcesados['Glucose'][x+y], datosProcesados['SlowInsuline'][x+y], datosProcesados['FastInsuline'][x+y], datosProcesados['Carbohydrates'][x+y]])
            x_muestras.append([datosProcesados['Glucose'][x+y], datosProcesados['SlowInsuline'][x+y], datosProcesados['FastInsuline'][x+y], datosProcesados['Carbohydrates'][x+y], datosProcesados['Glucose'][x+y]-datosProcesados['Glucose'][x+y-1]])
        x_glucose.append(x_muestras)     
        y_glucose.append(datosProcesados['Glucose'][x+time_span]-datosProcesados['Glucose'][x+time_span-1])
        glucose.append(datosProcesados['Glucose'][x+time_span-1])
        
x = np.array(x_glucose)
print('Tamaño de x: ', x.shape) 
x_array = x    

y = np.array(y_glucose)
print('Tamaño de y: ', y.shape)
y_array = y

g = np.array(glucose)
print('Tamaño de glucose: ', g.shape)


# xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.1, random_state = 0)

xTrain = []
xTest = x_array 
yTrain = [] 
yTest = y_array
gTrain = []

xTrain = np.array(xTrain)
yTrain = np.array(yTrain)
gTrain = np.array(gTrain)


includeall = True

for x in range(0,len(g)-posicion_glucosa):
    if (np.sum(datavalues[x+time_span:x+time_span+posicion_glucosa,3:]) == 0 or includeall):
        xTrain = np.append(xTrain, x_array[x,:,:])
        yTrain = np.append(yTrain, y[x])
        g_temp = []
        for index in range(posicion_glucosa):
            g_temp.append(g[x+index+1])
        gTrain = np.append(gTrain, g_temp)

xTrain = xTrain.reshape(len(yTrain),time_span,5)
gTrain = gTrain.reshape(len(yTrain),posicion_glucosa)
print(xTrain.shape)
print(yTrain.shape)
print(gTrain.shape)

"""
input1 = Input(shape=(time_span, 1))
x11 = LSTM(units=5, activation='relu', return_sequences=True)
x12 = x11(input1)
x13 = SimpleRNN(units=2, activation='relu')
x1 = x13(x12)
input2 = Input(shape=(time_span,1))
x21 = LSTM(units=5, activation='relu', return_sequences=True)
x22 = x21(input2)
x23 = SimpleRNN(units=2, activation='relu')
x2 = x23(x22)
input3 = Input(shape=(time_span, 1))
x31 = LSTM(units=5, activation='relu', return_sequences=True)
x32 = x31(input3)
x33 = SimpleRNN(units=2, activation='relu')
x3 = x33(x32)
input4= Input(shape=(time_span,1))
x41 = LSTM(units=5, activation='relu', return_sequences=True)
x42 = x41(input4)
x43 = SimpleRNN(units=2, activation='relu')
x4 = x43(x42)
added = Concatenate(axis=-1)([x2, x3, x4])
"""

unidades = 10
denseunits = 2
input1 = Input(shape=(time_span, 1))
x11 = LSTM(units=unidades, activation='relu', return_sequences=False)
x12 = x11(input1)
# x111 = LSTM(units=5, activation='relu', return_sequences=False)
# x112= x111(x12)
x13 = Dense(units=denseunits, activation='relu')
x1 = x13(x12)
input2 = Input(shape=(time_span,1))
x21 = LSTM(units=unidades, activation='relu', return_sequences=False)
x22 = x21(input2)
x23 = Dense(units=3, activation='relu')
x2 = x23(x22)
input3 = Input(shape=(time_span, 1))
x31 = LSTM(units=unidades, activation='relu', return_sequences=False)
x32 = x31(input3)
x33 = Dense(units=3, activation='relu')
x3 = x33(x32)
input4= Input(shape=(time_span,1))
x41 = LSTM(units=unidades, activation='relu', return_sequences=False)
x42 = x41(input4)
x43 = Dense(units=3, activation='relu')
x4 = x43(x42)
added = Concatenate(axis=-1)([x2, x3, x4])

out1 = Dense(1)(added)
added2 = Concatenate(axis=-1)([x1, added])
out2 = Dense(1)(added2)


model = Model(inputs=[input1, input2, input3, input4], outputs=[out1, out2])
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.001))

history = model.fit([xTrain[0:2000,:,4:5],xTrain[0:2000,:,1:2],xTrain[0:2000,:,2:3],xTrain[0:2000,:,3:4]/100], [yTrain[0:2000], yTrain[0:2000]],
    epochs=100,
    batch_size=bs,
    validation_split=0.2,
    verbose=1,
    shuffle=False)

xTest_ini= xTest

inicial = 0
iter = 0
for index in range(0,posicion_glucosa):
    [y_pred1, y_pred] = model.predict((xTest[inicial:1100,:,4:5],xTest[inicial:1100,:,1:2],xTest[inicial:1100,:,2:3],xTest[inicial:1100,:,3:4]/100))
    if iter == 0: 
        ypred1=y_pred1
        iter = 1
    for i in range(len(xTest[inicial:1100])):
        xTest[i+inicial,:,:] = np.roll(xTest[i+inicial,:,:], -1, axis=0)
        xTest[i+inicial,time_span-1,:] = [xTest[i+inicial,time_span-2,0]+y_pred[i],0,0,0,y_pred[i]]

g_est = np.zeros(len(xTest[inicial:1100])-posicion_glucosa)
y_true = np.zeros(len(xTest[inicial:1100])-posicion_glucosa)

for i in range(len(xTest[inicial:1100])-posicion_glucosa):
    g_est[i] = xTest[i+inicial,time_span-1,0] 
    # y_true[i] = g[i+800+posicion_glucosa-1]
    y_true[i] = xTest[i+inicial+posicion_glucosa, time_span-1-posicion_glucosa,0]

    
xTest = xTest_ini
    
# score = model.evaluate((xTrain[:,:,0:1],xTrain[:,:,1:2],xTrain[:,:,2:3],xTrain[:,:,3:4]), yTest, batch_size=32)
# print('El modelo de predicción tiene un score: ',score)

"""
includeall = True

g_est =[]
y_true =[]
for x in range(len(g)-time_span-posicion_glucosa):
    if (np.sum(datavalues[x+time_span:x+time_span+3,2:]) == 0 or includeall):
        g_est.append(g[x+time_span-1]+y_pred[x])
        y_true.append(g[x+time_span+posicion_glucosa-1])
"""

iniciar = 152
timeline=np.arange(0,24,0.08333334)
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(timeline,18*y_true[iniciar:iniciar+288], color='blue', label='measured glucose')
ax.plot(timeline,18*g_est[iniciar:iniciar+288], color='red', label='estimated glucose')
plt.title('Prediction horizon of 30 minutes for a real participant (model trained with a different user)')
plt.xlabel('hour of the day (h)')
plt.ylabel('Glucose (mg/dL)')
leg = ax.legend();
plt.show()

"""
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(y_true[400:600], color='blue', label='glucosa medida')
ax.plot(g_est[400:600], color='red', label='glucosa estimada')
plt.title('Resultados obtenidos predicción a media hora')
plt.xlabel('Número de muestra')
plt.ylabel('Glucosa')
leg = ax.legend();
plt.show()

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(y_true[800:1000], color='blue', label='glucosa medida')
ax.plot(g_est[800:1000], color='red', label='glucosa estimada')
plt.title('Resultados obtenidos predicción a media hora')
plt.xlabel('Número de muestra')
plt.ylabel('Glucosa')
leg = ax.legend();
plt.show()

model.save('model-2.h5')

# print(model.get_weights())

plt.plot(datavalues[37:,5]/100)
plt.plot(datavalues[37:,3])
plt.plot(y_true)
plt.plot(ypred1)
plt.show()

from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True, show_layer_names=False)
# plot_model(model, to_file='model.png')

scale = 18
print(scale*mean_squared_error(y_true[:1100], g_est[:1100])) 

plt.plot(y_true)
plt.plot(g_est)
plt.show()

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(y_true[1000:1350], color='blue', label='glucosa medida')
ax.plot(g_est[1000:1350], color='red', label='glucosa estimada')
plt.title('Resultados obtenidos predicción a media hora')
plt.xlabel('Número de muestra')
plt.ylabel('Glucosa')
leg = ax.legend();
plt.show()
"""

scale = 18
print(scale*mean_squared_error(y_true[iniciar:iniciar+288], g_est[iniciar:iniciar+288])) 

constant = 0
esc_corr = 1*scale
plot, zone60 = clarke_error_grid(esc_corr*y_true-constant, esc_corr*g_est-constant, "60 minute predictions real participant") 
print(zone60)
model.save('model-2.h5')