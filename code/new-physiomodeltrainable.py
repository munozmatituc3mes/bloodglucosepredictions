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

root = './AIDA_patients/'

posicion_glucosa = 4
time_span = 24
bs=16
trainable = False
model= keras.models.load_model('model-2-aida.h5')
printing = True
mem_cells = 10

# pacientes = ['p3','p5']
pacientes = ['p2']
x_glucose = []
y_glucose = []
glucose = []
datavalues = []

for paciente in pacientes:

    data = pd.read_csv(root+paciente+'.csv', delimiter=';', names=['Time', 'Glucose', 'SlowInsuline', 'FastInsuline', 'Carbohydrates'], skiprows=1)
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
    if (np.sum(datavalues[x+time_span:x+time_span+posicion_glucosa,2:]) == 0 or includeall):
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
if (trainable):
    input1 = Input(shape=(time_span, 1))
    x11 = LSTM(units=mem_cells, activation='relu', return_sequences=False)
    x12 = x11(input1)
    x13 = Dense(units=3, activation='relu')
    x1 = x13(x12)
    input2 = Input(shape=(time_span,1))
    x21 = LSTM(units=mem_cells, activation='relu', return_sequences=False)
    x22 = x21(input2)
    x23 = Dense(units=3, activation='relu')
    x2 = x23(x22)
    input3 = Input(shape=(time_span, 1))
    x31 = LSTM(units=mem_cells, activation='relu', return_sequences=False)
    x32 = x31(input3)
    x33 = Dense(units=3, activation='relu')
    x3 = x33(x32)
    input4= Input(shape=(time_span,1))
    x41 = LSTM(units=mem_cells, activation='relu', return_sequences=False)
    x42 = x41(input4)
    x43 = Dense(units=3, activation='relu')
    x4 = x43(x42)
    added = Concatenate(axis=-1)([x2, x3, x4])
    
    out1 = Dense(1)(added)
    added2 = Concatenate(axis=-1)([x1, added])
    out2 = Dense(1)(added2)
    
    model = Model(inputs=[input1, input2, input3, input4], outputs=[out1, out2])
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.001))
    
    history = model.fit([xTrain[:,:,4:5],xTrain[:,:,1:2],xTrain[:,:,2:3],xTrain[:,:,3:4]], [yTrain, yTrain],
        epochs=100,
        batch_size=bs,
        validation_split=0.05,
        verbose=1,
        shuffle=False)



xTest_ini= xTest

iter = 0
for index in range(0,posicion_glucosa):
    [y_pred1, y_pred] = model.predict((xTest[:,:,4:5],xTest[:,:,1:2],xTest[:,:,2:3],xTest[:,:,3:4]))
    if iter == 0: 
        ypred1=y_pred1
        iter = 1
    for i in range(len(xTest)):
        xTest[i,:,:] = np.roll(xTest[i,:,:], -1, axis=0)
        xTest[i,time_span-1,:] = [xTest[i,time_span-2,0]+y_pred[i],0,0,0,y_pred[i]]

g_est = np.zeros(len(xTest)-posicion_glucosa)
y_true = np.zeros(len(xTest)-posicion_glucosa)

for i in range(len(xTest)-posicion_glucosa):
    g_est[i] = xTest[i,time_span-1,0] 
    y_true[i] = g[i+posicion_glucosa-1]
    
    
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

timedata = np.arange(0,24,0.25)
timedata2 = np.arange(0.35,8,0.25/24)

scale=18
if (printing):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(timedata, scale*y_true[72:72+96], color='blue', label='real glucose')
    ax.plot(timedata, scale*g_est[72:72+96], color='red', label='estimated glucose')
    plt.title('Prediction horizon 60 minutes')
    plt.xlabel('time of the day')
    plt.ylabel('Glucose level (mg/dL)')
    leg = ax.legend()
    plt.show()
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(timedata, scale*y_true[72+96:72+96+96], color='blue', label='real glucose')
    ax.plot(timedata, scale*g_est[72+96:72+96+96], color='red', label='estimated glucose')
    plt.title('Prediction horizon 60 minutes')
    plt.xlabel('time of the day')
    plt.ylabel('Glucose level (mg/dL)')
    leg = ax.legend()
    plt.show()
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(timedata2, scale*y_true, color='blue', label='real glucose')
    ax.plot(timedata2, scale*g_est, color='red', label='estimated glucose')
    plt.title('Prediction horizon 60 minutes')
    plt.xlabel('day number')
    plt.ylabel('Glucose level (mg/dL)')
    leg = ax.legend()
    plt.show()

model.save('model-2-aida.h5')

# print(model.get_weights())

if (printing):
    fig, ax = plt.subplots(figsize=(10,6))
    plt.plot(datavalues[25:,4])
    plt.plot(datavalues[25:,3])
    plt.plot(datavalues[25:,2])
    plt.plot(y_true)
    plt.plot(ypred1*10)
    plt.show()

from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True, show_layer_names=False)
# plot_model(model, to_file='model.png')

print(scale*mean_absolute_error(y_true, g_est))

print(scale*mean_squared_error(y_true, g_est)) 

constant = 30
esc_corr = 1.1*scale
plot, zone60 = clarke_error_grid(esc_corr*y_true-constant, esc_corr*g_est-constant, "60 minute predictions") 
print(zone60)


posicion_glucosa = 2
time_span = 24
bs=16

# pacientes = ['p3','p5']
pacientes = ['p2']
x_glucose = []
y_glucose = []
glucose = []
datavalues = []

for paciente in pacientes:

    data = pd.read_csv(root+paciente+'.csv', delimiter=';', names=['Time', 'Glucose', 'SlowInsuline', 'FastInsuline', 'Carbohydrates'], skiprows=1)
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
    if (np.sum(datavalues[x+time_span:x+time_span+posicion_glucosa,2:]) == 0 or includeall):
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


xTest_ini= xTest

iter = 0
for index in range(0,posicion_glucosa):
    [y_pred1, y_pred] = model.predict((xTest[:,:,4:5],xTest[:,:,1:2],xTest[:,:,2:3],xTest[:,:,3:4]))
    if iter == 0: 
        ypred1=y_pred1
        iter = 1
    for i in range(len(xTest)):
        xTest[i,:,:] = np.roll(xTest[i,:,:], -1, axis=0)
        xTest[i,time_span-1,:] = [xTest[i,time_span-2,0]+y_pred[i],0,0,0,y_pred[i]]

g_est = np.zeros(len(xTest)-posicion_glucosa)
y_true = np.zeros(len(xTest)-posicion_glucosa)

for i in range(len(xTest)-posicion_glucosa):
    g_est[i] = xTest[i,time_span-1,0] 
    y_true[i] = g[i+posicion_glucosa-1]
    
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

timedata = np.arange(0,24,0.25)
scale=18

if (printing):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(timedata, scale*y_true[72:72+96], color='blue', label='real glucose')
    ax.plot(timedata, scale*g_est[72:72+96], color='red', label='estimated glucose')
    plt.title('Prediction horizon 30 minutes')
    plt.xlabel('time of the day')
    plt.ylabel('Glucose level')
    leg = ax.legend()
    plt.show()
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(timedata, scale*y_true[72+96:72+96+96], color='blue', label='real glucose')
    ax.plot(timedata, scale*g_est[72+96:72+96+96], color='red', label='estimated glucose')
    plt.title('Prediction horizon 30 minutes')
    plt.xlabel('time of the day (mg/dL)')
    plt.ylabel('Glucose level (mg/dL)')
    leg = ax.legend()
    plt.show()
    
    timedata2 = np.arange(0.31,8,0.25/24)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(timedata2, scale*y_true, color='blue', label='real glucose')
    ax.plot(timedata2, scale*g_est, color='red', label='estimated glucose')
    plt.title('Prediction horizon 30 minutes')
    plt.xlabel('day number')
    plt.ylabel('Glucose level (mg/dL)')
    leg = ax.legend()
    plt.show()

model.save('model-2-aida.h5')

# print(model.get_weights())

timedata3 = np.arange(0.27,8,0.25/24)
timedata4 = np.arange(0.29,8,0.25/24)


if (printing):
    fig, ax = plt.subplots(figsize=(10,6))
    plt.plot(timedata3,datavalues[25:,4], color='green', label='Carbohydrate (grams)')
    plt.plot(timedata3,datavalues[25:,3], color='blue', label='fast acting insulin Dose (units)')
    plt.plot(timedata3,datavalues[25:,2], color='black', label='slow acting insulin Dose (units)')
    # plt.plot(timedata2,y_true)
    plt.plot(timedata4,scale*ypred1, color='orange', label='real glucose')
    plt.show()

from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True, show_layer_names=False)
# plot_model(model, to_file='model.png')

print(scale*mean_absolute_error(y_true, g_est))

print(scale*mean_squared_error(y_true, g_est)) 

timedata3 = np.arange(0.27,8,0.25/24)
timedata4 = np.arange(0.29,8,0.25/24)


if (printing):
    fig, ax = plt.subplots(figsize=(10,6))
    plt.plot(timedata,datavalues[25+72:25+72+96,4], color='green', linestyle='--', label='Carbohydrate (grams)')
    plt.plot(timedata,datavalues[25+72:25+72+96,3], color='blue', linestyle='-.', label='fast acting insulin Dose (units)')
    plt.plot(timedata,datavalues[25+72:25+72+96,2], color='black', linestyle='-.', label='slow acting insulin Dose (units)')
    # plt.plot(timedata2,y_true)
    plt.plot(timedata,scale*ypred1[72:72+96], color='orange', label='estimated glucose increments')
    plt.title('Prediction horizon 30 minutes')
    plt.xlabel('hour of the day')
    plt.ylabel('Glucose increments estimation (mg/dL)')
    leg = ax.legend()
    plt.show()

from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True, show_layer_names=False)
# plot_model(model, to_file='model.png')

print(scale*mean_absolute_error(y_true, g_est))

print(scale*mean_squared_error(y_true, g_est)) 

constant = 30
esc_corr = 1.1*scale
plot, zone30 = clarke_error_grid(esc_corr*y_true-constant, esc_corr*g_est-constant, "30 minute predictions") 
print(zone30)




"""
Testing another day
"""

posicion_glucosa = 4

pacientes = ['p2test']
# pacientes = ['p5']
x_glucose = []
y_glucose = []
glucose = []
datavalues = []
times = []

for paciente in pacientes:

    data = pd.read_csv(root+paciente+'.csv', delimiter=';', names=['Time', 'Glucose', 'SlowInsuline', 'FastInsuline', 'Carbohydrates'], skiprows=1)
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
        times.append(datosProcesados['Time'][x+time_span-1])
        
x = np.array(x_glucose)
print('Tamaño de x: ', x.shape) 
x_array = x    

y = np.array(y_glucose)
print('Tamaño de y: ', y.shape)
y_array = y

g = np.array(glucose)
print('Tamaño de glucose: ', g.shape)

time = np.array(times)

# xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.1, random_state = 0)

xTrain = []
xTest = x_array 
yTrain = [] 
yTest = y_array
gTrain = []

xTest_ini= xTest

iter = 0
for index in range(0,posicion_glucosa):
    [y_pred1, y_pred] = model.predict((xTest[:,:,4:5],xTest[:,:,1:2],xTest[:,:,2:3],xTest[:,:,3:4]))
    if iter == 0: 
        ypred1=y_pred1
        iter = 1
    for i in range(len(xTest)):
        xTest[i,:,:] = np.roll(xTest[i,:,:], -1, axis=0)
        xTest[i,time_span-1,:] = [xTest[i,time_span-2,0]+y_pred[i],0,0,0,y_pred[i]]

g_est = np.zeros(len(xTest)-posicion_glucosa)
y_true = np.zeros(len(xTest)-posicion_glucosa)

for i in range(len(xTest)-posicion_glucosa):
    g_est[i] = xTest[i,time_span-1,0] 
    y_true[i] = g[i+posicion_glucosa-1]
    
xTest = xTest_ini
    
timegen = np.linspace(start = 6, stop = 6+len(y_true)/4, num=len(y_true))

if (printing):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(timegen[72:]-24, scale*y_true[72:], color='blue', label='measured glucose')
    ax.plot(timegen[72:]-24, scale*g_est[72:], color='red', label='estimated glucose')
    plt.title('Prediction horizon of 60 minutes')
    plt.xlabel('hour of the day (h)')
    plt.ylabel('Glucose (mg/dL)')
    leg = ax.legend()
    plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error

print(18*mean_absolute_error(y_true, g_est))

print(18*mean_squared_error(y_true, g_est)) 


posicion_glucosa = 2


pacientes = ['p2test']
# pacientes = ['p5']
x_glucose = []
y_glucose = []
glucose = []
datavalues = []
times = []

for paciente in pacientes:

    data = pd.read_csv(root+paciente+'.csv', delimiter=';', names=['Time', 'Glucose', 'SlowInsuline', 'FastInsuline', 'Carbohydrates'], skiprows=1)
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
        times.append(datosProcesados['Time'][x+time_span-1])
        
x = np.array(x_glucose)
print('Tamaño de x: ', x.shape) 
x_array = x    

y = np.array(y_glucose)
print('Tamaño de y: ', y.shape)
y_array = y

g = np.array(glucose)
print('Tamaño de glucose: ', g.shape)

time = np.array(times)

# xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.1, random_state = 0)

xTrain = []
xTest = x_array 
yTrain = [] 
yTest = y_array
gTrain = []

xTest_ini= xTest

iter = 0
for index in range(0,posicion_glucosa):
    [y_pred1, y_pred] = model.predict((xTest[:,:,4:5],xTest[:,:,1:2],xTest[:,:,2:3],xTest[:,:,3:4]))
    if iter == 0: 
        ypred1=y_pred1
        iter = 1
    for i in range(len(xTest)):
        xTest[i,:,:] = np.roll(xTest[i,:,:], -1, axis=0)
        xTest[i,time_span-1,:] = [xTest[i,time_span-2,0]+y_pred[i],0,0,0,y_pred[i]]

g_est = np.zeros(len(xTest)-posicion_glucosa)
y_true = np.zeros(len(xTest)-posicion_glucosa)

for i in range(len(xTest)-posicion_glucosa):
    g_est[i] = xTest[i,time_span-1,0] 
    y_true[i] = g[i+posicion_glucosa-1]
    
xTest = xTest_ini
    
timegen = np.linspace(start = 6, stop = 6+len(y_true)/4, num=len(y_true))

if (printing):
   fig, ax = plt.subplots(figsize=(10,6))
   ax.plot(timegen[72:]-24, scale*y_true[72:], color='blue', label='measured glucose')
   ax.plot(timegen[72:]-24, scale*g_est[72:], color='red', label='estimated glucose')
   plt.title('Prediction horizon of 30 minutes')
   plt.xlabel('hour of the day (h)')
   plt.ylabel('Glucose (mg/dL)')
   leg = ax.legend();
   plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error

print(18*mean_absolute_error(y_true, g_est))

print(18*mean_squared_error(y_true, g_est)) 

constant = 30
esc_corr = 1.1*scale
plot, zone = clarke_error_grid(esc_corr*y_true-constant, esc_corr*g_est-constant, "60 minute predictions") 
print(zone)
