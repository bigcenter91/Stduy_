# 함수형으로 만들어

import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def m1():
    model = Sequential()
    model.add(Dense(3, input_dim = 1, name = 'hidden1'))
    model.add(Dense(2, name = 'hidden2'))
    model.add(Dense(1, name = 'hidden3'))
    
    model.layers[0].trainable = False # hidden1
    pd.set_option('max_colwidth', -1)

    layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
    # print(layers)

    results = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])
    print(results) # 가중치 False

def m2():
    model = Sequential()
    model.add(Dense(3, input_dim = 1, name = 'hidden1'))
    model.add(Dense(2, name = 'hidden2'))
    model.add(Dense(1, name = 'hidden3'))
    
    model.layers[1].trainable = False # hidden1
    pd.set_option('max_colwidth', -1)

    layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
    # print(layers)

    results = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])
    print(results) # 가중치 False

def m3():
    model = Sequential()
    model.add(Dense(3, input_dim = 1, name = 'hidden1'))
    model.add(Dense(2, name = 'hidden2'))
    model.add(Dense(1, name = 'hidden3'))
    
    model.layers[2].trainable = False # hidden1
    pd.set_option('max_colwidth', -1)

    layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
    # print(layers)

    results = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])
    print(results) # 가중치 False

def_list = [m1,
            m2,
            m3]
for d in range(len(def_list)):
    if d == 0:
        m1()
    elif d == 1:
        m2()
    elif d == 2:
        m3()