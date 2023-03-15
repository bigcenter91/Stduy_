from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
import pandas as pd
import numpy as np

#1. 데이터
path = './_data/dacon_wine/'
path_save = './_save/dacon_wine/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0)

print(train_csv) # 5497, 13

test_csv = pd.read_csv(path + 'test.csv',
                       index_col=0)

print(test_csv) # 1000, 12

print(type(train_csv)) # 데이터 타입을 알 수 있다

