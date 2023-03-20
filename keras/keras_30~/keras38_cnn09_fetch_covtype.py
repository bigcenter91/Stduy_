from sklearn.datasets import fetch_covtype
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터

datasets = fetch_covtype()
x = datasets.data
y = datasets['target']

print(x.shape) # (581012, 54)
print(y.shape) # (581012, )