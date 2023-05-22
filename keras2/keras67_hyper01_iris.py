import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout
from tensorflow.keras.optimizers import Adam
# node와 lr 적용해서 소스 완성
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
