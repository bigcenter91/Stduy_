from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers import Conv1D, Reshape, LSTM, Input
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)/255.
x_test = x_test.reshape(10000, 28, 28, 1)/255.

#2. 모델 구성
# model= Sequential()
# model.add(Conv2D(filters=64, kernel_size=(3,3),
#                  padding='same', input_shape=(28, 28, 1)))
# model.add(MaxPooling2D())
# model.add(Conv2D(32, (3,3)))
# model.add(Conv2D(10, 3))
# model.add(MaxPooling2D())
# model.add(Flatten())                        # (N, 250)
# model.add(Reshape(target_shape=(25, 10)))
# model.add(Conv1D(10, 3, padding='same'))
# model.add(LSTM(784))
# model.add(Reshape(target_shape=(28, 28, 1)))
# model.add(Conv2D(32, (3,3), padding='same'))
# model.add(Flatten())
# model.add(Dense(10, activation='softmax'))

input1 = Input(shape=(28, 28, 1))
conv2_1 = Conv2D(64, (3,3), padding='same')(input1)
max = MaxPooling2D()(conv2_1)
conv2_2 = Conv2D(32, (3,3))(max)
conv2_3 = Conv2D(10, 3,)(conv2_2)
max = MaxPooling2D()(conv2_3)
flat = Flatten()(max)
re1 = Reshape(target_shape=(25, 10))(flat)
ls1 = LSTM(784)(re1)
re2 = Reshape(target_shape=(28, 28, 1))(ls1)
conv2_4 = Conv2D(32, (3,3), padding='same')(re2)
flat = Flatten()(conv2_4)
output1 = Dense(10, activation='softmax')(flat)
model = Model(inputs=input1, outputs=output1)

model.summary()