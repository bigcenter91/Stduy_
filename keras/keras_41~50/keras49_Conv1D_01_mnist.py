from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 784, 1)
x_test = x_test.reshape(10000, 784, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape) # (60000, 784)
print(x_test.shape) # (10000, 784)
print(y_test.shape) # (10000, 10)
print(y_train.shape) # (60000, 10) 

#2. 모델 구성
model = Sequential()

model.add(Conv1D(100, 2, input_shape=(784, 1)))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_loss', patience=10, mode='min',
                   verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=50, batch_size=64,
                 validation_split=0.2, verbose=1, callbacks=[es])

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)

acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1))
print('acc : ', acc)

import matplotlib.pyplot as plt
# plt.plot(hist.history['val_loss'], label='val_acc')
plt.imshow(x_train[333], 'gray')
plt.show()
