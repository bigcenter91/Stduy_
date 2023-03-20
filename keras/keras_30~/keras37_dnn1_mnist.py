from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical


# [실습]

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape)
print(x_test.shape)


#2. 모델구성
model = Sequential()
model.add(Dense(64, input_shape=(784, )))
# model.add(Dense(64, input_shape=(28*28, )))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 64)                50240

#  dense_1 (Dense)             (None, 64)                4160

#  dense_2 (Dense)             (None, 32)                2080

#  dense_3 (Dense)             (None, 10)                330

print(np.unique(y_train, return_counts=True))

#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_loss', patience=50, mode='min',
                   verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=200, batch_size=1000,
                 validation_split=0.2, verbose=1, callbacks=[es])

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)

acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1))
print('acc : ', acc)

import matplotlib.pyplot as plt
plt.plot(hist.history['val_loss'], label='val_acc')
plt.show()


# result :  [0.32610443234443665, 0.9420999884605408] // acc :  0.9421
# result :  [0.41293367743492126, 0.9386000037193298] // acc :  0.9386
# result :  [0.30116724967956543, 0.9358000159263611] // acc :  0.9358