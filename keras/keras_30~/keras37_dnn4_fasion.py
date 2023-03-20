from tensorflow.keras.datasets import fashion_mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# 두 개의 튜플로 나눠서 할당하기 위해 사용

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)


x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

print(x_train.shape, x_test.shape) # (60000, 784) (10000, 784)

print(np.max(x_train), np.min(x_train)) # 1.0 0.0 / 이미지는 이렇게 쓰는게 괜찮다
print(np.unique(y_train, return_counts=True))

print(y_test.shape) # (10000, )
print(y_train.shape) # (60000, )

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

print(y_test.shape) # (10000, 10)

#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_shape=(784, )))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary() # 요약하여 출력

# print(np.unique(y_train, return_counts=True))


#3. 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer='Adagrad', metrics='acc')

es = EarlyStopping(monitor='val_loss', patience=30, mode='min',
                   verbose=1, restore_best_weights=True)

import time
start_time = time.time()

hist = model.fit(x_train, y_train, epochs=100, batch_size=500,
                 validation_split=0.2, verbose=1, callbacks=[es])

end_time = time.time()
print('training time : ', round(end_time-start_time, 2))


#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)

acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1))
print('acc : ', acc)

import matplotlib.pyplot as plt
plt.plot(hist.history['val_loss'], label='val_acc')
# plt.imshow(x_train[333])
plt.show()

# result :  [0.8409841060638428, 0.7874000072479248]
# acc :  0.7874