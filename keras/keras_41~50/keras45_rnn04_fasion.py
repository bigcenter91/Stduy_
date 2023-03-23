from tensorflow.keras.datasets import fashion_mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# 두 개의 튜플로 나눠서 할당하기 위해 사용

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000, )

print(x_train)

scaler = MinMaxScaler() # Minmax : 최대값 - 최소값을 0과 1로 표현 
x_train = scaler.fit_transform(x_train.reshape(-1, 28*28))
x_test = scaler.transform(x_test.reshape(-1, 28*28)) 

x_train = x_train.reshape(-1, 28*28, 1)
x_test = x_test.reshape(-1, 28*28, 1)

# .은 python의 부동소수를 보여주기 위해

print(np.max(x_train), np.min(x_train)) # 1.0 0.0 / 이미지는 이렇게 쓰는게 괜찮다
print(np.unique(y_train, return_counts=True))

print(y_test.shape) # (10000,)
print(y_train.shape) # (60000,)

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)
print(y_test.shape) # (10000, 10) / 이진 벡터로 변환

#2. 모델 구성
model = Sequential()

model.add(SimpleRNN(100, input_shape=(28*28, 1)))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_loss', patience=50, mode='min',
                   verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=100, batch_size=5000,
                 validation_split=0.2, verbose=1, callbacks=[es])

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)

acc = accuracy_score(np.argmax(y_test, axis=1), 
                     np.argmax(y_predict, axis=1))
print('acc : ', acc)

import matplotlib.pyplot as plt
# plt.plot(hist.history['val_loss'], label='val_acc')
plt.imshow(x_train[222])
plt.show()