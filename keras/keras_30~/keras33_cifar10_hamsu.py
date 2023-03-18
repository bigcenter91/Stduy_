from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# 두 개의 튜플로 나눠서 할당하기 위해 사용

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

scaler = MinMaxScaler() # Minmax : 최대값 - 최소값을 0과 1로 표현
x_train = x_train/255.
x_test = x_test/255. # .은 python의 부동소수를 보여주기 위해

print(np.max(x_train), np.min(x_train)) # 1.0 0.0 / 이미지는 이렇게 쓰는게 괜찮다
print(np.unique(y_train, return_counts=True))

print(y_test.shape) # (10000, 1)
print(y_train.shape) # (50000, 1)

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)
print(y_test.shape) # (10000, 10) / 이진 벡터로 변환

#2. 모델 구성
model = Sequential()
model.add(Conv2D(16, 2,
                 padding='same',
                 input_shape=(32, 32, 3)))

model.add(MaxPooling2D())
# (2,2)중 가장 큰 값 뽑아서 반의 크기(14x14)로 재구성함
# Maxpooling안에 디폴트가 (2,2)로 중첩되지 않도록 설정되어있음

model.add(Conv2D(16, 2, padding='valid', activation='relu'))
model.add(Conv2D(16, 2))
model.add(Conv2D(16, 2, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary() # 요약하여 출력

# print(np.unique(y_train, return_counts=True))


#3. 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer='Adagrad', metrics='acc')

es = EarlyStopping(monitor='val_loss', patience=30, mode='max',
                   verbose=1, restore_best_weights=True)

import time
start_time = time.time()

hist = model.fit(x_train, y_train, epochs=100, batch_size=2000,
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
plt.show()