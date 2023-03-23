from tensorflow.keras.datasets import cifar100
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
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

ry_test = to_categorical(y_test)
y_train = to_categorical(y_train)
print(y_test.shape) # (10000, 10) / 이진 벡터로 변환

#2. 모델 구성

input1 = Input(shape=(32, 32, 3))
conv1 = Conv2D(16, 2, padding='same', activation='relu')(input1)
max1 = MaxPooling2D(pool_size=(2,2))(conv1)
conv2 = Conv2D(32, 3, padding='same')(max1)
conv3 = Conv2D(64, 3, padding='same', activation='relu')(conv2)
max2 = MaxPooling2D(pool_size=(2,2))(conv3)
flat1 = Flatten()(max2)
output1 = Dense(100, activation='softmax')(flat1)
model = Model(inputs=input1, outputs=output1)

model.summary()

#print(np.unique(y_train, return_counts=True))

#3. 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics='acc')

es = EarlyStopping(monitor='val_loss', patience=30, mode='min',
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
# plt.plot(hist.history['val_loss'], label='val_acc')
plt.imshow(x_train[555])
plt.show()
