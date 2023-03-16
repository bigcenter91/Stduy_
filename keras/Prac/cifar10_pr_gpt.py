from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

scaler = MinMaxScaler() # Minmax : 최대값 - 최소값
x_train = x_train/255.
x_test = x_test/255. # .은 python의 부동소수를 보여주기 위해

print(np.max(x_train), np.min(x_train)) # 1.0 0.0 / 이미지는 이렇게 쓰는게 괜찮다
print(np.unique(y_train, return_counts=True))

print(y_test.shape) # (10000, 1)
print(y_train.shape)

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)
print(y_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()

print(np.unique(y_train, return_counts=True))

#3. 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)

import time
start_time = time.time()

hist = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.2, verbose=1, callbacks=[es])

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


# training time :  2460.45
# 313/313 [==============================] - 6s 18ms/step - loss: 0.5861 - accuracy: 0.8063
# result :  [0.5861232280731201, 0.8062999844551086]
# acc :  0.8063