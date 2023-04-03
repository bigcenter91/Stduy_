from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical

path_save = 'd:/study_data/_save/mnist/'
x_train = np.load(path_save + 'keras58_2_mnist_x_train.npy')
x_test = np.load(path_save + 'keras58_2_mnist_x_test.npy')
y_train = np.load(path_save + 'keras58_2_mnist_y_train.npy')
y_test = np.load(path_save + 'keras58_2_mnist_y_test.npy')

# 2. 모델
model = Sequential()
model.add(Conv2D(64, 2, input_shape=(28, 28, 1)))
model.add(Conv2D(64, 2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='acc', mode='auto', patience=20, restore_best_weights=True)
model.fit(x_train, y_train, epochs=1, batch_size=128, validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print(y_predict.shape)
print(y_test.shape)
acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1))
print('acc : ', acc)