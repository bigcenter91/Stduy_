import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터

path = 'd:/study_data/_save/fmnist/'

x_train = np.load(path + 'keras58_fmnist_x_train.npy')
x_test = np.load(path + 'keras58_fmnist_x_test.npy')
y_train = np.load(path + 'keras58_fmnist_y_train.npy')
y_test = np.load(path + 'keras58_fmnist_y_test.npy')

print(x_train.shape, x_test.shape) # (70000, 28, 28, 1) (10000, 28, 28, 1)


#2. 모델
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(28,28, 1), activation='relu'))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

es = EarlyStopping(monitor='acc', mode='auto', patience=5, restore_best_weights=True)

model.fit(x_train, y_train, epochs=50, batch_size=128, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print(y_predict.shape) # (10000, 10)
print(y_test.shape) # (10000, 10)
acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1))
print('acc : ', acc)