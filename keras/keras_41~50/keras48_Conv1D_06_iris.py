from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd

datasets = load_iris()

x = datasets.data
y = datasets['target']

print(x.shape, y.shape) # (150, 4) (150,)

x = x.reshape(150, 2, 2)

y = to_categorical(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=777)

print(x.shape,y.shape)

#2. 모델 구성
model = Sequential()

model.add(Conv1D(10, 2, input_shape=(2, 2)))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics='acc')

es = EarlyStopping(monitor='val_loss', patience=5, mode='min',
                   verbose=1, restore_best_weights=True)

model.fit(x_train,y_train, epochs=10, batch_size=200,
          validation_split=0.2, verbose=1, callbacks=[es])

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

y_test_acc = np.argmax(y_test,axis=1)
y_predict = np.argmax(y_predict,axis=1)

acc = accuracy_score(y_test_acc,y_predict)
print("acc : ", acc)