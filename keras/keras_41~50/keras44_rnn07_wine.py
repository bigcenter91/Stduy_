from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd

# 1. 데이터
datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target.reshape(-1,1)

print(x.shape,y.shape) # (178, 13) (178,)
print(np.unique(y)) # [0 1 2]

y = to_categorical(y)

x = x.reshape(178,13,1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, test_size=0.8, stratify=y,
)

print(x_train.shape) # (35, 13, 1)
print(x_test.shape) # (143, 13, 1)

#2. 모델 구성
model = Sequential()

model.add(SimpleRNN(50, input_shape=(13, 1)))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics='acc')
es = EarlyStopping(monitor='val_loss', patience=20, mode='min',
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