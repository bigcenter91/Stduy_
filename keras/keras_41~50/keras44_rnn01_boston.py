from sklearn.datasets import load_boston
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

#1. 데이터

datasets = load_boston()
x = datasets.data
y = datasets['target']

print (x.shape) # (506, 13)
print (y.shape) # (506, )

x = x.reshape(506, 13, 1)

print(x.shape) # (506, 13, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123,
)

#2. 모델 구성
model = Sequential()
model.add(SimpleRNN(32, input_shape=(13, 1)))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='loss', patience=50, mode='min',
                    verbose=1, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=200, batch_size=1000,
                 validation_split=0.2, verbose=1, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)