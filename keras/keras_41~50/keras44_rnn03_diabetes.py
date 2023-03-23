from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import accuracy_score, r2_score

#1.데이터
datasets = load_diabetes()
x = datasets.data
y = datasets['target']

print(x.shape) # 442, 10

x = x.reshape(442, 10, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123)

#2. 모델 구성
model = Sequential()

model.add(SimpleRNN(32, input_shape=(10, 1)))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
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