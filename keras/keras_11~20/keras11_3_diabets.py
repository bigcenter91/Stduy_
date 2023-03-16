from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
import numpy as np


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)  # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.90, shuffle=True, random_state=123)

# [실습]
# R2 0.62 이상 // 잘맞을라면 데이터 정제를 잘해야한다

#2. 모델 구성
model = Sequential()
model.add(Dense(20, input_dim=10, activation = 'sigmoid'))
model.add(Dense(50, activation=LeakyReLU()))
model.add(Dense(70, activation=LeakyReLU()))
model.add(Dense(90, activation=LeakyReLU()))
model.add(Dense(80, activation=LeakyReLU()))
model.add(Dense(45, activation=LeakyReLU()))
model.add(Dense(30, activation=LeakyReLU()))
model.add(Dense(20, activation=LeakyReLU()))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=10) # r2 스코어 :  0.6357998381006098


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

#R2 r2 결정계수

from sklearn.metrics import r2_score #predict 예측한 y값
r2 = r2_score(y_test, y_predict)

print('r2 스코어 : ', r2)