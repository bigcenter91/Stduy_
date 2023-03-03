from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, LeakyReLU
import numpy as np



#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.8, shuffle=True, random_state=20)


# [실습]
# R2 0.55 ~ 0.6 이상

#2. 모델 구성
model = Sequential()
model.add(Dense(20, input_dim=8, activation = 'sigmoid'))
model.add(Dense(30, activation=LeakyReLU()))
model.add(Dense(40, activation=LeakyReLU()))
model.add(Dense(80, activation=LeakyReLU()))
model.add(Dense(60, activation=LeakyReLU()))
model.add(Dense(40, activation=LeakyReLU()))
model.add(Dense(20, activation=LeakyReLU()))
model.add(Dense(15, activation=LeakyReLU()))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer='adam' )
model.fit(x_train, y_train, epochs=10, batch_size=8)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

#R2 r2 결정계수

from sklearn.metrics import r2_score #predict 예측한 y값
r2 = r2_score(y_test, y_predict)

print('r2 스코어 : ', r2)