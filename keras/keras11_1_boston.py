from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, LeakyReLU
import numpy as np



#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=100)

#print(x)
#print(y)
#1~ 1조라면 0~1로 바꿔줘야한다 1조로 나누면 가능

print(datasets.feature_names)
'''['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
 'B' 'LSTAT']'''
print(datasets.DESCR)

print(x.shape, y.shape) # (506, 13) / (506, )

# [실습]
#1. train 0.7
#2. r2 0.8 이상
#######################

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=13, activation = 'sigmoid'))
model.add(Dense(200, activation=LeakyReLU()))
model.add(Dense(250, activation=LeakyReLU()))
model.add(Dense(100, activation=LeakyReLU()))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=7)

loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

#R2 r2 결정계수

from sklearn.metrics import r2_score #predict 예측한 y값
r2 = r2_score(y_test, y_predict)

print('r2 스코어 : ', r2)