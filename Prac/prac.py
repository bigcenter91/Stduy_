import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x =np.array([1,2,3,4,5])
y =np.array([4,5,6,7,8])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y
    test_size=0.3, shuffle=True, random_state=123)


#2.모델 구성
model = Sequential()
model.add(Dense(4, input_dim=10))
model.add(Dense(6))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(15))
model.add(Dense(7))
model.add(Dense(1))


#3 컴파일, 훈련
model.comlie(loss='mse', optimize='adam')
model.fit(x_train, y_train, epochs=200, batch_size=5)

loss = model.evaluate(x_test, y_test)
print("loss : ", loss)


y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('r2 스코어 : ', r2)