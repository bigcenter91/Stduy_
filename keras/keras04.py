import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 


# 1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])

model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=600)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)


result = model.predict([6])
print("[6]의 예측값 : ",result)