from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. 데이터
x_train = np.array(range(1,11)) # 10개의 데이터죠?
y_train = np.array(range(1,11)) # range = 10 스칼라 1 벡터(특성이 하나) 1 디맨션

x_val = np.array([14,15,16]) #16개 데이터 중에 3개도 되고, 13개 중 3개
y_val = np.array([14,15,16])


x_test = np.array([11,12,13])
y_test = np.array([11,12,13])


#2. 모델
model= Sequential()
model.add(Dense(5, activation='linear', input_dim=1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1)) #특성, 피쳐 하나

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val)) #train data가 제일크긴해야한다

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print('17의 예측값 : ', result)

#loss :  1.515824466814808e-12 // 17의 예측값 :  [[17.]] // epochs=100, batch_size=1,// 5 5 5 3 1 
#loss :  4.108793749679762e-09 // 17의 예측값 :  [[17.00009]] // epochs=100, batch_size=1,// 5 5 5 5 1 