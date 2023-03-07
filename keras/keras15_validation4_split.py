from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,17)) # 10개의 데이터죠?
y = np.array(range(1,17)) # range = 10 스칼라 1 벡터(특성이 하나) 1 디맨션

# x_val = np.array([14,15,16]) #16개 데이터 중에 3개도 되고, 13개 중 3개
# y_val = np.array([14,15,16])
# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.2, random_state=123, shuffle=True)



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
          validation_split=0.2) #20%는 validation으로 쓰겠다라는 말

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print('17의 예측값 : ', result)