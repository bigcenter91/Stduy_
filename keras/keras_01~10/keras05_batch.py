import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 


# 1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])

# 2.모델구성
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=3)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)


result = model.predict([6])
print("[6]의 예측값 : ",result)

#batch 얼마나 묶어서 훈련하느냐 많아지면 좋아질 수 있으나
#느리고 오래걸린다 디폴트 : 32