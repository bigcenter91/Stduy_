import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터


# 행무시, 열우선
x = np.array(
   [[1,2,3,4,5,6,7,8,9,10],
    [1,1,1,1,2,1.3,1.4,1.5, 1.6, 1.4]]
)


y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

# x = x.transpose() #행열 바꾸는 함수_transpose
x = x.T #전치행렬 가로와 세로를 바꾼다

print(x.shape)  # (2, 10) > 10개의 특성을 가진 2개의 데이터 / 최소단위부터 센다
print(y.shape)  # (2, ) 2개의 피쳐?


model = Sequential()
model.add(Dense(3, input_dim=2)) # 열이 두개라서 '2' (차원)
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer ='adam')
model.fit(x, y, epochs=30, batch_size=3) #batch = 3개를 묶어서 훈련

#4. 평가, 예측
loss = model.evaluate(x, y)  #평가는 model.fit에서 나온 값으로 한다
print('loss : ', loss)
result = model.predict([[10, 1.4]])
print("[[10, 1.4]]의 예측값 : ",result)