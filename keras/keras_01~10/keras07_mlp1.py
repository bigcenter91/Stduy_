import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터


# 행무시, 열우선
x = np.array(
   [[1, 1],
    [2, 1],
    [3, 1],
    [4, 1],
    [5, 2],
    [6, 1.3],
    [7, 1.4],
    [8, 1.5],
    [9, 1.6],
    [10, 1.4]]
)

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

print(x.shape)  # (10, 2) > 2개의 특성을 가진 10개의 데이터 / 
                # 최소단위부터 센다_2개의 특성을 가진 10개의 데이터
print(y.shape)  # (10, ) 10 스칼라

# 행: 데이터 갯수
# 열: 특성, 컬럼, 피쳐
#열을 보고 모델 갯수 파악

model = Sequential()
model.add(Dense(3, input_dim=2)) # 열이 두개라서 '2' (차원) / 노드 2개로 시작
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer ='adam')
model.fit(x, y, epochs=30, batch_size=3) #batch = 3번에 나눠서 훈련

#4. 평가, 예측
loss = model.evaluate(x, y)  #평가는 model.fit에서 나온 값으로 한다
print('loss : ', loss)
result = model.predict([[10, 1.4]])
print("[[10, 1.4]]의 예측값 : ",result)
