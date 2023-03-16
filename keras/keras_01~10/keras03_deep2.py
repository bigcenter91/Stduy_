# 맨처음 층 inputlayer 아래는 output layer

#1. 데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
import tensorflow as tf
from tensorflow.keras.models import Sequential #순차적으로 가기 때문에 Sequense
from tensorflow.keras.layers import Dense 
#각 층을 이루고있어서 layers
#Dense는 y = ax+b 


# 연산단계를 모르니까 hidden layer
model = Sequential()
model.add(Dense(3, input_dim=1)) #상위에 있는 add가 input으로 아래 입력하지 않아도 됨
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))

# 한 라인 삭제 Shift + Delete
# 취소 Ctrl + Z


# 맨처음 층 inputlayer 아래는 output layer

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)
# mse, mae, epochs(훈련량), node, 레이어 수를 추가 및 삭제하여 변화 가능


#4. 평가, 예측
loss = model.evaluate(x, y) # 추후에는 훈련하지 않은 값을 넣어서 평가한다 / predict 예측하다
print('loss : ', loss)
result = model.predict([4])
print("4의 예측값 : ",result)