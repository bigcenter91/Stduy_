# x는 1개
# y는 3개

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


x = np.array([range(10)]) 

x = x.T
print(x.shape) #(3, 10)

#print(x)  #0 1 2 3 4 5 6 7 8 9 컴퓨터 시작 숫자는 '0' 0부터 10-1로 생각하면 된다


y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
              [9,8,7,6,5,4,3,2,1,0]])  # (3, 10)
              
y = y.T # (10, 3)

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer ='adam')
model.fit(x, y, epochs=200, batch_size=1)  

#훈련에 사용한 데이터는 평가에 사용하지 않는다.

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([[9]])
print("[[9]]의 예측값 : ",result)  # 예상 값 10, 1.9, 0

# loss :  0.021128108724951744 // [[9]]의 예측값 :  [[10.048254    1.9233867  -0.06694499]] // model.fit(x, y, epochs=200, batch_size=1)  