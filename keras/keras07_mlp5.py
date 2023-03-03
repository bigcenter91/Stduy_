# x는 3개
# y는 2개

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


x = np.array([range(10), range(21,31), range(201, 211)]) 

x = x.T
print(x.shape) #(3, 10)

#print(x)  #0 1 2 3 4 5 6 7 8 9 컴퓨터 시작 숫자는 '0' 0부터 10-1로 생각하면 된다


y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]])  # (2, 10)
              
y = y.T # (10, 2)


#실습 
#예측 : [9, 30, 210] > 예상 [10, 1.9]


model = Sequential()
model.add(Dense(4, input_dim=3))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(2))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer ='adam')
model.fit(x, y, epochs=200, batch_size=3)  

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([[9, 30, 210]])
print("[[9, 30, 210]]의 예측값 : ",result)

#[[9, 30, 210]]의 예측값 :  [[9.472487  1.7591251]] _ epochs=200, batch_size=3_loss :  0.2525573670864105)