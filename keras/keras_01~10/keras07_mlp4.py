import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터


x = np.array([range(10), range(21,31), range(201, 211)]) 

x = x.T
print(x.shape) #(3, 10)

#print(x)  #0 1 2 3 4 5 6 7 8 9 컴퓨터 시작 숫자는 '0' 0부터 10-1로 생각하면 된다

'''
[[  0  21 201]
 [  1  22 202]
 [  2  23 203]
 [  3  24 204]
 [  4  25 205]
 [  5  26 206]
 [  6  27 207]
 [  7  28 208]
 [  8  29 209]
 [  9  30 210]]
'''

y = np.array([[1,2,3,4,5,6,7,8,9,10]])  # (1, 10)
y = y.T # (10, 1)

print(y.shape)

model = Sequential()
model.add(Dense(4, input_dim=3))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))   #행 무시, 열 우선

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer ='adam')
model.fit(x, y, epochs=200, batch_size=10) 

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([[9, 30, 210]])
print("[[9, 30, 210]]의 예측값 : ",result)



