import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([10,9,8,7,6,5,4,3,2,1,])


#[실습] 넘파이 리스트의 슬라이싱 7:3으로 잘라라

x_train = x[:7] #컴퓨터 숫자는 0부터니까
#print(x_train) # [1 2 3 4 5 6 7]
x_test = x[7:]
#print(x_test) # [8 9 10]
y_train = y[:7]
#print(y_train)
y_test = y[7:]
#print(y_test)
#훈련을 시킬 때는 범위 내에 훈련을 시키되 평가데이터에 쓰면 안된다
#앞으로는 전체 범위에서 비율로 뽑아 낼 것이다


print(x_train.shape, x_test.shape) #(7,) (3,)
print(y_train.shape, y_test.shape) #(7,) (3,)
