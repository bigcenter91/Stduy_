import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

# y=?

x=np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],
            [5,6,7,8,9]]) # 8, 9, 10 y가 있기 때문에

y=np.array([6, 7 ,8 ,9 ,10])

print(x.shape, y.shape) # (5, 5) (5,) 
# rnn의 구조는 3차원이 되야한다
# x의 shpae = (행, 열, 몇개씩 훈련하는지!!!)
x = x.reshape(5, 5, 1) # [[1], [2], [3]], [[2], [3], [4],....]]
print(x.shape) # (5, 5, 1) = RNN의 SHAPE를 맞춰줄려고 변경

#2. 모델 구성
model = Sequential()                         # batch, timesteps:열 / 여기서는 5로 짤랐지, feature
                                             # batch, input_legth, input_dim으로 가능
                                             
# model.add(LSTM(10, input_legth=5, input_dim=1))
model.add(LSTM(10, input_dim=1, input_legth=5)) # 가독성은 떨어져
model.add(Dense(7))
model.add(Dense(1)) 

model.summary()