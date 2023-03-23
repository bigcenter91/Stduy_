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
model.add(LSTM(10, input_shape=(5, 1))) # 10*(10+1+1) = 120 / RNN은 출력 3차원 받아서 2차원으로
# units * (feature + bias + units) = 120
model.add(Dense(7)) # (10 + 1) x 7 = 77
model.add(Dense(1)) # (7 + 1) x 1 = 8

model.summary()