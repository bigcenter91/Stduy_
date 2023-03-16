import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))


model.summary()

#파라미터가 아래와 같은 이유는 
#wx+b