import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(337)

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
# [[-1.175985  , -0.20179522, -1.1501358 ]]
model.add(Dense(2)) # 3개가 2개로 되지 연산량은 6번이고 = (N, 1) * (1, 3)
model.add(Dense(2)) 
model.add(Dense(1))
model.summary()

###################################
# model.trainable = False # ★★★
###################################
model.summary()

model.compile(loss='mse', optimizer='adam')

model.fit(x, y, batch_size=1, )

y_predict = model.predict(x)
print(y_predict)
