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
model.add(Dense(2, input_dim=1))
# [[-1.175985  , -0.20179522, -1.1501358 ]]
model.add(Dense(1))


print(model.weights)
###################################
model.trainable = False # ★★★
###################################

model.compile(loss='mse', optimizer='adam')

model.fit(x, y, batch_size=1, epochs=10)

y_predict = model.predict(x)
print(y_predict)

# [[-0.7544218]
#  [-1.5088435]
#  [-2.2632656]
#  [-3.017687 ]
#  [-3.7721088]]

# 순전파 1 epochs

# Trainable params: 0 // 훈련이 필요없는게 있을 수 있다
# 다른거 가져왔을 때 굳이 훈련 시키지 않는다 이상해질 수 있다
# 입력과 출력만 커스터마이징 해주면 된다 가운데는 바꾸지않아도 된다