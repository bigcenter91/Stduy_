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

print(model.weights) # bias까지 같이 연산량을 보여주고 있다
# [<tf.Variable 'dense/kernel:0' shape=(1, 3) > kernel이 weight다
# dtype=float32, numpy=array([[-0.2190435 , -0.48635262,  0.07527459]], 
# dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, 
# numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
# array([[ 0.4822924 ,  0.5086595 ],
#        [-0.21094984,  0.12488556],
#        [ 0.10581172,  0.9635434 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
# array([[-0.4670033],
#        [-0.2673074]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

# 할때마다 바뀌는건 nan 수치가 없다는 얘기야
# 고정해도 초기값은 같지만 연산하면서 달라진다 그래서 가중치 저장이 중요한거야

print("=========================")
print(model.trainable_weights)
print("=========================")

print(len(model.weights))               #6 // 한개의 레이어당 weight 하나 bias 하나
print(len(model.trainable_weights))

###################################
model.trainable = False # ★★★
###################################

print(len(model.weights))               #6
print(len(model.trainable_weights))

print("=========================")
print(model.trainable_weights)

model.summary()