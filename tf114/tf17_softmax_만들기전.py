import tensorflow as tf
import numpy as np
tf.set_random_seed(337)

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],        # 1
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],        # 0
          [1, 0, 0]]

#2. 모델 구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
w = tf.compat.v1.Variable(tf.random_normal([4, 3]))
b = tf.compat.v1.Variable(tf.zeros([1, 3]), name = 'bias')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
# N, 4 / 4, 3 == 4가 퉁퉁 되니까 (4, 3) // hypothesis 또는 y_predict가 되겠지?


hypothesis = tf.compat.v1.matmul(x, w) + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-2)
# train = optimizer.minimize(loss)

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(loss)


# [실습]
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# Training
epochs = 3000
for epoch in range(epochs):
    _, loss_val = sess.run([train, loss], feed_dict={x: x_data, y: y_data})
    if epoch % 10 == 0:
        print("Epoch:", epoch, "Loss:", loss_val)
        

# x_test = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
from sklearn.metrics import mean_squared_error, r2_score

y_pred = sess.run(hypothesis, feed_dict={x:x_data})
r2 = r2_score(y_data, y_pred)
print('R2:', r2)


# w1x1 +w2x2 + w3x3 + b = y