import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

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


hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x, w) + b)

#3-1. 컴파일
# loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y)) # mse
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
# loss의 categorical_crossentropy

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-2)
# train = optimizer.minimize(loss)

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(loss)


# [실습]
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2001

# Training
for epoch in range(epochs):
    loss_val,_, w_val, b_val = sess.run([loss, train, w, b],
                feed_dict={x:x_data, y : y_data})
    
    if epoch % 10 ==0:
        print(epoch, 'loss :', loss_val)

#4. 평가, 예측            
x_test = tf.compat.v1.placeholder(tf.float32, shape = [None, 4])
y_predict = tf.matmul(x_test, w_val) + b_val
y_data =np.argmax(y_data, axis =1)
y_aaa = np.argmax(sess.run(y_predict, feed_dict={x_test:x_data}),axis=1)
print(type(y_aaa))  # <class 'numpy.ndarray'>

acc = accuracy_score(y_aaa, y_data)
print('acc : ', acc)

mse = mean_squared_error(y_aaa, y_data)
print('mse : ', mse)

sess.close()