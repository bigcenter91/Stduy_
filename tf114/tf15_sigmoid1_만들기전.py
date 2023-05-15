import tensorflow as tf
tf.compat.v1.set_random_seed(337)
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]     # (6, 2)
y_data = [[0], [0], [0], [1], [1], [1]]                 # (6, 1)

###############################
# [실습] 시그모이드 빼고 걍 만들어봐!!!
###############################
'''
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')


#2. 모델
hypothesis = tf.compat.v1.matmul(x, w) + b


#3. 컴파일, 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-3)
train = optimizer.minimize(cost)

epochs = 1001
for step in range(epochs):
    _, cost_val = sess.run([train, cost],
                           feed_dict={x:x_data, y:y_data})
    if step %20 == 0:
    
        print(epochs, 'loss:', cost_val)
        
y_predict = sess.run(hypothesis, feed_dict={x: x_data})
r2 = r2_score(y_data, y_predict)
print('R2:', r2)
'''
############################# yys

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 1]), dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), dtype=tf.float32)

#2. 모델
hypothesis = tf.compat.v1.matmul(x, w) + b


loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-2)
train = optimizer.minimize(loss)

#3. 컴파일, 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b], feed_dict={x:x_data, y:y_data})
    
    if step % 20 == 0:
        print(epochs, 'loss :', cost_val)


print(w_val)

print(type(w_val), type(b_val))
print(b_val)

# 평가, 예측
#########################################################
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])

# y_predict = x_data * w_val + b_val            # 넘파이랑 텐서랑 행렬곱했더니 에러생긴다. 그래서 맷멀써!!!
y_predict = tf.matmul(x_test, w_val) + b_val

y_aaa = sess.run(y_predict, feed_dict={x_test:x_data})
print(type(y_aaa)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

r2 = r2_score(y_aaa, y_data)
print('r2 :', r2)

mse = mean_squared_error(y_aaa, y_data)
print('mse :', mse)


sess.close()

# 이진분류야
# r2 : 0.6517273256496418
# mse : 0.047364102519423724
