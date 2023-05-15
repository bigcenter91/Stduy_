import tensorflow as tf
tf.compat.v1.set_random_seed(337)
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

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
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)
# sigmoid를 통과한 식을 만들어준다
# activate 함수라는게 뭐야? 나온 값에 activation을 넣어준거다?

loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
# mse를 쓸 수가 없지?

# loss = 'binary_crossentropy' 
loss = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)* tf.log(1-hypothesis))
# 1일 때 (1-y)* tf.log(1-hypothesis) 1-1이니까 (y*tf.log(hypothesis) 얘가돈다


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
y_predict = tf.sigmoid(tf.matmul(x_test, w_val) + b_val)
y_predict = tf.cast(y_predict>0.5, dtype=tf.float32)
# y_predict>0.5 트루일 경우에 float형태로 바꿔줘라
# round 쳐도된다


y_aaa = sess.run(y_predict, feed_dict={x_test:x_data})
print(y_aaa)
print(type(y_aaa)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

acc = accuracy_score(y_aaa, y_data)
print('acc :', acc)

mse = mean_squared_error(y_aaa, y_data)
print('mse :', mse)

# acc : 0.8333333333333334
# mse : 0.16666666666666666

sess.close()

# 이진분류야

# 다중분류 카테고리컬 크로스엔트로피
# 시그모이드 바이너리 크로스엔트로피
# 활성화 함수라고도 하지만 한정함수
# 0과 1 아니야 0~1 사이


# r2 : 0.6517273256496418
# mse : 0.047364102519423724
