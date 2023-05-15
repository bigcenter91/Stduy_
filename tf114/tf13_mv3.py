import tensorflow as tf

tf.compat.v1.set_random_seed(337)

x_data = [[73, 51, 65],                         # (5, 3)
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]
y_data = [[152], [185], [180], [205], [142]]    # (5, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')

#2. 모델
# hypothesis = x1*w1 + x2*w2 + x3*w3 + b
# hypothesis = x * w + b
hypothesis = tf.compat.v1.matmul(x, w) + b # y랑 shape이 같다
# matmul: 행렬곱

# x.shape = (5, 3)
# y.shape = (5, 1)
# hy = x * w + b
#    = (5, 3) * w + b = (5, 1)
   
# (5, 3) * (?, ?) = (5, 1) 답은 (3, 1)

# 컴파일, 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())


from sklearn.metrics import r2_score, mean_absolute_error

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.000005)
train = optimizer.minimize(cost)

epochs = 5001
for step in range(epochs):
    _, cost_val = sess.run([train, cost], feed_dict={x: x_data, y: y_data})
    
    if step %20 == 0:
    
        print(epochs, 'loss:', cost_val)

y_pred = sess.run(hypothesis, feed_dict={x: x_data})
r2 = r2_score(y_data, y_pred)
print('R2:', r2)