import tensorflow as tf
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
tf.set_random_seed(915)
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error



#1. 데이터
x, y = load_wine(return_X_y=True)
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape) # (178, 3)
print(x.shape, y.shape) # (178, 13) (178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=337,
)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([13,3]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([3]), name='bias')


#2. 모델구성
hypothesis = tf.compat.v1.nn.softmax(tf.matmul(x, w) + b)


# 3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy
train = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)


# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    _, hy_val, cost_val, b_val = sess.run([train, hypothesis, loss, b], feed_dict={x:x_train, y:y_train})
    if step % 20 == 0:
        print(step, cost_val, hy_val)


y_pred = sess.run(hypothesis, feed_dict={x: x_test})

# Convert one-hot encoded predictions to class labels
y_pred_arg = sess.run(tf.argmax(y_pred, 1))
# print(y_pred_arg) # [2 2 2 1 1 1 0 0]

y_data_arg = np.argmax(y_test, 1)
# print(y_data_arg)

acc = accuracy_score(y_pred_arg, y_data_arg)
# mse = mean_squared_error(y_test, y_pred)
print('acc : ', acc)