import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
tf.set_random_seed(337)
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, mean_absolute_error


#1. 데이터
x, y = load_iris(return_X_y=True)
y = to_categorical(y)
print(x.shape, y.shape) # (150, 4) (150, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=337,
)

#2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w1 = tf.Variable(tf.random_normal([4, 8]), name='weight1')
b1 = tf.Variable(tf.zeros([8]), name='bias1')
layer1 = tf.compat.v1.matmul(x, w1) + b1

w2 = tf.Variable(tf.random_normal([8, 16]), name='weight2')
b2 = tf.Variable(tf.zeros([16]), name='bias2')
layer2 = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(layer1, w2) + b2)

w3 = tf.Variable(tf.random_normal([16, 24]), name='weight3')
b3 = tf.Variable(tf.zeros([24]), name='bias3')
layer3 = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(layer2, w3) + b3)

w4 = tf.Variable(tf.random_normal([24, 3]), name='weight4')
b4 = tf.Variable(tf.zeros([3]), name='bias4')
hypothesis = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(layer3, w4) + b4)

# cost = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))
cost = tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00005).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    
    for step in range(1001) :
        _, hy_val, cost_val, b_val = sess.run([train, hypothesis, cost, b4], feed_dict={x:x_train, y:y_train})
    if step % 20 == 0 :
        print(step, cost_val, hy_val)
        
    _, y_predict = sess.run([train, hypothesis], feed_dict={x:x_test, y:y_test})

    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1))
    print('acc : ',acc)

    sess.close()
