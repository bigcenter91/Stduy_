# pip install keras==1.2.2
# from tensorflow.keras.datasets import mnist
from keras.datasets import mnist
import keras
# print(keras.__version__)
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error



(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) (60000, 28, 28) (60000,) 
# print(x_test.shape, y_test.shape) (10000, 28, 28) (10000,)

# [실습] 맹그러
# 60000, 784로 하면 되겠지?

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

y_train = to_categorical(y_train) # (60000, 10)
y_test = to_categorical(y_test)

print(y_train.shape) # (60000, 10)
print(y_test.shape) # (10000, 10)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

w1 = tf.Variable(tf.random_normal([784, 64]), name='weight1')
b1 = tf.Variable(tf.zeros([64]), name='bias1')
layer1 = tf.compat.v1.matmul(x, w1) + b1

w2 = tf.Variable(tf.random_normal([64, 64]), name='weight2')
b2 = tf.Variable(tf.zeros([64]), name='bias2')
layer2 = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(layer1, w2) + b2)

w3 = tf.Variable(tf.random_normal([64, 64]), name='weight3')
b3 = tf.Variable(tf.zeros([64]), name='bias3')
layer3 = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(layer2, w3) + b3)

w4 = tf.Variable(tf.random_normal([64, 32]), name='weight4')
b4 = tf.Variable(tf.zeros([32]), name='bias4')
layer4 = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(layer3, w4) + b4)

w5 = tf.Variable(tf.random_normal([32, 10]), name='weight5')
b5 = tf.Variable(tf.zeros([10]), name='bias5')
hypothesis = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(layer4, w5) + b5)

# cost = tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
cost = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, hypothesis))

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00005).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

# with tf.Session() as sess :
#     sess.run(tf.global_variables_initializer())
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2001
for step in range(epochs) :
    _, hy_val, cost_val, b_val = sess.run([train, hypothesis, cost, b4], feed_dict={x:x_train, y:y_train})
    
    if step % 20 == 0 :
        print(epochs, cost_val, hy_val)
        
    _, y_predict = sess.run([train, hypothesis], feed_dict={x:x_test, y:y_test})

acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1))
print('acc : ',acc)
    
sess.close()