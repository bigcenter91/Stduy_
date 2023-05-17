# pip install keras==1.2.2
# from tensorflow.keras.datasets import mnist
from keras.datasets import mnist
import keras
# print(keras.__version__)
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error
import time

tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()

# tf.random.set_random_seed(337) # 이거 1.x되
tf.random.set_seed(337) # 이거 2.x 되


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) (60000, 28, 28) (60000,) 
# print(x_test.shape, y_test.shape) (10000, 28, 28) (10000,)

# [실습] 맹그러
# 60000, 784로 하면 되겠지?

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(x_train.shape, y_train.shape) # (60000, 784) (60000, 10)
# print(x_test.shape, y_test.shape) # (10000, 10) (10000, 10)

#2. 모델구성
x = tf.compat.v1.placeholder('float', [None, 784])
y = tf.compat.v1.placeholder('float', [None, 10])

w1 = tf.Variable(tf.random_normal([784, 128]), name='w1')
b1 = tf.Variable(tf.zeros([128]), name='b1')
layer1 = tf.compat.v1.matmul(x, w1) + b1
dropout1 = tf.compat.v1.nn.dropout(layer1, rate=0.3)

w2 = tf.Variable(tf.random_normal([128, 64]), name='w2')
b2 = tf.Variable(tf.zeros([64]), name='b2')
layer2 = tf.compat.v1.matmul(dropout1, w2) + b2

w3 = tf.Variable(tf.random_normal([64, 32]), name='w2')
b3 = tf.Variable(tf.zeros([32]), name='b3')
layer3 = tf.nn.softmax(tf.compat.v1.matmul(layer2, w3) + b3)

w4 = tf.Variable(tf.random_normal([32, 10]), name='w4')
b4 = tf.Variable(tf.zeros([10]), name='b4')
# hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer3, w4) + b4)
hypothesis = tf.matmul(layer3, w4) + b4


#3. 컴파일, 훈련
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy
# loss = tf.reduce_mean(-tf.reduce_sum(y * tf.nn.log_softmax(hypothesis), axis=1)) # categorical_crossentropy
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=y)) # categorical_crossentropy
# loss = tf.losses.softmax_cross_entropy(y, hypothesis)

train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

# [실습]
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

batch_size = 100
total_batch = int(len(x_train)/batch_size)


# Training
epochs = 100

start_time = time.time()
for step in range(epochs):
    avg_loss = 0
    for i in range(int(total_batch)):           # 100개씩 600번 도는거야
        start = i * batch_size                  # 0,   1   2 스타트
        end = start + batch_size                # 100, 200 300
        
        # x_train[start:end], y_train[start:end]
    
        loss_val, _, w_val, b_val = sess.run([loss, train, w4, b4],
            feed_dict={x: x_train[start:end], y: y_train[start:end]})
    
        avg_loss += loss_val / total_batch
    
    print('Epochs : ', step + 1, 'loss : {:.9f}'.format(avg_loss))
    
    
    
end_time = time.time()
print("훈련 끗")


    # if step % 10 == 0:
    #     print(step, "Loss:", loss_val)
        
y_pred = sess.run(hypothesis, feed_dict={x: x_test})
print(y_pred) # [2 2 2 1 0 1 0 0]

y_pred_arg = sess.run(tf.argmax(y_pred, 1))
print(y_pred_arg) # [2 2 2 1 1 1 0 0]

y_data_arg = np.argmax(y_test, 1)
print(y_data_arg)

acc = accuracy_score(y_pred_arg, y_data_arg)

print("acc:", acc)
print("tf", tf.__version__, "걸린시간 : ", end_time - start_time)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

acc = sess.run([accuracy], feed_dict={x:x_test, y:y_test})