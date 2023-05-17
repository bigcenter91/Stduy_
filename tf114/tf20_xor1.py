import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
tf.set_random_seed(337)

#1. 데이터
x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

#2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.compat.v1.Variable(tf.random_normal([1]), name='bias')


hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)
# sigmoid 통과해서 0~1 사이가 될거아냐

#3-1. 컴파일
cost = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))


train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# 0.7이 네개가 된다고 하면
# 0.7이라면 0.5보다 크니까 True로 반환하겠지
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))
# tf.equal(predicted, y = true나 false로 반환되
# 0 1 1 0이 되면 2/4로 되고 엔빵치니까 0.5

 
with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    
    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict = {x:x_data, y:y_data})
        
        if step % 200 == 0:
            print(step, cost_val)
            
    h, p, a = sess.run([hypothesis, predicted, accuracy],
             feed_dict={x:x_data, y:y_data})
    print("예측값:",  h, "\n 원래값 :", p, "\n Accuracy :", a)


# with tf.compat.v1.Session() as sess : 
#     sess.run(tf.global_variables_initializer())
    
#     epochs = 1001
#     for step in range(epochs):
#         cost_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
#         if step % 20 == 0 :
#             print(epochs, 'loss :', cost_val, '\n', hy_val)
    
   
#     y_predict = sess.run(hypothesis, feed_dict={x: x_data}) # 이 값이 참이면 1 거짓이면 0

#     acc = accuracy_score(y_data, y_predict)
#     print('acc :', acc)