import tensorflow as tf
tf.compat.v1.set_random_seed(337)
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

print(x.shape, y.shape) # (569, 30) (569,)
y = y.reshape(-1, 1)
print(y.shape) # (569, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=337,
)

print(x_train.shape, y_train.shape) # (455, 30) (455, 1)
print(x_test.shape, y_test.shape) # (114, 30) (114, 1)

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([30, 1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')

hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(xp, w) + b)

#3. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - yp)) # mse
loss = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)* tf.log(1-hypothesis))

optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)

train = optimizer.minimize(loss)


#3-1 훈려
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 1001
for epochs in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b],
            feed_dict={xp:x_train, yp:y_train})
    
    if epochs %20 == 0:
        print(epochs, 'loss: ', cost_val)

x_test = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])



# y_predict = x_data * w_val + b_val            # 넘파이랑 텐서랑 행렬곱했더니 에러생긴다. 그래서 맷멀써!!!
y_predict = tf.sigmoid(tf.matmul(x_test, w_val) + b_val)
y_predict = tf.cast(y_predict>0.5, dtype=tf.float32)
# y_predict>0.5 트루일 경우에 float형태로 바꿔줘라
# round 쳐도된다


y_aaa = sess.run(y_predict, feed_dict={xp:x_test})
print(y_aaa)
print(type(y_aaa)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

acc = accuracy_score(y_aaa, y_test)
print('acc :', acc)

mse = mean_squared_error(y_aaa, y_test)
print('mse :', mse)