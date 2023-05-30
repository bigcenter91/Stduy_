import tensorflow as tf
import numpy as np

x_train = [1,2,3] # [1]
y_train = [1,2,3] # [2]
x_test = [4,5,6]
y_test = [4,5,6]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
# ============= 아래로 옵티마이저여 =============
lr = 0.1
# gradient = tf.reduce_mean((w * x - y) * x)
gradient = tf.reduce_mean((x * w - y) * x)  # w = w-lr x ae/aw 연관이 있다 // loss의 미분값
# gradient = tf.reduce_mean((hypothesis - y) * x)

descent = w - lr * gradient

update = w.assign(descent) # w = w - lr * gradient

# ============= 아래로 옵티마이저여 =============


w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(21):
    _, loss_v, w_v =sess.run([update, loss, w], feed_dict={x:x_train, y:y_train})
    print(step, '\t', loss_v, '\t', w_v)
    
    w_history.append(w_v)
    loss_history.append(loss_v)

sess.close()

# R2 and mae evaluation
# predictions = sess.run(hypothesis, feed_dict={x: x_train})
# r2 = 1 - np.sum((y_train - predictions) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
# mae = np.mean(np.abs(y_train - predictions))


# print("============= w history =============")
# print(w_history)
# print("============= loss history =============")
# print(loss_history)

############### [실습] R2, mae 맹그러!!! ###############
from sklearn.metrics import r2_score, mean_absolute_error

y_predict = x_test * w_v
print(y_predict) # [4.00006676 5.00008345 6.00010014]

r2 = r2_score(y_predict, y_test)
print("r2 :", r2) # r2 : 0.999999989276847

mae = mean_absolute_error(y_predict, y_test)
print("mae :", mae) # mae : 8.344650268554688e-05


# print("R2: {:.2f}".format(r2))
# print("mae: {:.2f}".format(mae))

