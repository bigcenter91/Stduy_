import tensorflow as tf
import matplotlib.pyplot as plt

# x = [1,2,3]
# y = [1,2,3]
x = [1,2]
y = [1,2]

w = tf.compat.v1.placeholder(tf.float32) # -30부터하면

hypothesis = x * w 
loss = tf.reduce_mean(tf.square(hypothesis - y))
# 로컬미니마로 어떻게든 찾아간다
# 변화량은 기울기

w_history = []
loss_history = []

with tf.compat.v1.Session() as sess :
    for i in range(-30, 50):
        curr_w = i
        curr_loss = sess.run(loss, feed_dict={w:curr_w})
        
        w_history.append(curr_w)
        loss_history.append(curr_loss)
        
print("============= w history =============")
print(w_history)
print("============= loss history =============")
print(loss_history)


plt.plot(w_history, loss_history)
plt.xlabel('weight')
plt.ylabel('loss')
plt.show()

# loss가 가장 낮은 지점을 찾는거지?
# 기울기가 0이면 loss가 가장 낮은 지점이겠지
# 로스로 웨이트를 미분한건 양수겠지