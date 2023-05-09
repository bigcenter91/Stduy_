# 실습
# lr 수정해서 epoch 100번 이하로 줄여
# step = 100, 이하, w=1.99, b=0.99

import tensorflow as tf
tf.set_random_seed(337)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

#1. 데이터

x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)
# uniform 균등분포

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(w)) #[-0.4121612]

# with tf.compat.v1.Session() as sess:



#2. 모델 구성
# y = wx + b
# weight가 앞에 있느냐 뒤에 있느냐 값이 다르다

hypothesis = x * w + b


#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
# learning_rate만큼 줄어든다 epochs가 지날 수록 우리가 원하는 값으로 되겠지?

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.008)
train = optimizer.minimize(loss)
# model.compile(loss='mse', optimizer='sgd')

#3-2 . 훈련
loss_val_list = []
w_val_list = []



with tf.compat.v1.Session() as sess:
# sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    epochs = 101
    for step in range(epochs):
        # sess.run
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                             feed_dict={x:[1,2,3,4,5], y:[2,4,6,8,10]})
        if step %20 == 0:
            # print(step, sess.run(loss), sess.run(w), sess.run(b)) # verbose가 되겠지?
            print(step, loss_val, w_val, b_val) # verbose가 되겠지?
            
        loss_val_list.append(loss_val)
        w_val_list.append(w_val)
    # y_predict = sess.run(hypothesis, feed_dict={x: [6, 7, 8]})
    # print("예측 값:", y_predict)
        
        
# 그래프 연산방식은 그래프를 만들어 그리고 한방에 툭!
# placeholer에는 feed_dict이 따라다닌다

######################## [실습] ########################

# x_data = [6,7,8]
# 예측값을 뽑아라
########################################################
# placeholder를 정의하고 

    x_data = [6,7,8]

    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

    y_predict = x_test * w_val + b_val

    print('[6,7,8]의 예측 값 : ',
        sess.run(y_predict, feed_dict={x_test:x_data}))
    
# print(loss_val_list)
# print(w_val_list)

# plt.plot(loss_val_list)
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.show()

# plt.plot(w_val_list)
# plt.xlabel('epochs')
# plt.ylabel('weight')
# plt.show()

# plt.scatter(w_val_list, loss_val_list)
# plt.xlabel('weight')
# plt.ylabel('loss')
# plt.show()




fig, axs = plt.subplots(1, 3, figsize=(15,5))   # 1행 3열의 subplot 생성

# 첫번째 subplot
axs[0].plot(loss_val_list)
axs[0].set_xlabel('epochs')
axs[0].set_ylabel('loss')

# 두번째 subplot
axs[1].plot(w_val_list)
axs[1].set_xlabel('epochs')
axs[1].set_ylabel('weight')

# 세번째 subplot
axs[2].scatter(w_val_list, loss_val_list)
axs[2].set_xlabel('weight')
axs[2].set_ylabel('loss')

plt.show()





