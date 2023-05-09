import tensorflow as tf
tf.set_random_seed(337)

#1. 데이터
x = [1,2,3,4,5]
y = [2,4,6,8,10]

# w = tf.Variable(111, dtype=tf.float32)
# b = tf.Variable(100, dtype=tf.float32)

# w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
# b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
# uniform 균등분포

#w는 통상 랜덤값/ b는 통상0을 줌 
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)   #tf.random_normal :정규분포(중심값 근처) /  tf.random_uniform :균등분포(n, 동일한 확률)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)   #[1] : 단 하나의 요소만을 가지는 텐서/ 즉, 스칼라 값과 같은 개념
# w = tf.random_normal([1])
# b = tf.random_normal([1])

# sess = tf.compat.v1.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(w)) #[-0.4121612]


# with tf.compat.v1.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(w))  #[-0.4121612]


#2. 모델 구성
# y = wx + b
# weight가 앞에 있느냐 뒤에 있느냐 값이 다르다

hypothesis = x * w + b


#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
# learning_rate만큼 줄어든다 epochs가 지날 수록 우리가 원하는 값으로 되겠지?

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)
# model.compile(loss='mse', optimizer='sgd')

#3-2 . 훈련
with tf.compat.v1.Session() as sess:
# sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 2001
    for step in range(epochs):
        sess.run(train)
        if step %20 == 0:
            print(step, sess.run(loss), sess.run(w), sess.run(b)) # verbose가 되겠지?
        
    # 그래프 연산방식은 그래프를 만들어 그리고 한방에 툭!
    # sess.close()
    #with문은 자동으로 close됨
