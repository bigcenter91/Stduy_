import tensorflow as tf
tf.set_random_seed(337)

#1. 데이터
x = [1, 2, 3]
y = [1, 2, 3]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

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
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    sess.run(train)
    if step %20 == 0:
        print(step, sess.run(loss), sess.run(w), sess.run(b)) # verbose가 되겠지?
        
# 그래프 연산방식은 그래프를 만들어 그리고 한방에 툭!
sess.close()