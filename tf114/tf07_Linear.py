import tensorflow as tf
tf.set_random_seed(337)

#1. 데이터
x = [1, 2, 3]
y = [1, 2, 3]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

#2. 모델 구성

# weight가 앞에 있느냐 뒤에 있느냐 값이 다르다
# y = wx + b
hypothesis = x * w + b
# 식 두개의 값은 완전 다르다 // 앞뒤가 다르다


#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
# learning_rate만큼 줄어든다 epochs가 지날 수록 우리가 원하는 값으로 되겠지?

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss) # 풀어쓴거긴 하지만 어느정도 잡아준거야
# model.compile(loss='mse', optimizer='sgd')

#3-2 . 훈련
with tf.compat.v1.Session() as sess:
# sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 2001
    for step in range(epochs):
        sess.run(train)
        if step %20 == 0: # 20번마다 출력을 할거야
            print(step, sess.run(loss), sess.run(w), sess.run(b)) # verbose가 되겠지?
            # loss는 사실상 중복실행이야 3-1을 한번 더 실행
        
# 그래프 연산방식은 그래프를 만들어 그리고 한방에 툭!
# sess.close()

# y = wx+b를 미분하면 w가 나오지// 경사하강법
# 미분한다는건 그 시점의 기울기를 찾는다는 얘기

# 미분 = 기울기 = 그 지점의 변화량