import tensorflow as tf
print(tf.__version__)

print("hello world")

aaa = tf.constant('hello world')
print(aaa) # Tensor("Const:0", )

# 그래프 연산방식?
# sess.run을 더 거친다 = sess.run을 넣어줘야 프린트가 된다

# sess = tf.Session() 버전 차이다
sess = tf.compat.v1.Session()
print(sess.run(aaa))

# 텐서1은 한 단계 더 거친다
# 텐서1은 그래프 방식
# 텐서2는 즉시 실행