import tensorflow as tf
print(tf.__version__)

# 현재버전이 1.0이면 그냥 출력
# 현재버전이 2.0이면 즉시 실행모드를 끄고 출력
print(tf.__version__[0])


if int(tf.__version__[0])>1:
    tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())       # False 

aaa = tf.constant('hello world')

sess = tf.compat.v1.Session()
print(sess.run(aaa))
print(aaa)