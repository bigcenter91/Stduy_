###텐서1 변수 차이점!!###
#변수를 항상 초기화 해줘야한다!!

import tensorflow as tf

sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype=tf.float32)
y = tf.Variable([3], dtype=tf.float32)
# 텐서플로어는 디폴트가 행렬연산이야

init = tf.compat.v1.global_variables_initializer()
sess.run(init)

print(sess.run(x + y))


