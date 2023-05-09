import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # 텐서플로우에 상수형

node3 = node1 + node2
# node3 = tf.add(node1, node2) 둘다 가능

print(node1)
print(node2)
print(node3)

# 7을 원하는거잖아

sess = tf.compat.v1.Session()
print(sess.run(node3))      # 7.0
print(sess.run(node1))      # 3.0


