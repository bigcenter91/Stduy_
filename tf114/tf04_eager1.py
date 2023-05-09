import tensorflow as tf
print(tf.__version__) # 1.14.0

# 즉시실행모드!!! // 2에서는 즉시실행모드로 바뀐다
print(tf.executing_eagerly()) # False > True // #tf114cpu(그래프 연산방식) : False //=>>  ##tf273cpu(즉시 실행방식) : True

tf.compat.v1.disable_eager_execution() # 즉시실행모드 꺼 / 텐서2.0을 1.0 방식으로

print(tf.executing_eagerly()) # False > True

tf.compat.v1.enable_eager_execution() # 즉시실행모드 꺼 / 텐서2.0을 1.0 방식으로

print(tf.executing_eagerly()) # 


aaa = tf.constant('hello world')

sess = tf.compat.v1.Session() 
print(sess.run(aaa))           #tf2에서는 session없어짐, 그냥 print(aaa)출력 : 즉시실행모드 = 텐서플로2

print(aaa)                     #텐서플로2 즉시실행 

