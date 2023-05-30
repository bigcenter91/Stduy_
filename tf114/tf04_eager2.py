###### 현재버전이 1.0 이면 그냥 출력
###### 현재버전이 2.0 이면 즉시실행모드를 끄고 출력
###### if문 써서 1번 소스를 변경!!!

import tensorflow as tf
print(tf.__version__) # 1.14.0

# 즉시실행모드!!! // 2에서는 즉시실행모드로 바뀐다
print(tf.executing_eagerly()) # True

tf.compat.v1.disable_eager_execution() # 즉시실행모드 꺼 / 텐서2.0을 1.0 방식으로
print(tf.executing_eagerly()) # False

tf.compat.v1.enable_eager_execution() # 즉시실행모드 켜 / 텐서1.0을 2.0 방식으로
print(tf.executing_eagerly()) # 


aaa = tf.constant('hello world')

sess = tf.compat.v1.Session()
# print(sess.run(aaa)) # 텐서2에서는 sess.run 안써


# if tf.__version__.startswith('1.'):
#     tf.compat.v1.disable_eager_execution()
#     print(tf.executing_eagerly())  # False
#     sess = tf.compat.v1.Session()
    
# else tf.__version__.startswith('1.') :
#     print(tf.executing_eagerly())  # True
#     aaa = tf.constant('hello world')
#     print(aaa)


v_list = ['v1', 'v2']

for v in v_list:
    if v == 'v1':
        tf.compat.v1.disable_eager_execution()
        print('즉시실행 off 버전2 > 버전1')
        print(tf.executing_eagerly())
    elif v == 'v2':
        tf.compat.v1.enable_eager_execution()
        print('즉시실행 on 버전1 > 버전2')
        print(tf.executing_eagerly())