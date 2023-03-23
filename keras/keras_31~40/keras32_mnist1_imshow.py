import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() # train, test 분리해줬다

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,) reshape 해준다 / (60000, 28, 28, 1)


# x_train = x_train.reshape(60000, 28, 28, 1)/255.
# x_test = x_test.reshape(10000, 28, 28, 1)/255.
# 이렇게 할 수도 있다


print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)

print(x_train)
print(y_train) # 5 0 4 ... 5 6 8 / 0번째 숫자는 5
#  [[0 0 0 ... 0 0 0]
#   [0 0 0 ... 0 0 0]
#   [0 0 0 ... 0 0 0]
#   ...
#   [0 0 0 ... 0 0 0]
#   [0 0 0 ... 0 0 0]
#   [0 0 0 ... 0 0 0]]] 배경이니까 0으로 나오겠지?

print(x_train[0])
print(y_train[0])

import matplotlib.pyplot as plt
plt.imshow(x_train[222], 'gray')
plt.show()