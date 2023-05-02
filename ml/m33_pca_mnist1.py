from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.utils import to_categorical
(x_train, y_train), ( x_test, y_test) = mnist.load_data()
# 60000, 28, 28


x = np.concatenate((x_train, x_test), axis=0)
x = np.append(x_train, x_test, axis=0) # 생각 나는거 쓰면 되는거야
print(x.shape) # (70000, 28, 28)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
print(x.shape) # (70000, 784)

################실습
# pca를 통해 0.95 이상인 n_components 몇개
# 0.95 몇개?
# 0.99 몇개?
# 0.999 몇개?
# 1.0 몇개??

pca = PCA(n_components=784)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_
cumsum = np.cumsum(pca_EVR)
print(cumsum)
print(np.argmax(cumsum >= 0.95)+ 1) # 154
print(np.argmax(cumsum >= 0.99)+ 1) # 331
print(np.argmax(cumsum >= 0.999)+ 1) # 486
print(np.argmax(cumsum >= 1.0)+ 1) # 713
