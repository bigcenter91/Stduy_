from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.utils import to_categorical

###################### 실습 #########################
# 모델 만들어서 비교

# acc
#1. 나의 최고의 CNN
#2. 나의 최고의 DNN
#3. PCA 0.95 :
#3. PCA 0.99 :
#3. PCA 0.9999 :
#3. PCA 1.0 :

(x_train, y_train), (x_test, y_test) = mnist.load_data() # 데이터를 할당을 안함. _

x_train = x_train.reshape(60000, 784)/255.
x_test = x_test.reshape(10000, 784)/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

pca_ratios = [0.95, 0.99, 0.9999, 784]

pca = PCA(n_components = 784) # [154, 331, 486, 713]

for ratio in pca_ratios:
    pca = PCA(n_components=ratio)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    x_train_pca = x_train_pca.reshape(-1, x_train_pca.shape[1], 1, 1)
    x_test_pca = x_test_pca.reshape(-1, x_test_pca.shape[1], 1, 1)

    model = Sequential()
    model.add(Conv2D(32, (1, 1), activation='relu', input_shape=(x_train_pca.shape[1], 1, 1)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train_pca, y_train, epochs=2, batch_size=128, verbose=1)

    loss, acc = model.evaluate(x_test_pca, y_test, verbose=0)
    print(f"PCA Ratio: {ratio}, Test Accuracy: {acc}")