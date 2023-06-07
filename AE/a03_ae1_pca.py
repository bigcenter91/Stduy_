import numpy as np
from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()
# _ 하면 땡기지 않을거야라는 뜻

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape) #정규분포방식에 값이 랜덤하게 들어가겠지
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

print(x_train_noised.shape, x_test_noised.shape)
print(np.max(x_train_noised), np.min(x_train_noised))                   # 1.5201940490862174 -0.5946990593420122
print(np.max(x_test_noised), np.min(x_test_noised))                     # 1.4667683674397987 -0.5065540455759264
# 평균 0 표편 0.1

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
print(np.max(x_train_noised), np.min(x_train_noised))                   # 1.0 0.0
print(np.max(x_test_noised), np.min(x_test_noised))                     # 1.0 0.0

#### 아까 만든거 가지고 확인

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,)))
    model.add(Dense(784, activation='sigmoid'))
    return model

# model = autoencoder(hidden_layer_size=154)  #PCA 95프로 성능
# model = autoencoder(hidden_layer_size=331)  #PCA 99프로 성능
# model = autoencoder(hidden_layer_size=486)  #PCA 99.9프로 성능
model = autoencoder(hidden_layer_size=713)  #PCA 100프로 성능


# print(np.argmax(cumsum >= 0.95)+ 1) # 154
# print(np.argmax(cumsum >= 0.99)+ 1) # 331
# print(np.argmax(cumsum >= 0.999)+ 1) # 486
# print(np.argmax(cumsum >= 1.0)+ 1) # 713



#3. 컴파일, 훈련
model.compile(optimizer='adam', loss='mse')

model.fit(x_train_noised, x_train, epochs=30, batch_size=128)

#4. 평가, 예측
decoded_imgs = model.predict(x_test_noised)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test_noised[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()

