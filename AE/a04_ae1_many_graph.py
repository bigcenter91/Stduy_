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
# model_01 = autoencoder(hidden_layer_size=1)  #PCA 100프로 성능
# model_08 = autoencoder(hidden_layer_size=8)  #PCA 100프로 성능
# model_32 = autoencoder(hidden_layer_size=32)  #PCA 100프로 성능
# model_64 = autoencoder(hidden_layer_size=64)  #PCA 100프로 성능
# model_154 = autoencoder(hidden_layer_size=154)  #PCA 100프로 성능
# model_331 = autoencoder(hidden_layer_size=331)  #PCA 100프로 성능
# model_486= autoencoder(hidden_layer_size=486)  #PCA 100프로 성능
# model_713= autoencoder(hidden_layer_size=713)  #PCA 99.9프로 성능

hidden_layer_sizes = [1, 8, 32, 64, 154, 331, 486, 713]

models = []

# print(np.argmax(cumsum >= 0.95)+ 1) # 154
# print(np.argmax(cumsum >= 0.99)+ 1) # 331
# print(np.argmax(cumsum >= 0.999)+ 1) # 486
# print(np.argmax(cumsum >= 1.0)+ 1) # 713



#3. 컴파일, 훈련
# print("================ node 1개의 시작 ================")
# model_01.compile(optimizer='adam', loss='mse')
# model_01.fit(x_train_noised, x_train, epochs=10, batch_size=32)

# print("================ node 8개의 시작 ================")
# model_08.compile(optimizer='adam', loss='mse')
# model_08.fit(x_train_noised, x_train, epochs=10, batch_size=32)

# print("================ node 32개의 시작 ================")
# model_32.compile(optimizer='adam', loss='mse')
# model_32.fit(x_train_noised, x_train, epochs=10, batch_size=32)

# print("================ node 64개의 시작 ================")
# model_64.compile(optimizer='adam', loss='mse')
# model_64.fit(x_train_noised, x_train, epochs=10, batch_size=32)

# print("================ node 154개의 시작 ================")
# model_154.compile(optimizer='adam', loss='mse')
# model_154.fit(x_train_noised, x_train, epochs=10, batch_size=32)

# print("================ node 331개의 시작 ================")
# model_331.compile(optimizer='adam', loss='mse')
# model_331.fit(x_train_noised, x_train, epochs=10, batch_size=32)

# print("================ node 486개의 시작 ================")
# model_486.compile(optimizer='adam', loss='mse')
# model_486.fit(x_train_noised, x_train, epochs=10, batch_size=32)

# print("================ node 713개의 시작 ================")
# model_713.compile(optimizer='adam', loss='mse')
# model_713.fit(x_train_noised, x_train, epochs=10, batch_size=32)


for hidden_size in hidden_layer_sizes:
    model = autoencoder(hidden_layer_size=hidden_size)
    print("================ node {}개의 시작 ================".format(hidden_size))
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train_noised, x_train, epochs=5, batch_size=32)
    models.append(model)


#4. 평가, 예측
# decoded_imgs_01 = model_01.predict(x_test_noised)
# decoded_imgs_08 = model_08.predict(x_test_noised)
# decoded_imgs_32 = model_32.predict(x_test_noised)
# decoded_imgs_64 = model_64.predict(x_test_noised)
# decoded_imgs_154 = model_154.predict(x_test_noised)
# decoded_imgs_331 = model_331.predict(x_test_noised)
# decoded_imgs_486 = model_486.predict(x_test_noised)
# decoded_imgs_713 = model_713.predict(x_test_noised)

# Generate reconstructed images for each model
decoded_imgs = []
for model in models:
    decoded_imgs.append(model.predict(x_test_noised))


##############################################################

from matplotlib import pyplot as plt
import random

# fig, axes = plt.subplots(9 , 5, figsize=(15, 15))

# random_imgs = random.sample(range(decoded_imgs_01.shape[0]), 5)
# outputs = [x_test, decoded_imgs_01, decoded_imgs_08, decoded_imgs_32, decoded_imgs_64,
#            decoded_imgs_154, decoded_imgs_331, decoded_imgs_486, decoded_imgs_713]

# for row_num, row in enumerate(axes):
#     for col_num, ax in enumerate(row):
#         ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28, 28),
#                   cmap='gray')
#         ax.grid(False)
#         ax.set_yticks([])
#         ax.set_xticks([])
        
# plt.show()

fig, axes = plt.subplots(len(hidden_layer_sizes), 5, figsize=(15, 15))

random_imgs = random.sample(range(decoded_imgs[0].shape[0]), 5)
outputs = [x_test] + decoded_imgs

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28, 28), cmap='gray')
        # ax.set_ylabel(outputs[row_num])
        ax.grid(False)
        ax.set_yticks([])
        ax.set_xticks([])
        
plt.show()