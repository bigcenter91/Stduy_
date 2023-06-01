# a3_ae2 카피해서 모델을 직접구성

# 인코더 
# conv
# maxpool
# conv
# maxpool

# # 디코더
# conv
# Upsampling(2, 2)
# conv
# Upsampling(2, 2)


import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, UpSampling2D, MaxPooling2D, Conv2D

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

def autoencoder(hidden_layer_size):
    model = Sequential()
# 인코더
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D()) # 그대로 내려오고 맥스풀 때려주니까 n, 14, 14, 1 // 실질적으로 2, 2 생략 되있는거야
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D()) # n, 7, 7, 1
        
# 디코더
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same')) # 14, 14, 16
    model.add(UpSampling2D()) 
    model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same')) # 14, 14, 16

    model.summary()
    return model
# 맥스풀이 나눴을 떄 제일 큰놈이잖아

#  conv2d (Conv2D)             (None, 28, 28, 16)        160

#  max_pooling2d (MaxPooling2D  (None, 14, 14, 16)       0
#  )

#  conv2d_1 (Conv2D)           (None, 14, 14, 8)         1160

#  max_pooling2d_1 (MaxPooling  (None, 7, 7, 8)          0
#  2D)

#  conv2d_2 (Conv2D)           (None, 7, 7, 8)           584

#  up_sampling2d (UpSampling2D  (None, 14, 14, 8)        0
#  )

#  conv2d_3 (Conv2D)           (None, 14, 14, 16)        1168

#  up_sampling2d_1 (UpSampling  (None, 28, 28, 16)       0
#  2D)

#  conv2d_4 (Conv2D)           (None, 28, 28, 1)         145

# =================================================================
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

##############################################################

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
      (ax11, ax12, ax13, ax14, ax15)) = \
        plt.subplots(3, 5, figsize=(20, 7))
        
# 이미지 다섯 개를 무작위로 고른다
random_images = random.sample(range(decoded_imgs.shape[0]), 5)


# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈가 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(decoded_imgs[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


plt.tight_layout()
plt.show()
