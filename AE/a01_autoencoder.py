# 가장 큰 장점은 잡음제거
# 평균 사진외에 다른 사진의 특징들? 필요없는 부분
# x를 x로 훈련시킨다 = y값이 필요가 없다

# 비지도: y가 없다
# 준지도: y 대신 x를 쓴다

import numpy as np
from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()
# _ 하면 땡기지 않을거야라는 뜻

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#2. 모델
input_img = Input(shape=(784,))
encoded = Dense(16, activation='tanh')(input_img)
# encoded = Dense(1, activation='relu')(input_img)
# encoded = Dense(32, activation='relu')(input_img)
# encoded = Dense(1024, activation='relu')(input_img)

decoded = Dense(784, activation='sigmoid')(encoded)
# decoded = Dense(784, activation='tanh')(encoded)
# decoded = Dense(784, activation='relu')(encoded) # 어차피 양수니까 써도된다 relu

autoencoder = Model(input_img, decoded)
autoencoder.summary()

# 노드가 5 > 3 > 5
# 큰놈은 남아있을거고 작은 놈은 사라진다
# 기미, 주근깨는 사라진다
# 특성이 낮은 사진은 뿌옇게 된다
# 노이즈가 있는 사진을 프레딕트 하면 없는 사진으로 프레딕트 한다
# 오토인코더 기본개념은 동일 사진으로 훈련시킨다
# 결국은 784 > 64 > 784로 원복

#3. 컴파일, 훈련
# autoencoder.compile(optimizer='adam', loss='mse', metrics=['acc'])
autoencoder.compile(optimizer='adam', loss='mse',) # acc를 빼버릴 수도 있겠지? // 중요한건 loss가 감소하는걸 봐야되는거야
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

autoencoder.fit(x_train, x_train, epochs=30, batch_size=128,
                validation_split=0.2)

# 라벨 값을 찾는게 아니라 784개의 특성값을 찾는거라서 acc 의미가 없다 
# mnist니까 acc 넣어야지하면 잘못된 생각이야 실질적으로 loss가 중요하다
# sigmoid에서도 mse 써도 된다
# autoencoder나 gan쪽 너무 loss에 의존하게 되면 안된다 물론 좋아야겠지만
# 소비자에 판매할 떄는 눈으로 한번 봐야한다 // 초창기 GAN은 LOSS를 믿지 말라고했어

#4. 평가, 예측
decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()

# 인코더 4개의 히든과
# 디코드 4개의 엑티베이션
# 그리고 2개의 컴파일-로스 부분의 경우의 수
# 총 32가지의 로스를 정리
# 눈으로 비교

# 통상적으로 오토인코더에서 sigmoid와 mse 많이 써

# 375/375 [==============================] - 1s 3ms/step - loss: 0.0094 - val_loss: 0.0094 // linear_linear // 784, 64, 784
# 375/375 [==============================] - 1s 3ms/step - loss: 0.0126 - val_loss: 0.0126 // sigmoid_relu // 784, 64, 784
# 375/375 [==============================] - 1s 3ms/step - loss: 0.0865 - val_loss: 0.0858 // relu_relu // 784, 1, 784 : 밑에 글자 거의 안보임
# 375/375 [==============================] - 1s 3ms/step - loss: 0.0100 - val_loss: 0.0102 // relu_sigmoid
# 375/375 [==============================] - 1s 3ms/step - loss: 0.0100 - val_loss: 0.0102 // relu_sigmoid // 784, 1024, 784
