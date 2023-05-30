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

#2. 모델
input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img)
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
decoded_imgs = autoencoder.predict(x_test_noised)

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

# 특성이 있는 가중치를 빼내는거야
# 노이즈, 노이즈// un노이즈, un노이즈 해도 상관없다 