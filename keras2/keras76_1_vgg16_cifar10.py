# 맹그러!!
# 가중치 동결과 동결하지 않았을 때 성능비교

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import GlobalAveragePooling2D

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


print(x_train.shape)

x_train = x_train.reshape(50000, 32, 32, 3) 
x_test = x_test.reshape(10000, 32, 32, 3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape)


#2. 모델 구성

vgg16 = VGG16(weights='imagenet', include_top=False,    # True로 하면 에러
              input_shape=(32, 32, 3))

# vgg16.trainable = True # 디폴트 트루 // 가중치 동결

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

vgg16.trainable = False

model.summary()

print(len(model.weights))
print(len(model.trainable_weights))


import time
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5)

start = time.time()
model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[es, reduce_lr], batch_size=128)
end = time.time()

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss :', result[0])
print('acc :', result[1])
print('걸린시간 : ', end-start)


# vgg16.trainable = False
# loss : 1.159711241722107
# acc : 0.6086000204086304
# 걸린시간 :  186.13470482826233

# vgg16.trainable = True
# loss : 2.3026065826416016
# acc : 0.10000000149011612
# 걸린시간 :  408.11228609085083


# vgg16.trainable = True // summary 위에
# loss : 1.2057491540908813
# acc : 0.8015000224113464
# 걸린시간 :  410.02353954315186