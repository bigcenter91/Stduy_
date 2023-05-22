# cifar100으로 바꿔서 GAP이 이기면 끗

from tensorflow.keras.datasets import cifar100
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import time
import tensorflow as tf
tf.random.set_seed(377)


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()


####### 실습 ####### 맹그러봐

print(x_train.shape)

x_train = x_train.reshape(50000, 32, 32, 3) # (60000, 28, 14, 2)도 가능하다 곱하거나 나눠야한다 그렇게해서 원값이 나와야한다
                                            # shape와 크기는 같아야한다 
                                            # 행은 건들면 안된다 / 4차원 데이터
x_test = x_test.reshape(10000, 32, 32, 3)
#reshape 할 때 데이터 구조만 바뀌지 순서와 내용은 바뀌지 않는다 / transepose는 열과 행을 바뀐다 반대로 구조가 바뀌는거다


# print(x_test.shape)
# print(y_test.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape) # (60000, 10)
print(y_test.shape) # (10000, 10)
# y가 N, 10이 되야겠지 그럼 y 원핫해야겠지


#2. 모델 구성
model = Sequential()

model.add(Conv2D(64, (2,2), padding='same', input_shape=(32,32,3)))
model.add(MaxPooling2D()) 
# 반이 된다, 중첩되지 않아 있는거에서 제일 큰거 뽑아내 2,2로 준다
# Maxpooling 안에 커널 사이즈 있겠지 그 디폴트가 2,2이고 중첩되지 않는다

model.add(Conv2D(filters=64, kernel_size=(2,2), padding='valid', activation='relu'))
model.add(Conv2D(32, 2)) # 2는 커널 사이즈 2,2 라는 표현
# model.add(Flatten()) # (None, 4608) > (None, 12, 12, 32) 곱한거
model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(100, activation='softmax'))
model.summary()

####################### Flatten 연산량 #######################
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 32, 32, 64)        832
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 16, 16, 64)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 14, 14, 64)        36928
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 12, 12, 32)        18464
# _________________________________________________________________
# flatten (Flatten)            (None, 4608)              0
# _________________________________________________________________
# dense (Dense)                (None, 100)               460900
# =================================================================
# Total params: 517,124
# Trainable params: 517,124
# Non-trainable params: 0
# _________________________________________________________________


####################### GlovaAveragePooling 연산량 #######################
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 32, 32, 64)        832
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 16, 16, 64)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 14, 14, 64)        36928
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 12, 12, 32)        18464
# _________________________________________________________________
# module_wrapper (ModuleWrappe (None, 32)                0
# _________________________________________________________________
# dense (Dense)                (None, 100)               3300
# =================================================================
# Total params: 59,524
# Trainable params: 59,524
# Non-trainable params: 0
# _________________________________________________________________
print(np.unique(y_train, return_counts=True))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
# es = EarlyStopping(monitor='val_loss', patience=50, mode='min',
#                     verbose=1, restore_best_weights=True)

start = time.time()
model.fit(x_train, y_train, epochs=50, batch_size=128,)
end = time.time()

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss :', result[0])
print('acc :', result[1])
print('걸린시간 : ', end-start)


# #4. 평가, 예측
# result = model.evaluate(x_test, y_test)
# print('result : ', result)

# y_predict = model.predict(x_test)

# acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1))
# print('acc : ', acc)

# import matplotlib.pyplot as plt
# plt.plot(hist.history['val_loss'], label='val_acc')
# plt.show()


# Flatten_30
# loss : 11.44886589050293
# acc : 0.15690000355243683
# 걸린시간 :  64.62471604347229

# GlobalAver_30
# loss : 2.4710371494293213
# acc : 0.37119999527931213
# 걸린시간 :  64.8893141746521

# Flatten
# loss : 8.223716735839844
# acc : 0.21739999949932098
# 걸린시간 :  109.18703150749207

# GlobalAver_60_dense 32 추가
# loss : 2.3885250091552734
# acc : 0.399399995803833
# 걸린시간 :  110.88184523582458


# Flatten_dense 100
# loss : 21.501333236694336
# acc : 0.181099995970726
# 걸린시간 :  111.28081583976746

# GlobalAverage_dense 100 
# loss : 2.509286403656006
# acc : 0.3691999912261963
# 걸린시간 :  121.78449368476868