import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

@profile
def my_function():
    
#1. 데이터

path = 'd:/study_data/_save/_npy/'
# np.save(path + 'keras55_1_x_train.npy', arr=xy_train[0][0] )
# np.save(path + 'keras55_1_x_test.npy', arr=xy_train[0][0] )
# np.save(path + 'keras55_1_y_train.npy', arr=xy_train[0][1] )
# np.save(path + 'keras55_1_y_test.npy', arr=xy_train[0][1] )



x_train = np.load(path + 'keras55_1_x_train.npy')
x_test = np.load(path + 'keras55_1_x_test.npy')
y_train = np.load(path + 'keras55_1_y_train.npy')
y_test = np.load(path + 'keras55_1_y_test.npy')

# print(x_train)
print(x_train.shape, x_test.shape) # (160, 100, 100, 1)
print(y_train.shape, y_test.shape) # (160, 100, 100, 1)

# 배치사이즈 크기, X,Y

print("================================================")

# print(type(xy_train)) # <class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0])) # <class 'tuple'> 리스트와 튜플의 차이 바꾸지 못한다
# print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
# print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

# 현재 (5, 200, 200, 1) 짜리 데이터가 32덩어리

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(100, 100, 1), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

hist = model.fit(x_train, y_train, epochs=30, # xy_train을 넣으면 x,y 데이터/ 배치 사이즈까지 끝난거다
                    steps_per_epoch=32,  # 전체 데이터 크기/batch = 160/5 = 32 // 32(계산한만큼) 이상주면 에러난다 // 안써줘도 돌아간다
                    validation_data=(x_test, y_test),
                    validation_steps=24, # 발리데이터/batch = 120/5 = 24
                    )

# fit_generator 대신에 fit을 쓰면 된다

# numpy로 수치를 저장해놓으면 좋다

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']


# # print(acc[-1])
# print('loss : ', loss[-1])
# print('val_loss : ', val_loss[-1])
# print('acc : ', acc[-1])
# print('val_acc : ', val_acc[-1])

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.subplot(2,1,1)
# plt.plot(loss)
# plt.plot(val_loss)

# plt.subplot(2,1,2)
# plt.plot(acc)
# plt.plot(val_acc)

# plt.show()




#4. 평가, 예측
