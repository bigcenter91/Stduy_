import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. 데이터

path = 'd:/study_data/_save/rps/'

rps_x_train = np.load(path + 'keras56_7_rps_x_train.npy')
rps_x_test = np.load(path + 'keras56_7_rps_x_test.npy')
rps_y_train = np.load(path + 'keras56_7_rps_y_train.npy')
rps_y_test = np.load(path + 'keras56_7_rps_y_test.npy')

print(rps_x_train.shape, rps_y_train.shape) # (24998, 100, 100, 3) (24998,)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(100, 100, 3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

hist = model.fit(rps_x_train, rps_y_train, epochs=30, # xy_train을 넣으면 x,y 데이터/ 배치 사이즈까지 끝난거다
                    steps_per_epoch=32,  # 전체 데이터 크기/batch = 160/5 = 32 // 32(계산한만큼) 이상주면 에러난다 // 안써줘도 돌아간다
                    validation_data=(rps_x_test, rps_y_test),
                    validation_steps=24, # 발리데이터/batch = 120/5 = 24
                    )

# fit_generator 대신에 fit을 쓰면 된다

# numpy로 수치를 저장해놓으면 좋다

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']