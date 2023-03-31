import numpy as  np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten


#1. 데이터

path = 'd:/study_data/_save/horse_or_human/'


hh_x_train = np.load(path + 'horse_or_human5_hh_x_train.npy')
hh_x_test = np.load(path + 'horse_or_human5_hh_x_test.npy')
hh_y_train = np.load(path + 'horse_or_human5_hh_y_train.npy')
hh_y_test = np.load(path + 'horse_or_human5_hh_y_test.npy')

print(hh_x_train.shape, hh_x_test.shape) 
# (70, 150, 150, 3) (30, 150, 150, 3)

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

hist = model.fit(hh_x_train, hh_y_train, epochs=30, # xy_train을 넣으면 x,y 데이터/ 배치 사이즈까지 끝난거다
                    steps_per_epoch=32,  # 전체 데이터 크기/batch = 160/5 = 32 // 32(계산한만큼) 이상주면 에러난다 // 안써줘도 돌아간다
                    validation_data=(hh_x_test, hh_y_test),
                    validation_steps=24, # 발리데이터/batch = 120/5 = 24
                    )

# fit_generator 대신에 fit을 쓰면 된다

# numpy로 수치를 저장해놓으면 좋다

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']