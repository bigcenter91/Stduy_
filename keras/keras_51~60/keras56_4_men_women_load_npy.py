import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. 데이터

path = 'd:/study_data/_save/men_women/'

mw_x_train = np.load(path + 'keras56_7_mw_x_train.npy')
mw_x_test = np.load(path + 'keras56_7_mw_x_test.npy')
mw_y_train = np.load(path + 'keras56_7_mw_y_train.npy')
mw_y_test = np.load(path + 'keras56_7_mw_y_test.npy')

print(mw_x_train.shape, mw_y_train.shape) # (716, 100, 100, 3) (716,)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(100, 100, 3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

hist = model.fit(mw_x_train, mw_y_train, epochs=50, # xy_train을 넣으면 x,y 데이터/ 배치 사이즈까지 끝난거다
                    # steps_per_epoch=32,  # 전체 데이터 크기/batch = 160/5 = 32 // 32(계산한만큼) 이상주면 에러난다 // 안써줘도 돌아간다
                    validation_data=(mw_x_test, mw_y_test),
                    # validation_steps=24, # 발리데이터/batch = 120/5 = 24
                    )


# 4. 평가, 예측
loss = model.evaluate(mw_x_test, mw_y_test)
print('loss : ', loss)

y_predict = np.round(model.predict(mw_x_test))
from sklearn.metrics import accuracy_score
acc = accuracy_score(mw_y_test, y_predict)
print('acc : ', acc)