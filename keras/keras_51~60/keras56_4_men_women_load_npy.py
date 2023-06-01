import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator # 전처리
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1 데이터
path = 'c:/study_data/_data/men_women/'
save_path = 'c:/study_data/_save/men_women/'

x = np.load(save_path + 'keras56_men_women_x_train_x.npy')
y = np.load(save_path + 'keras56_men_women_y_train_y.npy')
# print(np.unique(y),return count)
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size = 0.7,
                                                 shuffle = True,
                                                 random_state = 123)


# print(x_train)
# print(x_train.shape, x_test.shape) # (160, 100, 100, 1) (160, 100, 100, 1)
# print(y_train.shape, y_test.shape) # (160,) (160,)



# print("################################################")
# print(type(x_train, y_train)) # <class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(x_train, y_train[0])) # <class 'tuple'> 튜플 값 못바꿈
# print(type(x_train, y_train[0][0])) # <class 'numpy.ndarray'>
# print(type(x_train, y_train[0][1])) # <class 'numpy.ndarray'>

# 현재 x는 (5, 200, 200, 1) 짜리 데이터가 32덩어리

#2 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(40, (8, 8), input_shape = (250, 250, 3), activation = 'relu'))
model.add(Conv2D(70, (8, 8), activation = 'relu'))
model.add(Flatten())
model.add(Dense(30, activation = 'relu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#3 컴파일, 훈련
s_time = time.time()
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['acc'])
# model.fit(xy_train[:][0], xy_train[:][1], epochs = 10) # 에러
# hist = model.fit(xy_train[0][0], xy_train[0][1],
#           epochs = 10,
#           batch_size = 16,
#           validation_data = (xy_test[0][0], xy_test[0][1])) # 통배치 넣으면 이것도 가능
# hist = model.fit_generator(xy_train, epochs = 1,
#                     steps_per_epoch = 32, # 전체데이터크기/batch = 160/5 = 32 이렇게 해주는게 적당하다. 결과값 이상으로 주면 안된다.
#                     validation_data = xy_test,
#                     validation_steps = 24 # 발리데이터/batch = 120/5 = 24
#                     )
hist = model.fit(x_train, y_train,
                 epochs = 50,
                 steps_per_epoch = 30, # 전체데이터크기/batch = 160/5 = 32 이렇게 해주는게 적당하다. 결과값 이상으로 주면 안된다.
                 validation_data = [x_test ,y_test], # 모델의 성능을 평가할 때 사용할 검증 데이터
                 validation_steps = 30 # 발리데이터/batch = 120/5 = 24
                )
e_time = time.time()
print('걸리는 시간 : ', e_time - s_time, 2)
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

# print(acc)
print('loss : ', loss[-1]) # 마지막 값만 출력
print('val_loss : ', val_loss[-1])
print('acc : ', acc[-1]) # 마지막 값만 출력
print('val_acc : ', val_acc[-1])

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = np.round(model.predict(x_test))

acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)