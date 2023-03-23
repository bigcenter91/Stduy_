import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping


#1. 데이터
x = np.array([[1,2,3], [2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_predict = np.array([50, 60, 70]) # 80

print (x.shape, y.shape) # (13, 3) (13,)

x = x.reshape(13,3,1)
print (x.shape)

#2. 모델 구성

model = Sequential()

model.add(LSTM(64, input_shape=(3, 1)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

#3, 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', patience=100, mode='min',
                 verbose=1, restore_best_weights=True)

model.fit(x, y, epochs=5000, callbacks=[es])
model.save_weights('./_save/LSTM_1.h5')



#4. 평가, 예측
loss = model.evaluate(x, y)
x_predict = np.array([50, 60, 70]).reshape(1, 3, 1) 

print(x_predict.shape)


result = model.predict(x_predict)
print('loss :', loss)
print('[56, 60, 70]의 결과: ', result)