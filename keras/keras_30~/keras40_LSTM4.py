import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping


#2. 모델 구성
model = Sequential()                         # batch, timesteps:열 / 여기서는 5로 짤랐지, feature
model.add(LSTM(10, input_shape=(5, 1))) # 10*(10+1+1) = 120 / RNN은 출력 3차원 받아서 2차원으로
# units * (feature + bias + units) = 120
model.add(Dense(7)) # (10 + 1) x 7 = 77
model.add(Dense(1)) # (7 + 1) x 1 = 8

model.summary()


#3, 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='loss', patience=100, verbose=1, 
                   mode='min', restore_best_weights=True)

hist = model.fit(x, y, epochs=1000, callbacks=[es])

# es = EarlyStopping(monitor='val_loss', patience=5, mode='min',
#                    verbose=1, restore_best_weights=True)



#4. 평가, 예측
loss = model.evaluate(x, y)
x_predict = np.array([6, 7, 8, 9, 10]).reshape(1, 5, 1) 
#6, 7, 8, 9, 10이 1차원이니까 reshape 해준다 // #[[[7], [8], [9], [10]]]

print(x_predict.shape)



result = model.predict(x_predict)
print('loss :', loss)
print('[6, 7, 8, 9, 10]의 결과: ', result)
