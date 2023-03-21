import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

# y=?

x=np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],
            [5,6,7,8,9]]) # 8, 9, 10 y가 있기 때문에

y=np.array([6, 7 ,8 ,9 ,10])

print(x.shape, y.shape) # (5, 5) (5,) 
# rnn의 구조는 3차원이 되야한다
# x의 shpae = (행, 열, 몇개씩 훈련하는지!!!)
x = x.reshape(5, 5, 1) # [[1], [2], [3]], [[2], [3], [4],....]]
print(x.shape) # (5, 5, 1) = RNN의 SHAPE를 맞춰줄려고 변경

#2. 모델 구성
model = Sequential()
model.add(SimpleRNN(64, input_shape=(5, 1))) # 행빼고 나머지를 입력하죠? 32 output 노드의 갯수겠지
# model.add(Dense(32, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(100, activation='relu'))
model.add(Dense(1))

#3, 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

# es = EarlyStopping(monitor='loss', patience=100, verbose=1, 
#                    mode='min', restore_best_weights=True)
import time
start_time = time.time()
model.fit(x, y, epochs=1000)

end_time = time.time()
print('training time : ', round(end_time-start_time, 2))
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
print("걸린 시간 : ", round(end_time - start_time, 2))

# loss : 1.7280808606301434e-05 [8, 9, 10]의 결과:  [[10.811575]] // epochs 1000
# loss : 2.1646816094289534e-05 [8, 9, 10]의 결과:  [[10.838179]] 
# loss : 5.456968427478004e-13  [6, 7, 8, 9, 10]의 결과:  [[10.634054]] training time :  14.43 gpu
# loss : 2.728484213739002e-13 [6, 7, 8, 9, 10]의 결과:  [[10.684405]] training time :  2.94 cpu


# loss : 0.00010419584577903152
# [6, 7, 8, 9, 10]의 결과:  [[10.539905]]
# 걸린 시간 :  18.27 // GPU
# loss : 2.7034907361667138e-06
# [6, 7, 8, 9, 10]의 결과:  [[10.401791]]
# 걸린 시간 :  11.4 // CPU