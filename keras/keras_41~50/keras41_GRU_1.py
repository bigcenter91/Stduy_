import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

# y=?

x=np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7],
            [5,6,7,8],[6,7,8,9]]) # 8, 9, 10 y가 있기 때문에

y=np.array([5, 6, 7 ,8 ,9 ,10])

print(x.shape, y.shape) # (6, 4) (6,) 
# rnn의 구조는 3차원이 되야한다
# x의 shpae = (행, 열, 몇개씩 훈련하는지!!!)
x = x.reshape(6, 4, 1) # [[1], [2], [3]], [[2], [3], [4],....]]
print(x.shape) # (6, 4, 1) = RNN의 SHAPE를 맞춰줄려고 변경

#2. 모델 구성
model = Sequential()
model.add(GRU(32, input_shape=(4, 1))) # 행빼고 나머지를 입력하죠? 32 output 노드의 갯수겠지
model.add(Dense(16))
model.add(Dense(1))
# 사용법은 똑같다 로직만 다를 뿐


#3, 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='loss', patience=200, verbose=1, 
                   mode='min', restore_best_weights=True)

model.fit(x, y, epochs=100)


#4. 평가, 예측
loss = model.evaluate(x, y)
x_predict = np.array([7, 8, 9, 10]).reshape(1, 4, 1) #7, 8, 9, 10이 1차원이니까 reshape 해준다 // #[[[7], [8], [9], [10]]]

print(x_predict.shape)



result = model.predict(x_predict)
print('loss :', loss)
print('[7, 8, 9, 10]의 결과: ', result)
