import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping

#1 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
# y 값 ?

x = np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7],[5,6,7,8],[6,7,8,9]]) # 값 출력을 위해 9까지만

y = np.array([5,6,7,8,9,10])

print(x.shape,y.shape) # (6, 4) (6,)
# x의 shape = (행, 열, 몇개씩 훈련하는지!!!)
x = x.reshape(6, 4, 1) # [[[1],[2],[3],[4]],[[2],[3],[4],[5] .......]]
print(x.shape) # (6, 4, 1)

#2. 모델 구성
model = Sequential()
model.add(SimpleRNN(32, input_shape=(4, 1))) # 행빼고 나머지를 입력하죠? 32 output 노드의 갯수겠지
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

#3, 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='loss', patience=200, verbose=1, 
                   mode='min', restore_best_weights=True)

model.fit(x, y, epochs=1000)


#4. 평가, 예측
loss = model.evaluate(x, y)
x_predict = np.array([7, 8, 9, 10]).reshape(1, 4, 1) #8, 9, 10이 1차원이니까 reshape 해준다 // #[[[8], [9], [10]]]

print(x_predict.shape)



result = model.predict(x_predict)
print('loss :', loss)
print('[7, 8, 9, 10]의 결과: ', result)