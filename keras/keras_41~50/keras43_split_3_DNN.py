import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping

dataset = np.array(range(1,101))
timesteps = 5
x_predict = np.array(range(96, 106))

# 96, 97, 98, 99
# 97, 98, 99, 100
# 98, 99, 100, 101
# ...
# 102, 103, 104, 105

def split_x(dataset, timesteps): 
    aaa = []
    for i in range(len(dataset) - timesteps +1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

x_predict = split_x(x_predict, 4)

# x_predict = x_predict[:, : -1]
# print(x_predict)
# print(x_predict.shape)
# 6번을 반복하겠어요
# i는 카운트 하나씩 올라간다
# if문, 반복문(for)
x_predict = x_predict.reshape(7, 4, 1)

bbb = split_x(dataset, timesteps)
print(bbb)
print(bbb.shape) # 96

# x = bbb[:, :4] 둘다 같은 표현
x = bbb[:, : -1]
y = bbb[:,  -1]

print(x)
print(x.shape) #(96, 3)
print(y)
print(y.shape) #(96, )

x = x.reshape(96, 4, 1)


#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_shape=(4, )))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

#. 컴파일, 훈련
model.compile(optimizer='adam', loss='mse')

es = EarlyStopping(monitor='loss', patience=10, verbose=1, 
                   mode='min', restore_best_weights=True)

model.fit(x, y, epochs=1000, callbacks=[es])

print(x_predict)

#4. 평가, 예측
loss = model.evaluate(x,y)
# bb = timesteps-1

result = model.predict(x_predict)
print('loss :', loss)
print('[100 ~ 106]의 결과: ', result)
