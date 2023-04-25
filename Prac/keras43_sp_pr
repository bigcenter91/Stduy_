import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping

dataset = np.array(range(1, 101))
timesteps = 4
x_predict = np.array(range(96, 106)) # 100 ~ 106 예상 값

def split_x(dataset, timesteps): 
    aaa = []
    for i in range(len(dataset) - timesteps):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(dataset, timesteps)

# 데이터셋 분할
x_train = bbb[:, : -1]
y_train = bbb[:, -1]

# 모델 구성
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(timesteps-1,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# 모델 컴파일
model.compile(optimizer='adam', loss='mse')

# 학습
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1, callbacks=[early_stopping])

# 예측
x_predict = x_predict.reshape(1, -1)
x_predict = split_x(x_predict[0], timesteps)
y_predict = model.predict(x_predict)

print("predicted value: ", y_predict)