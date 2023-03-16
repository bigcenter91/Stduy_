from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1.데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets['target']

print(type(x)) #<class 'numpy.ndarray'>
print(x)

print(np.min(x), np.max(x))
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
print(np.min(x), np.max(x))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123,
)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))

#2. 모델
# input1 = Input(shape=(8,))
# dense1 = Dense(20)(input1)
# drop1 = Dropout(0.2)(dense1)
# dense2 = Dense(50)(drop1)
# drop2 = Dropout(0.2)(dense2)
# dense3 = Dense(20)(drop2)
# drop3 = Dropout(0.3)(dense3)
# output1 = Dense(1)(drop3)
# model = Model(inputs=input1, outputs=output1)

input1 = Input(shape=(8,))
dense1 = Dense(10, activation='relu')(input1)
dense2 = Dense(20, activation='relu')(dense1)
dense3 = Dense(20, activation='relu')(dense2)
dense4 = Dense(20, activation='relu')(dense3)
output1 = Dense(1, activation='relu')(dense4)
model = Model(inputs=input1, outputs=output1)




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

import datetime
date = datetime.datetime.now()

date = date.strftime("%m%d_%H%M")

filepath = './_save/MCP/keras28/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='min',
                   verbose=1,
                   )

mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
                      verbose=1,
                      save_best_only=True,
                      filepath="".join([filepath, 'k28_', date, '_', filename]))

model.fit(x_train, y_train, epochs=100,
          callbacks=[es, ],
          validation_split=0.2)


#4. 평가, 예측
from sklearn.metrics import r2_score

print("=================== 1. 기본 출력 ===================")

loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss)
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

