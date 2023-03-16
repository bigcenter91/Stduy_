from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np


#1. 데이터
dataset = load_diabetes()

x = dataset.data
y = dataset.target
print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, test_size=0.2
)

#2. 모델구성
# model = Sequential()
# model.add(Dense(15, activation='relu', input_dim=10))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(5, activation='relu'))
# model.add(Dense(1))

input1= Input(shape=(10, ))
dense1 = Dense(15, activation='relu')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(20, activation='relu')(drop1)
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(10, activation='relu')(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(5, activation='relu')(drop3)
drop4 = Dropout(0.2)(dense4)
output1 = Dense(1)(drop4)
model = Model(inputs=input1, outputs=output1)




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

import datetime
date = datetime.datetime.now()

date = date.strftime("%m%d_%H%M")

filepath = './_save/MCP/keras30/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5, mode='min',
                   verbose=1,
                   )

mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
                      verbose=1,
                      save_best_only=True,
                      filepath="".join([filepath, 'k30_', date, '_', filename]))

model.fit(x_train, y_train, epochs=100,
          callbacks=[es, ],
          validation_split=0.2)

#4. 평가, 예측
from sklearn.metrics import r2_score

print("=================== 1. 기본 출력 ===================")

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 스코어 :', r2)


# loss :  2973.235595703125
# r2 스코어 : 0.5280703528986179

"""
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.plot(hist.history['loss'], marker='.', c='red', label='로스')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='발로스')
plt.title('당뇨병')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend()
plt.grid()
plt.show()
"""