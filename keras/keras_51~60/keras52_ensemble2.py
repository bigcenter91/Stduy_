#1. 데이터

import numpy as np
x1_datasets = np.array([range(100), range(301, 401)]) # 삼성, 아모레
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)])
x3_datasets = np.array([range(201, 301), range(511, 611), range(1300, 1400)])

# 온도, 습도, 강수량

print(x1_datasets.shape)    # (2, 100)
print(x2_datasets.shape)    # (3, 100)

x1 = np.transpose(x1_datasets)
x2 = x2_datasets.T
x3 = x3_datasets.T

print(x1.shape)    # (100, 2)
print(x2.shape)    # (100, 3)
print(x3.shape)

y = np.array([range(2001, 2101)]).T # 환율

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    x1, x2, x3, y, train_size=0.7, random_state=333
)


# y_train, y_test = train_test_split(
#     y, train_size=0.7, random_state=333
# )

print (x1_train.shape, x1_test.shape)
print (x2_train.shape, x2_test.shape)
print (x3_train.shape, x3_test.shape)
print (y_train.shape, y_test.shape)


#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#2. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='stock1')(input1)
# (input_size * dense1_units) + dense1_units = (2 * 10) + 10 = 30
dense2 = Dense(20, activation='relu', name='stock2')(dense1)
dense3 = Dense(30, activation='relu', name='stock3')(dense2)
output1 = Dense(1, activation='relu', name='output1')(dense3)


#2. 모델2
input2 = Input(shape=(3,))
dense11 = Dense(10, name='weather1')(input2)
dense12 = Dense(10, name='weather2')(dense11)
dense13 = Dense(10, name='weather3')(dense12)
dense14 = Dense(10, name='weather4')(dense13)
output2 = Dense(1, name='weather5')(dense14)

from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([output1, output2], name='mg1') # 2개 이상은 뭐? 리스트! 리스트 형태로 받아들일거다
merge2 = Dense(2, activation='relu', name='mg2')(merge1)
merge3 = Dense(3, activation='relu', name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs = [input1, input2], outputs = last_output)

model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', patience=5, mode='min',
                    verbose=1, restore_best_weights=True)

model.fit([x1_train, x2_train, x3_train], y_train, validation_split=0.2, epochs=50, verbose=1, callbacks=[es])


#4. 평가, 예측
from sklearn.metrics import r2_score, mean_squared_error

loss = model.evaluate([x1_test, x2_test, x3_test], y_test)
print("loss : ", loss)

y_pred = model.predict([x1_test, x2_test, x3_test])

r2 = r2_score(y_pred, y_test)
print('r2 스코어 : ', r2)

def RMSE(y_pred, y_test):
    return np.sqrt(mean_squared_error(y_pred, y_test))
rmse = RMSE(y_pred, y_test)
print("RMSE : ", rmse)
