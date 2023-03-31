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

y1 = np.array(range(2001, 2101)) # 환율
y2 = np.array(range(1001, 1101)) # 금리

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, \
y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, x2, x3, y1, y2, train_size=0.7, random_state=333
)

print(x1_train.shape, x2_train.shape, x3_train.shape, y1_train.shape, y2_train.shape)


# y_train, y_test = train_test_split(
#     y, train_size=0.7, random_state=333
# )

print (x1_train.shape, x1_test.shape)
print (x2_train.shape, x2_test.shape)
print (x3_train.shape, x3_test.shape)
print (y1_train.shape, y1_test.shape)
print (y2_train.shape, y2_test.shape)


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
output2 = Dense(1, name='output2')(dense14)


#2. 모델3
input3 = Input(shape=(3,))
dense21 = Dense(10, name='weather11')(input3)
dense22 = Dense(10, name='weather12')(dense21)
dense23 = Dense(10, name='weather13')(dense22)
dense24 = Dense(10, name='weather14')(dense23)
output3 = Dense(1, name='output3')(dense24)

from tensorflow.keras.layers import concatenate, Concatenate
merge1 = Concatenate([output1, output2, output3], name='mg1') # 2개 이상은 뭐? 리스트! 리스트 형태로 받아들일거다
merge2 = Dense(2, activation='relu', name='mg2')(merge1)
merge3 = Dense(3, activation='relu', name='mg3')(merge2)
hidden_output = Dense(1, name='last')(merge3)

#2-5. 분기1
bungi1 = Dense(10, activation='selu',name='bg1')(hidden_output)
bungi2 = Dense(10, name='bg2')
last_output1 = Dense(1, name='last1')(bungi2)

#2-5 분기2
bungi11 = Dense(1, activation='linear', name='bg11')(hidden_output)
last_output2 = Dense(1, activation='relu', name='last2')(bungi11)


model = Model(inputs = [input1, input2, input3], outputs = [last_output1, last_output2])

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train],
          validation_split=0.2, epochs=50, verbose=1)


#4. 평가, 예측

from sklearn.metrics import r2_score, mean_squared_error

result = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
print("result : ", result)

y_pred = model.predict([x1_test, x2_test, x3_test])
print(len(y_pred), len(y_pred[0]))
#리스트는 shape로 확인이 안되고 len으로 확인을 해야한다 / list는 파이썬 자료형
#shape의 특성자체가 없다 리스트는 위에 y_pred를 np.y_pred로 해줘야 shape로 가능하다

r2_1 = r2_score(y1_test, y_pred[0])
r2_2 = r2_score(y2_test, y_pred[1])

print('r2 스코어 : ', (r2_1 + r2_2)/2)

def RMSE(y_pred, y_test):
    return np.sqrt(mean_squared_error(y_pred, [y1_test, y2_test]))
rmse = RMSE(y_pred, [y1_test, y2_test])
print("RMSE : ", rmse)

#앙상블은 인풋이 여러개 아웃풋이 여러개일 뿐이다..?
#깃허브에 concatenate / Concatenate 차이점 정리