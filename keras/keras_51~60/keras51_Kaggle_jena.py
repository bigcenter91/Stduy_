import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error

#조건문과 반복문만 할줄 알면 된다
#시대의 흐름에 잘맞춰야한다 

#1. 데이터
path = './_data/kaggle_jena/'

datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
# print (datasets) # [420551 rows x 15 columns]

# print (datasets.columns)
# print (datasets.info()) # 결측이 하나도 없다는걸 알 수 있죠?
# print (datasets.describe())
# #df.head() 찍으면 위에 다섯개만 보인다

# print(datasets['T (degC)']) #index까지 같이 붙는다
# print(datasets['T (degC)'].values) 
# print(datasets['T (degC)'].to_numpy()) #pandas를 numpy로 바꿔줘야한다? // values, to_numpy 똑같다

x = datasets.drop(['T (degC)'], axis=1)
y = datasets['T (degC)']

def split_x(datasets, timestep):
    gen= (datasets[i : i + timestep]for i in range(len(datasets)-timestep))
    return np.array(list(gen))

timestep = 5

x = split_x(x, timestep)
y = y[timestep:]

# print (x)
# print (y)


print (x.shape, y.shape) # (420551, 13) (420551,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.7, shuffle=False, random_state=123,
)

print(x_test.shape, y_test.shape) # (126164, 5, 13) (126164,)
print (x_train.shape, y_train.shape) # (294382, 5, 13) (294382,)

x_test, x_predict, y_test, y_predict = train_test_split(
    x_test, y_test, test_size=2/3, shuffle=False, random_state=123,
)




#2. 모델 구성
model = Sequential()
model.add(Conv1D(16, kernel_size=2, input_shape=(5, 13)))
model.add(Conv1D(16, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='loss', patience=5, mode='min',
                    verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=20, batch_size=128,
                 validation_split=0.2, verbose=1, callbacks=[es])


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)
# print(x_predict)


import matplotlib.pyplot as plt
plt.plot(datasets['T (degC)'].values)
plt.show()
#규칙성 있는 시계열이다