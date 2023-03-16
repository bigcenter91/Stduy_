import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


#1. 데이터
path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0)

print(train_csv) 
print(train_csv.shape) # 10886, 11

test_csv = pd.read_csv(path + 'test.csv',
                        index_col=0)
print(test_csv)
print(test_csv.shape) # 6493, 8

######결측치 처리######
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())
print(train_csv.shape) # 10886, 11

######train_csv 데이터에서 x와 y를 분리

#2. 모델 구성
print(type(train_csv))
x = train_csv.drop(['casual','registered','count'], axis=1)
y = train_csv['count']

print(np.min(x), np.max(x)) # x의 최소값 / (0.0 711.0)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=456
)

scaler = MaxAbsScaler()
scaler.fit(x_train) # x_train만큼 범위 잡아라
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)
print(np.min(x_test), np.max(x_test))

print(x_train.shape, x_test.shape)
print(y_train, y_test.shape)

######train_csv 데이터에서 x와 y를 분리

# model = Sequential()
# model.add(Dense(10, activation='relu', input_dim=8))
# model.add(Dense(5, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(5, activation='relu'))
# model.add(Dense(1))

input1 = Input(shape=(8,))
dense1 = Dense(10, activation='relu')(input1)
dense2 = Dense(5, activation='relu')(dense1)
dense3 = Dense(20, activation='relu')(dense2)
dense4 = Dense(10, activation='relu')(dense3)
dense5 = Dense(5, activation='relu')(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=20, mode='min',
                   verbose=1, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8,
                 validation_split=0.2, verbose=1, callbacks=[es])

print("=========발로스=========")
print(hist.history['val_loss'])
print("=========발로스=========")



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 :', r2)

def RMSE(y_test, y_predict): #재사용 할 때 함수를 쓴다/ RMSE라는 함수를 정의 할거야
    return np.sqrt(mean_squared_error(y_test, y_predict)) #sqrt 루트를 씌우는 놈이다
rmse = RMSE(y_test, y_predict) # 정의한 RMSE 사용
print("RMSE : ", rmse)



y_submit = model.predict(test_csv)
print(y_submit)

submission = pd.read_csv(path + 'samplesubmission.csv', index_col=0)
print(submission)
submission['count'] = y_submit
print(submission)

submission.to_csv(path_save + 'submit_0313_1203_MA.csv')

'''
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.plot(hist.history['loss'], marker='.', c='red', label='로스')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='발로스')
plt.title('캐글_자전거')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend()
plt.grid()
plt.show()
'''