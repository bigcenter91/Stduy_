from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


#1. 데이터
path = './_data/ddarung/'
path_save = './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)

print(train_csv) 
print(train_csv.shape) # 1459, 10

test_csv = pd.read_csv(path + 'test.csv', index_col=0)

print(test_csv.shape) #(715, 9)


print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.shape) # (1328, 10)


x = train_csv.drop(['count'],axis=1)
y = train_csv['count']

print (x.shape)

x = np.array(x)
x = x.reshape(-1, 9, 1)
print(x.shape) # (1328, 9, 1)
print(test_csv.shape)
test_csv = np.array(test_csv)
test_csv = test_csv.reshape(-1, 9, 1)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7, random_state=333
)


print(np.unique(y))
print(x_train.shape, y_train.shape) # (929, 9, 1) (929,) 
print(x_test.shape, y_test.shape) # (399, 9, 1) (399,)

print(type(train_csv)) # <class 'pandas.core.frame.DataFrame'>

#2. 모델 구성
model = Sequential()

model.add(SimpleRNN(32, input_shape=(9, 1)))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.summary()


#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

es = EarlyStopping(monitor = 'val_loss', patience = 20, mode = 'min',
                   verbose = 1, restore_best_weights=True)

model.fit(x_train,y_train, epochs = 1000, batch_size = 9,
          validation_split=0.2, verbose=1, callbacks =[es])



#4 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
print(y_predict.shape)
r2 = r2_score(y_test, y_predict)
print("r2 스코어 : ", r2)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

# y_submit = model.predict(test_csv)

# submission = pd.read_csv(path + 'submission.csv', index_col = 0)
# submission['count'] = y_submit
# submission.to_csv(path_save + 'submit_0310_1658.csv')

