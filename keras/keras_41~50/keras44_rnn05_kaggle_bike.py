from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np

#1. 데이터
path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'


train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(train_csv.shape) # (10886, 11)
print(test_csv.shape) # (6493, 8)

x = train_csv.drop(['count','casual','registered'], axis = 1)
y = train_csv['count']

x = np.array(x)
x = x.reshape(10886, 8, 1)

print(x.shape)

test_csv = np.array(test_csv)
test_csv = test_csv.reshape(-1, 8, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123,
)


#2. 모델 구성
model = Sequential()

model.add(SimpleRNN(50, input_shape=(8, 1)))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping(monitor='val_loss', patience=5,
                   mode='min', verbose=1, restore_best_weights=True,
                   )
model.fit(x_train,y_train, epochs=100, batch_size=8,
          validation_split=0.2, verbose=1, callbacks =[es])

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 스코어 : ", r2)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

# y_submit = model.predict(test_csv)
# submission = pd.read_csv(path + 'sampleSubmission.csv', index_col = 0)
# submission['count'] = y_submit
# submission.to_csv(path_save + 'submit_0310_1658.csv')