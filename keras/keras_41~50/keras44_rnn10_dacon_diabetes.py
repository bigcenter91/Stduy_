import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, SimpleRNN
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. 데이터
path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0)

print(train_csv)
print(train_csv.shape) # (652, 9)

test_csv = pd.read_csv(path + 'test.csv',
                       index_col=0)

print(test_csv)
print(test_csv.shape) # 116, 8

print(train_csv.isnull().sum())
train_csv = train_csv.dropna()

print(type(train_csv))
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']
print(x.shape) # (652, 8)

x = np.array(x)
x = x.reshape(652,4,2)
test_csv = np.array(test_csv)
print(test_csv.shape)

test_csv = test_csv.reshape(-1,4,2)

#2. 모델 구성
model = Sequential()

model.add(SimpleRNN(100, input_shape=(6, 9)))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results : ', results)

y_predict = np.round(model.predict(x_test))

from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)


# y_submit = np.round(model.predict(test_csv))
# print(y_submit)

# submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
# print(submission)
# submission['Outcome'] = y_submit
# print(submission)

# submission.to_csv(path_save + 'submit_0313_1127.csv')