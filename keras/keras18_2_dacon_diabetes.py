import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import pandas as pd

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
print(train_csv.isnull().sum())
print(train_csv.shape)

#2. 모델구성
print(type(train_csv))
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=400
)

print(x_train.shape, x_test.shape)

model = Sequential()
model.add(Dense(10, activation='linear', input_dim=8))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam',
              metrics=['acc', 'mse'])
es = EarlyStopping(monitor='val_loss', patience=300, mode='min',
                   verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=5000, batch_size=4,
          validation_split=0.2,
          verbose=1, callbacks=[es]
          )

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results : ', results)

y_predict = np.round(model.predict(x_test))

from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)




y_submit = np.round(model.predict(test_csv))
print(y_submit)

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
print(submission)
submission['Outcome'] = y_submit
print(submission)

submission.to_csv(path_save + 'submit_0312_2323.csv')