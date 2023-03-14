import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
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
print(train_csv.isnull().sum())
print(train_csv.shape)

#2. 모델구성
print(type(train_csv))
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

print(np.min(x), np.max(x)) # x의 최소값 / (0.0 711.0)
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
print(np.min(x), np.max(x))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7, random_state=400
)

print(x_train.shape, x_test.shape)

input1 = Input(shape=(8,))
dense1 = Dense(10, activation='linear')(input1)
dense2 = Dense(80, activation='relu')(dense1)
dense3 = Dense(50, activation='relu')(dense2)
dense4 = Dense(100, activation='relu')(dense3)
dense5 = Dense(100, activation='relu')(dense4)
dense6 = Dense(50, activation='relu')(dense5)
dense7 = Dense(30, activation='relu')(dense6)
output1 = Dense(1, activation='sigmoid')(dense7)
model = Model(inputs=input1, outputs=output1)

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


# y_submit = np.round(model.predict(test_csv))
# print(y_submit)

# submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
# print(submission)
# submission['Outcome'] = y_submit
# print(submission)

# submission.to_csv(path_save + 'submit_0313_1127.csv')