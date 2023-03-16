from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical


#1. 데이터
path = './_data/dacon_call/'
path_save = './_save/dacon_call/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0)

print(train_csv) # 30200, 13

test_csv = pd.read_csv(path + 'test.csv',
                       index_col=0)

print(test_csv) # 12943, 12

# print(train_csv.isnull().sum())
# print(train_csv.info())
# train_csv = train_csv.dropna()

print(type(train_csv)) # <class 'pandas.core.frame.DataFrame'>

x = train_csv.drop(['전화해지여부'], axis=1)
y = train_csv['전화해지여부']


print(x.shape, y.shape) # (30200, 12) (30200,)
print(x)
print(y)


print('y의 라벨값 : ', np.unique(y)) # [0 1]


# y = to_categorical(y)
# print(y.shape) # 5497, 10
# print(x.shape, y.shape)

# y = y[:, 3:]
#y = np.delet(y, 0, axis=1)
# 하지만, 이 방법은 모든 라벨 값이 순서대로 0, 1, 2, 3, ... 으로 
# 지정된 경우에만 제대로 작동합니다. 즉, 이 방법은 라벨 값의 범위가 
# 0부터 시작하는 경우에만 사용할 수 있습니다.

# 따라서, 이 코드가 문제를 일으키는 원인 중 하나가 됩니다. 해당 코드 대신에
#  y = y[:, 3:]와 
#  같이 인덱싱을 사용하여 불필요한 열을 삭제하는 것이 더욱 안전한 방법입니다.


print(y)
print(y.shape)

print(np.min(x), np.max(x))


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=444
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)


print(x_train.shape, x_test.shape)


#2. 모델 구성

input1 = Input(shape=(12, ))
dense1 = Dense(50, activation='relu')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(100, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(40, activation='relu')(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(50, activation='relu')(drop3)
drop4 = Dropout(0.3)(dense4)
output1 = Dense(1, activation='sigmoid')(drop4)
model = Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=20, mode='min',
                   verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=1000, batch_size=128,
          validation_split=0.2,
          verbose=1, callbacks=[es])


#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('results : ' , result)

y_predict = np.round(model.y_predict(x_test))

acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)



# #내보내기
# submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
# y_submit = model.predict(test_csv)

# y_submit = np.argmax(y_submit, axis=1)
# y_submit+=3 # 라벨 값하고 맞춰줄려고
# #print(y_submit)

# submission['quality'] = y_submit

# submission.to_csv(path_save + 'submit_0314_1758.csv')