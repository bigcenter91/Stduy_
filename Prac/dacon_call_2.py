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

x = train_csv.drop(['전화해지여부'], axis=1)
y = train_csv['전화해지여부']


# 1. 데이터 전처리
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=2022)

# 데이터 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

# 레이블 원핫인코딩
y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)

# 2. 모델 구성
input_shape = x_train.shape[1]

model = Sequential([
    Input(shape=(input_shape, )),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

# 3. 컴파일, 훈련
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=32, callbacks=[es])


#4. 평가, 예측
result = model.evaluate(x_val, y_val)
print('results : ' , result)

y_predict = model.predict(x_val)
y_predict = np.argmax(y_predict, axis=-1)
y_test = np.argmax(y_val, axis=-1)

acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)

import matplotlib.pyplot as plt
plt.plot(hist.history['val_loss'], label='val_acc')
plt.show()