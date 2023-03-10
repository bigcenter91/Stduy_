from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

#1.데이터
datasets = fetch_covtype()
#print(datasets.DESCR)
#print(datasets.feature_names)

x = datasets.data
y = datasets['target']
#print(x.shape, y.shape) # (581012, 54) (581012,)
#print(x)
#print(y) # [5 5 2 ... 3 3 3]

#print('y의 라벨값 : ', np.unique(y)) # y의 라벨값 :  [1 2 3 4 5 6 7]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape) # (581012, 8)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,
    test_size=0.8,
    stratify=y,
)

print(y_train)
print(np.unique(y_train, return_counts=True))



#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=54))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(7, activation='softmax'))
#히든에서 어느정도 커져야 잘먹힌다

model.summary()


#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              #loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

import time # 시간 나오게 하는 코드
start_time = time.time()

model.fit(x_train, y_train, epochs=1, batch_size=300, #batch가 크면 터진다 자꾸 터지면 배치를 줄이면 된다
          validation_split=0.2,
          verbose=1,
          )
end_time = time.time() 

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results)
print('loss : ', results[0])
print('acc : ', results[1])

print("걸린시간 : ", round(end_time - start_time, 2))


'''
y_pred = model.predict(x_test)

y_test_acc = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

acc = accuracy_score(y_test_acc, y_pred)
print('accuracy_acc : ', acc)
'''