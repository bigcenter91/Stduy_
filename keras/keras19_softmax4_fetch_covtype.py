from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow.keras.utils import to_categorical

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


y = to_categorical(y, num_classes=0)
print(y.shape) # (581012, 8)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,
    test_size=0.2,
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

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=1000,
          validation_split=0.3,
          verbose=1,
          )


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results)
print('loss : ', results[0])
print('acc : ', results[1])

y_pred = model.predict(x_test)

y_test_acc = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

acc = accuracy_score(y_test_acc, y_pred)
print('accuracy_acc : ', acc)
