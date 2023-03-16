import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
datasets = load_digits()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target.reshape(-1,1)
print(x.shape, y.shape) # (1797, 64) (1797, 1)

print(x)
print(y)

print('y의 라벨값 : ', np.unique(y)) # [0 1 2 3 4 5 6 7 8 9]

encoder = OneHotEncoder()
y = encoder.fit_transform(y).toarray()
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
x, y, shuffle=True,
test_size=0.2,
stratify=y,
)

print(y_train)
print(np.unique(y_train, return_counts=True))

#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=64))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=8,
validation_split=0.2,
verbose=1,
)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results : ', results)

y_predict = np.round(model.predict(x_test))
acc = np.round(accuracy_score(y_test, y_predict))
print('acc : ', acc)