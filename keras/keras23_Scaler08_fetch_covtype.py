from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1.데이터
datasets = fetch_covtype()
#print(datasets.DESCR)
#print(datasets.feature_names)

x = datasets.data
y = datasets['target']


print(type(x)) #
print(x)

print(np.min(x), np.max(x)) # x의 최소값 / (0.0 711.0)
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
print(np.min(x), np.max(x))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123,
)
scaler = MaxAbsScaler()
scaler.fit(x_train) # x_train만큼 범위 잡아라
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
print(np.min(x_test), np.max(x_test))

#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=54))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=1000)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss )