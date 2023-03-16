from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import r2_score, accuracy_score


datasets = load_breast_cancer()
x = datasets.data
y = datasets['target']

print(type(x)) # <class 'numpy.ndarray'>
print(x)


print(np.min(x), np.max(x)) # x의 최소값 / (0.0 711.0)
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
print(np.min(x), np.max(x)) # (0.0 1.0)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123,
)

#scaler = MinMaxScaler()
#scaler = StandardScaler() # MinMaxScaler, StandardScaler 둘 중 하나 / 하나로 모아줘야하면 스탠다드, 그 반대는 민맥스
scaler = MaxAbsScaler()
scaler.fit(x_train) # x_train만큼 범위 잡아라
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
print(np.min(x_test), np.max(x_test))# fit할 필요가 없다/ x_train의 범위에 맞춰서한다 이후 변한만 해주면 된다 
#결과 좋은 거 선택해서 사용하면 된다


#2. 모델
model = Sequential()
model.add(Dense(50, input_dim=30))
model.add(Dense(100, activation='linear'))
model.add(Dense(40, activation='linear'))
model.add(Dense(20, activation='linear'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=50)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results : ', results)

y_predict = np.round(model.predict(x_test))

acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)