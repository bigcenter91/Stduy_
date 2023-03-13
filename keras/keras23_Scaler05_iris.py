import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


#1. 데이터
datasets = load_iris()
print(datasets.DESCR) #
print(datasets.feature_names) # 판다스


x = datasets.data
y = datasets['target']
# print(x.shape, y.shape) #(150,4) / (150,)
# print(x)
# print(y)
# print('y의 라벨값 : ', np.unique(y)) #y의 라벨값 :  [0 1 2]

print(np.min(x), np.max(x)) # 
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
print(np.min(x), np.max(x))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=333,
   )

scaler = MaxAbsScaler()
scaler.fit(x_train) # x_train만큼 범위 잡아라
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
print(np.min(x_test), np.max(x_test))


#2. 모델구성
model = Sequential()
model.add(Dense(50, activation='relu', input_dim=4))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax')) 

# 다중분류에서는 무조건 softmax 안바뀐다 / 출력값은 150, 3이된다 
#다중분류 문제는 마지막 레이아웃(out layer)에 y의 라벨 값의 노드를 적어준다
#마지막 히든 레이어 3은 y의 라벨 값의 갯수
# 0~1사이 한정 시키는 놈이다


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100)

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print("loss : ", loss )
