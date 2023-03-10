from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
#위 네가지 스케일러 제일 많이 쓴다.

datasets = load_boston()
x = datasets.data
y = datasets['target']

print(type(x)) # <class 'numpy.ndarray'>
print(x)
#연산은 실수에 최적화 되어있다
#[0,1,2,.......백만] > 0~1로 바꿔준다 이렇게 바꿔주는걸 '정규화' 작업이라한다
#연산을 시킬려면 numpy를 사용 부동소수점에 최적화 되어있으니 / normalization
#y=ax+b 스케일링은 x에만 해당된다
#성능이 좋아질 수도 있지만 안좋을 수도 있다 아닐 경우 다른 기법을 쓰면 된다 / 0~1사이로 만들어주는것
#최대값으로 나눠버린다

print(np.min(x), np.max(x)) # x의 최소값 / (0.0 711.0)
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
print(np.min(x), np.max(x)) # (0.0 1.0)
#훈련데이터만 정규화한다 0~1사이로
#테스트 데이터 정규화 하는데 훈련데이터를 잡고 훈련데이터의 비율에 맞춰서
#x_predict 미래를 알고싶은놈
#ex) 110-0 / 100-0 = 1.1 0은 훈련데이터에서 온 것
#분리한다음에 스케일링 한다

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123,
)

#scaler = MinMaxScaler()
#scaler = StandardScaler() # MinMaxScaler, StandardScaler 둘 중 하나 / 하나로 모아줘야하면 스탠다드, 그 반대는 민맥스
scaler = MaxAbsScaler 
scaler.fit(x_train) # x_train만큼 범위 잡아라
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
print(np.min(x_test), np.max(x_test))# fit할 필요가 없다/ x_train의 범위에 맞춰서한다 이후 변한만 해주면 된다 
#결과 좋은 거 선택해서 사용하면 된다


#2. 모델
model = Sequential()
model.add(Dense(1, input_dim=13))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss )