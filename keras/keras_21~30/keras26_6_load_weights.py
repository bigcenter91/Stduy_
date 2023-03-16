from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

datasets = load_boston()
x = datasets.data
y = datasets['target']

print(type(x)) # <class 'numpy.ndarray'>
print(x)


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
scaler = MaxAbsScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)

x_train = scaler.fit_transform(x_train) # 위에 두줄과 같은 한줄이다
x_test = scaler.transform(x_test) 
print(np.min(x_test), np.max(x_test))


#2. 모델

#model = load_model('./_save/keras26_3_save_model.h5')
input1 = Input(shape=(13,))
dense1 = Dense(30)(input1)
dense2 = Dense(20)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)




#가중치까지 같이 저장되기 때문에 몇번을 돌려도 loss 값이 같다
#컴파일, 훈련을 다시 하면 변경된다
###########################################################
#model.save('./_save/keras26_1_save_model.h5') #초기 랜덤값의 웨이트만 저장되어있다
###########################################################

model.load_weights('./_save/keras26_5_save_weights1.h5')

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
#model.fit(x_train, y_train, epochs=10)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss )