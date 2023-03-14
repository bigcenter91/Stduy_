# 저장할 때 평가결과값, 훈련시간 파일에 넣어줘


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. 데이터
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

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123,
)

scaler = MaxAbsScaler()


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 
print(np.min(x_test), np.max(x_test))


#2. 모델

input1 = Input(shape=(13,))
dense1 = Dense(30)(input1)
dense2 = Dense(20)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)

# model.save('./_save/keras26_1_save_model.h5') #모델파일 *.h5


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

import datetime
date = datetime.datetime.now()

print(date) # 2023-03-14 11:11:09.391441
date = date.strftime("%m%d_%H%M") #문자로 바꿔야 파일명에 들어갈거 아냐 / _는 그냥 문자다
print(date) # 0314_1116

filepath = './_save/MCP/keras27_4/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' #history에서 봤지?

# {epoch:04d}: 이 부분은 현재 epoch(에포크)의 값을 4자리로 0으로 채워진 문자열로 변환합니다. 예를 들어, 2번째 epoch일 경우 "0002"로 변환됩니다.

# epoch은 Keras가 실행 중인 현재 epoch 번호를 가리키는 변수입니다.
# :04d는 4자리의 10진수 정수를 표시하기 위한 포맷 코드입니다. 0은 빈 자리를 0으로 채웁니다.
# {val_loss:.4f}: 이 부분은 현재 검증 손실(validation loss) 값의 소수점 이하 4자리까지를 포함한 부동 소수점(floating-point number) 값을 문자열로 변환합니다.

# val_loss는 Keras가 실행 중인 현재 검증 손실 값을 가리키는 변수입니다.
# .4f는 소수점 이하 4자리까지 출력하기 위한 포맷 코드입니다. f는 부동 소수점 값을 표시하는 코드입니다.



from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='min',
                   verbose=1, 
                   #restore_best_weights=True /
                   #가장 좋은 지점이라고 하는데 train에서는 맞는건데 patience에서 밀린게 더 좋은 경우도 있다.
                   #데이터 보고 판단해야한다
                   )

mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
            verbose=1,
            save_best_only=True, #가장 좋은 지점에서 적용 시켜라
            filepath="".join([filepath, 'k27_', date, '_', filename]) # join 뭔가를 합진다 / 문자는 +가 되면 그냥 합쳐져
)


model.fit(x_train, y_train, epochs=100, 
          callbacks=[es, mcp],
          validation_split=0.2) # 두개 이상은 리스트, 훈련시키는게 쟤말고 많이 있어서



#4. 평가, 예측

from sklearn.metrics import r2_score

print("=================== 1. 기본 출력 ===================")

loss = model.evaluate(x_test, y_test, verbose=0)
print("loss : ", loss )
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


#modelCheckPoint : 지점마다 저장한다

