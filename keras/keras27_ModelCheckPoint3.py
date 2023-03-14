#1. save model과 비교
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
            filepath='./_save/MCP/keras27_3_ModelCheckPoint.hdf5' #거의 동일한 가중치가 적용된다
)


model.fit(x_train, y_train, epochs=1000, 
          callbacks=[es, mcp],
          validation_split=0.2) # 두개 이상은 리스트, 훈련시키는게 쟤말고 많이 있어서


model.save('./_save/MCP/keras27_3_save_model.h5')


#4. 평가, 예측

from sklearn.metrics import r2_score

print("=================== 1. 기본 출력 ===================")

loss = model.evaluate(x_test, y_test, verbose=0)
print("loss : ", loss )
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

print("=================== 2. load_model 출력 ===================")

model2 = load_model('./_save/MCP/keras27_3_save_model.h5')
loss = model2.evaluate(x_test, y_test, verbose=0)
print("loss : ", loss )
y_predict = model2.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

print("=================== 3. MCP 출력 ===================")

model3 = load_model('./_save/MCP/keras27_3_ModelCheckPoint.hdf5')
loss = model3.evaluate(x_test, y_test, verbose=0)
print("loss : ", loss )
y_predict = model3.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


#modelCheckPoint : 지점마다 저장한다

# =================== 1. 기본 출력 ===================
# loss :  31.45586395263672
# r2스코어 :  0.6198032760435302
# =================== 2. load_model 출력 ===================
# loss :  31.45586395263672
# r2스코어 :  0.6198032760435302
# =================== 3. MCP 출력 ===================
# loss :  31.447837829589844
# r2스코어 :  0.6199002732630958
