from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
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
                   verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
            verbose=1,
            save_best_only=True, #가장 좋은 지점에서 적용 시켜라
            filepath='./_save/MCP/keras27_ModelCheckPoint.hdf5' #거의 동일한 가중치가 적용된다
)

model.fit(x_train, y_train, epochs=1000, 
          callbacks=[es, mcp],
          validation_split=0.2) # 두개 이상은 리스트, 훈련시키는게 쟤말고 많이 있어서

#val_loss improved from 29.55315 to 29.27658, saving model to ./_save/MCP\keras27_ModelCheckPoint.hdf5
#개선 됐으니까 save 되겠지? / 10번째 개선이 없을 때 Early Stopping 걸린다


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss )
