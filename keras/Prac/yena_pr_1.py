import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Conv1D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1 데이터
path = './_data/_kaggle_jena/'
path_save = './_save/kaggle_jena/'

datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv',index_col=0)

x = datasets.drop(['T (degC)'],axis=1)
y = datasets['T (degC)']

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,shuffle=False)
x_test,x_predict,y_test,y_predict = train_test_split(x_test,y_test,train_size=2/3,shuffle=False)
print(x_train.shape,y_train.shape) # (294385, 13) (294385,)

timesteps = 10

def split_x(datasets, timesteps): # split_x라는 함수를 정의
    aaa = [] # aaa 에 빈칸의 리스트를 만든다.
    for i in range(len(datasets) - timesteps):
        subset = datasets[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

x_train_split = split_x(x_train,10)
x_test_split = split_x(x_test,timesteps)
x_predict_split = split_x(x_predict,timesteps)

y_train_split = y_train[timesteps:]
print(y_train_split)
y_test_split = y_test[timesteps:]
y_predict_split = y_predict[timesteps:]
print(x_train.shape,y_train.shape) # (294385, 13) (294385,)
print(np.unique(y_train))

#2 모델
model = load_model('./_save/MCP/kaggle_jena/kaggle_jena_0326_0154.h5')

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

#4 평가, 예측
loss = model.evaluate(x_test_split,y_test_split)
print('loss : ', loss)

predict = model.predict(x_predict_split)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(predict,y_predict_split)
print("RMSE : ", rmse)

print(predict[-2:])