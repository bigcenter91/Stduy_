import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler,Normalizer
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Conv2D, SimpleRNN, Concatenate, concatenate, Dropout, Bidirectional, Flatten, MaxPooling2D, Input, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

predict_date = 31

def split_x(dt, ts):
    a = []
    for i in range(len(dt)-ts-predict_date):
        b = dt[i:(i+ts)]
        a.append(b)
    return np.array(a)

def f1(x):
    a = []
    for i in range(31):
        b = x.iloc[3*i+2]
        a.append(b)
    return pd.DataFrame(np.array(a).reshape(-1, 1)).replace(',','', regex=True).astype(int)

path = "d:/study_data/_data/project_p/"
save_path = "d:/study_data/_save/project_p/"

for j in range(3):
    if j==0:
        j='g'
        k='광주'
    elif j==1:
        j='j'
        k='전주'
    else:
        j='m'
        k='목포'
    for i in range(5):
            globals()['weather_'+j+str(i+18)] = pd.read_csv(path + str(i+18)+'08_'+k+'날씨.csv', index_col=0, encoding='cp949')

for i in range(5):
    globals()['express_g'+str(i+18)] = f1(pd.read_csv(path + str(i+18)+'08_광주고속도로교통량.csv', 
                                                      encoding='cp949').drop(['일자', '입출구'], axis=1))


weather_g = pd.concat([weather_g18, weather_g19, weather_g20, weather_g21, weather_g22])
weather_j = pd.concat([weather_j18, weather_j19, weather_j20, weather_j21, weather_j22])
weather_m = pd.concat([weather_m18, weather_m19, weather_m20, weather_m21, weather_m22])
express_g_x = pd.concat([express_g18, express_g19, express_g20, express_g21, express_g22])

weather_g_x = np.array(weather_g.drop(['지점명','일시','10분 최다 강수량(mm)', '일강수량(mm)', '1시간 최다강수량(mm)'], axis=1))
weather_g_y = np.array(weather_g['일강수량(mm)'])
weather_j_x = np.array(weather_j.drop(['지점명','일시','10분 최다 강수량(mm)', '일강수량(mm)', '1시간 최다강수량(mm)'], axis=1))
weather_j_y = np.array(weather_j['일강수량(mm)'])
weather_m_x = np.array(weather_m.drop(['지점명','일시','10분 최다 강수량(mm)', '일강수량(mm)', '1시간 최다강수량(mm)'], axis=1))
weather_m_y = np.array(weather_m['일강수량(mm)'])

print(weather_g_x)

weather_g_y = np.array(pd.DataFrame(weather_g_y).fillna(0))
weather_j_y = np.array(pd.DataFrame(weather_j_y).fillna(0))
weather_m_y = np.array(pd.DataFrame(weather_m_y).fillna(0))

scaler = MinMaxScaler()
weather_g_x = scaler.fit_transform(weather_g_x)
weather_j_x = scaler.transform(weather_j_x)
weather_m_x = scaler.transform(weather_m_x)

timesteps = 5

weather_g_x_split = split_x(weather_g_x, timesteps)
weather_j_x_split = split_x(weather_j_x, timesteps)
weather_m_x_split = split_x(weather_m_x, timesteps)
express_g_x_split = split_x(express_g_x, timesteps)

weather_g_y_split = weather_g_y[(timesteps+predict_date):]
weather_j_y_split = weather_j_y[(timesteps+predict_date):]
weather_m_y_split = weather_m_y[(timesteps+predict_date):]

split = int(np.round(weather_g_x_split.shape[0]*0.7))
weather_g_x_split_test = weather_g_x_split[split:]
weather_j_x_split_test = weather_g_x_split[split:]
weather_m_x_split_test = weather_g_x_split[split:]
express_g_x_split_test = express_g_x_split[split:]

weather_g_y_split_test = weather_g_y_split[split:]
weather_j_y_split_test = weather_g_y_split[split:]
weather_m_y_split_test = weather_g_y_split[split:]

# 2. 모델 구성
# #2.1 모델1
input1 = Input(shape=(timesteps, weather_g_x.shape[1]))
dense11 = LSTM(16, name='gwangju1')(input1)
dense12 = Dense(64, activation='relu', name='gwangju2')(dense11)
drop11 = Dropout(0.125)(dense12)
dense13 = Dense(64, activation='relu', name='gwangju3')(drop11)
drop12 = Dropout(0.125)(dense13)
dense14 = Dense(64, activation='relu', name='gwangju4')(drop12)
drop13 = Dropout(0.125)(dense14)
dense15 = Dense(64, activation='relu', name='gwangju5')(drop13)
drop14 = Dropout(0.125)(dense15)
output1 = Dense(32, name='output1')(drop14)

#2.2 모델2
input2 = Input(shape=(timesteps, weather_j_x.shape[1]))
dense21 = LSTM(16, name='jeonju1')(input2)
dense22 = Dense(64, activation='relu', name='jeonju2')(dense21)
drop21 = Dropout(0.125)(dense22)
dense23 = Dense(64, activation='relu', name='jeonju3')(drop21)
drop22 = Dropout(0.125)(dense23)
dense24 = Dense(64, activation='relu', name='jeonju4')(drop22)
output2 = Dense(64, name='output2')(dense24)

#2.2 모델3
input3 = Input(shape=(timesteps, weather_m_x.shape[1]))
dense31 = LSTM(16, name='mokpo1')(input3)
dense32 = Dense(128, activation='relu', name='mokpo2')(dense31)
drop31 = Dropout(0.125)(dense32)
dense33 = Dense(128, activation='relu', name='mokpo3')(drop31)
drop32 = Dropout(0.125)(dense33)
dense34 = Dense(256, activation='relu', name='mokpo4')(drop32)
output3 = Dense(23, name='output3')(dense34)

#2.3 모델4
input4 = Input(shape=(timesteps, express_g_x.shape[1]))
dense41 = LSTM(16, name='express1')(input4)
dense42 = Dense(128, activation='relu', name='express2')(dense41)
drop41 = Dropout(0.125)(dense42)
dense43 = Dense(128, activation='relu', name='express3')(drop41)
drop42 = Dropout(0.125)(dense43)
dense44 = Dense(256, activation='relu', name='express4')(drop42)
output4 = Dense(23, name='output4')(dense44)


#2.4 머지
merge1 = Concatenate(name='mg1')([output1, output2, output3, output4])
merge2 = Dense(128, activation='relu', name='mg2')(merge1)
merge3 = Dense(64, activation='relu', name='mg3')(merge2)
hidden_output = Dense(32, name='last')(merge3)

#2.5 분기1
bungi1 = Dense(64, activation='relu')(hidden_output)
bungi1 = Dense(32)(bungi1)
last_output1 = Dense(1, name='last1')(bungi1)

#2.6 분기2
bungi2 = Dense(64, activation='relu')(hidden_output)
bungi2 = Dense(32)(bungi2)
last_output2 = Dense(1, activation='linear', name='last2')(bungi2)

#2.7 분기3
bungi3 = Dense(64, activation='relu')(hidden_output)
bungi3 = Dense(32)(bungi3)
last_output3 = Dense(1, activation='linear', name='last3')(bungi3)
model = Model(inputs=[input1, input2, input3, input4], outputs=[last_output1, last_output2, last_output3])

model.summary()


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)

hist = model.fit([weather_g_x_split, weather_j_x_split, weather_m_x_split, express_g_x_split], 
          [weather_g_y_split, weather_j_y_split, weather_m_y_split], 
           epochs=1000, batch_size=64, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate([weather_g_x_split_test, weather_j_x_split_test, weather_m_x_split_test, express_g_x_split_test], 
                      [weather_g_y_split_test, weather_j_y_split_test, weather_m_y_split_test])

print('loss : ', loss)

for i in range(predict_date):
    weather_g_x_predict = weather_g_x[(len(weather_g_x)-predict_date-timesteps+1+i):(len(weather_g_x)-predict_date+1+i)]
    weather_g_x_predict = weather_g_x_predict.reshape(1, timesteps, weather_g_x.shape[1])

    weather_j_x_predict = weather_g_x[(len(weather_j_x)-predict_date-timesteps+1+i):(len(weather_j_x)-predict_date+1+i)]
    weather_j_x_predict = weather_g_x_predict.reshape(1, timesteps, weather_j_x.shape[1])

    weather_m_x_predict = weather_m_x[(len(weather_m_x)-predict_date-timesteps+1+i):(len(weather_m_x)-predict_date+1+i)]
    weather_m_x_predict = weather_m_x_predict.reshape(1, timesteps, weather_m_x.shape[1])

    express_g_x_predict = express_g_x[(len(express_g_x)-predict_date-timesteps+1+i):(len(express_g_x)-predict_date+1+i)]
    express_g_x_predict = np.array(express_g_x_predict).reshape(1, timesteps, express_g_x.shape[1])
    
    predict_result = model.predict([weather_g_x_predict, weather_j_x_predict, weather_m_x_predict, express_g_x_predict])
    print('2023.08.', (i+1),'일 광주의 강수량은 : ', np.round(predict_result[0], 2))



# plt.plot(hist.history['loss'], label='loss')
# plt.plot(hist.history['val_loss'], label='val_loss')
# plt.legend()
# plt.show()