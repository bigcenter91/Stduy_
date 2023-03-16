from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
#노란줄 같은거 발생할 때 버전차이가 있을 수 있다
#소문자 함수 대문자 class
#파이선은 되도록 _로 잡는게 좋아

#1. 데이터
dataset = load_boston()

x = dataset.data
y = dataset['target']
print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, test_size=0.2
)

#2. 모델구성
model = Sequential()
model.add(Dense(20, activation='relu', input_dim=13))
model.add(Dense(20, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=10, batch_size=2,
                 validation_split=0.2, 
                 verbose=1) # 훈련을 하는 과정은 fit에 있다
'''
print("=================")
print(hist)
#<tensorflow.python.keras.callbacks.History object at 0x0000023A9A148A60>
print("=================")
print(hist.history)
print("=================")
print(hist.history['loss'])
print("=================")
print(hist.history['val_loss'])
print("=================")
'''


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'gothic'
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], marker='.', c='red', label='로스') # 순서대로 있을 땐 x를 명시하지 않아도 된다
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='발_로스')
plt.title('보스턴')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend()
plt.grid()
plt.show()


#loss보단 val_loss를 기준으로 보는게 낫다
#그래프모양과 과적합 모양을 보여주기 위해

