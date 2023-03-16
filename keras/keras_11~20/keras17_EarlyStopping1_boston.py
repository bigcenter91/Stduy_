from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

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

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, mode='min',
                   verbose=1, restore_best_weights=True) # restore_best_weights 디폴트가 false 

#loss보다 val_loss가 낫다 / 갱신되면 멈춘다
#r2가 된다면 최대값을 한다
#min, max 말고 auto라는 놈이 있다

hist = model.fit(x_train, y_train, epochs=500, batch_size=3,
                 validation_split=0.2, 
                 verbose=1, 
                 callbacks=[es]) # 훈련을 하는 과정은 fit에 있다




'''
print("=================")
print(hist)
#<tensorflow.python.keras.callbacks.History object at 0x0000023A9A148A60>
print("=================")
print(hist.history)
print("=================")
print(hist.history['loss'])
'''
print("===========발로스======")
print(hist.history['val_loss'])
print("===========발로스======")

#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)






import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
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

#4/4 loss값 : 끊기 전지점의 값의 w값으로 loss 값을 구함