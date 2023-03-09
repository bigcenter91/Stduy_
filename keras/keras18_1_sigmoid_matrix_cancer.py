import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping

#1.데이터
datasets = load_breast_cancer()
#print(datasets)
# 사전 (:) = key와 value 한쌍으로 되어있다 {}로 되어있다
#ex) 

print(datasets.DESCR) # pandas : .describe()
print(datasets.feature_names) # pandas : columns()

x = datasets['data']
y = datasets.target

print(x.shape, y.shape) # (569, 30) (569,) = 569개의 스칼라, 벡터
#print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2
    )

#2. 모델구성

model = Sequential()
model.add(Dense(10, activation='relu', input_dim=30))
model.add(Dense(9, activation='linear'))
model.add(Dense(8, activation='linear'))
model.add(Dense(7, activation='linear'))
model.add(Dense(1, activation='sigmoid'))
#활성화 함수 중 0,1로 분류
#이진 분류는 그냥 sigmoid를 쓴다
#이진 분류는 마지막 레이어에 sigmoid를 준다
#loss는 binary_crossentropy를 입력해 준다 무조건이다 sigmoid/ binary_crossentropy

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy', 'mse'])#'acc', 'mean_squared_error']
es = EarlyStopping(monitor='val_loss', patience=5, mode='min',
                   verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=1000, batch_size=8,
          validation_split=0.2,
          verbose=1, callbacks=[es]
          )

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results : ', results)

y_predict = np.round(model.predict(x_test))
# print("============================")
# print(y_test[:5])
# print(y_predict[:5])
# print(np.round(y_predict[:5]))
# print("============================")

from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)

#loss: 0.1455 - accuracy: 0.9561 - mse: 0.0395
#loss 이후 두번째에는 metrix에 넣어놓은 두개가 출력된다

#2개 분류하면 2진 분류라 한다
#여러개에서 분류 다중분류 / 2진과 다중이 있다
#2개 이상은 리스트 []를 보면 리스트라고 받아들여야한다
#key, value = 딕셔너리