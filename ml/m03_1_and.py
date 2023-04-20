#And : (곱하기) 두번다 참(1)이어야 참 (한번이라도 거짓(0)이라면, 거짓) 
#OR : (더하기   한번이라도 참(1)이면 참이다 
#Xor : and,or 조합해서 만듦 / 같으면0, 다르면1
 

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#1. 데이터 
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,0,0,1]

#2. 모델구성 
model = LinearSVC()

#3. 훈련
model.fit(x_data,y_data)

#4. 평가, 예측 
y_predict = model.predict(x_data)

results = model.score(x_data, y_data)
print("model_score:", results)

acc = accuracy_score(y_data, y_predict)
print("accuracy_score:", acc)

# model_score: 1.0
# accuracy_score: 1.0