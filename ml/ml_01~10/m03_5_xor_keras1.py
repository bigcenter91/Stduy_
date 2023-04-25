# [실습] 1.0만들기

import numpy as np
from sklearn.svm import LinearSVC, SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score


#1. 데이터 
x_data = [[0,0],[0,1],[1,0],[1,1]]  #(4,2)
y_data = [0,1,1,0]                  #이진분류 -> sigmoid 

#2. 모델구성 
# model = LinearSVC()
# model = Perceptron()
# model = SVC()

model = Sequential()
model.add(Dense(1, input_dim=2, activation='linear'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='linear'))
model.add(Dense(1, activation='sigmoid'))


#3. 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['acc'])
model.fit(x_data,y_data, batch_size=1, epochs = 100)

#4. 평가, 예측 
y_predict = model.predict(x_data)

# results = model.score(x_data, y_data)
results = model.evaluate(x_data, y_data)
print("model_score:", results[1])
acc =  accuracy_score(y_data, np.round(y_predict)) 
print("accuracy_score:", acc)

###[주의]############################################################
# print(y_predict)
'''
[[0.49666756]
 [0.7068602 ]
 [0.3204532 ]
 [0.5353965 ]]
'''
#ValueError: Classification metrics can't handle a mix of binary and continuous targets
#acc =  accuracy_score(y_data, y_predict) 
#sigmoid, binary=> 소수점나오니까 에러뜸 따라서, round해줘야함 
#######################################################################

'''
model_score: 0.5
accuracy_score: 0.5
'''
'''
model_score: 1.0
accuracy_score: 1.0
'''