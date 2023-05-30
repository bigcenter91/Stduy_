import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import pandas as pd
from sklearn.metrics import accuracy_score
#1.데이터

#(x_train, y_train), (x_test,y_test) = mnist.load_data()
(x_train, y_train), (x_test,y_test) = mnist.load_data() #이렇게도 가능.

#print(__.shape) #메모리 할당이 됨. 변수를 특수문자를 사용해도됨.

#x = np.concatenate((x_train,x_test),axis = 0)
#x = np.append(x_train,x_test, axis= 0)
#print(x.shape)

#x = x.reshape(70000, -1)
# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

x_train = x_train.reshape(60000, -1)
x_test = x_test.reshape(10000, -1)

y_train=np.array(pd.get_dummies(y_train))
y_test=np.array(pd.get_dummies(y_test))

# x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
# print(x.shape)
cum_list = ['154','331','486','713']

# #-> 쫙피면 70000, 784의 컬럼으로 볼 수 있음.
for i in enumerate(cum_list):
    pca = PCA(n_components=int(i)) # [154,331,486,713]
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)#print(x.shape)
    print(x_train.shape, x_test.shape)
    
    # 2. 모델구성
    model = Sequential()
    model.add(Dense(64, input_shape=(int(i),)))
    # model.add(Dense(64, input_shape=(28*28,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(100, activation='softmax'))
    #3. 훈련
    hist = model.fit(x_train, y_train, 
                     epochs=10, 
                     batch_size=5000, 
                     verbose=1,
                     validation_split=0.2)
    
    # 4. 평가, 예측
    result = model.evaluate(x_test, y_test)
    print('result : ', result)

    y_predict = model.predict(x_test)
    acc = accuracy_score(np.argmax(y_test,axis=1), np.argmax(y_predict,axis=1))
    print(f'acc for pca {i+1} : {acc}')

# pca_EVR = pca.explained_variance_ratio_ #EVR임
# cumsum = np.cumsum(pca_EVR)
# #print(cumsum)

# print(np.argmax(cumsum >= 0.95) + 1) #154 +1을한 이유 0부터 시작하기 위해서
# print(np.argmax(cumsum >= 0.99) + 1) #331 +1을한 이유 0부터 시작하기 위해서
# print(np.argmax(cumsum >= 0.999) + 1) #486 +1을한 이유 0부터 시작하기 위해서
# print(np.argmax(cumsum >= 1.0) + 1) #712 +1을한 이유 0부터 시작하기 위해서
#########################################################################################
########################################실습#############################################
#########################################################################################
#모델 맹그러서 비교

#                         acc
#1. 나의 최고의 CNN :   0.00000
#2. 나의 최고의 DNN :   0.00000
#3. pca 0.95       :
#4. pca 0.99       :
#5. pca 0.999      :
#6. pca 1.0        :