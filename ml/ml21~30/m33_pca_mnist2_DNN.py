# import numpy as np
# from tensorflow.keras.datasets import mnist
# from sklearn.decomposition import PCA
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
# import pandas as pd
# from sklearn.metrics import accuracy_score

# # 1. 데이터 전처리
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(60000, -1)
# x_test = x_test.reshape(10000, -1)

# y_train = np.array(pd.get_dummies(y_train))
# y_test = np.array(pd.get_dummies(y_test))

# cum_list = ['154', '331', '486', '713']
# cum_name_list = ['pca0.95','pca0.99','pca0.999','pca1']

# result_list = []
# for i in range(len(cum_list)):
#     pca = PCA(n_components=int(cum_list[i]))
#     x_train_pca = pca.fit_transform(x_train)
#     x_test_pca = pca.transform(x_test)

#     # 모델 구성
#     model = Sequential()
#     model.add(Dense(64, input_shape=(int(cum_list[i]),)))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(16, activation='relu'))
#     model.add(Dense(10, activation='softmax'))

#     # 모델 컴파일
#     model.compile(optimizer='adam', loss='categorical_crossentropy')

#     # 모델 훈련
#     hist = model.fit(x_train_pca, y_train, 
#                      epochs=50, 
#                      batch_size=5000, 
#                      verbose=1,
#                      validation_split=0.2)

#     # 모델 평가
#     result = model.evaluate(x_test_pca, y_test)
#     result_list.append(result)
    
#     y_predict = model.predict(x_test_pca)
#     acc = accuracy_score(np.argmax(y_test,axis=1), np.argmax(y_predict,axis=1))    
# for i in range(len(cum_list)):
#     print(f"{cum_name_list[i]} : {result_list[i]}")
#     print(f"{cum_name_list[i]} : {acc:.4f}")



#1. 나의 최고의 CNN : 0.2951
#2. 나의 최고의 DNN : 0.9714
# pca0.95 : 0.8897522687911987
# pca0.95 : 0.8645
# pca0.99 : 0.67267906665802
# pca0.99 : 0.8645
# pca0.999 : 0.747768759727478
# pca0.999 : 0.8645
# pca1 : 0.7328537702560425
# pca1 : 0.8645

# for i in cum_list:
#     pca = PCA(n_components=int(i))
#     x_train_pca = pca.fit_transform(x_train)
#     x_test_pca = pca.transform(x_test)
#     #print(x_train_pca.shape, x_test_pca.shape)
    
#     # 2. 모델 구성
#     model = Sequential()
#     model.add(Dense(64, input_shape=(int(i),)))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(16, activation='relu'))
#     model.add(Dense(10, activation='softmax'))
#     # 3. 모델 컴파일
#     model.compile(optimizer='adam', loss='categorical_crossentropy')

#     # 4. 모델 훈련
#     hist = model.fit(x_train_pca, y_train, 
#                      epochs=10, 
#                      batch_size=5000, 
#                      verbose=1,
#                      validation_split=0.2)
    
#     # 5. 모델 평가
#     result = model.evaluate(x_test_pca, y_test)
#     print('result : ', result)

#     y_predict = model.predict(x_test_pca)
#     acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1))
#     print(f'acc for pca {i} : {acc}')


import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
import pandas as pd
from sklearn.metrics import accuracy_score

# 1. 데이터 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, -1)
x_test = x_test.reshape(10000, -1)

y_train = np.array(pd.get_dummies(y_train))
y_test = np.array(pd.get_dummies(y_test))

cum_list = ['154', '331', '486', '713']
cum_name_list = ['pca0.95','pca0.99','pca0.999','pca1']

for i in range(len(cum_list)):
    pca = PCA(n_components=int(cum_list[i]))
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    # 모델 구성
    model = Sequential()
    model.add(Dense(64, input_shape=(int(cum_list[i]),)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # 모델 컴파일
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # 모델 훈련
    hist = model.fit(x_train_pca, y_train, 
                     epochs=5, 
                     batch_size=5000, 
                     verbose=1,
                     validation_split=0.2)

    # 모델 평가
    result = model.evaluate(x_test_pca, y_test)
    
    y_predict = model.predict(x_test_pca)
    acc = accuracy_score(np.argmax(y_test,axis=1), np.argmax(y_predict,axis=1))    

    print(f"{cum_name_list[i]} : {result}")
    print(f"{cum_name_list[i]} : {acc:.4f}")
